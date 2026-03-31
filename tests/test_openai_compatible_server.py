"""
End-to-end tests for the OpenAI-compatible NeuroMem proxy.
"""

from __future__ import annotations

import json
import socketserver
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler

from fastapi.testclient import TestClient

from neuromem.integrations.config import NeuromemSettings
from neuromem.server.openai_compatible import create_app


class _ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class MockOpenAIHandler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, format, *args):  # pragma: no cover - silence test output
        return

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        payload = json.loads(raw_body or "{}")
        MockOpenAIHandler.requests.append(
            {
                "path": self.path,
                "payload": payload,
            }
        )

        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        messages = payload.get("messages", [])
        saw_memory_prompt = any(
            message.get("role") == "system"
            and "Relevant NeuroMem context:" in str(message.get("content", ""))
            for message in messages
        )
        last_user_message = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user_message = str(message.get("content", ""))
                break

        response_payload = {
            "id": "chatcmpl_mock",
            "object": "chat.completion",
            "created": 0,
            "model": payload.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "memory_context={0}; last_user={1}".format(
                            "yes" if saw_memory_prompt else "no",
                            last_user_message,
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        }

        encoded = json.dumps(response_payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class OpenAICompatibleServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        MockOpenAIHandler.requests = []
        cls.upstream_server = _ReusableTCPServer(("127.0.0.1", 0), MockOpenAIHandler)
        cls.upstream_thread = threading.Thread(
            target=cls.upstream_server.serve_forever,
            daemon=True,
        )
        cls.upstream_thread.start()
        cls.upstream_base_url = "http://127.0.0.1:{0}/v1".format(
            cls.upstream_server.server_address[1]
        )

    @classmethod
    def tearDownClass(cls):
        cls.upstream_server.shutdown()
        cls.upstream_server.server_close()
        cls.upstream_thread.join(timeout=5)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = "{0}/proxy.db".format(self.temp_dir.name)
        settings = NeuromemSettings.from_dict(
            {
                "chat_model": {
                    "provider": "openai_compatible",
                    "model": "mock-model",
                    "base_url": self.upstream_base_url,
                },
                "embedding_model": {
                    "provider": "local_lexical",
                    "model": "lexical-hash-512",
                },
                "memory": {
                    "db_path": db_path,
                    "retrieval_top_k": 5,
                    "auto_store": True,
                    "auto_associate": True,
                    "load_existing": True,
                },
                "server": {
                    "host": "127.0.0.1",
                    "port": 8080,
                },
            }
        )
        self.client = TestClient(create_app(settings=settings))

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_chat_completion_retrieves_prior_session_memory(self):
        first_response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [
                    {"role": "user", "content": "Remember alpha architecture decisions."}
                ],
                "neuromem": {"session_id": "project-alpha", "top_k": 5},
            },
        )
        self.assertEqual(first_response.status_code, 200)
        self.assertNotIn("neuromem", first_response.json())

        second_response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [
                    {"role": "user", "content": "What should you remember about alpha?"}
                ],
                "neuromem": {"session_id": "project-alpha", "top_k": 5},
            },
        )
        self.assertEqual(second_response.status_code, 200)
        body = second_response.json()
        self.assertIn("neuromem", body)
        self.assertGreaterEqual(body["neuromem"]["retrieved_count"], 1)
        retrieved_contents = [item["content"] for item in body["neuromem"]["retrieved_memories"]]
        self.assertTrue(
            any("Remember alpha architecture decisions." in content for content in retrieved_contents)
        )
        self.assertIn(
            "memory_context=yes",
            body["choices"][0]["message"]["content"],
        )

    def test_memory_search_is_strictly_session_scoped(self):
        for session_id, content in [
            ("project-alpha", "alpha-only roadmap"),
            ("project-beta", "beta-only roadmap"),
        ]:
            response = self.client.post(
                "/v1/memory/records",
                json={
                    "content": content,
                    "session_id": session_id,
                    "memory_type": "semantic",
                },
            )
            self.assertEqual(response.status_code, 200)

        alpha_search = self.client.post(
            "/v1/memory/search",
            json={
                "query": "roadmap",
                "session_id": "project-alpha",
                "top_k": 10,
            },
        )
        self.assertEqual(alpha_search.status_code, 200)
        alpha_contents = [item["content"] for item in alpha_search.json()["data"]]
        self.assertIn("alpha-only roadmap", alpha_contents)
        self.assertNotIn("beta-only roadmap", alpha_contents)

        beta_search = self.client.post(
            "/v1/memory/search",
            json={
                "query": "roadmap",
                "session_id": "project-beta",
                "top_k": 10,
            },
        )
        self.assertEqual(beta_search.status_code, 200)
        beta_contents = [item["content"] for item in beta_search.json()["data"]]
        self.assertIn("beta-only roadmap", beta_contents)
        self.assertNotIn("alpha-only roadmap", beta_contents)

    def test_responses_api_uses_user_as_default_session_id(self):
        seed_response = self.client.post(
            "/v1/memory/records",
            json={
                "content": "alice session memory",
                "session_id": "alice",
                "memory_type": "semantic",
            },
        )
        self.assertEqual(seed_response.status_code, 200)

        response = self.client.post(
            "/v1/responses",
            json={
                "model": "mock-model",
                "input": "What do you remember for Alice?",
                "user": "alice",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["neuromem"]["session_id"], "alice")
        self.assertGreaterEqual(body["neuromem"]["retrieved_count"], 1)
        self.assertEqual(body["output_text"].startswith("memory_context=yes"), True)


if __name__ == "__main__":
    unittest.main()

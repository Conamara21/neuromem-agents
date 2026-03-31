"""
Protocol-level tests for the NeuroMem MCP server.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client


REPO_ROOT = Path("/home/conamara/Desktop/ai memory/neuromem-agents")


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout_seconds: float = 10.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for {host}:{port}")


class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = "{0}/mcp.db".format(self.temp_dir.name)
        self.config_path = "{0}/mcp.json".format(self.temp_dir.name)
        self.base_config = {
            "embedding_model": {
                "provider": "local_lexical",
                "model": "lexical-hash-512",
            },
            "memory": {
                "db_path": self.db_path,
                "retrieval_top_k": 5,
                "auto_store": True,
                "auto_associate": True,
                "load_existing": True,
            },
            "mcp": {
                "server_name": "neuromem-mcp-test",
                "transport": "stdio",
                "host": "127.0.0.1",
                "port": 8765,
                "streamable_http_path": "/mcp",
                "json_response": True,
                "stateless_http": True,
                "log_level": "ERROR",
            },
        }

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

    def _write_config(self, payload):
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    async def test_stdio_transport_exposes_tools_resources_and_prompts(self):
        self._write_config(self.base_config)

        server = StdioServerParameters(
            command=str(REPO_ROOT / "venv/bin/python"),
            args=["-m", "neuromem.mcp.server", "--config", self.config_path, "--transport", "stdio"],
            cwd=REPO_ROOT,
            env=dict(os.environ),
        )

        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                tools = await session.list_tools()
                tool_names = {tool.name for tool in tools.tools}
                self.assertTrue(
                    {
                        "create_memory",
                        "search_memory",
                        "list_memories",
                        "associate_memories",
                        "get_memory_stats",
                        "get_observability_metrics",
                        "consolidate_memory",
                    }.issubset(tool_names)
                )

                created = await session.call_tool(
                    "create_memory",
                    {
                        "content": "stdio alpha memory",
                        "session_id": "alpha",
                        "memory_type": "semantic",
                    },
                )
                self.assertFalse(created.isError)
                memory_id = created.structuredContent["result"]["id"]

                search = await session.call_tool(
                    "search_memory",
                    {"query": "alpha", "session_id": "alpha", "top_k": 5},
                )
                self.assertFalse(search.isError)
                self.assertGreaterEqual(search.structuredContent["result"]["count"], 1)

                listing = await session.call_tool(
                    "list_memories",
                    {"session_id": "alpha", "limit": 10},
                )
                self.assertEqual(listing.structuredContent["result"]["count"], 1)
                self.assertEqual(listing.structuredContent["result"]["items"][0]["id"], memory_id)

                resource = await session.read_resource("memory://records/{0}".format(memory_id))
                self.assertEqual(len(resource.contents), 1)
                self.assertIn("stdio alpha memory", resource.contents[0].text)

                observability = await session.call_tool("get_observability_metrics")
                self.assertFalse(observability.isError)
                self.assertGreaterEqual(
                    observability.structuredContent["result"]["counters"]["memory_search_total"],
                    1,
                )

                observability_resource = await session.read_resource("memory://stats/observability")
                self.assertEqual(len(observability_resource.contents), 1)
                self.assertIn('"memory_search_total"', observability_resource.contents[0].text)

                prompt = await session.get_prompt(
                    "memory_recall_query",
                    {"session_id": "alpha", "objective": "Summarize the alpha project."},
                )
                self.assertGreaterEqual(len(prompt.messages), 1)
                self.assertIn("alpha", prompt.messages[0].content.text.lower())

    async def test_streamable_http_transport_is_session_scoped(self):
        port = _pick_free_port()
        config = dict(self.base_config)
        config["mcp"] = dict(self.base_config["mcp"])
        config["mcp"]["transport"] = "streamable-http"
        config["mcp"]["port"] = port
        self._write_config(config)

        process = subprocess.Popen(
            [
                str(REPO_ROOT / "venv/bin/python"),
                "-m",
                "neuromem.mcp.server",
                "--config",
                self.config_path,
                "--transport",
                "streamable-http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(os.environ),
        )

        try:
            await asyncio.to_thread(_wait_for_port, "127.0.0.1", port, 10.0)
            async with streamable_http_client("http://127.0.0.1:{0}/mcp".format(port)) as (
                read_stream,
                write_stream,
                _get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    alpha = await session.call_tool(
                        "create_memory",
                        {
                            "content": "alpha streamable memory",
                            "session_id": "project-alpha",
                            "memory_type": "semantic",
                        },
                    )
                    beta = await session.call_tool(
                        "create_memory",
                        {
                            "content": "beta streamable memory",
                            "session_id": "project-beta",
                            "memory_type": "semantic",
                        },
                    )
                    self.assertFalse(alpha.isError)
                    self.assertFalse(beta.isError)

                    alpha_search = await session.call_tool(
                        "search_memory",
                        {"query": "memory", "session_id": "project-alpha", "top_k": 10},
                    )
                    self.assertFalse(alpha_search.isError)
                    alpha_items = alpha_search.structuredContent["result"]["items"]
                    alpha_contents = [item["content"] for item in alpha_items]
                    self.assertIn("alpha streamable memory", alpha_contents)
                    self.assertNotIn("beta streamable memory", alpha_contents)

                    stats = await session.read_resource("memory://stats/overview")
                    self.assertEqual(len(stats.contents), 1)
                    self.assertIn('"total_nodes"', stats.contents[0].text)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()

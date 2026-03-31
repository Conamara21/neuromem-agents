"""
Tests for the LlamaIndex NeuroMem adapter.
"""

from __future__ import annotations

import json
import tempfile
import unittest

from llama_index.core.llms.mock import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine

from neuromem.frameworks.llamaindex import (
    NeuroMemLlamaIndexRetriever,
    create_llamaindex_openai_like,
    create_llamaindex_query_engine,
)
from neuromem.integrations.config import load_settings
from neuromem.integrations.engine import MemoryAugmentedProxy


class LlamaIndexAdapterTests(unittest.TestCase):
    def test_retriever_returns_session_scoped_nodes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = "{0}/llamaindex.json".format(temp_dir)
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "embedding_model": {
                            "provider": "local_lexical",
                            "model": "lexical-hash-512",
                        },
                        "memory": {
                            "db_path": "{0}/llamaindex.db".format(temp_dir),
                            "retrieval_top_k": 5,
                        },
                    },
                    handle,
                )

            proxy = MemoryAugmentedProxy(load_settings(config_path))
            proxy.create_memory("alpha roadmap memory", session_id="alpha", memory_type="semantic")
            proxy.create_memory("beta roadmap memory", session_id="beta", memory_type="semantic")

            retriever = NeuroMemLlamaIndexRetriever(
                settings_path=config_path,
                session_id="alpha",
                top_k=5,
            )
            nodes = retriever.retrieve("roadmap")
            contents = [item.node.text for item in nodes]

            self.assertIn("alpha roadmap memory", contents)
            self.assertNotIn("beta roadmap memory", contents)
            self.assertEqual(nodes[0].node.metadata["memory_type"], "semantic")

    def test_openai_like_helper_defaults_to_chat_proxy_mode(self):
        llm = create_llamaindex_openai_like(
            model="test-model",
            base_url="http://127.0.0.1:8080/v1",
            api_key="neuromem-local",
        )

        self.assertEqual(llm.api_base, "http://127.0.0.1:8080/v1")
        self.assertEqual(llm.api_key, "neuromem-local")
        self.assertEqual(llm.is_chat_model, True)

    def test_query_engine_helper_wraps_retriever(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = "{0}/query-engine.json".format(temp_dir)
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "embedding_model": {
                            "provider": "local_lexical",
                            "model": "lexical-hash-512",
                        },
                        "memory": {
                            "db_path": "{0}/query-engine.db".format(temp_dir),
                            "retrieval_top_k": 5,
                        },
                    },
                    handle,
                )

            proxy = MemoryAugmentedProxy(load_settings(config_path))
            proxy.create_memory(
                "alpha launch plan and delivery schedule",
                session_id="alpha",
                memory_type="semantic",
            )

            retriever = NeuroMemLlamaIndexRetriever(
                settings_path=config_path,
                session_id="alpha",
                top_k=5,
            )
            query_engine = create_llamaindex_query_engine(
                retriever,
                llm=MockLLM(max_tokens=32),
            )
            response = query_engine.query("Summarize the alpha launch plan")

            self.assertIsInstance(query_engine, RetrieverQueryEngine)
            self.assertTrue(str(response).strip())


if __name__ == "__main__":
    unittest.main()

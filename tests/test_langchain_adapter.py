"""
Tests for the LangChain NeuroMem adapter.
"""

from __future__ import annotations

import json
import tempfile
import unittest

from neuromem.frameworks.langchain import NeuroMemRetriever
from neuromem.integrations.config import load_settings
from neuromem.integrations.engine import MemoryAugmentedProxy


class LangChainAdapterTests(unittest.TestCase):
    def test_retriever_returns_session_scoped_documents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = "{0}/langchain.json".format(temp_dir)
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "embedding_model": {
                            "provider": "local_lexical",
                            "model": "lexical-hash-512",
                        },
                        "memory": {
                            "db_path": "{0}/langchain.db".format(temp_dir),
                            "retrieval_top_k": 5,
                        },
                    },
                    handle,
                )

            proxy = MemoryAugmentedProxy(load_settings(config_path))
            proxy.create_memory("alpha roadmap memory", session_id="alpha", memory_type="semantic")
            proxy.create_memory("beta roadmap memory", session_id="beta", memory_type="semantic")

            retriever = NeuroMemRetriever(
                settings_path=config_path,
                session_id="alpha",
                top_k=5,
            )
            documents = retriever.invoke("roadmap")
            contents = [document.page_content for document in documents]

            self.assertIn("alpha roadmap memory", contents)
            self.assertNotIn("beta roadmap memory", contents)
            self.assertEqual(documents[0].metadata["memory_type"], "semantic")


if __name__ == "__main__":
    unittest.main()

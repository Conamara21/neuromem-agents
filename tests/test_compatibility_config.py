"""
Unit tests for compatibility-layer configuration.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from neuromem.integrations.config import ProviderConfig, load_settings


class CompatibilityConfigTests(unittest.TestCase):
    def test_file_settings_are_not_overridden_by_env_defaults(self):
        payload = {
            "chat_model": {
                "provider": "ollama",
                "model": "qwen2.5-coder",
                "base_url": "http://localhost:11435/v1",
            },
            "server": {
                "port": 9999,
            },
            "memory": {
                "db_path": "custom_proxy.db",
            },
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            config_path = handle.name

        try:
            with patch.dict(os.environ, {}, clear=True):
                settings = load_settings(config_path)
            self.assertEqual(settings.chat_model.provider, "ollama")
            self.assertEqual(settings.chat_model.model, "qwen2.5-coder")
            self.assertEqual(settings.chat_model.base_url, "http://localhost:11435/v1")
            self.assertEqual(settings.server.port, 9999)
            self.assertEqual(settings.memory.db_path, "custom_proxy.db")
        finally:
            os.unlink(config_path)

    def test_env_overrides_specific_fields(self):
        payload = {
            "chat_model": {
                "provider": "openai",
                "model": "from-file",
            },
            "server": {
                "port": 8080,
            },
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            config_path = handle.name

        try:
            with patch.dict(
                os.environ,
                {
                    "NEUROMEM_CHAT_MODEL": "from-env",
                    "NEUROMEM_CHAT_PROVIDER": "anthropic",
                    "NEUROMEM_SERVER_PORT": "9000",
                },
                clear=True,
            ):
                settings = load_settings(config_path)
            self.assertEqual(settings.chat_model.provider, "anthropic")
            self.assertEqual(settings.chat_model.model, "from-env")
            self.assertEqual(settings.server.port, 9000)
        finally:
            os.unlink(config_path)

    def test_provider_alias_defaults(self):
        provider = ProviderConfig(provider="lmstudio", model="local-model").with_defaults()
        self.assertEqual(provider.resolved_kind(), "openai_compatible")
        self.assertEqual(provider.resolved_base_url(), "http://localhost:1234/v1")

    def test_mcp_settings_load_from_file_and_env(self):
        payload = {
            "mcp": {
                "transport": "streamable-http",
                "port": 9100,
                "streamable_http_path": "/custom-mcp",
            }
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            config_path = handle.name

        try:
            with patch.dict(
                os.environ,
                {
                    "NEUROMEM_MCP_TRANSPORT": "stdio",
                    "NEUROMEM_MCP_PORT": "9200",
                },
                clear=True,
            ):
                settings = load_settings(config_path)
            self.assertEqual(settings.mcp.transport, "stdio")
            self.assertEqual(settings.mcp.port, 9200)
            self.assertEqual(settings.mcp.streamable_http_path, "/custom-mcp")
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    unittest.main()

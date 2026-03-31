"""
Validation tests for IDE MCP config packs.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path("/home/conamara/Desktop/ai memory/neuromem-agents")


class MCPClientConfigPackTests(unittest.TestCase):
    def _load_json(self, relative_path: str):
        with (REPO_ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_vscode_stdio_config_pack_uses_current_schema(self):
        payload = self._load_json("examples/ide/vscode/mcp_stdio.example.json")
        self.assertIn("servers", payload)
        self.assertEqual(payload["servers"]["neuromem"]["type"], "stdio")
        self.assertIn("--transport", payload["servers"]["neuromem"]["args"])

    def test_vscode_http_config_pack_uses_current_schema(self):
        payload = self._load_json("examples/ide/vscode/mcp_streamable_http.example.json")
        self.assertIn("servers", payload)
        self.assertEqual(payload["servers"]["neuromem"]["type"], "http")
        self.assertEqual(payload["servers"]["neuromem"]["url"], "http://127.0.0.1:8765/mcp")

    def test_jetbrains_stdio_config_pack_uses_import_schema(self):
        payload = self._load_json("examples/ide/jetbrains/mcp_stdio.example.json")
        self.assertIn("mcpServers", payload)
        self.assertIn("command", payload["mcpServers"]["neuromem"])
        self.assertIn("env", payload["mcpServers"]["neuromem"])

    def test_jetbrains_http_config_pack_uses_import_schema(self):
        payload = self._load_json("examples/ide/jetbrains/mcp_streamable_http.example.json")
        self.assertIn("mcpServers", payload)
        self.assertEqual(payload["mcpServers"]["neuromem"]["url"], "http://127.0.0.1:8765/mcp")


if __name__ == "__main__":
    unittest.main()

# MCP Integration

NeuroMem now ships with a first-party MCP server so IDE agents and MCP-capable runtimes can use the memory layer without going through the OpenAI-compatible proxy.

## Supported Transports

- `stdio` for local IDE integrations and desktop tools.
- `streamable-http` for remote or containerized deployment.

## Exposed MCP Tools

- `create_memory`
- `search_memory`
- `list_memories`
- `associate_memories`
- `get_memory_stats`
- `get_observability_metrics`
- `consolidate_memory`

## Exposed MCP Resources

- `memory://stats/overview`
- `memory://stats/observability`
- `memory://sessions/{session_id}/summary`
- `memory://records/{memory_id}`

## Exposed MCP Prompts

- `memory_recall_query`
- `project_handoff_brief`

## Start With Stdio

```bash
pip install 'neuromem-agents[mcp]'
neuromem-mcp-server --config examples/configs/mcp_stdio.example.json --transport stdio
```

## Start With Streamable HTTP

```bash
pip install 'neuromem-agents[mcp]'
neuromem-mcp-server \
  --config examples/configs/mcp_streamable_http.example.json \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8765
```

The default endpoint is `http://127.0.0.1:8765/mcp`.

## Ready-To-Use Config Packs

- VS Code stdio: `examples/ide/vscode/mcp_stdio.example.json`
- VS Code streamable-http: `examples/ide/vscode/mcp_streamable_http.example.json`
- JetBrains stdio import: `examples/ide/jetbrains/mcp_stdio.example.json`
- JetBrains streamable-http import: `examples/ide/jetbrains/mcp_streamable_http.example.json`

## VS Code Example

Copy `examples/ide/vscode/mcp_stdio.example.json` into `.vscode/mcp.json`, or use this equivalent config:

```json
{
  "servers": {
    "neuromem": {
      "type": "stdio",
      "command": "${workspaceFolder}/venv/bin/neuromem-mcp-server",
      "args": [
        "--config",
        "${workspaceFolder}/examples/configs/mcp_stdio.example.json",
        "--transport",
        "stdio"
      ],
      "env": {
        "NEUROMEM_MEMORY_DB_PATH": "${workspaceFolder}/neuromem_mcp.db",
        "NEUROMEM_MCP_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

For a remote MCP deployment, copy `examples/ide/vscode/mcp_streamable_http.example.json`.

## JetBrains Example

Import `examples/ide/jetbrains/mcp_stdio.example.json` into JetBrains AI Assistant after replacing `<ABSOLUTE_PATH_TO_REPO>`, or use this equivalent snippet:

```json
{
  "mcpServers": {
    "neuromem": {
      "command": "<ABSOLUTE_PATH_TO_REPO>/venv/bin/neuromem-mcp-server",
      "args": [
        "--config",
        "<ABSOLUTE_PATH_TO_REPO>/examples/configs/mcp_stdio.example.json",
        "--transport",
        "stdio"
      ],
      "env": {
        "NEUROMEM_MEMORY_DB_PATH": "<ABSOLUTE_PATH_TO_REPO>/neuromem_mcp.db",
        "NEUROMEM_MCP_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

For a remote MCP deployment, import `examples/ide/jetbrains/mcp_streamable_http.example.json`.

## Design Notes

- Session scoping is enforced through `session:<id>` tags to avoid memory leakage across projects.
- `streamable-http` runs in stateless JSON-response mode by default for easier deployment behind modern HTTP infrastructure.
- Relative `memory.db_path` values are resolved against the JSON config file location, so IDE launches do not depend on the current working directory.
- The MCP server shares the same SQLite-backed NeuroMem store as the OpenAI-compatible proxy when pointed at the same `db_path`.

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
- `consolidate_memory`

## Exposed MCP Resources

- `memory://stats/overview`
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

## VS Code Example

Create `.vscode/mcp.json`:

```json
{
  "servers": {
    "neuromem": {
      "type": "stdio",
      "command": "/absolute/path/to/venv/bin/neuromem-mcp-server",
      "args": [
        "--config",
        "/absolute/path/to/examples/configs/mcp_stdio.example.json",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

## JetBrains Example

Use the MCP server settings in JetBrains AI Assistant and either:

- point to the `stdio` command `neuromem-mcp-server --config ... --transport stdio`
- or point to the `streamable-http` URL `http://127.0.0.1:8765/mcp`

## Design Notes

- Session scoping is enforced through `session:<id>` tags to avoid memory leakage across projects.
- `streamable-http` runs in stateless JSON-response mode by default for easier deployment behind modern HTTP infrastructure.
- The MCP server shares the same SQLite-backed NeuroMem store as the OpenAI-compatible proxy when pointed at the same `db_path`.

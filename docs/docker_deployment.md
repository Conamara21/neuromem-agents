# Docker Deployment

NeuroMem now ships with a single Docker image that can run either:

- the OpenAI-compatible proxy
- the MCP server

## Build

```bash
docker build -t neuromem-agents:latest .
```

## Run The Proxy

```bash
docker run --rm -p 8080:8080 \
  -e NEUROMEM_CHAT_PROVIDER=openai \
  -e NEUROMEM_CHAT_MODEL=your-upstream-model \
  -e OPENAI_API_KEY=your_key_here \
  -e NEUROMEM_MEMORY_DB_PATH=/data/neuromem.db \
  -v neuromem_data:/data \
  neuromem-agents:latest \
  neuromem-openai-server --host 0.0.0.0 --port 8080
```

## Run The MCP Server

```bash
docker run --rm -p 8765:8765 \
  -e NEUROMEM_MEMORY_DB_PATH=/data/neuromem.db \
  -e NEUROMEM_MCP_TRANSPORT=streamable-http \
  -e NEUROMEM_MCP_HOST=0.0.0.0 \
  -e NEUROMEM_MCP_PORT=8765 \
  -v neuromem_data:/data \
  neuromem-agents:latest \
  neuromem-mcp-server --transport streamable-http --host 0.0.0.0 --port 8765
```

## Run Both With Compose

```bash
docker compose up --build
# or: docker-compose up --build
```

This starts:

- `neuromem-proxy` on `http://127.0.0.1:8080`
- `neuromem-mcp` on `http://127.0.0.1:8765/mcp`

Both services share the same Docker volume-backed SQLite database at `/data/neuromem.db`.

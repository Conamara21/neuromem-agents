# Compatibility Roadmap

This roadmap turns NeuroMem from a research-oriented memory module into a practical memory layer that can sit in front of mainstream developer tooling and mainstream model providers.

## Phase 1: Shipping Now

- Provider-agnostic settings for chat and embedding backends.
- OpenAI-compatible proxy server for `/v1/chat/completions` and `/v1/responses`.
- Memory augmentation that retrieves relevant memories before forwarding to an upstream model.
- Direct memory endpoints for manual ingestion, search, and stats inspection.
- Example configs for OpenAI, Anthropic, Gemini, and local OpenAI-compatible servers such as Ollama.

## Supported Provider Shapes

- `openai`: OpenAI's hosted API through the OpenAI-compatible adapter.
- `anthropic`: direct Anthropic Messages API adapter.
- `gemini`: direct Gemini GenerateContent adapter.
- `ollama`: OpenAI-compatible local server alias.
- `lmstudio`: OpenAI-compatible local server alias.
- `vllm`: OpenAI-compatible self-hosted server alias.
- `local_lexical`: zero-dependency local embedding backend for memory indexing.

## Supported Development Environments

- Any client that can target an OpenAI-compatible `base_url`.
- Python projects through the official OpenAI SDK.
- JavaScript and TypeScript projects through the official OpenAI SDK.
- IDE assistants and agent frameworks that already speak OpenAI-compatible HTTP.
- Local model workflows through Ollama, LM Studio, or vLLM without changing NeuroMem's memory layer.

## Recommended Default Deployment Pattern

1. Keep NeuroMem as the memory and routing layer.
2. Point its upstream chat backend to one of:
   - OpenAI
   - Anthropic
   - Gemini
   - Ollama
   - LM Studio
   - vLLM
3. Keep memory embeddings on `local_lexical` first for zero-friction setup.
4. Upgrade to an external embedding backend only when a workload needs stronger semantic matching.

## Why OpenAI-Compatible Proxying Comes First

- It minimizes client-side changes.
- It works across SDKs, IDE extensions, and orchestration frameworks.
- It lets NeuroMem add long-context memory without forcing users onto a custom protocol.
- It provides a stable bridge toward future MCP integration.

## Phase 2: Next Recommended Step

- Add an MCP server that exposes:
  - memory search
  - memory write
  - memory stats
  - session-scoped recall
- Provide ready-to-use configs for VS Code, JetBrains AI Assistant, Claude Desktop, and agent runtimes that support MCP.

## Phase 3: Broader Ecosystem Support

- LangChain wrapper with memory-aware retriever and chat model proxy helpers.
- LlamaIndex adapter for using NeuroMem as long-term conversational memory.
- Docker image and compose template for one-command local startup.
- Observability hooks for tracing retrieval latency, memory hit rate, and augmentation quality.

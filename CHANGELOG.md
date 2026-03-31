# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-03-31

### Added

- OpenAI-compatible proxy with session-scoped memory augmentation, direct memory APIs, and runtime metrics.
- First-party MCP server with `stdio` and `streamable-http` transports, observability tools, and prompt/resource surfaces.
- First-party LangChain and LlamaIndex adapters for retriever-based integration.
- Ready-to-use MCP client config packs for VS Code and JetBrains AI Assistant.
- Docker image and compose templates for local deployment.
- Reproducible benchmark outputs and visualization assets documented in the repository.

### Changed

- Reworked the README homepage to prioritize developer onboarding and integration paths.
- Resolved relative `memory.db_path` values against the config file location for more stable IDE and CLI launches.
- Unified package versioning at `0.3.0` across the published surfaces.

### Licensing

- Relicensed the project under GNU GPL v3.0.

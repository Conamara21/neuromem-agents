"""
MCP server for NeuroMem memory operations.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Sequence

from ..integrations.config import NeuromemSettings, load_settings
from ..integrations.engine import MemoryAugmentedProxy


def _normalize_tags(tags: Optional[Sequence[str]]) -> List[str]:
    normalized: List[str] = []
    for tag in tags or []:
        value = str(tag).strip()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _serialize_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


class MCPMemoryService:
    """Thin service layer reused by MCP tools and resources."""

    def __init__(self, settings: NeuromemSettings):
        self.settings = settings
        self.proxy = MemoryAugmentedProxy(settings)

    def create_memory(
        self,
        content: str,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return self.proxy.create_memory(
            content=content,
            session_id=session_id,
            memory_type=memory_type,
            tags=_normalize_tags(tags),
        )

    def search_memory(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        items = self.proxy.search_memory(
            query=query,
            session_id=session_id,
            top_k=top_k,
            tags=_normalize_tags(tags),
        )
        return {
            "query": query,
            "session_id": session_id or self.settings.default_session_id,
            "count": len(items),
            "items": items,
        }

    def list_memories(
        self,
        session_id: Optional[str] = None,
        limit: int = 20,
        tags: Optional[Sequence[str]] = None,
        memory_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        items = self.proxy.list_memories(
            session_id=session_id,
            limit=limit,
            tags=_normalize_tags(tags),
            memory_type=memory_type,
        )
        return {
            "session_id": session_id,
            "count": len(items),
            "items": items,
        }

    def get_memory(self, memory_id: str) -> Dict[str, Any]:
        item = self.proxy.get_memory(memory_id)
        if item is None:
            raise ValueError(f"Memory `{memory_id}` was not found.")
        return item

    def associate_memories(self, source_id: str, target_id: str, strength: float = 0.8) -> Dict[str, Any]:
        return self.proxy.associate_memories(source_id, target_id, strength=strength)

    def consolidate_memory(self) -> Dict[str, Any]:
        self.proxy.memory.consolidate()
        return self.proxy.memory_statistics()

    def session_summary(self, session_id: str, limit: int = 10) -> Dict[str, Any]:
        listing = self.list_memories(session_id=session_id, limit=limit)
        memory_types: Dict[str, int] = {}
        for item in listing["items"]:
            memory_type = str(item.get("memory_type", "unknown"))
            memory_types[memory_type] = memory_types.get(memory_type, 0) + 1

        return {
            "session_id": session_id,
            "memory_types": memory_types,
            "recent_memories": listing["items"],
            "count": listing["count"],
        }

    def memory_stats(self) -> Dict[str, Any]:
        return self.proxy.memory_statistics()

    def build_recall_prompt(self, session_id: str, objective: str, limit: int = 5) -> List[Dict[str, str]]:
        summary = self.session_summary(session_id=session_id, limit=limit)
        bullet_lines = []
        for index, item in enumerate(summary["recent_memories"], start=1):
            bullet_lines.append(
                "{0}. [{1}] {2}".format(
                    index,
                    item.get("memory_type", "unknown"),
                    item.get("content", ""),
                )
            )

        prompt_text = "\n".join(bullet_lines) if bullet_lines else "No memories found for this session."
        return [
            {
                "role": "user",
                "content": (
                    "Use the following NeuroMem session context to help with the task.\n\n"
                    "Session: {0}\n"
                    "Objective: {1}\n\n"
                    "Relevant memories:\n{2}"
                ).format(session_id, objective, prompt_text),
            }
        ]


def build_mcp_server(settings: Optional[NeuromemSettings] = None, config_path: Optional[str] = None):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "The MCP server requires the official MCP Python SDK. "
            "Install with `pip install 'neuromem-agents[mcp]'`."
        ) from exc

    resolved_settings = settings or load_settings(config_path)
    service = MCPMemoryService(resolved_settings)

    server = FastMCP(
        name=resolved_settings.mcp.server_name,
        instructions=(
            "NeuroMem MCP exposes long-term project and conversation memory. "
            "Use its tools to store session-scoped memories, retrieve them by query, "
            "inspect statistics, and attach memory context to agent tasks."
        ),
        host=resolved_settings.mcp.host,
        port=resolved_settings.mcp.port,
        streamable_http_path=resolved_settings.mcp.streamable_http_path,
        json_response=resolved_settings.mcp.json_response,
        stateless_http=resolved_settings.mcp.stateless_http,
        log_level=resolved_settings.mcp.log_level,
    )

    @server.tool(description="Store a NeuroMem memory record for a session.", structured_output=True)
    def create_memory(
        content: str,
        session_id: str = "default",
        memory_type: str = "episodic",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return service.create_memory(content, session_id=session_id, memory_type=memory_type, tags=tags)

    @server.tool(description="Search session-scoped NeuroMem memories by query.", structured_output=True)
    def search_memory(
        query: str,
        session_id: str = "default",
        top_k: int = 5,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return service.search_memory(query, session_id=session_id, top_k=top_k, tags=tags)

    @server.tool(description="List recent memories for a session or tag set.", structured_output=True)
    def list_memories(
        session_id: Optional[str] = None,
        limit: int = 20,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        return service.list_memories(
            session_id=session_id,
            limit=limit,
            tags=tags,
            memory_type=memory_type,
        )

    @server.tool(description="Create an associative connection between two memories.", structured_output=True)
    def associate_memories(source_id: str, target_id: str, strength: float = 0.8) -> Dict[str, Any]:
        return service.associate_memories(source_id, target_id, strength=strength)

    @server.tool(description="Get overall NeuroMem memory statistics.", structured_output=True)
    def get_memory_stats() -> Dict[str, Any]:
        return service.memory_stats()

    @server.tool(description="Run a consolidation pass on working memory.", structured_output=True)
    def consolidate_memory() -> Dict[str, Any]:
        return service.consolidate_memory()

    @server.resource(
        "memory://stats/overview",
        name="memory_stats",
        description="Current NeuroMem memory statistics.",
        mime_type="application/json",
    )
    def memory_stats_resource() -> str:
        return _serialize_json(service.memory_stats())

    @server.resource(
        "memory://sessions/{session_id}/summary",
        name="session_summary",
        description="Recent session-scoped memory summary.",
        mime_type="application/json",
    )
    def session_summary_resource(session_id: str) -> str:
        return _serialize_json(service.session_summary(session_id))

    @server.resource(
        "memory://records/{memory_id}",
        name="memory_record",
        description="A single NeuroMem record by id.",
        mime_type="application/json",
    )
    def memory_record_resource(memory_id: str) -> str:
        return _serialize_json(service.get_memory(memory_id))

    @server.prompt(
        name="memory_recall_query",
        description="Build a prompt seeded with recent NeuroMem memories for a session.",
    )
    def memory_recall_query(session_id: str, objective: str) -> List[Dict[str, str]]:
        return service.build_recall_prompt(session_id=session_id, objective=objective)

    @server.prompt(
        name="project_handoff_brief",
        description="Create a concise project handoff prompt from recent session memory.",
    )
    def project_handoff_brief(session_id: str, task: str = "Prepare a project handoff brief.") -> List[Dict[str, str]]:
        return service.build_recall_prompt(session_id=session_id, objective=task, limit=10)

    return server


def create_mcp_app(config_path: Optional[str] = None, settings: Optional[NeuromemSettings] = None):
    server = build_mcp_server(settings=settings, config_path=config_path)
    return server.streamable_http_app()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NeuroMem MCP server.")
    parser.add_argument("--config", help="Path to a JSON settings file.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        help="Transport to use. Defaults to the config value.",
    )
    parser.add_argument("--host", help="Override streamable-http bind host.")
    parser.add_argument("--port", type=int, help="Override streamable-http bind port.")
    parser.add_argument("--path", help="Override streamable-http path.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings(args.config)

    if args.transport:
        settings.mcp.transport = args.transport
    if args.host:
        settings.mcp.host = args.host
    if args.port:
        settings.mcp.port = args.port
    if args.path:
        settings.mcp.streamable_http_path = args.path

    server = build_mcp_server(settings=settings)
    server.run(transport=settings.mcp.transport)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

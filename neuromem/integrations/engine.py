"""
Memory-augmented proxy engine for chat model compatibility.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..core.memory_manager import MemoryManager, MemoryNode, MemoryType
from .adapters import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    create_chat_adapter,
    create_text_embedder,
    normalize_chat_messages,
)
from .config import NeuromemSettings


def _coerce_memory_type(value: Any) -> MemoryType:
    if isinstance(value, MemoryType):
        return value
    raw_value = getattr(value, "value", value)
    if isinstance(raw_value, str):
        return MemoryType(raw_value.lower())
    raise ValueError("Unsupported memory type: {0}".format(value))


def _now_epoch() -> int:
    return int(time.time())


def _usage_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


@dataclass
class MemoryProxyOptions:
    enabled: bool = True
    top_k: Optional[int] = None
    session_id: Optional[str] = None
    store_messages: Optional[bool] = None
    tags: List[str] = field(default_factory=list)
    retrieval_query: Optional[str] = None
    return_memories: bool = True

    @classmethod
    def from_payload(cls, payload: Optional[Dict[str, Any]]) -> "MemoryProxyOptions":
        if not payload:
            return cls()
        return cls(
            enabled=bool(payload.get("enabled", True)),
            top_k=payload.get("top_k"),
            session_id=payload.get("session_id"),
            store_messages=payload.get("store_messages"),
            tags=list(payload.get("tags", [])),
            retrieval_query=payload.get("retrieval_query"),
            return_memories=bool(payload.get("return_memories", True)),
        )


class MemoryAugmentedProxy:
    """OpenAI-compatible proxy service with NeuroMem augmentation."""

    def __init__(self, settings: NeuromemSettings):
        self.settings = settings
        embedder = create_text_embedder(settings.embedding_model)
        self.memory = MemoryManager(
            capacity=settings.memory.capacity,
            db_path=settings.memory.db_path,
            embedder=embedder,
        )
        if settings.memory.load_existing:
            try:
                self.memory.load_from_db()
            except Exception:
                pass

        self.chat_adapter = create_chat_adapter(settings.chat_model)

    def _session_tag(self, session_id: Optional[str]) -> str:
        resolved_session = session_id or self.settings.default_session_id
        return "{0}:{1}".format(self.settings.memory.session_tag_prefix, resolved_session)

    def _build_tags(self, session_id: Optional[str], tags: Optional[Sequence[str]]) -> List[str]:
        built_tags = [self._session_tag(session_id)]
        for tag in tags or []:
            normalized = str(tag).strip()
            if normalized and normalized not in built_tags:
                built_tags.append(normalized)
        return built_tags

    def _find_existing_memory(self, content: str, tags: Sequence[str]) -> Optional[str]:
        content_index = getattr(self.memory, "content_index", {})
        for node_id in content_index.get(content, set()):
            node = self.memory.memory_nodes.get(node_id)
            if node is None:
                continue
            node_tags = set(node.tags or [])
            if all(tag in node_tags for tag in tags):
                return node_id
        return None

    def _serialize_memory_node(self, node: MemoryNode, score: Optional[float] = None) -> Dict[str, Any]:
        payload = {
            "id": node.id,
            "content": node.content,
            "memory_type": getattr(node.memory_type, "value", str(node.memory_type)),
            "timestamp": node.timestamp,
            "tags": list(node.tags or []),
            "access_frequency": self.memory.access_frequency.get(node.id, 1),
        }
        if score is not None:
            payload["score"] = score
        return payload

    def create_memory(
        self,
        content: str,
        session_id: Optional[str] = None,
        memory_type: Optional[Any] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        combined_tags = self._build_tags(session_id, tags)
        existing_id = self._find_existing_memory(content, combined_tags)
        if existing_id and existing_id in self.memory.memory_nodes:
            return {
                "id": existing_id,
                "duplicate": True,
                "memory": self._serialize_memory_node(self.memory.memory_nodes[existing_id]),
            }

        node_id = self.memory.encode(
            content,
            _coerce_memory_type(memory_type or self.settings.memory.default_memory_type),
            tags=combined_tags,
        )
        return {
            "id": node_id,
            "duplicate": False,
            "memory": self._serialize_memory_node(self.memory.memory_nodes[node_id]),
        }

    def search_memory(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        resolved_top_k = top_k or self.settings.memory.retrieval_top_k
        built_tags = self._build_tags(session_id, tags)
        context = {
            "tags": built_tags,
            "required_tags": built_tags,
        }
        results = self.memory.retrieve(query, top_k=resolved_top_k, context=context)
        return [self._serialize_memory_node(node) for node in results]

    def list_memories(
        self,
        session_id: Optional[str] = None,
        limit: int = 20,
        tags: Optional[Sequence[str]] = None,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        required_tags = set()
        if session_id is not None:
            required_tags.add(self._session_tag(session_id))
        for tag in tags or []:
            normalized_tag = str(tag).strip()
            if normalized_tag:
                required_tags.add(normalized_tag)

        normalized_memory_type = str(memory_type).strip().lower() if memory_type else None
        items = []
        for node in self.memory.memory_nodes.values():
            node_tags = set(node.tags or [])
            if required_tags and not required_tags.issubset(node_tags):
                continue
            if (
                normalized_memory_type
                and getattr(node.memory_type, "value", "").lower() != normalized_memory_type
            ):
                continue
            items.append(self._serialize_memory_node(node))

        items.sort(key=lambda item: item["timestamp"], reverse=True)
        return items[: max(0, int(limit))]

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        node = self.memory.memory_nodes.get(memory_id)
        if node is None:
            return None
        return self._serialize_memory_node(node)

    def associate_memories(
        self,
        source_id: str,
        target_id: str,
        strength: float = 0.8,
    ) -> Dict[str, Any]:
        if source_id not in self.memory.memory_nodes:
            raise ValueError("Memory `{0}` was not found.".format(source_id))
        if target_id not in self.memory.memory_nodes:
            raise ValueError("Memory `{0}` was not found.".format(target_id))

        self.memory.associate(source_id, target_id, strength=float(strength))
        return {
            "source_id": source_id,
            "target_id": target_id,
            "strength": float(strength),
            "source_connections": len(self.memory.connections.get(source_id, [])),
            "target_connections": len(self.memory.connections.get(target_id, [])),
        }

    def memory_statistics(self) -> Dict[str, Any]:
        stats = self.memory.get_statistics()
        stats["db_path"] = self.settings.memory.db_path
        return stats

    def _latest_user_query(self, messages: Sequence[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        for message in reversed(messages):
            if message.role != "system":
                return message.content
        return ""

    def _build_memory_prompt(self, memories: Sequence[Dict[str, Any]]) -> str:
        if not memories:
            return ""
        lines = [
            "Relevant NeuroMem context:",
        ]
        for index, memory in enumerate(memories, start=1):
            memory_type = memory.get("memory_type", "unknown")
            tags = ", ".join(memory.get("tags", []))
            lines.append(
                "{0}. [{1}] {2}".format(index, memory_type, memory.get("content", ""))
            )
            if tags:
                lines.append("   tags: {0}".format(tags))
        lines.append(
            "Use this memory only when it helps answer the latest request accurately."
        )
        return "\n".join(lines)

    def _store_turn(
        self,
        messages: Sequence[ChatMessage],
        response: ChatResponse,
        session_id: Optional[str],
        extra_tags: Sequence[str],
        retrieved_memories: Sequence[Dict[str, Any]],
    ) -> Dict[str, Optional[str]]:
        last_user_message: Optional[ChatMessage] = None
        for message in reversed(messages):
            if message.role == "user":
                last_user_message = message
                break

        user_id: Optional[str] = None
        assistant_id: Optional[str] = None

        if last_user_message is not None:
            stored_user = self.create_memory(
                last_user_message.content,
                session_id=session_id,
                memory_type="episodic",
                tags=list(extra_tags) + ["role:user"],
            )
            user_id = stored_user["id"]

        if response.text.strip():
            stored_assistant = self.create_memory(
                response.text,
                session_id=session_id,
                memory_type="episodic",
                tags=list(extra_tags) + ["role:assistant", "model:{0}".format(response.model)],
            )
            assistant_id = stored_assistant["id"]

        if (
            self.settings.memory.auto_associate
            and user_id
            and assistant_id
            and user_id in self.memory.memory_nodes
            and assistant_id in self.memory.memory_nodes
        ):
            self.memory.associate(user_id, assistant_id, strength=0.9)

        if self.settings.memory.auto_associate and user_id:
            for memory in retrieved_memories[:3]:
                memory_id = memory.get("id")
                if memory_id and memory_id in self.memory.memory_nodes:
                    self.memory.associate(user_id, memory_id, strength=0.55)
                    if assistant_id and assistant_id in self.memory.memory_nodes:
                        self.memory.associate(assistant_id, memory_id, strength=0.45)

        return {
            "user_id": user_id,
            "assistant_id": assistant_id,
        }

    def generate_reply(
        self,
        messages: Sequence[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        proxy_options: Optional[MemoryProxyOptions] = None,
    ) -> Dict[str, Any]:
        if self.chat_adapter is None:
            raise ValueError("No upstream chat provider configured.")

        normalized_messages = normalize_chat_messages(messages)
        options = proxy_options or MemoryProxyOptions()
        session_id = options.session_id or self.settings.default_session_id
        store_messages = (
            self.settings.memory.auto_store
            if options.store_messages is None
            else bool(options.store_messages)
        )
        resolved_top_k = options.top_k or self.settings.memory.retrieval_top_k
        retrieval_query = options.retrieval_query or self._latest_user_query(normalized_messages)
        extra_tags = list(options.tags)

        retrieved_memories: List[Dict[str, Any]] = []
        augmented_messages = list(normalized_messages)
        if options.enabled and retrieval_query.strip() and self.memory.memory_nodes:
            retrieved_memories = self.search_memory(
                retrieval_query,
                session_id=session_id,
                top_k=resolved_top_k,
                tags=extra_tags,
            )
            memory_prompt = self._build_memory_prompt(retrieved_memories)
            if memory_prompt:
                augmented_messages = [
                    ChatMessage(role="system", content=memory_prompt)
                ] + augmented_messages

        response = self.chat_adapter.complete(
            ChatRequest(
                messages=augmented_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
        )

        stored_ids: Dict[str, Optional[str]] = {"user_id": None, "assistant_id": None}
        if store_messages:
            stored_ids = self._store_turn(
                normalized_messages,
                response,
                session_id=session_id,
                extra_tags=extra_tags,
                retrieved_memories=retrieved_memories,
            )

        return {
            "response": response,
            "session_id": session_id,
            "retrieved_memories": retrieved_memories,
            "stored_ids": stored_ids,
            "return_memories": options.return_memories,
        }

    def chat_completions_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("stream"):
            raise ValueError("Streaming is not implemented yet.")

        options = MemoryProxyOptions.from_payload(payload.get("neuromem"))
        if not options.session_id and payload.get("user"):
            options.session_id = str(payload.get("user"))

        result = self.generate_reply(
            messages=payload.get("messages", []),
            model=payload.get("model"),
            temperature=payload.get("temperature"),
            max_tokens=payload.get("max_tokens"),
            stop=payload.get("stop"),
            proxy_options=options,
        )
        response = result["response"]
        usage = dict(response.usage or {})

        api_response = {
            "id": "chatcmpl_{0}".format(uuid.uuid4().hex[:24]),
            "object": "chat.completion",
            "created": _now_epoch(),
            "model": response.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.text},
                    "finish_reason": response.finish_reason,
                }
            ],
            "usage": usage,
        }
        if result["retrieved_memories"]:
            api_response["neuromem"] = {
                "session_id": result["session_id"],
                "retrieved_count": len(result["retrieved_memories"]),
                "stored_ids": result["stored_ids"],
            }
            if result["return_memories"]:
                api_response["neuromem"]["retrieved_memories"] = result["retrieved_memories"]
        return api_response

    def responses_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("stream"):
            raise ValueError("Streaming is not implemented yet.")

        options = MemoryProxyOptions.from_payload(payload.get("neuromem"))
        if not options.session_id and payload.get("user"):
            options.session_id = str(payload.get("user"))

        messages: List[Dict[str, Any]] = []
        instructions = payload.get("instructions")
        if instructions:
            messages.append({"role": "system", "content": instructions})

        input_payload = payload.get("input", [])
        if isinstance(input_payload, str):
            messages.append({"role": "user", "content": input_payload})
        elif isinstance(input_payload, list):
            for item in input_payload:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                    continue
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content")
                    if content is None and "text" in item:
                        content = item.get("text")
                    messages.append({"role": role, "content": content})
        elif isinstance(input_payload, dict):
            messages.append(
                {
                    "role": input_payload.get("role", "user"),
                    "content": input_payload.get("content"),
                }
            )

        result = self.generate_reply(
            messages=messages,
            model=payload.get("model"),
            temperature=payload.get("temperature"),
            max_tokens=payload.get("max_output_tokens"),
            proxy_options=options,
        )
        response = result["response"]
        usage = dict(response.usage or {})
        input_tokens = _usage_int(
            usage.get("input_tokens", usage.get("prompt_tokens", 0))
        )
        output_tokens = _usage_int(
            usage.get("output_tokens", usage.get("completion_tokens", 0))
        )
        total_tokens = _usage_int(usage.get("total_tokens", input_tokens + output_tokens))

        api_response = {
            "id": "resp_{0}".format(uuid.uuid4().hex[:24]),
            "object": "response",
            "created_at": _now_epoch(),
            "status": "completed",
            "model": response.model,
            "output": [
                {
                    "id": "msg_{0}".format(uuid.uuid4().hex[:24]),
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": response.text,
                            "annotations": [],
                        }
                    ],
                }
            ],
            "output_text": response.text,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }
        if result["retrieved_memories"]:
            api_response["neuromem"] = {
                "session_id": result["session_id"],
                "retrieved_count": len(result["retrieved_memories"]),
                "stored_ids": result["stored_ids"],
            }
            if result["return_memories"]:
                api_response["neuromem"]["retrieved_memories"] = result["retrieved_memories"]
        return api_response

    def list_models(self) -> Dict[str, Any]:
        data: List[Dict[str, Any]] = []
        if self.settings.chat_model and self.settings.chat_model.model:
            data.append(
                {
                    "id": self.settings.chat_model.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": self.settings.chat_model.provider,
                }
            )
        if self.settings.embedding_model and self.settings.embedding_model.model:
            data.append(
                {
                    "id": self.settings.embedding_model.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": self.settings.embedding_model.provider,
                }
            )
        return {"object": "list", "data": data}

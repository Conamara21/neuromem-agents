"""
Provider adapters for chat completion and embedding generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..core.text_embeddings import LexicalHashingEmbedder, TextEmbedder
from .config import ProviderConfig
from .http import get_json, post_json


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return _normalize_content(content.get("text"))
        if "content" in content:
            return _normalize_content(content.get("content"))
        if "parts" in content:
            return _normalize_content(content.get("parts"))
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text", "")))
                    continue
                if "text" in item:
                    parts.append(str(item.get("text", "")))
                    continue
                if "content" in item:
                    parts.append(_normalize_content(item.get("content")))
                    continue
                if "parts" in item:
                    parts.append(_normalize_content(item.get("parts")))
        return "\n".join(part for part in parts if part)
    return str(content)


def normalize_chat_messages(messages: Iterable[Dict[str, Any]]) -> List["ChatMessage"]:
    normalized: List[ChatMessage] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        content = _normalize_content(message.get("content"))
        if not content.strip():
            continue
        normalized.append(ChatMessage(role=role, content=content.strip()))
    return normalized


def _collapse_system_messages(messages: Sequence["ChatMessage"]) -> (str, List["ChatMessage"]):
    system_parts: List[str] = []
    non_system: List[ChatMessage] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
        else:
            non_system.append(message)
    return "\n\n".join(system_parts).strip(), non_system


def _require_base_url(config: ProviderConfig) -> str:
    base_url = config.resolved_base_url()
    if not base_url:
        raise ValueError(
            "Provider `{0}` requires a base_url. Set it in JSON config or environment.".format(
                config.provider
            )
        )
    return base_url


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatRequest:
    messages: Sequence[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Sequence[str]] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    text: str
    model: str
    provider: str
    finish_reason: str = "stop"
    usage: Dict[str, Any] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)


class ChatModelAdapter:
    """Minimal interface for upstream chat generation."""

    def complete(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError


class EmbeddingAdapter:
    """Minimal interface for embedding generation."""

    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        raise NotImplementedError

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class AdapterTextEmbedder(TextEmbedder):
    """Wrap an embedding adapter behind the existing TextEmbedder interface."""

    def __init__(self, adapter: EmbeddingAdapter):
        self.adapter = adapter

    def encode(self, text: str) -> np.ndarray:
        return self.adapter.embed_text(text)


class LocalLexicalEmbeddingAdapter(EmbeddingAdapter):
    """Zero-dependency embedding backend based on NeuroMem's lexical hashing."""

    def __init__(self, dimension: int = 512):
        self.embedder = LexicalHashingEmbedder(dimension=dimension)

    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [self.embedder.encode(text) for text in texts]


class OpenAICompatibleChatAdapter(ChatModelAdapter):
    def __init__(self, config: ProviderConfig):
        self.config = config.with_defaults()

    def complete(self, request: ChatRequest) -> ChatResponse:
        if not self.config.model and not request.model:
            raise ValueError("Chat model is required for OpenAI-compatible providers.")

        body: Dict[str, Any] = {
            "model": request.model or self.config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.stop:
            body["stop"] = list(request.stop)
        body.update(self.config.extra_body)
        body.update(request.extra_body)

        headers = dict(self.config.extra_headers)
        api_key = self.config.resolved_api_key()
        if api_key:
            headers["Authorization"] = "Bearer {0}".format(api_key)
        base_url = _require_base_url(self.config)

        response = post_json(
            "{0}/chat/completions".format(base_url),
            payload=body,
            headers=headers,
            timeout=self.config.timeout_seconds,
        )
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("Upstream provider returned no chat choices.")
        choice = choices[0]
        message = choice.get("message", {})
        text = _normalize_content(message.get("content", ""))
        return ChatResponse(
            text=text,
            model=response.get("model") or str(body["model"]),
            provider=self.config.provider,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=response.get("usage", {}),
            raw_response=response,
        )


class AnthropicChatAdapter(ChatModelAdapter):
    def __init__(self, config: ProviderConfig):
        self.config = config.with_defaults()

    def complete(self, request: ChatRequest) -> ChatResponse:
        api_key = self.config.resolved_api_key()
        if not api_key:
            raise ValueError("Anthropic chat requests require an API key.")
        if not self.config.model and not request.model:
            raise ValueError("Anthropic chat requests require a model.")

        system_prompt, non_system_messages = _collapse_system_messages(request.messages)
        payload: Dict[str, Any] = {
            "model": request.model or self.config.model,
            "messages": [
                {
                    "role": "assistant" if message.role == "assistant" else "user",
                    "content": [{"type": "text", "text": message.content}],
                }
                for message in non_system_messages
            ],
            "max_tokens": request.max_tokens or 1024,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        payload.update(self.config.extra_body)
        payload.update(request.extra_body)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        headers.update(self.config.extra_headers)

        response = post_json(
            "{0}/messages".format(_require_base_url(self.config)),
            payload=payload,
            headers=headers,
            timeout=self.config.timeout_seconds,
        )
        text_parts = []
        for item in response.get("content", []):
            if item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return ChatResponse(
            text="\n".join(part for part in text_parts if part).strip(),
            model=response.get("model") or str(payload["model"]),
            provider=self.config.provider,
            finish_reason=response.get("stop_reason", "end_turn"),
            usage=response.get("usage", {}),
            raw_response=response,
        )


class GeminiChatAdapter(ChatModelAdapter):
    def __init__(self, config: ProviderConfig):
        self.config = config.with_defaults()

    def complete(self, request: ChatRequest) -> ChatResponse:
        api_key = self.config.resolved_api_key()
        if not api_key:
            raise ValueError("Gemini chat requests require an API key.")
        if not self.config.model and not request.model:
            raise ValueError("Gemini chat requests require a model.")

        system_prompt, non_system_messages = _collapse_system_messages(request.messages)
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "model" if message.role == "assistant" else "user",
                    "parts": [{"text": message.content}],
                }
                for message in non_system_messages
            ],
        }
        generation_config: Dict[str, Any] = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        if generation_config:
            payload["generationConfig"] = generation_config
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        payload.update(self.config.extra_body)
        payload.update(request.extra_body)

        response = post_json(
            "{0}/models/{1}:generateContent".format(
                _require_base_url(self.config),
                request.model or self.config.model,
            ),
            payload=payload,
            timeout=self.config.timeout_seconds,
            query={"key": api_key},
        )
        candidates = response.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))
        usage = response.get("usageMetadata", {})
        return ChatResponse(
            text=text.strip(),
            model=request.model or str(self.config.model),
            provider=self.config.provider,
            finish_reason=candidates[0].get("finishReason", "STOP"),
            usage=usage,
            raw_response=response,
        )


class OpenAICompatibleEmbeddingAdapter(EmbeddingAdapter):
    def __init__(self, config: ProviderConfig):
        self.config = config.with_defaults()

    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        if not self.config.model:
            raise ValueError("Embedding model is required for OpenAI-compatible providers.")

        headers = dict(self.config.extra_headers)
        api_key = self.config.resolved_api_key()
        if api_key:
            headers["Authorization"] = "Bearer {0}".format(api_key)
        base_url = _require_base_url(self.config)

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "input": list(texts),
            "encoding_format": "float",
        }
        payload.update(self.config.extra_body)
        response = post_json(
            "{0}/embeddings".format(base_url),
            payload=payload,
            headers=headers,
            timeout=self.config.timeout_seconds,
        )
        rows = sorted(response.get("data", []), key=lambda item: item.get("index", 0))
        return [
            np.asarray(item.get("embedding", []), dtype=np.float32)
            for item in rows
        ]


class GeminiEmbeddingAdapter(EmbeddingAdapter):
    def __init__(self, config: ProviderConfig):
        self.config = config.with_defaults()

    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        api_key = self.config.resolved_api_key()
        if not api_key:
            raise ValueError("Gemini embeddings require an API key.")
        if not self.config.model:
            raise ValueError("Gemini embeddings require a model.")

        embeddings: List[np.ndarray] = []
        for text in texts:
            payload = {
                "content": {
                    "parts": [{"text": text}],
                }
            }
            payload.update(self.config.extra_body)
            response = post_json(
                "{0}/models/{1}:embedContent".format(
                    _require_base_url(self.config),
                    self.config.model,
                ),
                payload=payload,
                timeout=self.config.timeout_seconds,
                query={"key": api_key},
            )
            values = response.get("embedding", {}).get("values", [])
            embeddings.append(np.asarray(values, dtype=np.float32))
        return embeddings


def create_chat_adapter(config: Optional[ProviderConfig]) -> Optional[ChatModelAdapter]:
    if config is None:
        return None

    kind = config.with_defaults().resolved_kind()
    if kind == "openai_compatible":
        return OpenAICompatibleChatAdapter(config)
    if kind == "anthropic":
        return AnthropicChatAdapter(config)
    if kind == "gemini":
        return GeminiChatAdapter(config)
    raise ValueError("Unsupported chat provider: {0}".format(config.provider))


def create_embedding_adapter(config: ProviderConfig) -> EmbeddingAdapter:
    resolved = config.with_defaults()
    kind = resolved.resolved_kind()
    if kind == "local_lexical":
        return LocalLexicalEmbeddingAdapter()
    if kind == "openai_compatible":
        return OpenAICompatibleEmbeddingAdapter(resolved)
    if kind == "gemini":
        return GeminiEmbeddingAdapter(resolved)
    raise ValueError(
        "Unsupported embedding provider: {0}. Use local_lexical, openai-compatible, or gemini.".format(
            config.provider
        )
    )


def create_text_embedder(config: ProviderConfig) -> TextEmbedder:
    if config.with_defaults().resolved_kind() == "local_lexical":
        return LexicalHashingEmbedder()
    return AdapterTextEmbedder(create_embedding_adapter(config))

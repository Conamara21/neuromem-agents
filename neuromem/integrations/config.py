"""
Provider-agnostic configuration for NeuroMem compatibility layers.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "kind": "openai_compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "openai_compatible": {
        "kind": "openai_compatible",
    },
    "ollama": {
        "kind": "openai_compatible",
        "base_url": "http://localhost:11434/v1",
    },
    "lmstudio": {
        "kind": "openai_compatible",
        "base_url": "http://localhost:1234/v1",
    },
    "vllm": {
        "kind": "openai_compatible",
        "base_url": "http://localhost:8000/v1",
    },
    "anthropic": {
        "kind": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "kind": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key_env": "GEMINI_API_KEY",
    },
    "local_lexical": {
        "kind": "local_lexical",
    },
}


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _parse_float(value: Optional[str], default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _parse_csv(value: Optional[str], default: List[str]) -> List[str]:
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def _environment_payload(include_defaults: bool) -> Dict[str, Any]:
    def read_env(name: str, default: Optional[str] = None) -> Optional[str]:
        if include_defaults:
            return os.getenv(name, default)
        return os.getenv(name)

    payload: Dict[str, Any] = {}

    memory_payload: Dict[str, Any] = {}
    memory_capacity = read_env("NEUROMEM_MEMORY_CAPACITY", "10000")
    if memory_capacity is not None:
        memory_payload["capacity"] = _parse_int(memory_capacity, 10000)
    memory_db_path = read_env("NEUROMEM_MEMORY_DB_PATH", "neuromem_proxy.db")
    if memory_db_path is not None:
        memory_payload["db_path"] = memory_db_path
    memory_top_k = read_env("NEUROMEM_MEMORY_TOP_K", "5")
    if memory_top_k is not None:
        memory_payload["retrieval_top_k"] = _parse_int(memory_top_k, 5)
    auto_store = read_env("NEUROMEM_AUTO_STORE", "true")
    if auto_store is not None:
        memory_payload["auto_store"] = _parse_bool(auto_store, True)
    auto_associate = read_env("NEUROMEM_AUTO_ASSOCIATE", "true")
    if auto_associate is not None:
        memory_payload["auto_associate"] = _parse_bool(auto_associate, True)
    default_memory_type = read_env("NEUROMEM_DEFAULT_MEMORY_TYPE", "episodic")
    if default_memory_type is not None:
        memory_payload["default_memory_type"] = default_memory_type
    session_tag_prefix = read_env("NEUROMEM_SESSION_TAG_PREFIX", "session")
    if session_tag_prefix is not None:
        memory_payload["session_tag_prefix"] = session_tag_prefix
    load_existing = read_env("NEUROMEM_LOAD_EXISTING", "true")
    if load_existing is not None:
        memory_payload["load_existing"] = _parse_bool(load_existing, True)
    if memory_payload:
        payload["memory"] = memory_payload

    server_payload: Dict[str, Any] = {}
    server_host = read_env("NEUROMEM_SERVER_HOST", "127.0.0.1")
    if server_host is not None:
        server_payload["host"] = server_host
    server_port = read_env("NEUROMEM_SERVER_PORT", "8080")
    if server_port is not None:
        server_payload["port"] = _parse_int(server_port, 8080)
    server_cors = read_env("NEUROMEM_SERVER_CORS_ORIGINS", "*")
    if server_cors is not None:
        server_payload["cors_origins"] = _parse_csv(server_cors, ["*"])
    server_reload = read_env("NEUROMEM_SERVER_RELOAD", "false")
    if server_reload is not None:
        server_payload["reload"] = _parse_bool(server_reload, False)
    server_log_level = read_env("NEUROMEM_SERVER_LOG_LEVEL", "info")
    if server_log_level is not None:
        server_payload["log_level"] = server_log_level
    if server_payload:
        payload["server"] = server_payload

    mcp_payload: Dict[str, Any] = {}
    mcp_server_name = read_env("NEUROMEM_MCP_SERVER_NAME", "neuromem-mcp")
    if mcp_server_name is not None:
        mcp_payload["server_name"] = mcp_server_name
    mcp_transport = read_env("NEUROMEM_MCP_TRANSPORT", "stdio")
    if mcp_transport is not None:
        mcp_payload["transport"] = mcp_transport
    mcp_host = read_env("NEUROMEM_MCP_HOST", "127.0.0.1")
    if mcp_host is not None:
        mcp_payload["host"] = mcp_host
    mcp_port = read_env("NEUROMEM_MCP_PORT", "8765")
    if mcp_port is not None:
        mcp_payload["port"] = _parse_int(mcp_port, 8765)
    mcp_path = read_env("NEUROMEM_MCP_STREAMABLE_HTTP_PATH", "/mcp")
    if mcp_path is not None:
        mcp_payload["streamable_http_path"] = mcp_path
    mcp_json_response = read_env("NEUROMEM_MCP_JSON_RESPONSE", "true")
    if mcp_json_response is not None:
        mcp_payload["json_response"] = _parse_bool(mcp_json_response, True)
    mcp_stateless_http = read_env("NEUROMEM_MCP_STATELESS_HTTP", "true")
    if mcp_stateless_http is not None:
        mcp_payload["stateless_http"] = _parse_bool(mcp_stateless_http, True)
    mcp_log_level = read_env("NEUROMEM_MCP_LOG_LEVEL", "INFO")
    if mcp_log_level is not None:
        mcp_payload["log_level"] = mcp_log_level
    if mcp_payload:
        payload["mcp"] = mcp_payload

    default_session_id = read_env("NEUROMEM_DEFAULT_SESSION_ID", "default")
    if default_session_id is not None:
        payload["default_session_id"] = default_session_id

    chat_provider = read_env("NEUROMEM_CHAT_PROVIDER")
    if chat_provider is not None:
        payload["chat_model"] = {
            "provider": chat_provider,
            "model": read_env("NEUROMEM_CHAT_MODEL"),
            "base_url": read_env("NEUROMEM_CHAT_BASE_URL"),
            "api_key": read_env("NEUROMEM_CHAT_API_KEY"),
            "api_key_env": read_env("NEUROMEM_CHAT_API_KEY_ENV"),
            "timeout_seconds": _parse_float(
                read_env("NEUROMEM_CHAT_TIMEOUT_SECONDS", "60.0"),
                60.0,
            ),
        }

    embedding_provider = read_env("NEUROMEM_EMBEDDING_PROVIDER")
    if embedding_provider is not None:
        payload["embedding_model"] = {
            "provider": embedding_provider,
            "model": read_env("NEUROMEM_EMBEDDING_MODEL"),
            "base_url": read_env("NEUROMEM_EMBEDDING_BASE_URL"),
            "api_key": read_env("NEUROMEM_EMBEDDING_API_KEY"),
            "api_key_env": read_env("NEUROMEM_EMBEDDING_API_KEY_ENV"),
            "timeout_seconds": _parse_float(
                read_env("NEUROMEM_EMBEDDING_TIMEOUT_SECONDS", "60.0"),
                60.0,
            ),
        }

    return payload


@dataclass
class ProviderConfig:
    """Configuration for an upstream chat or embedding provider."""

    provider: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout_seconds: float = 60.0
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def resolved_kind(self) -> str:
        defaults = PROVIDER_DEFAULTS.get(self.provider, {})
        return str(defaults.get("kind", self.provider))

    def resolved_base_url(self) -> Optional[str]:
        if self.base_url:
            return self.base_url.rstrip("/")
        defaults = PROVIDER_DEFAULTS.get(self.provider, {})
        base_url = defaults.get("base_url")
        if not base_url:
            return None
        return str(base_url).rstrip("/")

    def resolved_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.api_key_env and os.getenv(self.api_key_env):
            return os.getenv(self.api_key_env)
        defaults = PROVIDER_DEFAULTS.get(self.provider, {})
        env_name = defaults.get("api_key_env")
        if env_name and os.getenv(str(env_name)):
            return os.getenv(str(env_name))
        return None

    def with_defaults(self) -> "ProviderConfig":
        defaults = PROVIDER_DEFAULTS.get(self.provider, {})
        return ProviderConfig(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url or defaults.get("base_url"),
            api_key=self.api_key,
            api_key_env=self.api_key_env or defaults.get("api_key_env"),
            timeout_seconds=self.timeout_seconds,
            extra_headers=dict(self.extra_headers),
            extra_body=dict(self.extra_body),
        )


@dataclass
class MemoryRuntimeConfig:
    """Runtime configuration for NeuroMem-backed proxy behavior."""

    capacity: int = 10000
    db_path: str = "neuromem_proxy.db"
    retrieval_top_k: int = 5
    auto_store: bool = True
    auto_associate: bool = True
    default_memory_type: str = "episodic"
    session_tag_prefix: str = "session"
    load_existing: bool = True


@dataclass
class ServerConfig:
    """Network and process settings for the compatibility server."""

    host: str = "127.0.0.1"
    port: int = 8080
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    reload: bool = False
    log_level: str = "info"


@dataclass
class MCPServerConfig:
    """Network and transport settings for the MCP server."""

    server_name: str = "neuromem-mcp"
    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8765
    streamable_http_path: str = "/mcp"
    json_response: bool = True
    stateless_http: bool = True
    log_level: str = "INFO"


@dataclass
class NeuromemSettings:
    """Top-level settings object for the compatibility layer."""

    chat_model: Optional[ProviderConfig] = None
    embedding_model: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(
            provider="local_lexical",
            model="lexical-hash-512",
        )
    )
    memory: MemoryRuntimeConfig = field(default_factory=MemoryRuntimeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    mcp: MCPServerConfig = field(default_factory=MCPServerConfig)
    default_session_id: str = "default"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NeuromemSettings":
        chat_payload = payload.get("chat_model")
        embedding_payload = payload.get("embedding_model")
        memory_payload = payload.get("memory", {})
        server_payload = payload.get("server", {})
        mcp_payload = payload.get("mcp", {})

        chat_model = ProviderConfig(**chat_payload) if chat_payload else None
        embedding_model = (
            ProviderConfig(**embedding_payload)
            if embedding_payload
            else ProviderConfig(provider="local_lexical", model="lexical-hash-512")
        )

        return cls(
            chat_model=chat_model.with_defaults() if chat_model else None,
            embedding_model=embedding_model.with_defaults(),
            memory=MemoryRuntimeConfig(**memory_payload),
            server=ServerConfig(**server_payload),
            mcp=MCPServerConfig(**mcp_payload),
            default_session_id=payload.get("default_session_id", "default"),
        )

    @classmethod
    def from_file(cls, path: str) -> "NeuromemSettings":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    @classmethod
    def from_env(cls) -> "NeuromemSettings":
        return cls.from_dict(_environment_payload(include_defaults=True))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_settings(config_path: Optional[str] = None) -> NeuromemSettings:
    """Load settings from a JSON file, then override with environment variables."""

    file_payload: Dict[str, Any] = {}
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            file_payload = json.load(handle)

    if not config_path:
        return NeuromemSettings.from_env()

    env_payload = _environment_payload(include_defaults=False)
    merged_payload = _deep_merge(file_payload, env_payload)
    return NeuromemSettings.from_dict(merged_payload)

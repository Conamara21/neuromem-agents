"""
Compatibility and provider integration layer for NeuroMem.
"""

from importlib import import_module

_EXPORTS = {
    "AdapterTextEmbedder": (".adapters", "AdapterTextEmbedder"),
    "ChatMessage": (".adapters", "ChatMessage"),
    "ChatRequest": (".adapters", "ChatRequest"),
    "ChatResponse": (".adapters", "ChatResponse"),
    "create_chat_adapter": (".adapters", "create_chat_adapter"),
    "create_embedding_adapter": (".adapters", "create_embedding_adapter"),
    "create_text_embedder": (".adapters", "create_text_embedder"),
    "normalize_chat_messages": (".adapters", "normalize_chat_messages"),
    "MemoryRuntimeConfig": (".config", "MemoryRuntimeConfig"),
    "NeuromemSettings": (".config", "NeuromemSettings"),
    "ProviderConfig": (".config", "ProviderConfig"),
    "ServerConfig": (".config", "ServerConfig"),
    "load_settings": (".config", "load_settings"),
    "MemoryAugmentedProxy": (".engine", "MemoryAugmentedProxy"),
    "MemoryProxyOptions": (".engine", "MemoryProxyOptions"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)

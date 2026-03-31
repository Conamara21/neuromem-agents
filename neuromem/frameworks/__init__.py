"""
Framework adapters for NeuroMem.
"""

from importlib import import_module

_EXPORTS = {
    "NeuroMemRetriever": (".langchain", "NeuroMemRetriever"),
    "create_langchain_chat_openai": (".langchain", "create_langchain_chat_openai"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)

"""
MCP server entry points for NeuroMem.
"""

from importlib import import_module

_EXPORTS = {
    "build_mcp_server": (".server", "build_mcp_server"),
    "create_mcp_app": (".server", "create_mcp_app"),
    "main": (".server", "main"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)

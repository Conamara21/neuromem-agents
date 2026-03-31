"""
NeuroMem-Agents Experiments Module
"""

from importlib import import_module

__version__ = "0.3.0"

__all__ = [
    "ComparisonEngine"
]


def __getattr__(name):
    if name == "ComparisonEngine":
        return getattr(import_module(".comparison_engine", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
NeuroMem public package.
"""

from importlib import import_module

__version__ = "0.2.0"

_EXPORTS = {
    "MemoryManager": (".core", "MemoryManager"),
    "EnhancedMemoryManager": (".core", "EnhancedMemoryManager"),
    "HierarchicalMemoryManager": (".core", "HierarchicalMemoryManager"),
    "EfficiencyOptimizedMemoryManager": (".core", "EfficiencyOptimizedMemoryManager"),
    "SparseActivationManager": (".core", "SparseActivationManager"),
    "ProgressiveRefinementEngine": (".core", "ProgressiveRefinementEngine"),
    "QuantumInspiredOptimizer": (".core", "QuantumInspiredOptimizer"),
    "BrainRegion": (".core", "BrainRegion"),
    "MemoryLayer": (".core", "MemoryLayer"),
    "MemoryType": (".core", "MemoryType"),
    "MemoryNode": (".core", "MemoryNode"),
    "TraditionalRAGSystem": (".core", "TraditionalRAGSystem"),
    "ComparisonEngine": (".experiments", "ComparisonEngine"),
    "MemoryAugmentedProxy": (".integrations", "MemoryAugmentedProxy"),
    "NeuromemSettings": (".integrations", "NeuromemSettings"),
    "load_settings": (".integrations", "load_settings"),
}

__all__ = [
    "MemoryManager",
    "EnhancedMemoryManager",
    "HierarchicalMemoryManager",
    "EfficiencyOptimizedMemoryManager",
    "SparseActivationManager",
    "ProgressiveRefinementEngine",
    "QuantumInspiredOptimizer",
    "BrainRegion",
    "MemoryLayer",
    "MemoryType",
    "MemoryNode",
    "TraditionalRAGSystem",
    "ComparisonEngine",
    "MemoryAugmentedProxy",
    "NeuromemSettings",
    "load_settings",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)

"""
NeuroMem-Agents Core Module
"""

from importlib import import_module

__version__ = "0.1.0"

_EXPORTS = {
    "MemoryManager": (".memory_manager", "MemoryManager"),
    "EnhancedMemoryManager": (".enhanced_memory_manager", "EnhancedMemoryManager"),
    "HierarchicalMemoryManager": (".hierarchical_memory", "HierarchicalMemoryManager"),
    "EfficiencyOptimizedMemoryManager": (".efficiency_optimizer", "EfficiencyOptimizedMemoryManager"),
    "SparseActivationManager": (".efficiency_optimizer", "SparseActivationManager"),
    "ProgressiveRefinementEngine": (".efficiency_optimizer", "ProgressiveRefinementEngine"),
    "QuantumInspiredOptimizer": (".efficiency_optimizer", "QuantumInspiredOptimizer"),
    "BrainRegion": (".hierarchical_memory", "BrainRegion"),
    "MemoryLayer": (".hierarchical_memory", "MemoryLayer"),
    "MemoryType": (".neuromorphic_memory", "MemoryType"),
    "MemoryNode": (".neuromorphic_memory", "MemoryNode"),
    "TraditionalRAGSystem": (".traditional_rag", "TraditionalRAGSystem"),
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
    "TraditionalRAGSystem"
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)

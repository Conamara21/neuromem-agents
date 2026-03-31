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
    "build_mcp_server": (".mcp", "build_mcp_server"),
    "NeuroMemRetriever": (".frameworks", "NeuroMemRetriever"),
    "create_langchain_chat_openai": (".frameworks", "create_langchain_chat_openai"),
    "NeuroMemLlamaIndexRetriever": (".frameworks", "NeuroMemLlamaIndexRetriever"),
    "create_llamaindex_openai_like": (".frameworks", "create_llamaindex_openai_like"),
    "create_llamaindex_query_engine": (".frameworks", "create_llamaindex_query_engine"),
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
    "build_mcp_server",
    "NeuroMemRetriever",
    "create_langchain_chat_openai",
    "NeuroMemLlamaIndexRetriever",
    "create_llamaindex_openai_like",
    "create_llamaindex_query_engine",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)

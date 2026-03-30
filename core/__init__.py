"""
NeuroMem-Agents Core Module
"""

__version__ = "0.1.0"

from .memory_manager import MemoryManager
from .enhanced_memory_manager import EnhancedMemoryManager
from .hierarchical_memory import HierarchicalMemoryManager, BrainRegion, MemoryLayer
from .neuromorphic_memory import MemoryType, MemoryNode
from .traditional_rag import TraditionalRAGSystem
from .efficiency_optimizer import EfficiencyOptimizedMemoryManager, SparseActivationManager, ProgressiveRefinementEngine, QuantumInspiredOptimizer

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
"""
NeuroMem-Agents Core Module
"""

__version__ = "0.1.0"

from .memory_manager import MemoryManager
from .neuromorphic_memory import MemoryType, MemoryNode
from .traditional_rag import TraditionalRAGSystem

__all__ = [
    "MemoryManager",
    "MemoryType", 
    "MemoryNode",
    "TraditionalRAGSystem"
]
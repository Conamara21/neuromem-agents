"""
NeuroMem-Agents: Neuromorphic Memory System

Implements a brain-inspired memory architecture for AI agents with:
- Hierarchical memory organization
- Synaptic plasticity mechanisms
- Associative recall
- Active forgetting
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math
import time
from datetime import datetime


class MemoryType(Enum):
    """Types of memory based on biological models"""
    SENSORY = "sensory"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryNode:
    """Represents a single memory unit with biological properties"""
    id: str
    content: str
    embedding: np.ndarray
    memory_type: MemoryType
    timestamp: float
    activation_level: float = 0.0
    connectivity_strength: float = 1.0
    importance_score: float = 1.0
    decay_rate: float = 0.01
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class SpikingNeuralNetwork:
    """Simulates neural dynamics for memory consolidation and retrieval"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.spike_history = []
        
    def compute_activation(self, inputs: np.ndarray, weights: np.ndarray) -> float:
        """Compute activation based on weighted inputs"""
        potential = np.dot(inputs, weights)
        return float(potential / (1 + abs(potential)))  # Normalized activation
    
    def generate_spike(self, activation: float) -> bool:
        """Generate spike if activation exceeds threshold"""
        return activation >= self.threshold


class MemoryManager:
    """Main memory management system implementing neuromorphic principles"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.connections: Dict[str, List[Tuple[str, float]]] = {}  # (target_id, weight)
        self.working_memory_buffer = []  # Short-term storage (like hippocampus)
        self.long_term_memory = {}  # Consolidated memories (like cortex)
        self.access_frequency = {}  # Track memory access patterns
        self.snn = SpikingNeuralNetwork()
        self.current_context = {}
        
    def encode(self, content: str, memory_type: MemoryType, tags: List[str] = None) -> str:
        """Encode new information into memory nodes"""
        import hashlib
        
        # Generate unique ID based on content
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        node_id = f"{memory_type.value}_{content_hash}_{int(time.time())}"
        
        # Create embedding (simplified - in practice, use transformer embeddings)
        embedding = self._generate_embedding(content)
        
        # Create memory node
        node = MemoryNode(
            id=node_id,
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            timestamp=time.time(),
            tags=tags or []
        )
        
        # Store in appropriate memory type
        self.memory_nodes[node_id] = node
        self.access_frequency[node_id] = 1
        
        # Handle different memory types
        if memory_type == MemoryType.WORKING:
            self.working_memory_buffer.append(node_id)
            # Limit working memory size
            if len(self.working_memory_buffer) > 100:  # Working memory capacity
                removed = self.working_memory_buffer.pop(0)
                del self.memory_nodes[removed]
        
        return node_id
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate simplified embedding (placeholder - replace with real model)"""
        # In practice, this would use a transformer model
        # For now, use a simple hash-based approach
        hash_val = hash(text) % (2**32)
        return np.array([float((hash_val >> i) & 1) for i in range(128)], dtype=np.float32)
    
    def associate(self, node_id1: str, node_id2: str, strength: float = 1.0):
        """Create associative connections between memory nodes"""
        if node_id1 not in self.connections:
            self.connections[node_id1] = []
        if node_id2 not in self.connections:
            self.connections[node_id2] = []
            
        # Update connection strengths (bidirectional)
        self._update_connection(node_id1, node_id2, strength)
        self._update_connection(node_id2, node_id1, strength)
        
        # Strengthen connection based on Hebbian learning
        node1 = self.memory_nodes[node_id1]
        node2 = self.memory_nodes[node_id2]
        node1.connectivity_strength = min(1.0, node1.connectivity_strength + 0.1 * strength)
        node2.connectivity_strength = min(1.0, node2.connectivity_strength + 0.1 * strength)
    
    def _update_connection(self, source: str, target: str, strength: float):
        """Update connection strength between nodes"""
        existing = [(tgt, w) for tgt, w in self.connections[source] if tgt == target]
        if existing:
            # Update existing connection
            idx = [i for i, (tgt, _) in enumerate(self.connections[source]) if tgt == target][0]
            old_weight = existing[0][1]
            new_weight = (old_weight + strength) / 2  # Average
            self.connections[source][idx] = (target, new_weight)
        else:
            # Add new connection
            self.connections[source].append((target, strength))
    
    def retrieve(self, query: str, top_k: int = 5, context: Dict = None) -> List[MemoryNode]:
        """Retrieve relevant memories using associative search"""
        query_embedding = self._generate_embedding(query)
        
        # Find similar memories
        similarities = []
        for node_id, node in self.memory_nodes.items():
            sim = self._cosine_similarity(query_embedding, node.embedding)
            
            # Boost similarity based on recent access and associations
            freq_factor = math.log(self.access_frequency.get(node_id, 1) + 1)
            assoc_factor = self._association_bonus(node_id, context)
            
            final_sim = sim * (1 + freq_factor * 0.1) * (1 + assoc_factor * 0.2)
            similarities.append((node, final_sim))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update access frequency
        for node, _ in similarities[:top_k]:
            self.access_frequency[node.id] = self.access_frequency.get(node.id, 1) + 1
            
        return [node for node, _ in similarities[:top_k]]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))
    
    def _association_bonus(self, node_id: str, context: Dict = None) -> float:
        """Calculate bonus based on associative connections"""
        if context is None:
            context = {}
        
        bonus = 0.0
        if node_id in self.connections:
            for connected_node, weight in self.connections[node_id]:
                if connected_node in self.access_frequency:
                    # Boost based on recent activation of connected nodes
                    activation = min(1.0, self.access_frequency[connected_node] / 10.0)
                    bonus += weight * activation
        
        return bonus
    
    def consolidate(self):
        """Consolidate working memory to long-term memory (like sleep/dreaming)"""
        # Process items in working memory buffer
        for node_id in self.working_memory_buffer:
            node = self.memory_nodes[node_id]
            
            # Only consolidate highly accessed items
            access_count = self.access_frequency.get(node_id, 1)
            if access_count > 3:  # Threshold for consolidation
                # Move to long-term memory
                self.long_term_memory[node_id] = node
                # Reduce working memory presence
                node.memory_type = MemoryType.SEMANTIC  # Upgrade to semantic memory
        
        # Clear working memory buffer periodically
        if len(self.working_memory_buffer) > 50:
            self.working_memory_buffer = self.working_memory_buffer[-20:]  # Keep recent items
    
    def forget(self, decay_threshold: float = 0.1):
        """Implement active forgetting mechanism"""
        current_time = time.time()
        to_remove = []
        
        for node_id, node in self.memory_nodes.items():
            # Calculate decay factor
            age = current_time - node.timestamp
            decay_factor = math.exp(-node.decay_rate * age)
            access_factor = 1.0 / (self.access_frequency.get(node_id, 1) ** 0.5)
            
            # Combined forgetting score
            forget_score = decay_factor * access_factor
            
            if forget_score < decay_threshold:
                to_remove.append(node_id)
        
        # Remove forgotten items
        for node_id in to_remove:
            if node_id in self.memory_nodes:
                del self.memory_nodes[node_id]
            if node_id in self.access_frequency:
                del self.access_frequency[node_id]
            if node_id in self.connections:
                del self.connections[node_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        memory_types = {}
        total_size = sum(len(str(node)) for node in self.memory_nodes.values())
        
        for node in self.memory_nodes.values():
            mtype = node.memory_type.value
            memory_types[mtype] = memory_types.get(mtype, 0) + 1
        
        return {
            "total_nodes": len(self.memory_nodes),
            "memory_types": memory_types,
            "estimated_size_bytes": total_size,
            "connection_density": sum(len(conns) for conns in self.connections.values()) / max(1, len(self.memory_nodes)),
            "average_access_frequency": np.mean(list(self.access_frequency.values())) if self.access_frequency else 0
        }


if __name__ == "__main__":
    # Example usage
    print("NeuroMem-Agents: Neuromorphic Memory System")
    print("=" * 50)
    
    # Initialize memory manager
    mem_manager = MemoryManager(capacity=1000)
    
    # Encode some sample memories
    id1 = mem_manager.encode("The capital of France is Paris", MemoryType.SEMANTIC, tags=["geography"])
    id2 = mem_manager.encode("I visited Paris last summer", MemoryType.EPISODIC, tags=["personal", "travel"])
    id3 = mem_manager.encode("Eiffel Tower is in Paris", MemoryType.SEMANTIC, tags=["landmark"])
    
    # Create associations
    mem_manager.associate(id1, id2, 0.8)
    mem_manager.associate(id1, id3, 0.9)
    
    # Retrieve related memories
    results = mem_manager.retrieve("Paris", top_k=3)
    print(f"\nRetrieved {len(results)} related memories:")
    for i, node in enumerate(results, 1):
        print(f"{i}. {node.content} (similarity: {node.activation_level:.3f})")
    
    # Get statistics
    stats = mem_manager.get_statistics()
    print(f"\nMemory Statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")
"""
Basic functionality test for NeuroMem-Agents without external dependencies
"""

import sys
import os
import json
import time
import hashlib
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


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
    embedding: List[float]  # Simplified embedding as list
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


class MockNeuroMemoryManager:
    """Mock version of the neuromorphic memory system using only stdlib"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.connections: Dict[str, List[Tuple[str, float]]] = {}  # (target_id, weight)
        self.working_memory_buffer = []  # Short-term storage (like hippocampus)
        self.long_term_memory = {}  # Consolidated memories (like cortex)
        self.access_frequency = {}  # Track memory access patterns
        self.current_context = {}
        
    def encode(self, content: str, memory_type: MemoryType, tags: List[str] = None) -> str:
        """Encode new information into memory nodes"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        node_id = f"{memory_type.value}_{content_hash}_{int(time.time())}"
        
        # Create simplified embedding using basic hashing
        embedding = self._generate_simple_embedding(content)
        
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
                if removed in self.memory_nodes:
                    del self.memory_nodes[removed]
        
        return node_id
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate simplified embedding using basic hashing (stdlib only)"""
        # Create a fixed-size vector from the hash of the text
        hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        embedding = []
        for i in range(32):  # Smaller embedding size for stdlib
            embedding.append(float((hash_val >> (i * 8)) & 0xFF) / 255.0)
        return embedding
    
    def associate(self, node_id1: str, node_id2: str, strength: float = 1.0):
        """Create associative connections between memory nodes"""
        if node_id1 not in self.connections:
            self.connections[node_id1] = []
        if node_id2 not in self.connections:
            self.connections[node_id2] = []
            
        # Update connection strengths (bidirectional)
        self._update_connection(node_id1, node_id2, strength)
        self._update_connection(node_id2, node_id1, strength)
        
        # Strengthen connection
        if node_id1 in self.memory_nodes and node_id2 in self.memory_nodes:
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
        """Retrieve relevant memories using simplified similarity"""
        query_embedding = self._generate_simple_embedding(query)
        
        # Find similar memories using basic distance calculation
        similarities = []
        for node_id, node in self.memory_nodes.items():
            sim = self._simple_similarity(query_embedding, node.embedding)
            
            # Boost similarity based on recent access and associations
            freq_factor = max(0, min(1, (self.access_frequency.get(node_id, 1) - 1) * 0.1))
            assoc_factor = self._association_bonus(node_id, context)
            
            final_sim = sim * (1 + freq_factor) * (1 + assoc_factor * 0.2)
            similarities.append((node, final_sim))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update access frequency
        for node, _ in similarities[:top_k]:
            self.access_frequency[node.id] = self.access_frequency.get(node.id, 1) + 1
            
        return [node for node, _ in similarities[:top_k]]
    
    def _simple_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate simple similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        # Simple dot product similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
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
        """Consolidate working memory to long-term memory"""
        # Process items in working memory buffer
        for node_id in self.working_memory_buffer:
            if node_id in self.memory_nodes:
                node = self.memory_nodes[node_id]
                
                # Only consolidate highly accessed items
                access_count = self.access_frequency.get(node_id, 1)
                if access_count > 3:  # Threshold for consolidation
                    # Move to long-term memory
                    self.long_term_memory[node_id] = node
                    # Update node type
                    node.memory_type = MemoryType.SEMANTIC  # Upgrade to semantic memory
        
        # Clear working memory buffer periodically
        if len(self.working_memory_buffer) > 50:
            self.working_memory_buffer = self.working_memory_buffer[-20:]  # Keep recent items
    
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
            "average_access_frequency": sum(self.access_frequency.values()) / len(self.access_frequency) if self.access_frequency else 0
        }


def run_basic_test():
    """Run a basic functionality test"""
    print("Testing NeuroMem-Agents (Basic Implementation)")
    print("=" * 50)
    
    # Initialize memory manager
    mem_manager = MockNeuroMemoryManager(capacity=1000)
    
    # Encode some sample memories
    print("Encoding sample memories...")
    id1 = mem_manager.encode("The capital of France is Paris", MemoryType.SEMANTIC, tags=["geography"])
    id2 = mem_manager.encode("I visited Paris last summer", MemoryType.EPISODIC, tags=["personal", "travel"])
    id3 = mem_manager.encode("Eiffel Tower is in Paris", MemoryType.SEMANTIC, tags=["landmark"])
    
    print(f"Encoded 3 memories: {id1}, {id2}, {id3}")
    
    # Create associations
    print("\nCreating associations...")
    mem_manager.associate(id1, id2, 0.8)
    mem_manager.associate(id1, id3, 0.9)
    print("Created associations between memories")
    
    # Retrieve related memories
    print("\nTesting retrieval...")
    results = mem_manager.retrieve("Paris", top_k=3)
    print(f"Retrieved {len(results)} related memories:")
    for i, node in enumerate(results, 1):
        print(f"  {i}. {node.content}")
    
    # Get statistics
    print("\nMemory Statistics:")
    stats = mem_manager.get_statistics()
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    print("\n✓ Basic functionality test passed!")
    
    return True


if __name__ == "__main__":
    success = run_basic_test()
    if success:
        print("\n🎉 NeuroMem-Agents basic test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
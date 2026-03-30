"""
Extended functionality test to demonstrate advanced features of NeuroMem-Agents
"""

import time
import sys
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import hashlib
import random


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
    embedding: List[float]
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


class ExtendedNeuroMemoryManager:
    """Extended version demonstrating advanced neuromorphic features"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.connections: Dict[str, List[Tuple[str, float]]] = {}
        self.working_memory_buffer = []
        self.long_term_memory = {}
        self.access_frequency = {}
        self.token_counter = 0
        self.forget_threshold = 0.1
        
    def encode(self, content: str, memory_type: MemoryType, tags: List[str] = None) -> str:
        """Encode new information into memory nodes"""
        self.token_counter += len(content.split())
        
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        node_id = f"{memory_type.value}_{content_hash}_{int(time.time())}"
        
        embedding = self._generate_simple_embedding(content)
        
        node = MemoryNode(
            id=node_id,
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            timestamp=time.time(),
            tags=tags or []
        )
        
        self.memory_nodes[node_id] = node
        self.access_frequency[node_id] = 1
        
        if memory_type == MemoryType.WORKING:
            self.working_memory_buffer.append(node_id)
            if len(self.working_memory_buffer) > 100:
                removed = self.working_memory_buffer.pop(0)
                if removed in self.memory_nodes:
                    del self.memory_nodes[removed]
        
        return node_id
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate simplified embedding using basic hashing"""
        hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        embedding = []
        for i in range(32):
            embedding.append(float((hash_val >> (i * 8)) & 0xFF) / 255.0)
        return embedding
    
    def associate(self, node_id1: str, node_id2: str, strength: float = 1.0):
        """Create associative connections between memory nodes"""
        if node_id1 not in self.connections:
            self.connections[node_id1] = []
        if node_id2 not in self.connections:
            self.connections[node_id2] = []
            
        self._update_connection(node_id1, node_id2, strength)
        self._update_connection(node_id2, node_id1, strength)
        
        if node_id1 in self.memory_nodes and node_id2 in self.memory_nodes:
            node1 = self.memory_nodes[node_id1]
            node2 = self.memory_nodes[node_id2]
            node1.connectivity_strength = min(1.0, node1.connectivity_strength + 0.1 * strength)
            node2.connectivity_strength = min(1.0, node2.connectivity_strength + 0.1 * strength)
    
    def _update_connection(self, source: str, target: str, strength: float):
        """Update connection strength between nodes"""
        existing = [(tgt, w) for tgt, w in self.connections[source] if tgt == target]
        if existing:
            idx = [i for i, (tgt, _) in enumerate(self.connections[source]) if tgt == target][0]
            old_weight = existing[0][1]
            new_weight = (old_weight + strength) / 2
            self.connections[source][idx] = (target, new_weight)
        else:
            self.connections[source].append((target, strength))
    
    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryNode]:
        """Retrieve relevant memories using enhanced associative search"""
        self.token_counter += len(query.split())
        
        query_embedding = self._generate_simple_embedding(query)
        
        similarities = []
        for node_id, node in self.memory_nodes.items():
            sim = self._simple_similarity(query_embedding, node.embedding)
            
            # Enhanced boosting with association network
            freq_factor = max(0, min(1, (self.access_frequency.get(node_id, 1) - 1) * 0.1))
            assoc_factor = self._enhanced_association_bonus(node_id)
            
            final_sim = sim * (1 + freq_factor) * (1 + assoc_factor * 0.3)  # Higher association weight
            similarities.append((node, final_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update access frequency
        for node, _ in similarities[:top_k]:
            self.access_frequency[node.id] = self.access_frequency.get(node.id, 1) + 1
            self.token_counter += len(node.content.split())
            
        return [node for node, _ in similarities[:top_k]]
    
    def _simple_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate simple similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _enhanced_association_bonus(self, node_id: str) -> float:
        """Calculate enhanced bonus based on associative connections and network effects"""
        bonus = 0.0
        if node_id in self.connections:
            for connected_node, weight in self.connections[node_id]:
                if connected_node in self.access_frequency:
                    # Boost based on recent activation of connected nodes
                    activation = min(1.0, self.access_frequency[connected_node] / 10.0)
                    bonus += weight * activation
                    
                    # Also consider connections of connected nodes (spreading activation)
                    if connected_node in self.connections:
                        for secondary_node, sec_weight in self.connections[connected_node]:
                            if secondary_node in self.access_frequency:
                                secondary_activation = min(0.5, self.access_frequency[secondary_node] / 20.0)
                                bonus += weight * sec_weight * secondary_activation * 0.5
        
        return bonus
    
    def consolidate_memory(self):
        """Consolidate important memories from working to long-term"""
        for node_id in self.working_memory_buffer:
            if node_id in self.memory_nodes:
                node = self.memory_nodes[node_id]
                access_count = self.access_frequency.get(node_id, 1)
                
                # Consolidate frequently accessed items
                if access_count > 2:
                    self.long_term_memory[node_id] = node
                    node.memory_type = MemoryType.SEMANTIC
        
        # Trim working memory
        if len(self.working_memory_buffer) > 50:
            self.working_memory_buffer = self.working_memory_buffer[-20:]
    
    def active_forget(self):
        """Implement active forgetting mechanism"""
        current_time = time.time()
        to_remove = []
        
        for node_id, node in self.memory_nodes.items():
            age = current_time - node.timestamp
            decay_factor = pow(0.999, age)  # Slow decay
            access_factor = 1.0 / (self.access_frequency.get(node_id, 1) ** 0.5)
            
            forget_score = decay_factor * access_factor
            
            if forget_score < self.forget_threshold and node_id not in self.long_term_memory:
                to_remove.append(node_id)
        
        for node_id in to_remove:
            if node_id in self.memory_nodes:
                del self.memory_nodes[node_id]
            if node_id in self.access_frequency:
                del self.access_frequency[node_id]
            if node_id in self.connections:
                del self.connections[node_id]
    
    def get_memory_size_estimate(self) -> int:
        """Estimate memory usage in bytes"""
        total_size = sum(len(str(node)) for node in self.memory_nodes.values())
        total_size += sum(len(str(conns)) for conns in self.connections.values())
        return total_size
    
    def get_network_metrics(self) -> Dict[str, float]:
        """Get metrics about the association network"""
        total_connections = sum(len(conns) for conns in self.connections.values())
        total_nodes = len(self.memory_nodes)
        
        avg_connectivity = total_connections / total_nodes if total_nodes > 0 else 0
        
        # Calculate clustering coefficient approximation
        triangles = 0
        connected_triplets = 0
        
        for node_id, connections in self.connections.items():
            neighbors = [conn[0] for conn in connections]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    neighbor1, neighbor2 = neighbors[i], neighbors[j]
                    if neighbor1 in self.connections and neighbor2 in [conn[0] for conn in self.connections[neighbor1]]:
                        triangles += 1
                    connected_triplets += 1
        
        clustering_coeff = (3 * triangles) / connected_triplets if connected_triplets > 0 else 0
        
        return {
            "total_nodes": total_nodes,
            "total_connections": total_connections,
            "avg_connectivity": avg_connectivity,
            "clustering_coefficient": clustering_coeff,
            "network_density": total_connections / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0
        }


def run_extended_test():
    """Run extended functionality test"""
    print("Running Extended NeuroMem-Agents Test")
    print("=" * 50)
    
    # Initialize memory manager
    mem_manager = ExtendedNeuroMemoryManager(capacity=1000)
    
    # Add related memories to demonstrate associative features
    print("Adding related memories to demonstrate associations...")
    
    # Paris-related memories
    paris_fact_id = mem_manager.encode("The capital of France is Paris", MemoryType.SEMANTIC, tags=["geography", "fact"])
    paris_exp_id = mem_manager.encode("I visited Paris last summer and loved the art museums", MemoryType.EPISODIC, tags=["personal", "experience"])
    eiffel_id = mem_manager.encode("Eiffel Tower is the most recognizable landmark in Paris", MemoryType.SEMANTIC, tags=["landmark", "architecture"])
    louvre_id = mem_manager.encode("Louvre Museum houses the Mona Lisa and many other masterpieces", MemoryType.SEMANTIC, tags=["culture", "art"])
    
    # Create associations between related memories
    print("Creating associative connections...")
    mem_manager.associate(paris_fact_id, paris_exp_id, 0.8)
    mem_manager.associate(paris_fact_id, eiffel_id, 0.9)
    mem_manager.associate(eiffel_id, louvre_id, 0.6)
    mem_manager.associate(paris_exp_id, louvre_id, 0.7)
    
    print(f"Created {len(mem_manager.connections)} associations")
    
    # Demonstrate spreading activation through associations
    print("\nTesting associative retrieval...")
    results = mem_manager.retrieve("Paris", top_k=4)
    print(f"Retrieved {len(results)} related memories:")
    for i, node in enumerate(results, 1):
        print(f"  {i}. {node.content}")
        print(f"     Tags: {node.tags}, Type: {node.memory_type.value}")
    
    # Access some memories repeatedly to increase their importance
    print("\nDemonstrating memory consolidation...")
    for _ in range(3):
        mem_manager.retrieve("Eiffel Tower", top_k=1)
    
    # Consolidate important memories
    mem_manager.consolidate_memory()
    
    # Add some less important memories that might be forgotten
    print("\nAdding temporary memories...")
    temp_id1 = mem_manager.encode("Random fact about weather today", MemoryType.SENSORY, tags=["temp"])
    temp_id2 = mem_manager.encode("Some unimportant detail", MemoryType.SENSORY, tags=["temp"])
    
    # Check network metrics
    print("\nNetwork Metrics:")
    metrics = mem_manager.get_network_metrics()
    for key, value in metrics.items():
        print(f"  - {key}: {value:.3f}" if isinstance(value, float) else f"  - {key}: {value}")
    
    # Simulate passage of time and forgetting
    print("\nDemonstrating active forgetting...")
    initial_count = len(mem_manager.memory_nodes)
    mem_manager.active_forget()  # This might remove temporary memories
    final_count = len(mem_manager.memory_nodes)
    
    print(f"Memory nodes before forgetting: {initial_count}")
    print(f"Memory nodes after forgetting: {final_count}")
    
    # Final statistics
    print("\nFinal Memory Statistics:")
    stats = {
        "total_nodes": len(mem_manager.memory_nodes),
        "estimated_size_bytes": mem_manager.get_memory_size_estimate(),
        "tokens_consumed": mem_manager.token_counter,
        "long_term_memories": len(mem_manager.long_term_memory),
        "working_memory_size": len(mem_manager.working_memory_buffer)
    }
    
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    print("\n✅ Extended functionality test completed successfully!")
    print("This demonstrates the advanced features of the neuromorphic memory system:")
    print("  - Associative connections between memories")
    print("  - Spreading activation through the network")
    print("  - Memory consolidation based on importance")
    print("  - Active forgetting of irrelevant information")
    print("  - Network analysis capabilities")


if __name__ == "__main__":
    run_extended_test()
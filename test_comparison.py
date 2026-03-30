"""
Comparison test between neuromorphic and traditional memory systems
Using only standard library to avoid dependency issues
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


class MockNeuroMemoryManager:
    """Mock version of the neuromorphic memory system"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.connections: Dict[str, List[Tuple[str, float]]] = {}
        self.working_memory_buffer = []
        self.long_term_memory = {}
        self.access_frequency = {}
        self.token_counter = 0  # Track token consumption
        
    def encode(self, content: str, memory_type: MemoryType, tags: List[str] = None) -> str:
        """Encode new information into memory nodes"""
        # Count tokens (approximate by word count)
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
                    self.token_counter -= len(self.memory_nodes[removed].content.split()) if removed in self.memory_nodes else 0
        
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
        """Retrieve relevant memories using associative search"""
        # Count tokens for query
        self.token_counter += len(query.split())
        
        query_embedding = self._generate_simple_embedding(query)
        
        similarities = []
        for node_id, node in self.memory_nodes.items():
            sim = self._simple_similarity(query_embedding, node.embedding)
            
            # Boost similarity based on recent access and associations
            freq_factor = max(0, min(1, (self.access_frequency.get(node_id, 1) - 1) * 0.1))
            assoc_factor = self._association_bonus(node_id)
            
            final_sim = sim * (1 + freq_factor) * (1 + assoc_factor * 0.2)
            similarities.append((node, final_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update access frequency and token counter
        for node, _ in similarities[:top_k]:
            self.access_frequency[node.id] = self.access_frequency.get(node.id, 1) + 1
            # Add tokens for returned results
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
    
    def _association_bonus(self, node_id: str) -> float:
        """Calculate bonus based on associative connections"""
        bonus = 0.0
        if node_id in self.connections:
            for connected_node, weight in self.connections[node_id]:
                if connected_node in self.access_frequency:
                    activation = min(1.0, self.access_frequency[connected_node] / 10.0)
                    bonus += weight * activation
        
        return bonus
    
    def get_memory_size_estimate(self) -> int:
        """Estimate memory usage in bytes"""
        total_size = sum(len(str(node)) for node in self.memory_nodes.values())
        # Add connection storage overhead
        total_size += sum(len(str(conns)) for conns in self.connections.values())
        return total_size


class MockTraditionalRAG:
    """Mock version of traditional RAG system"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.documents = []
        self.embeddings = []
        self.doc_ids = []
        self.access_frequency = {}
        self.token_counter = 0  # Track token consumption
        
    def add_document(self, content: str, metadata: Dict = None) -> str:
        """Add a document to the traditional RAG system"""
        # Count tokens
        self.token_counter += len(content.split())
        
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        doc_id = f"doc_{content_hash}_{int(time.time())}"
        
        embedding = self._generate_simple_embedding(content)
        
        self.documents.append(content)
        self.embeddings.append(embedding)
        self.doc_ids.append(doc_id)
        self.access_frequency[doc_id] = 1
        
        if len(self.documents) > self.capacity:
            self.documents.pop(0)
            self.embeddings.pop(0)
            removed_id = self.doc_ids.pop(0)
            if removed_id in self.access_frequency:
                del self.access_frequency[removed_id]
        
        return doc_id
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate simplified embedding using basic hashing"""
        hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        embedding = []
        for i in range(32):
            embedding.append(float((hash_val >> (i * 8)) & 0xFF) / 255.0)
        return embedding
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve documents using traditional vector similarity"""
        # Count tokens for query
        self.token_counter += len(query.split())
        
        query_embedding = self._generate_simple_embedding(query)
        
        if not self.embeddings:
            return []
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self._simple_similarity(query_embedding, doc_embedding)
            similarities.append((self.documents[i], sim, self.doc_ids[i]))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update access frequency and token counter
        results = []
        for doc, sim, doc_id in similarities[:top_k]:
            self.access_frequency[doc_id] = self.access_frequency.get(doc_id, 1) + 1
            # Add tokens for returned results
            self.token_counter += len(doc.split())
            results.append((doc, sim))
        
        return results[:top_k]
    
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
    
    def get_memory_size_estimate(self) -> int:
        """Estimate memory usage in bytes"""
        total_size = sum(len(doc) for doc in self.documents)
        total_size += sum(len(str(embedding)) for embedding in self.embeddings)
        return total_size


def run_comparison_test():
    """Run comparison test between neuromorphic and traditional systems"""
    print("Running Memory System Comparison Test")
    print("=" * 50)
    
    # Create test data
    test_documents = [
        "The capital of France is Paris",
        "I visited Paris last summer", 
        "Eiffel Tower is in Paris",
        "Louvre Museum is located in Paris",
        "Notre Dame Cathedral in Paris",
        "French cuisine is famous worldwide",
        "Wine tasting in Bordeaux",
        "Art galleries in Montmartre",
        "Seine river cruise in Paris",
        "French language basics"
    ]
    
    test_queries = [
        "Paris",
        "France",
        "Eiffel Tower",
        "French culture"
    ]
    
    print(f"Test documents: {len(test_documents)}")
    print(f"Test queries: {len(test_queries)}")
    print()
    
    # Initialize both systems
    neuro_system = MockNeuroMemoryManager()
    trad_system = MockTraditionalRAG()
    
    # Add documents to both systems
    print("Adding documents to both systems...")
    
    # Add to neuromorphic system
    neuro_doc_ids = []
    for doc in test_documents:
        doc_type = MemoryType.SEMANTIC if random.random() > 0.5 else MemoryType.EPISODIC
        doc_id = neuro_system.encode(doc, doc_type)
        neuro_doc_ids.append(doc_id)
    
    # Create some associations in neuromorphic system
    for i in range(len(neuro_doc_ids) - 1):
        neuro_system.associate(neuro_doc_ids[i], neuro_doc_ids[i+1], 0.5)
    
    # Add to traditional system
    trad_doc_ids = []
    for doc in test_documents:
        doc_id = trad_system.add_document(doc)
        trad_doc_ids.append(doc_id)
    
    print(f"Neuro system tokens used during encoding: {neuro_system.token_counter}")
    print(f"Traditional system tokens used during encoding: {trad_system.token_counter}")
    print()
    
    # Run retrieval tests
    print("Running retrieval tests...")
    
    # Reset token counters before retrieval tests
    neuro_system.token_counter = 0
    trad_system.token_counter = 0
    
    neuro_retrieval_times = []
    trad_retrieval_times = []
    
    for query in test_queries:
        print(f"Testing query: '{query}'")
        
        # Test neuromorphic system
        start_time = time.time()
        neuro_results = neuro_system.retrieve(query, top_k=3)
        neuro_time = time.time() - start_time
        neuro_retrieval_times.append(neuro_time)
        
        # Test traditional system
        start_time = time.time()
        trad_results = trad_system.retrieve(query, top_k=3)
        trad_time = time.time() - start_time
        trad_retrieval_times.append(trad_time)
        
        print(f"  Neuro: {len(neuro_results)} results in {neuro_time:.4f}s")
        print(f"  Trad:  {len(trad_results)} results in {trad_time:.4f}s")
    
    print()
    
    # Final statistics
    neuro_avg_time = sum(neuro_retrieval_times) / len(neuro_retrieval_times)
    trad_avg_time = sum(trad_retrieval_times) / len(trad_retrieval_times)
    
    neuro_memory_size = neuro_system.get_memory_size_estimate()
    trad_memory_size = trad_system.get_memory_size_estimate()
    
    print("Final Results:")
    print("-" * 30)
    print(f"Neuro system:")
    print(f"  - Avg retrieval time: {neuro_avg_time:.4f}s")
    print(f"  - Tokens consumed: {neuro_system.token_counter}")
    print(f"  - Memory estimate: {neuro_memory_size} bytes")
    
    print(f"Traditional system:")
    print(f"  - Avg retrieval time: {trad_avg_time:.4f}s") 
    print(f"  - Tokens consumed: {trad_system.token_counter}")
    print(f"  - Memory estimate: {trad_memory_size} bytes")
    
    print()
    print("Analysis:")
    print("-" * 10)
    
    # Token efficiency
    if trad_system.token_counter > 0:
        token_efficiency = neuro_system.token_counter / trad_system.token_counter
        print(f"Token efficiency: {token_efficiency:.2f}x (lower is better)")
    
    # Speed comparison
    if trad_avg_time > 0:
        speed_ratio = neuro_avg_time / trad_avg_time
        print(f"Speed ratio: {speed_ratio:.2f}x (lower is better)")
    
    # Memory efficiency
    if trad_memory_size > 0:
        memory_efficiency = neuro_memory_size / trad_memory_size
        print(f"Memory efficiency: {memory_efficiency:.2f}x (lower is better)")
    
    print()
    if token_efficiency < 1.0:
        print("✅ Neuro system is more token-efficient!")
    elif token_efficiency > 1.0:
        print("❌ Traditional system is more token-efficient")
    else:
        print("➡️  Both systems have similar token efficiency")
    
    if speed_ratio < 1.0:
        print("✅ Neuro system is faster!")
    elif speed_ratio > 1.0:
        print("❌ Traditional system is faster")
    else:
        print("➡️  Both systems have similar speed")


if __name__ == "__main__":
    run_comparison_test()
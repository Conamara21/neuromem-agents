"""
Traditional RAG (Retrieval-Augmented Generation) System
Used as baseline for comparison with neuromorphic memory system
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import hashlib


@dataclass
class TraditionalMemoryNode:
    """Simple memory node for traditional RAG system"""
    id: str
    content: str
    embedding: np.ndarray
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TraditionalRAGSystem:
    """Baseline traditional RAG implementation for comparison"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory_nodes: Dict[str, TraditionalMemoryNode] = {}
        self.access_frequency = {}
        self.vector_store = []  # Simple list of embeddings
        self.node_ids = []      # Corresponding node IDs
        
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a document to the traditional RAG system"""
        # Generate unique ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        node_id = f"doc_{content_hash}_{int(time.time())}"
        
        # Create embedding (simplified)
        embedding = self._generate_embedding(content)
        
        # Create memory node
        node = TraditionalMemoryNode(
            id=node_id,
            content=content,
            embedding=embedding,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Store in memory
        self.memory_nodes[node_id] = node
        self.access_frequency[node_id] = 1
        
        # Add to vector store
        self.vector_store.append(embedding)
        self.node_ids.append(node_id)
        
        # Manage capacity
        if len(self.memory_nodes) > self.capacity:
            self._remove_least_recently_used()
        
        return node_id
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate simplified embedding (placeholder - replace with real model)"""
        # In practice, this would use a transformer model
        # For now, use a simple hash-based approach
        hash_val = hash(text) % (2**32)
        return np.array([float((hash_val >> i) & 1) for i in range(128)], dtype=np.float32)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[TraditionalMemoryNode, float]]:
        """Retrieve documents using traditional vector similarity"""
        query_embedding = self._generate_embedding(query).reshape(1, -1)
        
        if not self.vector_store:
            return []
        
        # Calculate similarities using manual cosine similarity
        similarities = []
        for i, stored_embedding in enumerate(self.vector_store):
            # Manual cosine similarity calculation
            dot_product = np.dot(query_embedding[0], stored_embedding)
            norm_query = np.linalg.norm(query_embedding[0])
            norm_stored = np.linalg.norm(stored_embedding)
            if norm_query == 0 or norm_stored == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_query * norm_stored)
            similarities.append((self.memory_nodes[self.node_ids[i]], float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update access frequency
        for node, _ in similarities[:top_k]:
            self.access_frequency[node.id] = self.access_frequency.get(node.id, 1) + 1
        
        return similarities[:top_k]
    
    def _remove_least_recently_used(self):
        """Remove least recently used item when capacity is exceeded"""
        if not self.memory_nodes:
            return
        
        # Find least accessed node
        least_accessed = min(self.access_frequency.items(), key=lambda x: x[1])
        node_id = least_accessed[0]
        
        # Remove from all structures
        if node_id in self.memory_nodes:
            del self.memory_nodes[node_id]
            del self.access_frequency[node_id]
            
            # Remove from vector store
            try:
                idx = self.node_ids.index(node_id)
                self.node_ids.pop(idx)
                self.vector_store.pop(idx)
            except ValueError:
                pass  # Node not found in vector store
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get traditional RAG system statistics"""
        total_size = sum(len(str(node)) for node in self.memory_nodes.values())
        
        return {
            "total_nodes": len(self.memory_nodes),
            "estimated_size_bytes": total_size,
            "average_access_frequency": np.mean(list(self.access_frequency.values())) if self.access_frequency else 0,
            "vector_store_size": len(self.vector_store)
        }


if __name__ == "__main__":
    # Example usage
    print("Traditional RAG System (Baseline)")
    print("=" * 40)
    
    # Initialize traditional RAG system
    rag_system = TraditionalRAGSystem(capacity=1000)
    
    # Add sample documents
    id1 = rag_system.add_document("The capital of France is Paris", {"category": "geography"})
    id2 = rag_system.add_document("I visited Paris last summer", {"category": "personal", "year": 2023})
    id3 = rag_system.add_document("Eiffel Tower is in Paris", {"category": "landmark"})
    
    # Retrieve related documents
    results = rag_system.retrieve("Paris", top_k=3)
    print(f"\nRetrieved {len(results)} related documents:")
    for i, (node, score) in enumerate(results, 1):
        print(f"{i}. {node.content} (similarity: {score:.3f})")
    
    # Get statistics
    stats = rag_system.get_statistics()
    print(f"\nRAG System Statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")

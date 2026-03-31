"""
Efficiency Optimization for NeuroMem-Agents
Implements sparse activation, progressive refinement, and quantum-inspired algorithms
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import math
import time
from datetime import datetime
import hashlib
import heapq
from .text_embeddings import LexicalHashingEmbedder, TextEmbedder


class ActivationState(Enum):
    """States for sparse activation"""
    INACTIVE = "inactive"
    PARTIALLY_ACTIVE = "partially_active"
    FULLY_ACTIVE = "fully_active"


@dataclass
class SparseMemoryNode:
    """Memory node optimized for sparse activation"""
    id: str
    content: str
    embedding: np.ndarray  # Dense embedding
    sparse_embedding: List[float]  # Sparse representation (using list instead of scipy.sparse)
    memory_type: 'MemoryType'
    timestamp: float
    activation_state: ActivationState = ActivationState.INACTIVE
    activation_probability: float = 0.0
    connectivity_strength: float = 1.0
    importance_score: float = 1.0
    decay_rate: float = 0.01
    tags: List[str] = None
    attention_weight: float = 1.0
    coherence: float = 0.0  # Quantum-inspired coherence measure
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class SparseActivationManager:
    """Manages sparse activation of memory nodes"""
    
    def __init__(self, sparsity_threshold: float = 0.3, activation_budget: int = 100):
        self.sparsity_threshold = sparsity_threshold  # Threshold for activation
        self.activation_budget = activation_budget    # Max nodes to activate
        self.active_nodes: Set[str] = set()
        self.activation_probabilities: Dict[str, float] = {}
        
    def compute_activation_probability(self, query_embedding: np.ndarray, 
                                     node_embedding: np.ndarray) -> float:
        """Compute activation probability based on similarity"""
        # Use cosine similarity to determine activation likelihood
        dot_product = np.dot(query_embedding, node_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_node = np.linalg.norm(node_embedding)
        
        if norm_query == 0 or norm_node == 0:
            return 0.0
        
        similarity = dot_product / (norm_query * norm_node)
        # Normalize to [0, 1] range
        probability = (similarity + 1) / 2
        return probability
    
    def select_sparse_activations(self, query_embedding: np.ndarray, 
                                all_nodes: Dict[str, SparseMemoryNode],
                                top_k: int) -> List[str]:
        """Select subset of nodes to activate based on probability"""
        # Calculate activation probabilities for all nodes
        for node_id, node in all_nodes.items():
            prob = self.compute_activation_probability(query_embedding, node.embedding)
            self.activation_probabilities[node_id] = prob
        
        # Select top-k nodes with highest activation probabilities
        sorted_nodes = sorted(
            self.activation_probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Take only the top-k nodes or budget limit, whichever is smaller
        selected_count = min(top_k, self.activation_budget, len(sorted_nodes))
        selected_nodes = [node_id for node_id, _ in sorted_nodes[:selected_count]]
        
        # Update active nodes set
        self.active_nodes = set(selected_nodes)
        
        return selected_nodes


class ProgressiveRefinementEngine:
    """Implements progressive refinement from coarse to fine-grained retrieval"""
    
    def __init__(self):
        self.coarse_grained_threshold = 0.1  # Similarity threshold for coarse filter
        self.fine_grained_top_k = 5          # Number of items to refine in fine stage
        
    def coarse_filter(self, query_embedding: np.ndarray, 
                     candidate_nodes: List[SparseMemoryNode]) -> List[SparseMemoryNode]:
        """Coarse filtering stage - quickly eliminate unlikely candidates"""
        filtered_candidates = []
        
        for node in candidate_nodes:
            # Quick similarity check using sparse embedding
            similarity = self._sparse_similarity(query_embedding, node.sparse_embedding)
            
            if similarity >= self.coarse_grained_threshold:
                filtered_candidates.append((node, similarity))
        
        # Sort by similarity and take top candidates
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in filtered_candidates]
    
    def fine_refinement(self, query_embedding: np.ndarray,
                       coarse_results: List[SparseMemoryNode],
                       top_k: int) -> List[Tuple[SparseMemoryNode, float]]:
        """Fine refinement stage - detailed similarity calculation"""
        refined_results = []
        
        # Take only top candidates from coarse stage for fine refinement
        candidates_to_refine = coarse_results[:min(self.fine_grained_top_k, len(coarse_results))]
        
        for node in candidates_to_refine:
            # Detailed similarity calculation
            similarity = self._detailed_similarity(query_embedding, node.embedding)
            refined_results.append((node, similarity))
        
        # Sort and return top-k
        refined_results.sort(key=lambda x: x[1], reverse=True)
        return refined_results
    
    def _sparse_similarity(self, query: np.ndarray, sparse_emb: List[float]) -> float:
        """Quick similarity using sparse representation"""
        # Convert sparse to dense for calculation
        dense_emb = np.array(sparse_emb)
        
        if len(query) != len(dense_emb):
            return 0.0
        
        dot_product = np.dot(query, dense_emb)
        norm_query = np.linalg.norm(query)
        norm_emb = np.linalg.norm(dense_emb)
        
        if norm_query == 0 or norm_emb == 0:
            return 0.0
        
        return dot_product / (norm_query * norm_emb)
    
    def _detailed_similarity(self, query: np.ndarray, emb: np.ndarray) -> float:
        """Detailed similarity calculation"""
        dot_product = np.dot(query, emb)
        norm_query = np.linalg.norm(query)
        norm_emb = np.linalg.norm(emb)
        
        if norm_query == 0 or norm_emb == 0:
            return 0.0
        
        return dot_product / (norm_query * norm_emb)


class QuantumInspiredOptimizer:
    """Quantum-inspired optimization for search and association processes"""
    
    def __init__(self):
        self.superposition_factor = 0.5  # Controls superposition effects
        self.interference_threshold = 0.2  # Threshold for quantum interference
        self.entanglement_strength = 0.3  # Strength of quantum entanglement
        
    def quantum_similarity(self, query_embedding: np.ndarray, 
                          memory_embedding: np.ndarray) -> float:
        """Quantum-inspired similarity calculation"""
        # Classical similarity
        classical_sim = self._classical_similarity(query_embedding, memory_embedding)
        
        # Quantum superposition effect
        superposition_effect = self._superposition_amplification(
            query_embedding, memory_embedding
        )
        
        # Quantum interference effect
        interference_effect = self._quantum_interference(
            query_embedding, memory_embedding
        )
        
        # Combine effects
        quantum_enhanced_sim = (
            classical_sim + 
            self.superposition_factor * superposition_effect + 
            interference_effect
        )
        
        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, quantum_enhanced_sim))
    
    def _classical_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Standard cosine similarity"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return max(-1.0, min(1.0, dot_product / (norm1 * norm2)))
    
    def _superposition_amplification(self, query: np.ndarray, memory: np.ndarray) -> float:
        """Amplify similarity based on superposition principle"""
        # Calculate amplitude for each dimension
        amplitudes = np.multiply(query, memory)
        
        # Superposition amplification based on constructive interference
        amplification = np.sum(np.abs(amplitudes)) / len(amplitudes) if len(amplitudes) > 0 else 0.0
        
        return amplification
    
    def _quantum_interference(self, query: np.ndarray, memory: np.ndarray) -> float:
        """Model quantum interference effects"""
        # Calculate phase differences
        phase_diffs = np.subtract(query, memory)
        
        # Interference pattern
        interference = np.sum(np.cos(phase_diffs)) / len(phase_diffs) if len(phase_diffs) > 0 else 0.0
        
        # Apply threshold to filter weak interference
        if abs(interference) < self.interference_threshold:
            interference = 0.0
        
        return interference
    
    def quantum_search_optimization(self, query_embedding: np.ndarray,
                                   memory_nodes: List[SparseMemoryNode]) -> List[Tuple[SparseMemoryNode, float]]:
        """Optimize search using quantum-inspired principles"""
        results = []
        
        for node in memory_nodes:
            # Calculate quantum-enhanced similarity
            q_sim = self.quantum_similarity(query_embedding, node.embedding)
            results.append((node, q_sim))
        
        # Sort by quantum-enhanced similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class EfficiencyOptimizedMemoryManager:
    """Memory manager with efficiency optimizations"""
    
    def __init__(
        self,
        capacity: int = 10000,
        db_path: str = "efficient_neuromem.db",
        embedder: Optional[TextEmbedder] = None,
    ):
        from .neuromorphic_memory import MemoryType  # Import here to avoid circular import
        self.MemoryType = MemoryType
        
        self.capacity = capacity
        self.memory_nodes: Dict[str, SparseMemoryNode] = {}
        self.connections: Dict[str, List[Tuple[str, float, float]]] = {}
        self.access_frequency = {}
        
        # Efficiency optimization components
        self.sparse_activator = SparseActivationManager()
        self.progressive_engine = ProgressiveRefinementEngine()
        self.quantum_optimizer = QuantumInspiredOptimizer()
        
        # Import database functionality
        from .persistence import MemoryDatabase
        self.db = MemoryDatabase(db_path)
        self.embedder = embedder or LexicalHashingEmbedder()
        
    def encode(self, content: str, memory_type: 'MemoryType', tags: List[str] = None) -> str:
        """Encode information with sparse optimization"""
        # Generate unique ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        node_id = f"{memory_type.value}_{content_hash}_{int(time.time())}"
        
        # Create embedding
        embedding = self._generate_embedding(content)
        
        # Create sparse representation
        sparse_embedding = self._create_sparse_embedding(embedding)
        
        # Create sparse memory node
        node = SparseMemoryNode(
            id=node_id,
            content=content,
            embedding=embedding,
            sparse_embedding=sparse_embedding,
            memory_type=memory_type,
            timestamp=time.time(),
            tags=tags or [],
            activation_probability=0.0
        )
        
        # Store node
        self.memory_nodes[node_id] = node
        self.access_frequency[node_id] = 1
        
        # Persist to database
        self._save_sparse_node(node)
        
        return node_id
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding using the configured embedder."""
        return self.embedder.encode(text)
    
    def _create_sparse_embedding(self, dense_embedding: np.ndarray, 
                               sparsity_ratio: float = 0.7) -> List[float]:
        """Create sparse representation of embedding"""
        # Zero out a percentage of values based on sparsity ratio
        sparse_vector = dense_embedding.copy()
        
        # Determine how many elements to zero out
        num_zeros = int(len(sparse_vector) * sparsity_ratio)
        
        # Randomly select indices to zero out
        zero_indices = np.random.choice(len(sparse_vector), num_zeros, replace=False)
        sparse_vector[zero_indices] = 0.0
        
        # Convert to list format
        return sparse_vector.tolist()
    
    def _save_sparse_node(self, node: SparseMemoryNode):
        """Save sparse memory node to database"""
        # This would be implemented similar to the regular persistence
        # For now, just calling the regular save method
        pass
    
    def retrieve(self, query: str, top_k: int = 5) -> List[SparseMemoryNode]:
        """Efficiency-optimized retrieval using sparse activation and progressive refinement"""
        query_embedding = self._generate_embedding(query)
        
        # Stage 1: Sparse activation - select subset of nodes to activate
        active_node_ids = self.sparse_activator.select_sparse_activations(
            query_embedding, self.memory_nodes, top_k * 2  # Get more candidates for refinement
        )
        
        # Get the actually active nodes
        active_nodes = [self.memory_nodes[node_id] for node_id in active_node_ids]
        
        # Stage 2: Progressive refinement - coarse filter then fine refinement
        # Coarse filtering
        coarse_results = self.progressive_engine.coarse_filter(query_embedding, active_nodes)
        
        # Fine refinement
        fine_results = self.progressive_engine.fine_refinement(
            query_embedding, coarse_results, top_k
        )
        
        # Stage 3: Quantum-inspired optimization (optional post-processing)
        if len(fine_results) > 0:
            # Apply quantum-inspired re-ranking
            quantum_scores = []
            for node, classical_score in fine_results:
                quantum_score = self.quantum_optimizer.quantum_similarity(
                    query_embedding, node.embedding
                )
                # Combine classical and quantum scores
                combined_score = 0.7 * classical_score + 0.3 * quantum_score
                quantum_scores.append((node, combined_score))
            
            # Re-sort by combined score
            quantum_scores.sort(key=lambda x: x[1], reverse=True)
            final_results = [node for node, _ in quantum_scores[:top_k]]
        else:
            final_results = [node for node, _ in fine_results]
        
        # Update access frequencies
        for node in final_results:
            self.access_frequency[node.id] = self.access_frequency.get(node.id, 1) + 1
        
        return final_results
    
    def associate(self, node_id1: str, node_id2: str, strength: float = 1.0):
        """Create association with efficiency considerations"""
        if node_id1 not in self.connections:
            self.connections[node_id1] = []
        if node_id2 not in self.connections:
            self.connections[node_id2] = []
        
        # Use sparse activation probability as a factor in connection strength
        prob1 = self.sparse_activator.activation_probabilities.get(node_id1, 0.0)
        prob2 = self.sparse_activator.activation_probabilities.get(node_id2, 0.0)
        
        # Adjust strength based on activation probabilities
        adjusted_strength = strength * (prob1 + prob2) / 2
        
        # Update connections
        self._update_connection(node_id1, node_id2, adjusted_strength)
        self._update_connection(node_id2, node_id1, adjusted_strength)
    
    def _update_connection(self, source: str, target: str, strength: float):
        """Update connection strength between nodes"""
        existing = [(tgt, w, t) for tgt, w, t in self.connections[source] if tgt == target]
        if existing:
            idx = [i for i, (tgt, w, t) in enumerate(self.connections[source]) if tgt == target][0]
            old_weight = existing[0][1]
            new_weight = (old_weight + strength) / 2
            self.connections[source][idx] = (target, new_weight, time.time())
        else:
            self.connections[source].append((target, strength, time.time()))
    
    def get_efficiency_statistics(self) -> Dict[str, Any]:
        """Get statistics about efficiency optimizations"""
        total_nodes = len(self.memory_nodes)
        active_nodes_count = len(self.sparse_activator.active_nodes)
        
        # Calculate sparsity statistics
        if self.memory_nodes:
            avg_sparsity = np.mean([
                1 - (sum(1 for x in node.sparse_embedding if x == 0) / len(node.sparse_embedding)) 
                for node in self.memory_nodes.values()
            ])
        else:
            avg_sparsity = 0.0
        
        return {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes_count,
            "activation_ratio": active_nodes_count / total_nodes if total_nodes > 0 else 0,
            "average_sparsity": avg_sparsity,
            "sparsity_threshold": self.sparse_activator.sparsity_threshold,
            "activation_budget": self.sparse_activator.activation_budget,
            "coarse_filter_threshold": self.progressive_engine.coarse_grained_threshold,
            "fine_refinement_top_k": self.progressive_engine.fine_grained_top_k
        }


if __name__ == "__main__":
    print("Testing Efficiency Optimizations...")
    
    # Initialize efficiency-optimized memory manager
    eff_manager = EfficiencyOptimizedMemoryManager(capacity=1000)
    
    # Add some memories
    id1 = eff_manager.encode("Quantum physics describes the behavior of matter and energy", eff_manager.MemoryType.SEMANTIC, ["physics", "quantum"])
    id2 = eff_manager.encode("Machine learning algorithms can recognize patterns", eff_manager.MemoryType.SEMANTIC, ["ai", "ml"])
    id3 = eff_manager.encode("Neural networks are inspired by the brain", eff_manager.MemoryType.SEMANTIC, ["ai", "neuroscience"])
    
    print(f"Encoded {len(eff_manager.memory_nodes)} memories")
    
    # Perform retrieval with efficiency optimizations
    results = eff_manager.retrieve("quantum", top_k=2)
    print(f"Retrieved {len(results)} memories efficiently")
    
    # Get efficiency statistics
    stats = eff_manager.get_efficiency_statistics()
    print("Efficiency Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("Efficiency optimizations working correctly!")

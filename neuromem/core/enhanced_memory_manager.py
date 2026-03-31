"""
Enhanced Memory Manager with Neural Plasticity Features
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math
import time
from datetime import datetime
import hashlib
import pickle
import json
from .persistence import MemoryDatabase, attach_persistence_methods
from .text_embeddings import LexicalHashingEmbedder, TextEmbedder


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
    last_activation_time: float = 0.0
    attention_weight: float = 1.0  # Attention gate weight
    
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


class STDPMechanism:
    """Spike-Timing Dependent Plasticity implementation"""
    
    def __init__(self, window_size: float = 0.02, tau_plus: float = 0.01, tau_minus: float = 0.01):
        self.window_size = window_size  # 20ms window
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.strengthen_factor = 0.1
        self.weaken_factor = 0.05
        
    def update_connection_strength(self, pre_activation_time: float, post_activation_time: float, 
                                  current_weight: float) -> float:
        """Update connection strength based on STDP rule"""
        time_diff = post_activation_time - pre_activation_time
        
        if abs(time_diff) < self.window_size:
            if time_diff > 0:  # Pre-synaptic neuron fires before post-synaptic
                # Long Term Potentiation (LTP) - strengthen connection
                weight_change = self.strengthen_factor * math.exp(-abs(time_diff) / self.tau_plus)
            else:  # Post-synaptic neuron fires before pre-synaptic
                # Long Term Depression (LTD) - weaken connection
                weight_change = -self.weaken_factor * math.exp(-abs(time_diff) / self.tau_minus)
            return current_weight + weight_change
        
        return current_weight


class MetaLearningController:
    """Controller for meta-learning and adaptive parameters"""
    
    def __init__(self):
        self.performance_history = []
        self.learning_rate = 0.1
        self.consolidation_threshold = 3
        self.forget_threshold = 0.1
        self.stdp_window = 0.02
        
    def update_parameters(self, task_performance: float, time_since_last_update: float):
        """Dynamically adjust learning parameters based on performance"""
        self.performance_history.append({
            'performance': task_performance,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        cutoff_time = time.time() - 3600  # Last hour
        self.performance_history = [p for p in self.performance_history 
                                   if p['timestamp'] > cutoff_time]
        
        if len(self.performance_history) > 3:
            # Calculate recent performance trend
            recent_perf = [p['performance'] for p in self.performance_history[-3:]]
            avg_perf = sum(recent_perf) / len(recent_perf)
            
            if avg_perf < 0.5:  # Poor performance
                # Increase learning rate to adapt faster
                self.learning_rate = min(0.5, self.learning_rate * 1.1)
                # Adjust consolidation threshold to retain more information
                self.consolidation_threshold = max(2, self.consolidation_threshold - 0.5)
            elif avg_perf > 0.8:  # Excellent performance
                # Decrease learning rate to stabilize
                self.learning_rate = max(0.01, self.learning_rate * 0.9)
                # Increase consolidation threshold
                self.consolidation_threshold = min(10, self.consolidation_threshold + 0.2)
                
        return {
            'learning_rate': self.learning_rate,
            'consolidation_threshold': self.consolidation_threshold,
            'forget_threshold': self.forget_threshold
        }


class AttentionGate:
    """Attention mechanism for selective memory processing"""
    
    def __init__(self, focus_strength: float = 0.5, relevance_threshold: float = 0.3):
        self.focus_strength = focus_strength
        self.relevance_threshold = relevance_threshold
        
    def compute_attention_weights(self, query: str, memory_nodes: Dict[str, MemoryNode], 
                                 query_embedding: np.ndarray) -> Dict[str, float]:
        """Compute attention weights for each memory node"""
        weights = {}
        
        for node_id, node in memory_nodes.items():
            # Calculate semantic similarity between query and memory
            relevance = self._calculate_relevance(query_embedding, node.embedding)
            
            # Apply attention gate based on relevance and focus
            attention_weight = self.focus_strength * relevance
            weights[node_id] = attention_weight
            
        return weights
    
    def _calculate_relevance(self, query_embedding: np.ndarray, memory_embedding: np.ndarray) -> float:
        """Calculate relevance between query and memory using cosine similarity"""
        dot_product = np.dot(query_embedding, memory_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_memory = np.linalg.norm(memory_embedding)
        
        if norm_query == 0 or norm_memory == 0:
            return 0.0
            
        similarity = dot_product / (norm_query * norm_memory)
        # Normalize to [0, 1] range
        return (similarity + 1) / 2


@attach_persistence_methods
class EnhancedMemoryManager:
    """Enhanced memory management system with neural plasticity features"""
    
    def __init__(
        self,
        capacity: int = 10000,
        db_path: str = "neuromem.db",
        embedder: Optional[TextEmbedder] = None,
    ):
        self.capacity = capacity
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.connections: Dict[str, List[Tuple[str, float, float]]] = {}  # (target_id, weight, last_update_time)
        self.working_memory_buffer = []  # Short-term storage (like hippocampus)
        self.long_term_memory = {}  # Consolidated memories (like cortex)
        self.access_frequency = {}  # Track memory access patterns
        self.snn = SpikingNeuralNetwork()
        self.current_context = {}
        self.db = MemoryDatabase(db_path)  # Initialize persistence layer
        self.embedder = embedder or LexicalHashingEmbedder()
        
        # Neural plasticity components
        self.stdp_mechanism = STDPMechanism()
        self.meta_controller = MetaLearningController()
        self.attention_gate = AttentionGate()

    def _coerce_memory_type(self, memory_type: Any) -> MemoryType:
        if isinstance(memory_type, MemoryType):
            return memory_type

        raw_value = getattr(memory_type, "value", memory_type)
        if isinstance(raw_value, str):
            return MemoryType(raw_value.lower())
        raise ValueError(f"Unsupported memory type: {memory_type!r}")
        
    def encode(self, content: str, memory_type: MemoryType, tags: List[str] = None) -> str:
        """Encode new information into memory nodes"""
        memory_type = self._coerce_memory_type(memory_type)

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
            tags=tags or [],
            last_activation_time=time.time()
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
        
        # Persist to database
        self.db.save_memory_node(node)
        
        return node_id
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding using the configured embedder."""
        return self.embedder.encode(text)
    
    def associate(self, node_id1: str, node_id2: str, strength: float = 1.0):
        """Create associative connections between memory nodes with STDP"""
        if node_id1 not in self.connections:
            self.connections[node_id1] = []
        if node_id2 not in self.connections:
            self.connections[node_id2] = []
            
        # Apply STDP to update connection strength
        current_time = time.time()
        if node_id1 in self.memory_nodes and node_id2 in self.memory_nodes:
            # Get last activation times
            time1 = self.memory_nodes[node_id1].last_activation_time
            time2 = self.memory_nodes[node_id2].last_activation_time
            
            # Update strength using STDP mechanism
            updated_strength = self.stdp_mechanism.update_connection_strength(
                time1, time2, strength
            )
        else:
            updated_strength = strength
            
        # Update connection strengths (bidirectional)
        self._update_connection(node_id1, node_id2, updated_strength, current_time)
        self._update_connection(node_id2, node_id1, updated_strength, current_time)
        
        # Persist connection to database
        self.db.save_connection(node_id1, node_id2, updated_strength)
        self.db.save_connection(node_id2, node_id1, updated_strength)
        
        # Strengthen connection based on Hebbian learning
        if node_id1 in self.memory_nodes and node_id2 in self.memory_nodes:
            node1 = self.memory_nodes[node_id1]
            node2 = self.memory_nodes[node_id2]
            node1.connectivity_strength = min(1.0, node1.connectivity_strength + 0.1 * updated_strength)
            node2.connectivity_strength = min(1.0, node2.connectivity_strength + 0.1 * updated_strength)
    
    def _update_connection(self, source: str, target: str, strength: float, update_time: float):
        """Update connection strength between nodes"""
        existing = [(tgt, w, t) for tgt, w, t in self.connections[source] if tgt == target]
        if existing:
            # Update existing connection
            idx = [i for i, (tgt, w, t) in enumerate(self.connections[source]) if tgt == target][0]
            old_weight = existing[0][1]
            new_weight = (old_weight + strength) / 2  # Average
            self.connections[source][idx] = (target, new_weight, update_time)
        else:
            # Add new connection
            self.connections[source].append((target, strength, update_time))
    
    def retrieve(self, query: str, top_k: int = 5, context: Dict = None) -> List[MemoryNode]:
        """Retrieve relevant memories using associative search with attention gating"""
        query_embedding = self._generate_embedding(query)
        
        # Compute attention weights for selective processing
        attention_weights = self.attention_gate.compute_attention_weights(
            query, self.memory_nodes, query_embedding
        )
        
        # Find similar memories
        similarities = []
        for node_id, node in self.memory_nodes.items():
            # Apply attention gating to similarity calculation
            attention_factor = attention_weights.get(node_id, 1.0)
            if attention_factor < self.attention_gate.relevance_threshold:
                continue  # Skip low-attention nodes
                
            sim = self._cosine_similarity(query_embedding, node.embedding)
            
            # Boost similarity based on recent access and associations
            freq_factor = math.log(self.access_frequency.get(node_id, 1) + 1)
            assoc_factor = self._association_bonus(node_id, context)
            
            # Apply attention weighting
            final_sim = sim * attention_factor * (1 + freq_factor * 0.1) * (1 + assoc_factor * 0.2)
            similarities.append((node, final_sim))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update access frequency and activation times
        current_time = time.time()
        for node, _ in similarities[:top_k]:
            self.access_frequency[node.id] = self.access_frequency.get(node.id, 1) + 1
            # Update last activation time for STDP calculations
            node.last_activation_time = current_time
            # Update attention weight based on recent relevance
            node.attention_weight = min(1.0, node.attention_weight + 0.05)
            
            # Persist updated access frequency to database
            self.db.save_access_frequency(node.id, self.access_frequency[node.id])
            
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
            for connected_node, weight, _ in self.connections[node_id]:
                if connected_node in self.access_frequency:
                    # Boost based on recent activation of connected nodes
                    activation = min(1.0, self.access_frequency[connected_node] / 10.0)
                    bonus += weight * activation
        
        return bonus
    
    def consolidate(self):
        """Consolidate working memory to long-term memory (like sleep/dreaming)"""
        # Get current parameters from meta controller
        params = self.meta_controller.update_parameters(task_performance=0.8, time_since_last_update=0)
        consolidation_threshold = params['consolidation_threshold']
        
        # Process items in working memory buffer
        for node_id in self.working_memory_buffer:
            if node_id not in self.memory_nodes:
                continue
                
            node = self.memory_nodes[node_id]
            
            # Only consolidate highly accessed items (adjusted by meta learning)
            access_count = self.access_frequency.get(node_id, 1)
            if access_count > consolidation_threshold:  # Threshold adjusted by meta learning
                # Move to long-term memory
                self.long_term_memory[node_id] = node
                # Persist to long-term memory table
                self.db.save_to_long_term_memory(node_id, "high_access_frequency")
                # Reduce working memory presence
                node.memory_type = MemoryType.SEMANTIC  # Upgrade to semantic memory
        
        # Clear working memory buffer periodically
        if len(self.working_memory_buffer) > 50:
            # Update database with current buffer state
            for pos, node_id in enumerate(self.working_memory_buffer[-20:]):
                self.db.save_to_working_memory_buffer(node_id, pos)
            self.working_memory_buffer = self.working_memory_buffer[-20:]  # Keep recent items
    
    def forget(self, decay_threshold: float = 0.1):
        """Implement active forgetting mechanism with meta learning"""
        # Get current parameters from meta controller
        params = self.meta_controller.update_parameters(task_performance=0.8, time_since_last_update=0)
        decay_threshold = params['forget_threshold']
        
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
            # Remove from database as well
            # Note: In a real implementation, you might want to mark as deleted rather than remove entirely
    
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
            "average_access_frequency": np.mean(list(self.access_frequency.values())) if self.access_frequency else 0,
            "working_memory_size": len(self.working_memory_buffer),
            "long_term_memory_size": len(self.long_term_memory),
            "learning_rate": self.meta_controller.learning_rate,
            "consolidation_threshold": self.meta_controller.consolidation_threshold,
            "stdp_window": self.stdp_mechanism.window_size
        }


if __name__ == "__main__":
    # Example usage with neural plasticity
    print("NeuroMem-Agents: Enhanced Neuromorphic Memory System with Neural Plasticity")
    print("=" * 80)
    
    # Initialize enhanced memory manager with plasticity features
    mem_manager = EnhancedMemoryManager(capacity=1000, db_path="test_enhanced_neuromem.db")
    
    # Encode some sample memories
    id1 = mem_manager.encode("The capital of France is Paris", MemoryType.SEMANTIC, tags=["geography"])
    id2 = mem_manager.encode("I visited Paris last summer", MemoryType.EPISODIC, tags=["personal", "travel"])
    id3 = mem_manager.encode("Eiffel Tower is in Paris", MemoryType.SEMANTIC, tags=["landmark"])
    
    # Create associations (with STDP mechanism)
    mem_manager.associate(id1, id2, 0.8)
    mem_manager.associate(id1, id3, 0.9)
    
    # Perform consolidation (with meta learning adjustments)
    mem_manager.consolidate()
    
    # Retrieve related memories (with attention gating)
    results = mem_manager.retrieve("Paris", top_k=3)
    print(f"\nRetrieved {len(results)} related memories:")
    for i, node in enumerate(results, 1):
        print(f"{i}. {node.content}")
    
    # Get statistics including plasticity parameters
    stats = mem_manager.get_statistics()
    print(f"\nMemory Statistics (with neural plasticity):")
    for key, value in stats.items():
        print(f"- {key}: {value}")
    
    # Save to database
    mem_manager.save_to_db()
    print("\nEnhanced memory system with neural plasticity saved to database!")
    
    # Create a new instance and load from database
    print("\nLoading from database...")
    new_mem_manager = EnhancedMemoryManager(capacity=1000, db_path="test_enhanced_neuromem.db")
    new_mem_manager.load_from_db()
    
    # Verify loaded data
    stats = new_mem_manager.get_statistics()
    print(f"Loaded enhanced memory system with {stats['total_nodes']} nodes")
    
    # Test retrieval on loaded system
    results = new_mem_manager.retrieve("Paris", top_k=3)
    print(f"\nRetrieved from loaded system: {len(results)} memories")
    for i, node in enumerate(results, 1):
        print(f"{i}. {node.content}")

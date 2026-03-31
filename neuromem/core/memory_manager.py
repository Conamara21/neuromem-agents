"""
Enhanced Memory Manager with Persistence Support
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import time
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .persistence import MemoryDatabase, attach_persistence_methods
from .text_embeddings import LexicalHashingEmbedder, TextEmbedder, TOKEN_PATTERN


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
        return float(potential / (1 + abs(potential)))

    def generate_spike(self, activation: float) -> bool:
        """Generate spike if activation exceeds threshold"""
        return activation >= self.threshold


@attach_persistence_methods
class MemoryManager:
    """Main memory management system implementing neuromorphic principles"""

    def __init__(
        self,
        capacity: int = 10000,
        db_path: str = "neuromem.db",
        embedder: Optional[TextEmbedder] = None,
    ):
        self.capacity = capacity
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.connections: Dict[str, List[Tuple[str, float]]] = {}
        self.working_memory_buffer = []
        self.long_term_memory = {}
        self.access_frequency = {}
        self.snn = SpikingNeuralNetwork()
        self.current_context = {}
        self.db = MemoryDatabase(db_path)
        self.embedder = embedder or LexicalHashingEmbedder()
        self._initialize_runtime_state()

    def _initialize_runtime_state(self):
        """Initialize ephemeral indices used to accelerate retrieval."""
        if not hasattr(self, "inverted_index"):
            self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        elif not isinstance(self.inverted_index, defaultdict):
            self.inverted_index = defaultdict(set, self.inverted_index)

        if not hasattr(self, "content_index"):
            self.content_index: Dict[str, Set[str]] = defaultdict(set)
        elif not isinstance(self.content_index, defaultdict):
            self.content_index = defaultdict(set, self.content_index)

        if not hasattr(self, "node_index_terms"):
            self.node_index_terms: Dict[str, Set[str]] = {}
        if not hasattr(self, "token_document_frequency"):
            self.token_document_frequency: Dict[str, int] = {}
        if not hasattr(self, "embedding_norms"):
            self.embedding_norms: Dict[str, float] = {}
        if not hasattr(self, "recent_accessed_nodes"):
            self.recent_accessed_nodes: Deque[str] = deque(maxlen=128)
        elif not isinstance(self.recent_accessed_nodes, deque):
            self.recent_accessed_nodes = deque(self.recent_accessed_nodes, maxlen=128)

    def _index_terms_for_node(self, node: MemoryNode) -> Set[str]:
        terms = set(TOKEN_PATTERN.findall(node.content.lower()))
        terms.add(f"type:{node.memory_type.value}")
        for tag in node.tags or []:
            normalized_tag = tag.strip().lower()
            if not normalized_tag:
                continue
            terms.add(f"tag:{normalized_tag}")
            terms.update(
                f"tag_token:{token}"
                for token in TOKEN_PATTERN.findall(normalized_tag)
            )
        return terms

    def _index_node(self, node: MemoryNode):
        self._initialize_runtime_state()
        self.content_index[node.content].add(node.id)
        terms = self._index_terms_for_node(node)
        self.node_index_terms[node.id] = terms
        for term in terms:
            postings = self.inverted_index[term]
            postings.add(node.id)
            self.token_document_frequency[term] = len(postings)
        self.embedding_norms[node.id] = float(np.linalg.norm(node.embedding))

    def _remove_node_from_indices(self, node: MemoryNode):
        self._initialize_runtime_state()

        content_postings = self.content_index.get(node.content)
        if content_postings is not None:
            content_postings.discard(node.id)
            if not content_postings:
                self.content_index.pop(node.content, None)

        for term in self.node_index_terms.pop(node.id, set()):
            postings = self.inverted_index.get(term)
            if postings is None:
                continue
            postings.discard(node.id)
            if postings:
                self.token_document_frequency[term] = len(postings)
            else:
                self.inverted_index.pop(term, None)
                self.token_document_frequency.pop(term, None)

        self.embedding_norms.pop(node.id, None)

    def _rebuild_indices(self):
        self._initialize_runtime_state()
        self.inverted_index.clear()
        self.content_index.clear()
        self.node_index_terms.clear()
        self.token_document_frequency.clear()
        self.embedding_norms.clear()
        self.recent_accessed_nodes.clear()

        for node in self.memory_nodes.values():
            self._index_node(node)

    def _remove_node(self, node_id: str):
        node = self.memory_nodes.get(node_id)
        if node is None:
            return

        self._remove_node_from_indices(node)
        del self.memory_nodes[node_id]
        self.access_frequency.pop(node_id, None)
        self.long_term_memory.pop(node_id, None)
        self.working_memory_buffer = [
            working_id for working_id in self.working_memory_buffer if working_id != node_id
        ]

        if node_id in self.connections:
            del self.connections[node_id]
        for source_id, linked_nodes in list(self.connections.items()):
            filtered_links = [link for link in linked_nodes if link[0] != node_id]
            if len(filtered_links) != len(linked_nodes):
                self.connections[source_id] = filtered_links

        self.recent_accessed_nodes = deque(
            (recent_id for recent_id in self.recent_accessed_nodes if recent_id != node_id),
            maxlen=self.recent_accessed_nodes.maxlen,
        )

    def _record_access(self, node_id: str):
        self._initialize_runtime_state()
        self.recent_accessed_nodes.append(node_id)

    def _candidate_budget(self, top_k: int) -> int:
        node_count = len(self.memory_nodes)
        if node_count <= 128:
            return node_count
        return min(
            node_count,
            max(128, top_k * 32, int(math.sqrt(node_count) * 10)),
        )

    def _candidate_seed_ids(self, context: Optional[Dict[str, Any]]) -> Set[str]:
        seeds = {
            node_id
            for node_id in list(self.recent_accessed_nodes)[-16:]
            if self.access_frequency.get(node_id, 1) > 1
        }
        if not context:
            return seeds

        for key in ("focus_nodes", "recent_nodes", "node_ids"):
            values = context.get(key, [])
            if isinstance(values, str):
                values = [values]
            seeds.update(value for value in values if value in self.memory_nodes)
        return seeds

    def _required_node_ids(self, context: Optional[Dict[str, Any]]) -> Optional[Set[str]]:
        if not context:
            return None

        raw_tags = context.get("required_tags", [])
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]

        normalized_tags = [
            str(tag).strip().lower()
            for tag in raw_tags
            if str(tag).strip()
        ]
        if not normalized_tags:
            return None

        required_ids: Optional[Set[str]] = None
        for tag in normalized_tags:
            tag_postings = set(self.inverted_index.get(f"tag:{tag}", set()))
            if required_ids is None:
                required_ids = tag_postings
            else:
                required_ids &= tag_postings

            if not required_ids:
                return set()

        return required_ids or set()

    def _expand_association_candidates(
        self,
        seeds: Iterable[str],
        candidate_scores: Dict[str, float],
        bonus_scale: float,
    ):
        for seed_id in seeds:
            for connection in self.connections.get(seed_id, []):
                target_id, weight = connection[:2]
                if target_id in self.memory_nodes:
                    candidate_scores[target_id] = (
                        candidate_scores.get(target_id, 0.0) + weight * bonus_scale
                    )

    def _candidate_node_ids(
        self,
        query: str,
        top_k: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[str]]:
        self._initialize_runtime_state()

        node_count = len(self.memory_nodes)
        if node_count <= self._candidate_budget(top_k):
            return None

        candidate_scores: Dict[str, float] = {}
        exact_ids = self.content_index.get(query, set())
        for node_id in exact_ids:
            candidate_scores[node_id] = candidate_scores.get(node_id, 0.0) + 1_000_000.0

        query_terms = set(TOKEN_PATTERN.findall(query.lower()))
        for term in query_terms:
            postings = self.inverted_index.get(term)
            if not postings:
                continue
            document_frequency = max(1, self.token_document_frequency.get(term, len(postings)))
            idf_weight = 1.0 + math.log((node_count + 1) / (document_frequency + 1))
            for node_id in postings:
                candidate_scores[node_id] = candidate_scores.get(node_id, 0.0) + idf_weight

        if context:
            for tag in context.get("tags", []):
                tag_term = f"tag:{str(tag).strip().lower()}"
                for node_id in self.inverted_index.get(tag_term, set()):
                    candidate_scores[node_id] = candidate_scores.get(node_id, 0.0) + 2.0

        association_seeds = self._candidate_seed_ids(context)
        association_seeds.update(exact_ids)
        self._expand_association_candidates(
            association_seeds,
            candidate_scores,
            bonus_scale=1.25,
        )

        if not candidate_scores:
            return None

        budget = self._candidate_budget(top_k)
        ranked_candidates = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        candidate_ids = [node_id for node_id, _ in ranked_candidates[:budget]]

        if exact_ids:
            exact_only = [node_id for node_id in exact_ids if node_id not in candidate_ids]
            candidate_ids = exact_only + candidate_ids
            candidate_ids = candidate_ids[:budget]

        if len(candidate_ids) < top_k:
            return None
        return candidate_ids

    def _coerce_memory_type(self, memory_type: Any) -> MemoryType:
        if isinstance(memory_type, MemoryType):
            return memory_type

        raw_value = getattr(memory_type, "value", memory_type)
        if isinstance(raw_value, str):
            return MemoryType(raw_value.lower())
        raise ValueError(f"Unsupported memory type: {memory_type!r}")

    def encode(self, content: str, memory_type: MemoryType, tags: List[str] = None) -> str:
        """Encode new information into memory nodes"""
        self._initialize_runtime_state()
        memory_type = self._coerce_memory_type(memory_type)

        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        node_id = f"{memory_type.value}_{content_hash}_{int(time.time())}"

        embedding = self._generate_embedding(content)

        node = MemoryNode(
            id=node_id,
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            timestamp=time.time(),
            tags=tags or [],
        )

        self.memory_nodes[node_id] = node
        self.access_frequency[node_id] = 1
        self._index_node(node)
        self._record_access(node_id)

        if memory_type == MemoryType.WORKING:
            self.working_memory_buffer.append(node_id)
            if len(self.working_memory_buffer) > 100:
                removed = self.working_memory_buffer.pop(0)
                self._remove_node(removed)

        self.db.save_memory_node(node)
        return node_id

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding using the configured embedder."""
        return self.embedder.encode(text)

    def associate(self, node_id1: str, node_id2: str, strength: float = 1.0):
        """Create associative connections between memory nodes"""
        self._initialize_runtime_state()
        if node_id1 not in self.connections:
            self.connections[node_id1] = []
        if node_id2 not in self.connections:
            self.connections[node_id2] = []

        self._update_connection(node_id1, node_id2, strength)
        self._update_connection(node_id2, node_id1, strength)

        save_batch = getattr(self.db, "save_connections_batch", None)
        if callable(save_batch):
            save_batch(
                [
                    (node_id1, node_id2, strength),
                    (node_id2, node_id1, strength),
                ]
            )
        else:
            self.db.save_connection(node_id1, node_id2, strength)
            self.db.save_connection(node_id2, node_id1, strength)

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

    def retrieve(self, query: str, top_k: int = 5, context: Dict = None) -> List[MemoryNode]:
        """Retrieve relevant memories using associative search"""
        self._initialize_runtime_state()
        query_embedding = self._generate_embedding(query)
        query_norm = float(np.linalg.norm(query_embedding))
        candidate_ids = self._candidate_node_ids(query, top_k=top_k, context=context)
        required_node_ids = self._required_node_ids(context)

        if required_node_ids == set():
            return []

        if candidate_ids is None:
            if required_node_ids is None:
                candidate_items = list(self.memory_nodes.items())
            else:
                candidate_items = [
                    (node_id, self.memory_nodes[node_id])
                    for node_id in required_node_ids
                    if node_id in self.memory_nodes
                ]
        else:
            scoped_candidate_ids = candidate_ids
            if required_node_ids is not None:
                scoped_candidate_ids = [
                    node_id for node_id in candidate_ids if node_id in required_node_ids
                ]
                if len(scoped_candidate_ids) < min(top_k, len(required_node_ids)):
                    scoped_candidate_ids.extend(
                        node_id
                        for node_id in required_node_ids
                        if node_id not in scoped_candidate_ids
                    )
            candidate_items = [
                (node_id, self.memory_nodes[node_id])
                for node_id in scoped_candidate_ids
                if node_id in self.memory_nodes
            ]

        similarities = []
        for node_id, node in candidate_items:
            sim = self._cosine_similarity_with_norms(
                query_embedding,
                query_norm,
                node.embedding,
                self.embedding_norms.get(node_id),
            )

            freq_factor = math.log(self.access_frequency.get(node_id, 1) + 1)
            assoc_factor = self._association_bonus(node_id, context)

            final_sim = sim * (1 + freq_factor * 0.1) * (1 + assoc_factor * 0.2)
            similarities.append((node, final_sim))

        similarities.sort(key=lambda item: item[1], reverse=True)

        updated_frequencies = []
        for node, _ in similarities[:top_k]:
            self.access_frequency[node.id] = self.access_frequency.get(node.id, 1) + 1
            updated_frequencies.append((node.id, self.access_frequency[node.id]))
            self._record_access(node.id)

        if updated_frequencies:
            save_batch = getattr(self.db, "save_access_frequencies", None)
            if callable(save_batch):
                save_batch(updated_frequencies)
            else:
                for node_id, frequency in updated_frequencies:
                    self.db.save_access_frequency(node_id, frequency)

        return [node for node, _ in similarities[:top_k]]

    def _cosine_similarity_with_norms(
        self,
        vec1: np.ndarray,
        norm1: float,
        vec2: np.ndarray,
        norm2: Optional[float] = None,
    ) -> float:
        """Calculate cosine similarity while reusing precomputed norms."""
        if norm2 is None:
            norm2 = float(np.linalg.norm(vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return self._cosine_similarity_with_norms(
            vec1,
            float(np.linalg.norm(vec1)),
            vec2,
            float(np.linalg.norm(vec2)),
        )

    def _association_bonus(self, node_id: str, context: Dict = None) -> float:
        """Calculate bonus based on associative connections"""
        if context is None:
            context = {}

        bonus = 0.0
        if node_id in self.connections:
            for connected_node, weight in self.connections[node_id]:
                if connected_node in self.access_frequency:
                    activation = min(1.0, self.access_frequency[connected_node] / 10.0)
                    bonus += weight * activation

        return bonus

    def consolidate(self):
        """Consolidate working memory to long-term memory (like sleep/dreaming)"""
        for node_id in self.working_memory_buffer:
            if node_id not in self.memory_nodes:
                continue

            node = self.memory_nodes[node_id]
            access_count = self.access_frequency.get(node_id, 1)
            if access_count > 3:
                self.long_term_memory[node_id] = node
                self.db.save_to_long_term_memory(node_id, "high_access_frequency")
                node.memory_type = MemoryType.SEMANTIC

        if len(self.working_memory_buffer) > 50:
            for pos, node_id in enumerate(self.working_memory_buffer[-20:]):
                self.db.save_to_working_memory_buffer(node_id, pos)
            self.working_memory_buffer = self.working_memory_buffer[-20:]

    def forget(self, decay_threshold: float = 0.1):
        """Implement active forgetting mechanism"""
        current_time = time.time()
        to_remove = []

        for node_id, node in self.memory_nodes.items():
            age = current_time - node.timestamp
            decay_factor = math.exp(-node.decay_rate * age)
            access_factor = 1.0 / (self.access_frequency.get(node_id, 1) ** 0.5)
            forget_score = decay_factor * access_factor

            if forget_score < decay_threshold:
                to_remove.append(node_id)

        for node_id in to_remove:
            self._remove_node(node_id)

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
            "connection_density": (
                sum(len(conns) for conns in self.connections.values()) / max(1, len(self.memory_nodes))
            ),
            "average_access_frequency": (
                np.mean(list(self.access_frequency.values())) if self.access_frequency else 0
            ),
            "working_memory_size": len(self.working_memory_buffer),
            "long_term_memory_size": len(self.long_term_memory),
        }


if __name__ == "__main__":
    print("NeuroMem-Agents: Neuromorphic Memory System with Persistence")
    print("=" * 60)

    mem_manager = MemoryManager(capacity=1000, db_path="test_neuromem.db")

    id1 = mem_manager.encode("The capital of France is Paris", MemoryType.SEMANTIC, tags=["geography"])
    id2 = mem_manager.encode("I visited Paris last summer", MemoryType.EPISODIC, tags=["personal", "travel"])
    id3 = mem_manager.encode("Eiffel Tower is in Paris", MemoryType.SEMANTIC, tags=["landmark"])

    mem_manager.associate(id1, id2, 0.8)
    mem_manager.associate(id1, id3, 0.9)
    mem_manager.consolidate()

    results = mem_manager.retrieve("Paris", top_k=3)
    print(f"\nRetrieved {len(results)} related memories:")
    for i, node in enumerate(results, 1):
        print(f"{i}. {node.content}")

    stats = mem_manager.get_statistics()
    print(f"\nMemory Statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")

    mem_manager.save_to_db()
    print("\nMemory system saved to database!")

    print("\nLoading from database...")
    new_mem_manager = MemoryManager(capacity=1000, db_path="test_neuromem.db")
    new_mem_manager.load_from_db()

    stats = new_mem_manager.get_statistics()
    print(f"Loaded memory system with {stats['total_nodes']} nodes")

    results = new_mem_manager.retrieve("Paris", top_k=3)
    print(f"\nRetrieved from loaded system: {len(results)} memories")
    for i, node in enumerate(results, 1):
        print(f"{i}. {node.content}")

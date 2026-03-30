"""
Comprehensive benchmark test comparing token consumption and memory usage
between neuromorphic and traditional memory systems
"""

import time
import sys
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import hashlib


class MemoryType(Enum):
    SENSORY = "sensory"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryNode:
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


class BenchmarkNeuroMemoryManager:
    """Benchmark version of neuromorphic memory system"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.connections: Dict[str, List[Tuple[str, float]]] = {}
        self.working_memory_buffer = []
        self.long_term_memory = {}
        self.access_frequency = {}
        self.token_counter = 0
        self.operation_log = []
        
    def encode(self, content: str, memory_type: MemoryType, tags: List[str] = None) -> str:
        """Encode with token counting"""
        tokens_used = len(content.split())
        self.token_counter += tokens_used
        self.operation_log.append(('encode', tokens_used, len(self.memory_nodes)))
        
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        node_id = f"{memory_type.value}_{content_hash}_{int(time.time())}"
        
        # Simple embedding
        hash_val = int(hashlib.sha256(content.encode()).hexdigest(), 16)
        embedding = [float((hash_val >> (i * 8)) & 0xFF) / 255.0 for i in range(16)]
        
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
    
    def associate(self, node_id1: str, node_id2: str, strength: float = 1.0):
        """Create association with token counting"""
        self.token_counter += 1  # Minimal overhead for association
        self.operation_log.append(('associate', 1, len(self.memory_nodes)))
        
        for node_id in [node_id1, node_id2]:
            if node_id not in self.connections:
                self.connections[node_id] = []
        
        self._update_connection(node_id1, node_id2, strength)
        self._update_connection(node_id2, node_id1, strength)
    
    def _update_connection(self, source: str, target: str, strength: float):
        """Update connection strength"""
        existing = [(tgt, w) for tgt, w in self.connections[source] if tgt == target]
        if existing:
            idx = [i for i, (tgt, _) in enumerate(self.connections[source]) if tgt == target][0]
            old_weight = existing[0][1]
            new_weight = (old_weight + strength) / 2
            self.connections[source][idx] = (target, new_weight)
        else:
            self.connections[source].append((target, strength))
    
    def retrieve(self, query: str, top_k: int = 3) -> List[MemoryNode]:
        """Retrieve with token counting"""
        query_tokens = len(query.split())
        self.token_counter += query_tokens
        
        # Simple embedding for query
        hash_val = int(hashlib.sha256(query.encode()).hexdigest(), 16)
        query_embedding = [float((hash_val >> (i * 8)) & 0xFF) / 255.0 for i in range(16)]
        
        similarities = []
        for node_id, node in self.memory_nodes.items():
            sim = self._simple_similarity(query_embedding, node.embedding)
            
            # Boost with associations
            assoc_boost = self._association_boost(node_id)
            final_sim = sim * (1 + assoc_boost * 0.3)
            similarities.append((node, final_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update access frequency and count result tokens
        results = []
        for node, _ in similarities[:top_k]:
            self.access_frequency[node.id] = self.access_frequency.get(node.id, 1) + 1
            result_tokens = len(node.content.split())
            self.token_counter += result_tokens
            results.append(node)
        
        self.operation_log.append(('retrieve', query_tokens + sum(len(r.content.split()) for r in results), len(self.memory_nodes)))
        return results
    
    def _simple_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Simple similarity calculation"""
        if len(vec1) != len(vec2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0.0
    
    def _association_boost(self, node_id: str) -> float:
        """Boost based on associations"""
        boost = 0.0
        if node_id in self.connections:
            for connected_node, weight in self.connections[node_id]:
                if connected_node in self.access_frequency:
                    activation = min(1.0, self.access_frequency[connected_node] / 10.0)
                    boost += weight * activation
        return boost
    
    def get_memory_size_estimate(self) -> int:
        """Get estimated memory usage using sys.getsizeof"""
        total_size = sum(sys.getsizeof(str(node.__dict__)) for node in self.memory_nodes.values())
        total_size += sum(sys.getsizeof(str(conns)) for conns in self.connections.values())
        total_size += sum(sys.getsizeof(freq) for freq in self.access_frequency.values())
        return total_size


class BenchmarkTraditionalRAG:
    """Benchmark version of traditional RAG system"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.documents = []
        self.embeddings = []
        self.doc_ids = []
        self.access_frequency = {}
        self.token_counter = 0
        self.operation_log = []
        
    def add_document(self, content: str, metadata: Dict = None) -> str:
        """Add document with token counting"""
        tokens_used = len(content.split())
        self.token_counter += tokens_used
        self.operation_log.append(('add_doc', tokens_used, len(self.documents)))
        
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        doc_id = f"doc_{content_hash}_{int(time.time())}"
        
        # Simple embedding
        hash_val = int(hashlib.sha256(content.encode()).hexdigest(), 16)
        embedding = [float((hash_val >> (i * 8)) & 0xFF) / 255.0 for i in range(16)]
        
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
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve with token counting"""
        query_tokens = len(query.split())
        self.token_counter += query_tokens
        
        # Simple embedding for query
        hash_val = int(hashlib.sha256(query.encode()).hexdigest(), 16)
        query_embedding = [float((hash_val >> (i * 8)) & 0xFF) / 255.0 for i in range(16)]
        
        if not self.embeddings:
            return []
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self._simple_similarity(query_embedding, doc_embedding)
            similarities.append((self.documents[i], sim, self.doc_ids[i]))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update access frequency and count result tokens
        results = []
        for doc, sim, doc_id in similarities[:top_k]:
            self.access_frequency[doc_id] = self.access_frequency.get(doc_id, 1) + 1
            result_tokens = len(doc.split())
            self.token_counter += result_tokens
            results.append((doc, sim))
        
        self.operation_log.append(('retrieve', query_tokens + sum(len(r[0].split()) for r in results), len(self.documents)))
        return results
    
    def _simple_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Simple similarity calculation"""
        if len(vec1) != len(vec2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0.0
    
    def get_memory_size_estimate(self) -> int:
        """Get estimated memory usage using sys.getsizeof"""
        total_size = sum(sys.getsizeof(doc) for doc in self.documents)
        total_size += sum(sys.getsizeof(str(embedding)) for embedding in self.embeddings)
        total_size += sum(sys.getsizeof(freq) for freq in self.access_frequency.values())
        return total_size


def run_comprehensive_benchmark():
    """Run comprehensive benchmark test"""
    print("🔬 COMPREHENSIVE BENCHMARK TEST")
    print("=" * 50)
    print("Comparing NeuroMem vs Traditional RAG Systems")
    print()
    
    # Define test scenarios
    test_documents = [
        "The human brain contains approximately 86 billion neurons",
        "Neural networks are computing systems inspired by the human brain",
        "Machine learning algorithms can recognize patterns in data",
        "Deep learning uses multiple layers of neural networks",
        "Artificial intelligence aims to simulate human cognitive functions",
        "Cognitive science studies the mind and its processes",
        "Memory consolidation transfers information from short-term to long-term storage",
        "Working memory holds information temporarily for cognitive tasks",
        "Episodic memory records personal experiences and events",
        "Semantic memory stores general knowledge and concepts"
    ]
    
    test_queries = [
        "neural networks",
        "human brain",
        "machine learning",
        "memory systems",
        "artificial intelligence"
    ]
    
    print(f"📊 Test Configuration:")
    print(f"   - Documents: {len(test_documents)}")
    print(f"   - Queries: {len(test_queries)}")
    print()
    
    # Initialize systems
    print("🏗️  Initializing Memory Systems...")
    neuro_system = BenchmarkNeuroMemoryManager()
    trad_system = BenchmarkTraditionalRAG()
    
    # Phase 1: Document Encoding
    print("\n📝 Phase 1: Document Encoding")
    print("-" * 30)
    
    # Add documents to neuromorphic system
    neuro_doc_ids = []
    for i, doc in enumerate(test_documents):
        doc_type = MemoryType.SEMANTIC if i % 2 == 0 else MemoryType.EPISODIC
        doc_id = neuro_system.encode(doc, doc_type)
        neuro_doc_ids.append(doc_id)
    
    # Create associations in neuromorphic system
    for i in range(len(neuro_doc_ids) - 1):
        neuro_system.associate(neuro_doc_ids[i], neuro_doc_ids[i+1], 0.5)
    
    # Add documents to traditional system
    trad_doc_ids = []
    for doc in test_documents:
        doc_id = trad_system.add_document(doc)
        trad_doc_ids.append(doc_id)
    
    print(f"   NeuroMem - Tokens used: {neuro_system.token_counter}")
    print(f"   NeuroMem - Memory estimate: {neuro_system.get_memory_size_estimate()} bytes")
    print(f"   Traditional - Tokens used: {trad_system.token_counter}")
    print(f"   Traditional - Memory estimate: {trad_system.get_memory_size_estimate()} bytes")
    
    # Phase 2: Query Processing
    print("\n🔍 Phase 2: Query Processing")
    print("-" * 30)
    
    neuro_total_retrieval_time = 0
    trad_total_retrieval_time = 0
    
    neuro_query_tokens = neuro_system.token_counter
    trad_query_tokens = trad_system.token_counter
    
    for query in test_queries:
        print(f"   Query: '{query}'")
        
        # Test neuromorphic system
        start_time = time.time()
        neuro_results = neuro_system.retrieve(query, top_k=3)
        neuro_time = time.time() - start_time
        neuro_total_retrieval_time += neuro_time
        
        # Test traditional system
        start_time = time.time()
        trad_results = trad_system.retrieve(query, top_k=3)
        trad_time = time.time() - start_time
        trad_total_retrieval_time += trad_time
        
        print(f"     NeuroMem: {len(neuro_results)} results in {neuro_time:.4f}s")
        print(f"     Traditional: {len(trad_results)} results in {trad_time:.4f}s")
    
    # Calculate final metrics
    neuro_final_tokens = neuro_system.token_counter
    trad_final_tokens = trad_system.token_counter
    
    neuro_final_memory = neuro_system.get_memory_size_estimate()
    trad_final_memory = trad_system.get_memory_size_estimate()
    
    # Phase 3: Results Analysis
    print("\n📈 RESULTS ANALYSIS")
    print("-" * 30)
    
    print("Token Consumption:")
    print(f"   NeuroMem Total: {neuro_final_tokens}")
    print(f"   Traditional Total: {trad_final_tokens}")
    if trad_final_tokens > 0:
        token_efficiency = neuro_final_tokens / trad_final_tokens
        print(f"   Token Efficiency Ratio: {token_efficiency:.2f}x")
        if token_efficiency < 1.0:
            print(f"   🎯 NeuroMem uses {(1-token_efficiency)*100:.1f}% fewer tokens")
        else:
            print(f"   ⚠️  Traditional uses {(token_efficiency-1)*100:.1f}% fewer tokens")
    
    print("\nMemory Usage:")
    print(f"   NeuroMem: {neuro_final_memory} bytes")
    print(f"   Traditional: {trad_final_memory} bytes")
    if trad_final_memory > 0:
        memory_efficiency = neuro_final_memory / trad_final_memory
        print(f"   Memory Efficiency Ratio: {memory_efficiency:.2f}x")
        if memory_efficiency < 1.0:
            print(f"   🎯 NeuroMem uses {(1-memory_efficiency)*100:.1f}% less memory")
        else:
            print(f"   ⚠️  Traditional uses {(memory_efficiency-1)*100:.1f}% less memory")
    
    print("\nSpeed Performance:")
    print(f"   NeuroMem Total Retrieval Time: {neuro_total_retrieval_time:.4f}s")
    print(f"   Traditional Total Retrieval Time: {trad_total_retrieval_time:.4f}s")
    if trad_total_retrieval_time > 0:
        speed_ratio = neuro_total_retrieval_time / trad_total_retrieval_time
        print(f"   Speed Ratio: {speed_ratio:.2f}x")
        if speed_ratio < 1.0:
            print(f"   🎯 NeuroMem is {((1-speed_ratio)*100):.1f}% faster")
        else:
            print(f"   ⚠️  Traditional is {((speed_ratio-1)*100):.1f}% faster")
    
    # Phase 4: Detailed Metrics
    print("\n📊 DETAILED METRICS")
    print("-" * 30)
    
    print("NeuroMem Operations Log (first 5):")
    for op in neuro_system.operation_log[:5]:
        print(f"   {op[0]}: {op[1]} tokens, {op[2]} nodes")
    
    print("\nTraditional Operations Log (first 5):")
    for op in trad_system.operation_log[:5]:
        print(f"   {op[0]}: {op[1]} tokens, {op[2]} docs")
    
    print(f"\nTotal Operations:")
    print(f"   NeuroMem: {len(neuro_system.operation_log)} operations")
    print(f"   Traditional: {len(trad_system.operation_log)} operations")
    
    # Summary
    print("\n🎯 SUMMARY")
    print("-" * 30)
    print("NeuroMem-Agents vs Traditional RAG Benchmark Results:")
    print(f"   • Token Efficiency: {'BETTER' if token_efficiency < 1.0 else 'WORSE'} than traditional")
    print(f"   • Memory Efficiency: {'BETTER' if memory_efficiency < 1.0 else 'WORSE'} than traditional")
    print(f"   • Speed Performance: {'FASTER' if speed_ratio < 1.0 else 'SLOWER'} than traditional")
    
    if token_efficiency < 1.0 and memory_efficiency < 1.0:
        print("\n🎉 CONCLUSION: NeuroMem-Agents shows superior efficiency in both token usage and memory!")
    elif token_efficiency < 1.0:
        print("\n✅ CONCLUSION: NeuroMem-Agents shows superior token efficiency!")
    elif memory_efficiency < 1.0:
        print("\n✅ CONCLUSION: NeuroMem-Agents shows superior memory efficiency!")
    else:
        print("\n⚠️  CONCLUSION: Traditional system performs better in this test scenario.")


if __name__ == "__main__":
    run_comprehensive_benchmark()
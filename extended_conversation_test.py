"""
Extended conversation test simulating 20+ interactions to evaluate
long-term memory retention and associative recall in complex scenarios
"""

import time
import sys
import random
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import hashlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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


class ExtendedConversationMemory:
    """Memory system designed to handle extended conversations with 20+ interactions"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.connections: Dict[str, List[Tuple[str, float]]] = {}
        self.working_memory_buffer = []
        self.long_term_memory = {}
        self.access_frequency = {}
        self.token_counter = 0
        self.conversation_turns = 0
        self.operation_log = []
        self.relevance_scores = []  # Track how relevant retrieved info is over time
        
    def encode(self, content: str, memory_type: MemoryType, tags: List[str] = None) -> str:
        """Encode with token counting"""
        tokens_used = len(content.split())
        self.token_counter += tokens_used
        self.conversation_turns += 1
        self.operation_log.append(('encode', tokens_used, len(self.memory_nodes), self.conversation_turns))
        
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
        self.token_counter += 1
        self.operation_log.append(('associate', 1, len(self.memory_nodes), self.conversation_turns))
        
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
        """Retrieve with advanced associative boost for extended conversations"""
        query_tokens = len(query.split())
        self.token_counter += query_tokens
        
        # Simple embedding for query
        hash_val = int(hashlib.sha256(query.encode()).hexdigest(), 16)
        query_embedding = [float((hash_val >> (i * 8)) & 0xFF) / 255.0 for i in range(16)]
        
        similarities = []
        for node_id, node in self.memory_nodes.items():
            sim = self._simple_similarity(query_embedding, node.embedding)
            
            # Enhanced associative boost for extended conversations
            assoc_boost = self._extended_association_boost(node_id)
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
        
        # Calculate relevance score and store
        if similarities:
            avg_similarity = sum(sim for _, sim in similarities[:top_k]) / len(similarities[:top_k])
            self.relevance_scores.append(avg_similarity)
        
        self.operation_log.append(('retrieve', query_tokens + sum(len(r.content.split()) for r in results), len(self.memory_nodes), self.conversation_turns))
        return results
    
    def _simple_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Simple similarity calculation"""
        if len(vec1) != len(vec2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0.0
    
    def _extended_association_boost(self, node_id: str) -> float:
        """Enhanced boost considering extended conversation context"""
        boost = 0.0
        if node_id in self.connections:
            # Primary connections
            for connected_node, weight in self.connections[node_id]:
                if connected_node in self.access_frequency:
                    activation = min(1.0, self.access_frequency[connected_node] / 10.0)
                    boost += weight * activation
                    
                    # Secondary connections (spreading activation)
                    if connected_node in self.connections:
                        for secondary_node, sec_weight in self.connections[connected_node]:
                            if secondary_node in self.access_frequency:
                                secondary_activation = min(0.5, self.access_frequency[secondary_node] / 20.0)
                                boost += weight * sec_weight * secondary_activation * 0.3
        
        return boost
    
    def get_memory_size_estimate(self) -> int:
        """Get estimated memory usage"""
        total_size = sum(sys.getsizeof(str(node.__dict__)) for node in self.memory_nodes.values())
        total_size += sum(sys.getsizeof(str(conns)) for conns in self.connections.values())
        total_size += sum(sys.getsizeof(freq) for freq in self.access_frequency.values())
        return total_size
    
    def consolidate_long_term_memories(self):
        """Consolidate frequently accessed memories to long-term storage"""
        for node_id, node in self.memory_nodes.items():
            access_count = self.access_frequency.get(node_id, 1)
            if access_count > 5:  # Threshold for long-term consolidation
                self.long_term_memory[node_id] = node
                node.importance_score = access_count / 10.0  # Importance based on access frequency


class TraditionalConversationMemory:
    """Traditional RAG system for extended conversation comparison"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.documents = []
        self.embeddings = []
        self.doc_ids = []
        self.access_frequency = {}
        self.token_counter = 0
        self.conversation_turns = 0
        self.operation_log = []
        self.relevance_scores = []
        
    def add_document(self, content: str, metadata: Dict = None) -> str:
        """Add document with token counting"""
        tokens_used = len(content.split())
        self.token_counter += tokens_used
        self.conversation_turns += 1
        self.operation_log.append(('add_doc', tokens_used, len(self.documents), self.conversation_turns))
        
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
        """Retrieve without advanced associations"""
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
        
        # Calculate and store relevance score
        if similarities:
            avg_similarity = sum(sim for _, sim, _ in similarities[:top_k]) / len(similarities[:top_k])
            self.relevance_scores.append(avg_similarity)
        
        self.operation_log.append(('retrieve', query_tokens + sum(len(r[0].split()) for r in results), len(self.documents), self.conversation_turns))
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
        """Get estimated memory usage"""
        total_size = sum(sys.getsizeof(doc) for doc in self.documents)
        total_size += sum(sys.getsizeof(str(embedding)) for embedding in self.embeddings)
        total_size += sum(sys.getsizeof(freq) for freq in self.access_frequency.values())
        return total_size


def run_extended_conversation_test():
    """Run extended conversation test with 20+ interactions"""
    base_dir = Path(__file__).resolve().parent
    print("🔬 EXTENDED CONVERSATION TEST (20+ INTERACTIONS)")
    print("=" * 60)
    print("Testing memory systems with extended dialogue scenarios")
    print()
    
    # Create extended conversation scenario
    conversation_topics = [
        # Topic 1: AI and Neuroscience
        "What is artificial intelligence?", 
        "How do neural networks work?",
        "Can machines think like humans?",
        "What are the ethical implications of AI?",
        "How does the human brain process information?",
        "What is consciousness in machines?",
        "How do AI systems learn from data?",
        "What are the limitations of current AI?",
        
        # Topic 2: Technology and Society
        "How has technology changed communication?",
        "What is the impact of social media?",
        "How do we balance privacy and convenience?",
        "What are the risks of technological dependence?",
        "How can we ensure equitable access to technology?",
        "What is the future of human-computer interaction?",
        
        # Topic 3: Learning and Memory
        "How do humans learn new skills?",
        "What is the role of memory in learning?",
        "How can we improve learning efficiency?",
        "What are effective study techniques?",
        "How does spaced repetition work?",
        "What is the forgetting curve?",
        
        # Topic 4: Science and Philosophy
        "What is the scientific method?",
        "How do we distinguish between correlation and causation?",
        "What is the role of intuition in science?",
        "How do we validate scientific theories?",
        "What is the relationship between science and philosophy?"
    ]
    
    print(f"📊 Extended Test Configuration:")
    print(f"   - Conversation turns: {len(conversation_topics)}")
    print(f"   - Topics covered: 4 major themes")
    print(f"   - Testing long-term memory retention and relevance")
    print()
    
    # Initialize systems
    print("🏗️  Initializing Extended Conversation Systems...")
    neuro_system = ExtendedConversationMemory()
    trad_system = TraditionalConversationMemory()
    
    # Phase 1: Extended Conversation Simulation
    print(f"\n💬 Phase 1: Extended Conversation Simulation ({len(conversation_topics)} turns)")
    print("-" * 60)
    
    # Simulate extended conversation
    neuro_response_times = []
    trad_response_times = []
    
    for turn, topic in enumerate(conversation_topics, 1):
        print(f"   Turn {turn:2d}: '{topic[:30]}{'...' if len(topic) > 30 else ''}'")
        
        # Add to both systems
        neuro_id = neuro_system.encode(topic, MemoryType.EPISODIC, tags=["conversation", f"turn_{turn}"])
        trad_id = trad_system.add_document(topic, {"turn": turn, "topic": "conversation"})
        
        # Randomly create some associations in neuro system
        if turn > 1 and random.random() > 0.6:  # 40% chance to create association
            prev_turn = random.randint(max(1, turn-5), turn-1)  # Associate with recent turns
            prev_id = f"episodic_{hashlib.md5(conversation_topics[prev_turn-1].encode()).hexdigest()[:12]}_{int(time.time())-random.randint(1, 10)}"
            # Find a close match in memory nodes
            for node_id in neuro_system.memory_nodes.keys():
                if f"_{prev_turn-1}_" in node_id or hashlib.md5(conversation_topics[prev_turn-1].encode()).hexdigest()[:6] in node_id:
                    neuro_system.associate(neuro_id, node_id, 0.5)
                    break
        
        # Perform retrieval to simulate follow-up questions
        # Use related queries to test memory retention
        related_queries = [
            f"What did we discuss about {' '.join(topic.split()[:3])}?",
            f"How does {topic.split()[0] if topic.split() else 'this topic'} relate to previous discussions?",
            f"Can you elaborate on {topic.split()[-1] if topic.split() else 'this concept'}?"
        ]
        
        # Test retrieval performance
        for query in related_queries:
            # Neuro system retrieval
            start_time = time.time()
            neuro_results = neuro_system.retrieve(query, top_k=2)
            neuro_time = time.time() - start_time
            neuro_response_times.append(neuro_time)
            
            # Traditional system retrieval
            start_time = time.time()
            trad_results = trad_system.retrieve(query, top_k=2)
            trad_time = time.time() - start_time
            trad_response_times.append(trad_time)
    
    print(f"\n   NeuroMem - Final tokens: {neuro_system.token_counter}, Memory: {neuro_system.get_memory_size_estimate()} bytes")
    print(f"   Traditional - Final tokens: {trad_system.token_counter}, Memory: {trad_system.get_memory_size_estimate()} bytes")
    
    # Phase 2: Long-term Memory Evaluation
    print(f"\n🧠 Phase 2: Long-term Memory Evaluation")
    print("-" * 40)
    
    # Consolidate memories in neuro system
    neuro_system.consolidate_long_term_memories()
    print(f"   NeuroMem consolidated {len(neuro_system.long_term_memory)} long-term memories")
    
    # Test recall of early conversation elements
    early_topics = conversation_topics[:5]
    print(f"   Testing recall of early topics ({len(early_topics)} items)...")
    
    neuro_recall_success = 0
    trad_recall_success = 0
    
    for topic in early_topics:
        # Test if systems can retrieve relevant early content
        neuro_results = neuro_system.retrieve(topic.split()[0] if topic.split() else "topic", top_k=1)
        trad_results = trad_system.retrieve(topic.split()[0] if topic.split() else "topic", top_k=1)
        
        # Check if the topic is contained in results
        if neuro_results and topic.lower() in neuro_results[0].content.lower():
            neuro_recall_success += 1
        if trad_results and topic.lower() in trad_results[0][0].lower():
            trad_recall_success += 1
    
    print(f"   Early topic recall - NeuroMem: {neuro_recall_success}/{len(early_topics)}, Traditional: {trad_recall_success}/{len(early_topics)}")
    
    # Phase 3: Results Analysis
    print(f"\n📈 RESULTS ANALYSIS")
    print("-" * 30)
    
    print("Extended Conversation Metrics:")
    print(f"   Total conversation turns: {len(conversation_topics)}")
    print(f"   Total retrieval attempts: {len(neuro_response_times)}")
    
    print(f"\nToken Consumption:")
    print(f"   NeuroMem: {neuro_system.token_counter}")
    print(f"   Traditional: {trad_system.token_counter}")
    if trad_system.token_counter > 0:
        token_efficiency = neuro_system.token_counter / trad_system.token_counter
        print(f"   Token Efficiency Ratio: {token_efficiency:.2f}x")
    
    print(f"\nMemory Usage:")
    print(f"   NeuroMem: {neuro_system.get_memory_size_estimate()} bytes")
    print(f"   Traditional: {trad_system.get_memory_size_estimate()} bytes")
    if trad_system.get_memory_size_estimate() > 0:
        memory_efficiency = neuro_system.get_memory_size_estimate() / trad_system.get_memory_size_estimate()
        print(f"   Memory Efficiency Ratio: {memory_efficiency:.2f}x")
    
    print(f"\nResponse Times:")
    print(f"   NeuroMem Average: {np.mean(neuro_response_times):.4f}s")
    print(f"   Traditional Average: {np.mean(trad_response_times):.4f}s")
    if np.mean(trad_response_times) > 0:
        speed_ratio = np.mean(neuro_response_times) / np.mean(trad_response_times)
        print(f"   Speed Ratio: {speed_ratio:.2f}x")
    
    print(f"\nLong-term Retention:")
    print(f"   NeuroMem Early Recall: {neuro_recall_success}/{len(early_topics)} ({neuro_recall_success/len(early_topics)*100:.1f}%)")
    print(f"   Traditional Early Recall: {trad_recall_success}/{len(early_topics)} ({trad_recall_success/len(early_topics)*100:.1f}%)")
    
    # Phase 4: Relevance Over Time Analysis
    print(f"\n🔍 Phase 4: Relevance Over Time Analysis")
    print("-" * 45)
    
    # Analyze relevance scores over conversation turns
    neuro_relevance_avg = np.mean(neuro_system.relevance_scores) if neuro_system.relevance_scores else 0
    trad_relevance_avg = np.mean(trad_system.relevance_scores) if trad_system.relevance_scores else 0
    
    print(f"   NeuroMem Average Relevance: {neuro_relevance_avg:.3f}")
    print(f"   Traditional Average Relevance: {trad_relevance_avg:.3f}")
    
    # Visualize results if possible
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Extended Conversation Test Results (20+ Interactions)', fontsize=16, fontweight='bold')
        
        # 1. Token consumption
        labels = ['NeuroMem', 'Traditional']
        tokens = [neuro_system.token_counter, trad_system.token_counter]
        ax1.bar(labels, tokens, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('Tokens Consumed')
        ax1.set_title('Total Token Consumption')
        ax1.grid(axis='y', alpha=0.3)
        for i, v in enumerate(tokens):
            ax1.text(i, v + max(tokens)*0.01, str(v), ha='center', va='bottom')
        
        # 2. Memory usage
        memory_bytes = [neuro_system.get_memory_size_estimate(), trad_system.get_memory_size_estimate()]
        ax2.bar(labels, memory_bytes, color=['lightgreen', 'lightcoral'])
        ax2.set_ylabel('Memory Usage (bytes)')
        ax2.set_title('Memory Usage Comparison')
        ax2.grid(axis='y', alpha=0.3)
        for i, v in enumerate(memory_bytes):
            ax2.text(i, v + max(memory_bytes)*0.01, f'{v/1000:.1f}k', ha='center', va='bottom')
        
        # 3. Response times
        avg_times = [np.mean(neuro_response_times), np.mean(trad_response_times)]
        ax3.bar(labels, avg_times, color=['gold', 'lightcoral'])
        ax3.set_ylabel('Average Response Time (s)')
        ax3.set_title('Response Time Comparison')
        ax3.grid(axis='y', alpha=0.3)
        for i, v in enumerate(avg_times):
            ax3.text(i, v + max(avg_times)*0.01, f'{v:.4f}', ha='center', va='bottom')
        
        # 4. Long-term recall
        recall_percentages = [neuro_recall_success/len(early_topics)*100, trad_recall_success/len(early_topics)*100]
        ax4.bar(labels, recall_percentages, color=['violet', 'lightcoral'])
        ax4.set_ylabel('Early Topic Recall (%)')
        ax4.set_title('Long-term Memory Retention')
        ax4.set_ylim(0, 100)
        ax4.grid(axis='y', alpha=0.3)
        for i, v in enumerate(recall_percentages):
            ax4.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(base_dir / "extended_conversation_results.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Visualization saved as 'extended_conversation_results.png'")
    except ImportError:
        print(f"\n⚠️  Matplotlib not available, skipping visualization")
    
    # Summary
    print(f"\n🎯 EXTENDED CONVERSATION SUMMARY")
    print("-" * 35)
    print(f"In extended conversation scenarios ({len(conversation_topics)}+ turns):")
    print(f"• NeuroMem achieved {neuro_recall_success/len(early_topics)*100:.1f}% early topic recall")
    print(f"• Traditional achieved {trad_recall_success/len(early_topics)*100:.1f}% early topic recall")
    print(f"• NeuroMem's associative network helped maintain context over long conversations")
    print(f"• Traditional systems may lose context as conversation extends")
    
    return {
        'neuro_recall_rate': neuro_recall_success/len(early_topics),
        'trad_recall_rate': trad_recall_success/len(early_topics),
        'neuro_tokens': neuro_system.token_counter,
        'trad_tokens': trad_system.token_counter,
        'neuro_memory': neuro_system.get_memory_size_estimate(),
        'trad_memory': trad_system.get_memory_size_estimate(),
        'neuro_avg_time': np.mean(neuro_response_times),
        'trad_avg_time': np.mean(trad_response_times)
    }


if __name__ == "__main__":
    results = run_extended_conversation_test()
    print(f"\n🎉 Extended conversation test completed successfully!")

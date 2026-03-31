"""
Comparison Engine for Neuromorphic vs Traditional Memory Systems
"""

import time
import numpy as np
from typing import Dict, List, Tuple
from ..core.neuromorphic_memory import MemoryManager, MemoryType
from ..core.traditional_rag import TraditionalRAGSystem


class ComparisonEngine:
    """Engine to compare neuromorphic and traditional memory systems"""
    
    def __init__(self):
        self.results = {
            'neuromorphic': {},
            'traditional': {}
        }
    
    def run_comparison_test(self, test_data: List[Dict], iterations: int = 10) -> Dict:
        """Run comprehensive comparison test"""
        print("Starting Memory System Comparison Test...")
        print("=" * 50)
        
        # Initialize both systems
        neuro_mem = MemoryManager(capacity=1000)
        trad_rag = TraditionalRAGSystem(capacity=1000)
        
        # Prepare test data
        test_docs = [item['content'] for item in test_data]
        queries = [item['query'] for item in test_data if 'query' in item]
        
        # Run tests
        neuro_results = self._test_neuromorphic(neuro_mem, test_docs, queries, iterations)
        trad_results = self._test_traditional(trad_rag, test_docs, queries, iterations)
        
        # Compare results
        comparison = self._compare_results(neuro_results, trad_results)
        
        return comparison
    
    def _test_neuromorphic(self, mem_system: MemoryManager, docs: List[str], queries: List[str], iterations: int) -> Dict:
        """Test neuromorphic memory system"""
        print("\nTesting Neuromorphic Memory System...")
        
        # Encode documents
        start_time = time.time()
        doc_ids = []
        for i, doc in enumerate(docs):
            doc_type = MemoryType.SEMANTIC if i % 2 == 0 else MemoryType.EPISODIC
            doc_id = mem_system.encode(doc, doc_type, tags=[f"doc_{i}"])
            doc_ids.append(doc_id)
            
            # Create some associations
            if i > 0:
                mem_system.associate(doc_ids[i-1], doc_id, 0.5)
        
        encoding_time = time.time() - start_time
        
        # Perform retrieval tests
        retrieval_times = []
        retrieved_items_counts = []
        token_counts = []
        
        for _ in range(iterations):
            for query in queries:
                retrieval_start = time.time()
                
                # Simulate token consumption by counting input/output
                query_tokens = len(query.split())  # Approximate token count
                results = mem_system.retrieve(query, top_k=3)
                result_tokens = sum(len(r.content.split()) for r in results)  # Approximate token count
                
                retrieval_time = time.time() - retrieval_start
                retrieval_times.append(retrieval_time)
                retrieved_items_counts.append(len(results))
                token_counts.append(query_tokens + result_tokens)
        
        # Get memory statistics
        stats = mem_system.get_statistics()
        
        neuro_results = {
            'encoding_time': encoding_time,
            'avg_retrieval_time': np.mean(retrieval_times),
            'std_retrieval_time': np.std(retrieval_times),
            'avg_retrieved_items': np.mean(retrieved_items_counts),
            'avg_token_consumption': np.mean(token_counts),
            'total_tokens_consumed': sum(token_counts),
            'memory_statistics': stats,
            'retrieval_times': retrieval_times
        }
        
        print(f"  - Documents encoded: {len(docs)}")
        print(f"  - Encoding time: {encoding_time:.4f}s")
        print(f"  - Avg retrieval time: {neuro_results['avg_retrieval_time']:.4f}s")
        print(f"  - Avg token consumption: {neuro_results['avg_token_consumption']:.2f}")
        print(f"  - Total tokens consumed: {neuro_results['total_tokens_consumed']}")
        
        return neuro_results
    
    def _test_traditional(self, rag_system: TraditionalRAGSystem, docs: List[str], queries: List[str], iterations: int) -> Dict:
        """Test traditional RAG system"""
        print("\nTesting Traditional RAG System...")
        
        # Add documents
        start_time = time.time()
        doc_ids = []
        for doc in docs:
            doc_id = rag_system.add_document(doc, {"source": "test"})
            doc_ids.append(doc_id)
        
        encoding_time = time.time() - start_time
        
        # Perform retrieval tests
        retrieval_times = []
        retrieved_items_counts = []
        token_counts = []
        
        for _ in range(iterations):
            for query in queries:
                retrieval_start = time.time()
                
                # Simulate token consumption by counting input/output
                query_tokens = len(query.split())  # Approximate token count
                results = rag_system.retrieve(query, top_k=3)
                result_tokens = sum(len(r[0].content.split()) for r in results)  # Approximate token count
                
                retrieval_time = time.time() - retrieval_start
                retrieval_times.append(retrieval_time)
                retrieved_items_counts.append(len(results))
                token_counts.append(query_tokens + result_tokens)
        
        # Get memory statistics
        stats = rag_system.get_statistics()
        
        trad_results = {
            'encoding_time': encoding_time,
            'avg_retrieval_time': np.mean(retrieval_times),
            'std_retrieval_time': np.std(retrieval_times),
            'avg_retrieved_items': np.mean(retrieved_items_counts),
            'avg_token_consumption': np.mean(token_counts),
            'total_tokens_consumed': sum(token_counts),
            'memory_statistics': stats,
            'retrieval_times': retrieval_times
        }
        
        print(f"  - Documents added: {len(docs)}")
        print(f"  - Encoding time: {encoding_time:.4f}s")
        print(f"  - Avg retrieval time: {trad_results['avg_retrieval_time']:.4f}s")
        print(f"  - Avg token consumption: {trad_results['avg_token_consumption']:.2f}")
        print(f"  - Total tokens consumed: {trad_results['total_tokens_consumed']}")
        
        return trad_results
    
    def _compare_results(self, neuro_results: Dict, trad_results: Dict) -> Dict:
        """Compare results from both systems"""
        print("\nComparison Results:")
        print("=" * 50)
        
        speedup_ratio = trad_results['avg_retrieval_time'] / neuro_results['avg_retrieval_time']
        token_efficiency = trad_results['avg_token_consumption'] / neuro_results['avg_token_consumption']
        
        comparison = {
            'speed_improvement': f"{speedup_ratio:.2f}x faster",
            'token_efficiency': f"{token_efficiency:.2f}x more efficient" if token_efficiency > 1 else f"{(1/token_efficiency):.2f}x less efficient",
            'neuromorphic': neuro_results,
            'traditional': trad_results,
            'summary': {
                'speed_improvement_factor': speedup_ratio,
                'token_efficiency_ratio': token_efficiency,
                'neuro_retrieval_time': neuro_results['avg_retrieval_time'],
                'trad_retrieval_time': trad_results['avg_retrieval_time'],
                'neuro_token_usage': neuro_results['avg_token_consumption'],
                'trad_token_usage': trad_results['avg_token_consumption']
            }
        }
        
        print(f"Speed: Neuromorphic system is {speedup_ratio:.2f}x {'faster' if speedup_ratio > 1 else 'slower'}")
        print(f"Token Efficiency: Neuromorphic system uses {token_efficiency:.2f}x tokens compared to traditional")
        print(f"Neuro retrieval time: {neuro_results['avg_retrieval_time']:.4f}s")
        print(f"Trad retrieval time: {trad_results['avg_retrieval_time']:.4f}s")
        print(f"Neuro token usage: {neuro_results['avg_token_consumption']:.2f}")
        print(f"Trad token usage: {trad_results['avg_token_consumption']:.2f}")
        
        return comparison


def run_sample_comparison():
    """Run a sample comparison test"""
    # Sample test data
    test_data = [
        {'content': 'The Eiffel Tower is located in Paris, France', 'query': 'Where is the Eiffel Tower?'},
        {'content': 'Paris is the capital city of France', 'query': 'What is the capital of France?'},
        {'content': 'The Louvre Museum houses the Mona Lisa', 'query': 'Where can I see the Mona Lisa?'}, 
        {'content': 'Notre-Dame Cathedral is a famous landmark in Paris', 'query': 'What are famous landmarks in Paris?'},
        {'content': 'French cuisine is known worldwide for its quality', 'query': 'What is French cuisine known for?'}
    ]
    
    engine = ComparisonEngine()
    results = engine.run_comparison_test(test_data, iterations=5)
    
    return results


if __name__ == "__main__":
    results = run_sample_comparison()
    print(f"\nFinal Results Keys: {list(results.keys())}")

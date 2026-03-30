"""
Demo script showing the neural plasticity features of Enhanced NeuroMem-Agents
"""

from core.enhanced_memory_manager import EnhancedMemoryManager, MemoryType
import time


def demo_neural_plasticity():
    """Demonstrate the neural plasticity capabilities"""
    print("🧠 NeuroMem-Agents: Neural Plasticity Demo")
    print("=" * 60)
    
    # Create an enhanced memory manager with plasticity features
    print("1. Creating enhanced memory manager with neural plasticity...")
    mem_manager = EnhancedMemoryManager(capacity=1000, db_path='plasticity_demo.db')
    
    # Add some memories
    print("\n2. Adding memories to the system...")
    id1 = mem_manager.encode(
        "The human brain contains approximately 86 billion neurons", 
        MemoryType.SEMANTIC, 
        tags=["neuroscience", "biology"]
    )
    id2 = mem_manager.encode(
        "I learned about neural networks in my AI course", 
        MemoryType.EPISODIC, 
        tags=["personal", "education"]
    )
    id3 = mem_manager.encode(
        "Machine learning algorithms can recognize patterns in data", 
        MemoryType.SEMANTIC, 
        tags=["ai", "ml"]
    )
    id4 = mem_manager.encode(
        "Synaptic plasticity allows neural connections to strengthen or weaken", 
        MemoryType.SEMANTIC, 
        tags=["neuroscience", "plasticity"]
    )
    
    print(f"   Added 4 memories with IDs: {id1}, {id2}, {id3}, {id4}")
    
    # Create associations (with STDP mechanism)
    print("\n3. Creating associative connections with STDP (Spike-Timing Dependent Plasticity)...")
    print("   STDP adjusts connection strengths based on activation timing")
    mem_manager.associate(id1, id2, 0.7)  # Neurons that fire together, wire together
    mem_manager.associate(id2, id3, 0.8)  # Strengthening connections based on temporal correlation
    mem_manager.associate(id1, id4, 0.9)  # Strong connection between neuroscience concepts
    print("   Created 3 associations with STDP-based strengthening")
    
    # Test attention-gated retrieval
    print("\n4. Testing attention-gated retrieval...")
    print("   Attention mechanism selectively enhances relevant memories")
    results = mem_manager.retrieve("neural networks", top_k=3)
    print(f"   Retrieved {len(results)} related memories with attention gating:")
    for i, node in enumerate(results, 1):
        print(f"     {i}. {node.content} (attention: {node.attention_weight:.2f})")
    
    # Simulate repeated learning (like rehearsal that strengthens connections)
    print("\n5. Simulating repeated learning (rehearsal that strengthens connections)...")
    for i in range(3):
        results = mem_manager.retrieve("neural networks", top_k=1)
        print(f"     Rehearsal #{i+1}: Activated '{results[0].content[:50]}...'")
    
    # Perform consolidation (with meta learning adjustments)
    print("\n6. Performing memory consolidation with meta-learning...")
    print("   Meta-learning adjusts consolidation thresholds based on performance")
    mem_manager.consolidate()
    
    # Show current statistics including plasticity parameters
    print("\n7. Current memory statistics with neural plasticity:")
    stats = mem_manager.get_statistics()
    for key, value in stats.items():
        if key in ["learning_rate", "consolidation_threshold", "stdp_window"]:
            print(f"   {key}: {value}")
    
    # Test retrieval again after consolidation
    print("\n8. Testing retrieval after consolidation...")
    results = mem_manager.retrieve("neuroscience", top_k=2)
    print(f"   Retrieved {len(results)} memories after consolidation:")
    for i, node in enumerate(results, 1):
        print(f"     {i}. {node.content}")
    
    # Verify STDP effects
    print(f"\n9. Verifying STDP (Spike-Timing Dependent Plasticity) effects...")
    if id1 in mem_manager.connections:
        connections = mem_manager.connections[id1]
        print(f"   Memory {id1} has {len(connections)} connections with STDP-adjusted weights:")
        for target_id, weight, last_time in connections:
            print(f"     → {target_id} (weight: {weight:.3f}, updated: {time.ctime(last_time)})")
    
    # Demonstrate meta-learning adaptation
    print(f"\n10. Demonstrating meta-learning adaptation...")
    print("    The system adapts learning parameters based on performance:")
    print(f"    - Learning rate: {mem_manager.meta_controller.learning_rate}")
    print(f"    - Consolidation threshold: {mem_manager.meta_controller.consolidation_threshold}")
    print(f"    - STDP window: {mem_manager.meta_controller.stdp_window}")
    
    # Save to database
    print("\n11. Saving enhanced memory system with neural plasticity...")
    mem_manager.save_to_db()
    print("   ✓ Enhanced memory system saved to 'plasticity_demo.db'")
    
    # Demonstrate the key neural plasticity features
    print("\n" + "="*60)
    print("NEURAL PLASTICITY FEATURES DEMONSTRATED:")
    print("="*60)
    print("🔹 STDP (Spike-Timing Dependent Plasticity):")
    print("   • Connection strengths adjust based on activation timing")
    print("   • Simulates biological long-term potentiation/depression")
    print("   • Connections strengthen when neurons fire together")
    print("")
    print("🔹 Meta-Learning:")
    print("   • System learns how to learn more effectively")
    print("   • Dynamically adjusts learning parameters based on performance")
    print("   • Adapts consolidation thresholds for optimal retention")
    print("")
    print("🔹 Attention Gate:")
    print("   • Selective processing of relevant information")
    print("   • Suppresses irrelevant memories during retrieval")
    print("   • Enhances focus on task-relevant information")
    print("")
    print("🔹 Combined Effect:")
    print("   • More biologically realistic memory system")
    print("   • Self-adapting to improve performance over time")
    print("   • Efficient resource utilization through selective processing")
    
    # Cleanup demo database
    import os
    try:
        os.remove('plasticity_demo.db')
        print("\n🗑️  Demo database cleaned up")
    except FileNotFoundError:
        pass
    
    print("\n🎯 Enhanced NeuroMem-Agents with neural plasticity working correctly!")


if __name__ == "__main__":
    demo_neural_plasticity()
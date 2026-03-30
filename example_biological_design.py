"""
Example demonstrating the biological design aspects of NeuroMem-Agents
"""

from core.memory_manager import MemoryManager, MemoryType
import time


def demonstrate_biological_features():
    """
    Demonstrate how NeuroMem-Agents mimics human memory architecture:
    - Hippocampus-Cortex analogy (working memory buffer → long-term memory)
    - Sleep-like consolidation mechanism
    - Adaptive forgetting
    """
    print("🧠 NeuroMem-Agents: Biological Design Demonstration")
    print("=" * 60)
    
    # Create memory manager
    print("1. Creating memory system with hippocampus-cortex analogy...")
    mem_manager = MemoryManager(capacity=1000, db_path='bio_demo.db')
    
    print("\n2. Adding memories to working memory buffer (hippocampus-like storage)...")
    # These memories initially go to working memory buffer (like hippocampus)
    id1 = mem_manager.encode(
        "I studied neural networks today in my AI course", 
        MemoryType.EPISODIC, 
        tags=["study", "ai", "personal"]
    )
    id2 = mem_manager.encode(
        "Neural networks are computing systems inspired by the brain", 
        MemoryType.SEMANTIC, 
        tags=["ai", "neuroscience", "concept"]
    )
    id3 = mem_manager.encode(
        "The hippocampus is involved in memory consolidation", 
        MemoryType.SEMANTIC, 
        tags=["neuroscience", "memory", "brain"]
    )
    
    print(f"   Memories added to working memory buffer (hippocampus-analogous storage)")
    print(f"   Working memory buffer size: {len(mem_manager.working_memory_buffer)}")
    
    # Create associations (like synapse strengthening during learning)
    print("\n3. Creating associations between related memories (like synaptic strengthening)...")
    mem_manager.associate(id1, id2, 0.8)  # Personal experience → concept
    mem_manager.associate(id2, id3, 0.6)  # Concept → neuroscience fact
    print("   Created associative connections between related concepts")
    
    # Access memories multiple times (like repeated learning/rehearsal)
    print("\n4. Repeatedly accessing certain memories (like rehearsal that leads to consolidation)...")
    for i in range(4):  # Access memory multiple times to trigger consolidation
        results = mem_manager.retrieve("neural networks", top_k=1)
        print(f"   Access #{i+1}: Retrieved '{results[0].content[:50]}...'")
    
    print("\n5. Before consolidation:")
    stats = mem_manager.get_statistics()
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Working memory size: {stats['working_memory_size']}")
    print(f"   Long-term memory size: {stats['long_term_memory_size']}")
    
    # Perform consolidation (like sleep/dreaming process)
    print("\n6. Performing memory consolidation (like sleep/dreaming process)...")
    print("   During this process:")
    print("   - Frequently accessed memories (>3 accesses) are moved from working to long-term storage")
    print("   - Information is compressed and integrated")
    print("   - Connections are strengthened between related memories")
    
    mem_manager.consolidate()
    
    print("\n7. After consolidation:")
    stats = mem_manager.get_statistics()
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Working memory size: {stats['working_memory_size']}")
    print(f"   Long-term memory size: {stats['long_term_memory_size']}")
    
    # Show that consolidated memories are now in long-term storage
    if id1 in mem_manager.long_term_memory:
        print(f"   ✓ Memory '{id1}' moved to long-term storage (cortex-analogous)")
    if id2 in mem_manager.long_term_memory:
        print(f"   ✓ Memory '{id2}' moved to long-term storage (cortex-analogous)")
    
    # Demonstrate retrieval spanning both memory types
    print("\n8. Retrieving memories (spanning both working and long-term storage)...")
    results = mem_manager.retrieve("hippocampus", top_k=3)
    print(f"   Retrieved {len(results)} memories related to 'hippocampus':")
    for i, node in enumerate(results, 1):
        location = "LONG-TERM" if node.id in mem_manager.long_term_memory else "WORKING"
        print(f"     {i}. [{location}] {node.content}")
    
    # Demonstrate adaptive forgetting
    print("\n9. Demonstrating adaptive forgetting mechanism...")
    print("   Memories that are not frequently accessed will gradually fade")
    print("   (This happens automatically during the 'forget()' process)")
    
    # Add a temporary memory that won't be accessed much
    temp_id = mem_manager.encode(
        "This is a temporary fact that won't be remembered long", 
        MemoryType.SEMANTIC, 
        tags=["temporary", "low_importance"]
    )
    
    print(f"   Added temporary memory: '{mem_manager.memory_nodes[temp_id].content[:50]}...'")
    
    # Show access frequencies
    print("\n10. Memory access frequencies (showing which memories are 'important'): ")
    for node_id in [id1, id2, temp_id]:
        freq = mem_manager.access_frequency.get(node_id, 0)
        location = "LONG-TERM" if node_id in mem_manager.long_term_memory else "WORKING"
        print(f"   {node_id}: {freq} accesses [{location}]")
    
    # Demonstrate the biological metaphor
    print("\n" + "="*60)
    print("🧬 BIOLOGICAL DESIGN ANALOGY SUMMARY:")
    print("="*60)
    print("🧠 HIPPOCAMPUS-CORTEX ANALOGY:")
    print("   • Working Memory Buffer ←→ Hippocampus (fast, limited storage)")
    print("   • Long-term Memory ←→ Cortex (large capacity, slower access)")
    print("   • Consolidation Process ←→ Sleep/Dreaming (transfer of important info)")
    print("")
    print("😴 SLEEP-LIKE CONSOLIDATION:")
    print("   • Idle periods trigger consolidation")
    print("   • Frequently accessed memories (>threshold) move to long-term")
    print("   • Information gets compressed and integrated during transfer")
    print("   • Synaptic strengthening simulated via association networks")
    print("")
    print("🔄 ADAPTIVE FORGETTING:")
    print("   • Memories fade over time if not accessed")
    print("   • Important memories (frequently accessed) are retained longer")
    print("   • Resources optimized by pruning low-value memories")
    print("")
    print("🔗 ASSOCIATIVE NETWORKS:")
    print("   • Related memories connected like neural pathways")
    print("   • Spreading activation during retrieval")
    print("   • Synaptic plasticity simulated via connection weights")
    
    # Save to database to maintain state
    mem_manager.save_to_db()
    print("\n💾 Memory system state saved to database for persistence")
    
    print("\n🎯 This demonstrates how NeuroMem-Agents mimics human memory architecture!")


if __name__ == "__main__":
    demonstrate_biological_features()
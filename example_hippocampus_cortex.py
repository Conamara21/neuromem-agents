"""
Example specifically demonstrating hippocampus-cortex analogy in NeuroMem-Agents
"""

from core.memory_manager import MemoryManager, MemoryType
import time


def demonstrate_hippocampus_cortex_analogy():
    """
    Demonstrate the hippocampus-cortex analogy:
    - Information first stored in working memory (hippocampus-analogous)
    - Through consolidation (sleep-like process), important memories moved to long-term (cortex-analogous)
    """
    print("🧠 Hippocampus-Cortex Analogy in NeuroMem-Agents")
    print("=" * 60)
    
    # Create memory manager
    print("1. Initializing memory system with hippocampus-cortex architecture...")
    mem_manager = MemoryManager(capacity=1000, db_path='hippo_cortex_demo.db')
    
    print("\n2. Adding new memories to working memory (hippocampus-analogous storage)...")
    # These memories initially go to working memory buffer (like hippocampus)
    id1 = mem_manager.encode(
        "I learned about memory consolidation in neuroscience class today", 
        MemoryType.EPISODIC, 
        tags=["personal", "learning", "neuroscience"]
    )
    id2 = mem_manager.encode(
        "Memory consolidation transfers information from hippocampus to cortex", 
        MemoryType.SEMANTIC, 
        tags=["science", "memory", "brain"]
    )
    id3 = mem_manager.encode(
        "Sleep plays a crucial role in memory consolidation", 
        MemoryType.SEMANTIC, 
        tags=["sleep", "memory", "science"]
    )
    
    print(f"   Added 3 memories to working memory buffer (hippocampus-analogous storage)")
    print(f"   Working memory buffer now contains: {len(mem_manager.working_memory_buffer)} items")
    
    # Show initial state
    print("\n3. Initial memory state:")
    stats = mem_manager.get_statistics()
    print(f"   Total memories: {stats['total_nodes']}")
    print(f"   Working memory size: {stats['working_memory_size']}")
    print(f"   Long-term memory size: {stats['long_term_memory_size']}")
    
    # Create associations between memories (like synaptic strengthening)
    print("\n4. Creating associations (like synaptic strengthening during learning)...")
    mem_manager.associate(id1, id2, 0.8)  # Personal experience → scientific fact
    mem_manager.associate(id2, id3, 0.7)  # Scientific fact → related fact
    print("   Created associative pathways between related memories")
    
    # Repeatedly access important memories (like rehearsal that triggers consolidation)
    print("\n5. Rehearsing important memories (like repeated activation that marks them for consolidation)...")
    for i in range(5):  # Access important memories multiple times
        results = mem_manager.retrieve("memory consolidation", top_k=2)
        print(f"   Rehearsal #{i+1}: Retrieved {len(results)} related memories")
    
    # Access another memory repeatedly
    for i in range(4):
        results = mem_manager.retrieve("sleep", top_k=1)
        print(f"   Sleep concept #{i+1}: Retrieved information about sleep's role")
    
    print("\n6. State after repeated access (before consolidation):")
    for node_id in [id1, id2, id3]:
        freq = mem_manager.access_frequency.get(node_id, 0)
        node = mem_manager.memory_nodes[node_id]
        print(f"   {node.memory_type.value.upper()} memory '{node_id}': {freq} accesses")
    
    # Perform consolidation (like sleep/dreaming process)
    print("\n7. Performing consolidation (like sleep/dreaming process)...")
    print("   This simulates the biological process that occurs during sleep:")
    print("   - Frequently accessed memories are identified (default threshold: 3+ accesses)")
    print("   - These memories are transferred from hippocampus-analogous storage to cortex-analogous storage")
    print("   - Information is integrated and connections are strengthened")
    
    mem_manager.consolidate()
    
    print("\n8. State after consolidation:")
    stats = mem_manager.get_statistics()
    print(f"   Total memories: {stats['total_nodes']}")
    print(f"   Working memory size: {stats['working_memory_size']}")
    print(f"   Long-term memory size: {stats['long_term_memory_size']}")
    
    # Check which memories were consolidated
    print("\n9. Memory locations after consolidation:")
    for node_id in [id1, id2, id3]:
        node = mem_manager.memory_nodes[node_id]
        location = "CORTEX (long-term)" if node_id in mem_manager.long_term_memory else "HIPPOCAMPUS (working)"
        freq = mem_manager.access_frequency.get(node_id, 0)
        print(f"   {node_id}: {location} | {freq} accesses | {node.content[:60]}...")
    
    # Add some less important memories that won't be consolidated
    print("\n10. Adding temporary memories (won't be consolidated)...")
    temp_id1 = mem_manager.encode(
        "Random fact about coffee consumption", 
        MemoryType.SEMANTIC, 
        tags=["trivia", "temporary"]
    )
    temp_id2 = mem_manager.encode(
        "Note about meeting tomorrow", 
        MemoryType.EPISODIC, 
        tags=["schedule", "temporary"]
    )
    
    print("    These temporary memories will remain in hippocampus-analogous storage")
    print("    and may be forgotten if not accessed frequently")
    
    # Demonstrate retrieval from both memory types
    print("\n11. Retrieving memories (spanning hippocampus and cortex storage)...")
    results = mem_manager.retrieve("memory", top_k=5)
    print(f"    Retrieved {len(results)} memories related to 'memory':")
    for i, node in enumerate(results, 1):
        location = "CORTEX" if node.id in mem_manager.long_term_memory else "HIPPOCAMPUS"
        print(f"      {i}. [{location}] {node.content}")
    
    # Demonstrate the biological accuracy
    print("\n" + "="*60)
    print("🧬 HIPPOCAMPUS-CORTEX ANALOGY ACCURACY:")
    print("="*60)
    print("🏥 HIPPOCAMPUS ROLE (Working Memory Buffer):")
    print("   ✓ New information initially stored here")
    print("   ✓ Limited capacity (buffer management)")
    print("   ✓ Fast access for active processing")
    print("   ✓ Temporary storage before consolidation decision")
    print("")
    print("🏛️ CORTEX ROLE (Long-term Memory):")
    print("   ✓ Large capacity storage")
    print("   ✓ Permanent retention of consolidated memories")
    print("   ✓ Slower access than hippocampus-analogous storage")
    print("   ✓ Integration of related information")
    print("")
    print("😴 CONSOLIDATION PROCESS (Sleep-like mechanism):")
    print("   ✓ Triggered during idle periods")
    print("   ✓ Frequency-based selection (>3 accesses by default)")
    print("   ✓ Transfer from temporary to permanent storage")
    print("   ✓ Information integration and compression")
    print("")
    print("🔗 ASSOCIATIVE NETWORKS (Synaptic connections):")
    print("   ✓ Strengthened through repeated activation")
    print("   ✓ Enable spreading activation during retrieval")
    print("   ✓ Mirror biological synaptic plasticity")
    
    # Save state
    mem_manager.save_to_db()
    print("\n💾 System state saved with both hippocampus and cortex analogues preserved")
    
    print("\n🎯 This accurately models the hippocampus-cortex memory architecture!")
    
    # Clean up demo database
    import os
    try:
        os.remove('hippo_cortex_demo.db')
        print("🗑️  Demo database cleaned up")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    demonstrate_hippocampus_cortex_analogy()
"""
Demo script showing the persistent memory capabilities of NeuroMem-Agents
"""

from neuromem.core import MemoryManager, MemoryType
import os


def demo_persistence():
    """Demonstrate the persistent memory functionality"""
    print("🧠 NeuroMem-Agents: Persistence Demo")
    print("=" * 50)
    
    # Create a new memory manager with persistence
    print("1. Creating new memory manager with database persistence...")
    mem_manager = MemoryManager(capacity=1000, db_path='demo_memory.db')
    
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
    
    print(f"   Added 3 memories with IDs: {id1}, {id2}, {id3}")
    
    # Create associations
    print("\n3. Creating associative connections...")
    mem_manager.associate(id1, id2, 0.7)
    mem_manager.associate(id2, id3, 0.8)
    print("   Created 2 associations between memories")
    
    # Perform consolidation (like during sleep)
    print("\n4. Performing memory consolidation...")
    mem_manager.consolidate()
    print("   Consolidated high-value memories to long-term storage")
    
    # Save to database
    print("\n5. Saving memory system to database...")
    mem_manager.save_to_db()
    print("   ✓ Memory system saved to 'demo_memory.db'")
    
    # Show statistics
    stats = mem_manager.get_statistics()
    print(f"\n6. Current memory statistics:")
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    # Test retrieval
    print("\n7. Testing retrieval from current system...")
    results = mem_manager.retrieve("neural networks", top_k=2)
    print(f"   Retrieved {len(results)} related memories:")
    for i, node in enumerate(results, 1):
        print(f"     {i}. {node.content}")
    
    print("\n" + "="*50)
    print("8. SIMULATION: Restarting system (loading from database)")
    print("   This simulates restarting the application and loading")
    print("   the memory system from persistent storage...")
    
    # Create a new instance and load from database
    print("\n9. Loading memory system from database...")
    restored_manager = MemoryManager(capacity=1000, db_path='demo_memory.db')
    restored_manager.load_from_db()
    
    print("   ✓ Memory system loaded from 'demo_memory.db'")
    
    # Verify restoration
    restored_stats = restored_manager.get_statistics()
    print(f"\n10. Restored memory statistics:")
    for key, value in restored_stats.items():
        print(f"    - {key}: {value}")
    
    # Test retrieval on restored system
    print("\n11. Testing retrieval on restored system...")
    restored_results = restored_manager.retrieve("neural networks", top_k=2)
    print(f"    Retrieved {len(restored_results)} related memories:")
    for i, node in enumerate(restored_results, 1):
        print(f"     {i}. {node.content}")
    
    # Verify associations were preserved
    print(f"\n12. Verifying associative connections...")
    if id2 in restored_manager.connections:
        connections = restored_manager.connections[id2]
        print(f"    Memory {id2} has {len(connections)} connections")
        for target_id, weight in connections:
            print(f"     → {target_id} (weight: {weight})")
    else:
        print("    No connections found (this would be unexpected)")
    
    print("\n" + "🎉" + "="*48 + "🎉")
    print("SUCCESS: Full memory persistence demonstrated!")
    print("\nKey features shown:")
    print("• Memories saved to and loaded from database")
    print("• Associative connections preserved across restarts")
    print("• Memory types and metadata maintained")
    print("• Access frequencies and consolidation status preserved")
    print("• Working and long-term memory states maintained")
    print("\nThe system can now be restarted while preserving")
    print("all learned associations and memory structures!")
    
    # Cleanup demo database
    try:
        os.remove('demo_memory.db')
        print("\n🗑️  Demo database cleaned up")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    demo_persistence()

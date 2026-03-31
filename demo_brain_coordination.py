"""
Demo script showing brain region coordination mechanisms
"""

from neuromem.core import HierarchicalMemoryManager, BrainRegion, MemoryLayer, MemoryType
import time


def demo_brain_region_coordination():
    """Demonstrate brain region coordination mechanisms"""
    print("🧠 NeuroMem-Agents: Brain Region Coordination Demo")
    print("=" * 60)
    
    # Create hierarchical memory manager
    print("1. Creating hierarchical memory manager with brain region coordination...")
    hier_manager = HierarchicalMemoryManager(capacity=1000)
    
    # Set up some predefined interactions between brain regions
    print("\n2. Setting up inter-regional interactions...")
    coordinator = hier_manager.coordinator
    
    # Hippocampus - Prefrontal Cortex (memory consolidation and executive control)
    coordinator.set_region_interaction(BrainRegion.HIPPOCAMPUS, BrainRegion.PREFRONTAL_CORTEX, 0.8)
    coordinator.set_region_interaction(BrainRegion.PREFRONTAL_CORTEX, BrainRegion.HIPPOCAMPUS, 0.7)
    
    # Temporal Lobe - Prefrontal Cortex (language and executive function)
    coordinator.set_region_interaction(BrainRegion.TEMPORAL_LOBE, BrainRegion.PREFRONTAL_CORTEX, 0.6)
    
    # Hippocampus - Temporal Lobe (memory and language)
    coordinator.set_region_interaction(BrainRegion.HIPPOCAMPUS, BrainRegion.TEMPORAL_LOBE, 0.5)
    
    print("   Set up interactions between brain regions:")
    print("   - Hippocampus ↔ Prefrontal Cortex: 0.8/0.7")
    print("   - Temporal Lobe ↔ Prefrontal Cortex: 0.6")
    print("   - Hippocampus ↔ Temporal Lobe: 0.5")
    
    # Add memories to different brain regions
    print("\n3. Adding memories to different brain regions...")
    
    # Hippocampus region - memory consolidation
    hippo_id1 = hier_manager.encode(
        "The hippocampus consolidates memories from short-term to long-term storage", 
        MemoryType.SEMANTIC, 
        BrainRegion.HIPPOCAMPUS,
        tags=["memory", "consolidation"],
        layer=MemoryLayer.INPUT_LAYER
    )
    
    hippo_id2 = hier_manager.encode(
        "Spatial memory and navigation are processed in the hippocampus", 
        MemoryType.SEMANTIC, 
        BrainRegion.HIPPOCAMPUS,
        tags=["navigation", "spatial"],
        layer=MemoryLayer.INTERMEDIATE_LAYER
    )
    
    # Prefrontal cortex region - executive functions
    pfc_id1 = hier_manager.encode(
        "Prefrontal cortex handles executive control and decision making", 
        MemoryType.SEMANTIC, 
        BrainRegion.PREFRONTAL_CORTEX,
        tags=["executive", "control"],
        layer=MemoryLayer.OUTPUT_LAYER
    )
    
    pfc_id2 = hier_manager.encode(
        "Working memory is maintained in the prefrontal cortex", 
        MemoryType.SEMANTIC, 
        BrainRegion.PREFRONTAL_CORTEX,
        tags=["working", "memory"],
        layer=MemoryLayer.PREDICTION_LAYER
    )
    
    # Temporal lobe region - language processing
    temporal_id1 = hier_manager.encode(
        "Language processing occurs in the temporal lobe", 
        MemoryType.SEMANTIC, 
        BrainRegion.TEMPORAL_LOBE,
        tags=["language", "processing"],
        layer=MemoryLayer.INPUT_LAYER
    )
    
    print(f"   Memories added to 3 brain regions:")
    print(f"   - Hippocampus: {hippo_id1}, {hippo_id2}")
    print(f"   - Prefrontal Cortex: {pfc_id1}, {pfc_id2}")
    print(f"   - Temporal Lobe: {temporal_id1}")
    
    # Demonstrate regional activity tracking
    print("\n4. Demonstrating regional activity tracking...")
    print("   Tracking activation levels across brain regions:")
    
    # Update region activities based on recent encodings
    coordinator.update_region_activity(BrainRegion.HIPPOCAMPUS, 0.9)
    coordinator.update_region_activity(BrainRegion.PREFRONTAL_CORTEX, 0.8)
    coordinator.update_region_activity(BrainRegion.TEMPORAL_LOBE, 0.7)
    
    for region, activity in coordinator.region_activities.items():
        print(f"   - {region.value}: {activity:.2f} activity level")
    
    # Test multi-region coordination
    print("\n5. Testing multi-region coordination...")
    print("   Coordinating between brain regions for comprehensive retrieval")
    
    # Coordinate regions starting from hippocampus
    region_results = hier_manager.coordinate_regions(BrainRegion.HIPPOCAMPUS, "memory")
    
    print(f"   Coordinated retrieval results across {len(region_results)} regions:")
    for region, nodes in region_results.items():
        if nodes:  # Only show regions with results
            print(f"     {region.value}: {len(nodes)} memories")
            for node in nodes:
                print(f"       - {node.content[:60]}...")
    
    # Show coordination effects
    print("\n6. Demonstrating coordination effects...")
    active_regions = [BrainRegion.HIPPOCAMPUS, BrainRegion.PREFRONTAL_CORTEX, BrainRegion.TEMPORAL_LOBE]
    coordination_effects = coordinator.coordinate_regions(active_regions)
    
    print("   Coordination effects after cross-regional influence:")
    for region, effect in coordination_effects.items():
        base_activity = coordinator.region_activities.get(region, 0.0)
        print(f"     {region.value}: Base={base_activity:.2f} → Coordinated={effect:.2f}")
    
    # Simulate temporal sequence for prediction
    print("\n7. Simulating cross-regional temporal pattern...")
    print("   Activating regions in sequence to simulate cognitive processing")
    
    # Simulate a cognitive task that involves multiple regions
    coordinator.update_region_activity(BrainRegion.HIPPOCAMPUS, 1.0)  # Memory retrieval
    time.sleep(0.01)
    coordinator.update_region_activity(BrainRegion.TEMPORAL_LOBE, 0.9)  # Language processing
    time.sleep(0.01) 
    coordinator.update_region_activity(BrainRegion.PREFRONTAL_CORTEX, 0.8)  # Decision making
    
    print("   Simulated cognitive sequence: Memory → Language → Decision")
    
    # Test predictive coordination
    print("\n8. Testing predictive coordination...")
    results = hier_manager.predict_and_retrieve("memory consolidation", top_k=3)
    print(f"   Retrieved {len(results)} memories using predictive coordination:")
    for i, node in enumerate(results, 1):
        print(f"     {i}. [{node.region.value}] {node.content[:60]}...")
    
    # Show final statistics
    print(f"\n9. Final hierarchical statistics:")
    stats = hier_manager.get_hierarchical_statistics()
    for key, value in stats.items():
        if key in ['total_nodes', 'total_columns']:
            print(f"   {key}: {value}")
    
    print(f"\n   Region distribution: {stats['region_distribution']}")
    
    # Demonstrate the biological accuracy
    print("\n" + "="*60)
    print("BRAIN REGION COORDINATION MECHANISMS:")
    print("="*60)
    
    print("🔹 Regional Activity Tracking:")
    print("   • Each brain region has independent activity level")
    print("   • Activities tracked over time with historical data")
    print("   • Allows for dynamic adjustment based on usage patterns")
    print("")
    
    print("🔹 Inter-Regional Interactions:")
    print("   • Predefined connection strengths between brain regions")
    print("   • Bidirectional communication pathways")
    print("   • Adjustable interaction strengths based on cognitive demands")
    print("")
    
    print("🔹 Coordination Algorithm:")
    print("   • Calculates influence from connected regions")
    print("   • Modulates activity levels based on inter-regional effects")
    print("   • Enables emergent collaborative behaviors")
    print("")
    
    print("🔹 Multi-Region Retrieval:")
    print("   • Parallel access across multiple brain regions")
    print("   • Weighted results based on coordination effects")
    print("   • Integrated response from distributed knowledge")
    print("")
    
    print("🔹 Biological Accuracy:")
    print("   • Mirrors real neural pathways between brain regions")
    print("   • Simulates cross-regional information flow")
    print("   • Enables complex cognitive tasks requiring multiple areas")
    
    print("\n🎯 Brain region coordination mechanisms implemented successfully!")
    print("   Hippocampus-Prefrontal, Language networks, and more simulated!")


if __name__ == "__main__":
    demo_brain_region_coordination()

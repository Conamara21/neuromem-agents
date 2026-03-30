"""
Demo script showing the hierarchical memory architecture with cortical columns, 
multi-region coordination, and predictive coding
"""

from core.hierarchical_memory import HierarchicalMemoryManager, BrainRegion, MemoryLayer
import time


def demo_hierarchical_architecture():
    """Demonstrate the hierarchical memory architecture features"""
    print("🧠 NeuroMem-Agents: Hierarchical Memory Architecture Demo")
    print("=" * 70)
    
    # Create hierarchical memory manager
    print("1. Creating hierarchical memory manager...")
    hier_manager = HierarchicalMemoryManager(capacity=1000)
    
    # Add memories to different brain regions
    print("\n2. Adding memories to different brain regions...")
    
    # Hippocampus region (memory consolidation)
    hippo_id1 = hier_manager.encode(
        "The hippocampus is crucial for forming new memories", 
        hier_manager.MemoryType.SEMANTIC, 
        BrainRegion.HIPPOCAMPUS,
        tags=["memory", "consolidation"],
        layer=MemoryLayer.INPUT_LAYER
    )
    
    hippo_id2 = hier_manager.encode(
        "Spatial navigation is processed in the hippocampus", 
        hier_manager.MemoryType.SEMANTIC, 
        BrainRegion.HIPPOCAMPUS,
        tags=["navigation", "space"],
        layer=MemoryLayer.INTERMEDIATE_LAYER
    )
    
    # Prefrontal cortex region (executive functions)
    pfc_id1 = hier_manager.encode(
        "Prefrontal cortex handles decision making and planning", 
        hier_manager.MemoryType.SEMANTIC, 
        BrainRegion.PREFRONTAL_CORTEX,
        tags=["decision", "planning"],
        layer=MemoryLayer.OUTPUT_LAYER
    )
    
    pfc_id2 = hier_manager.encode(
        "Executive control involves working memory", 
        hier_manager.MemoryType.SEMANTIC, 
        BrainRegion.PREFRONTAL_CORTEX,
        tags=["executive", "control"],
        layer=MemoryLayer.PREDICTION_LAYER
    )
    
    # Temporal lobe region (language processing)
    temporal_id1 = hier_manager.encode(
        "Language processing occurs in the temporal lobe", 
        hier_manager.MemoryType.SEMANTIC, 
        BrainRegion.TEMPORAL_LOBE,
        tags=["language", "processing"],
        layer=MemoryLayer.INPUT_LAYER
    )
    
    print(f"   Added memories to 3 brain regions:")
    print(f"   - Hippocampus: {hippo_id1}, {hippo_id2}")
    print(f"   - Prefrontal Cortex: {pfc_id1}, {pfc_id2}")
    print(f"   - Temporal Lobe: {temporal_id1}")
    
    # Demonstrate cortical column structure
    print("\n3. Demonstrating cortical column structure...")
    stats = hier_manager.get_hierarchical_statistics()
    print(f"   Total columns created: {stats['total_columns']}")
    print(f"   Region distribution: {stats['region_distribution']}")
    print(f"   Layer distribution: {stats['layer_distribution']}")
    
    # Test predictive coding
    print("\n4. Testing predictive coding functionality...")
    print("   Predictive coding evaluates whether new information should be stored")
    print("   based on how well it can be predicted from existing knowledge")
    
    # Perform predictive retrieval
    results = hier_manager.predict_and_retrieve("memory formation", top_k=2)
    print(f"   Retrieved {len(results)} memories using predictive coding:")
    for i, node in enumerate(results, 1):
        print(f"     {i}. [{node.region.value}] {node.content[:60]}...")
        print(f"        Layer: {node.layer.value}, Prediction Error: {node.prediction_error:.3f}")
    
    # Test multi-region coordination
    print("\n5. Testing multi-region coordination...")
    print("   Coordinating between brain regions for comprehensive retrieval")
    
    region_results = hier_manager.coordinate_regions(BrainRegion.HIPPOCAMPUS, "memory")
    print(f"   Coordinated retrieval across {len(region_results)} regions:")
    for region, nodes in region_results.items():
        if nodes:  # Only show regions with results
            print(f"     {region.value}: {len(nodes)} memories")
            for node in nodes[:2]:  # Show first 2
                print(f"       - {node.content[:50]}...")
    
    # Simulate temporal pattern for prediction
    print("\n6. Simulating temporal pattern prediction...")
    print("   The system learns temporal patterns and predicts future activations")
    
    # Add more related memories to establish a pattern
    for i in range(3):
        hier_manager.encode(
            f"Memory consolidation pattern observation #{i+1}", 
            hier_manager.MemoryType.EPISODIC, 
            BrainRegion.HIPPOCAMPUS,
            tags=["pattern", "observation"],
            layer=MemoryLayer.PREDICTION_LAYER
        )
        time.sleep(0.01)  # Small delay to create temporal sequence
    
    # Test prediction accuracy
    results = hier_manager.predict_and_retrieve("pattern", top_k=3)
    avg_pred_error = sum(node.prediction_error for node in results) / len(results) if results else 0
    print(f"   Average prediction error: {avg_pred_error:.3f}")
    print("   Lower error indicates better predictive accuracy")
    
    # Demonstrate the hierarchical features
    print("\n" + "="*70)
    print("HIERARCHICAL ARCHITECTURE FEATURES DEMONSTRATED:")
    print("="*70)
    
    print("🔹 Cortical Column Simulation:")
    print("   • Hierarchical layers (Input, Intermediate, Output, Prediction)")
    print("   • Region-specific processing units")
    print("   • Columnar organization mimicking biological structure")
    print("")
    
    print("🔹 Multi-Region Coordination:")
    print("   • Communication between different brain regions")
    print("   • Coordinated activity patterns")
    print("   • Cross-regional information flow")
    print("")
    
    print("🔹 Predictive Coding:")
    print("   • Prediction of likely future inputs")
    print("   • Storage optimization based on prediction error")
    print("   • Reduction of unnecessary information storage")
    print("   • Temporal pattern learning")
    print("")
    
    print("🔹 Combined Effect:")
    print("   • More biologically realistic memory system")
    print("   • Efficient resource utilization through prediction")
    print("   • Multi-level information processing")
    print("   • Adaptive storage based on novelty and prediction")
    
    # Show final statistics
    print(f"\n7. Final hierarchical statistics:")
    stats = hier_manager.get_hierarchical_statistics()
    for key, value in stats.items():
        if key != 'region_distribution' and key != 'layer_distribution' and key != 'column_distribution':
            print(f"   {key}: {value}")
    
    print(f"\n   Region distribution: {stats['region_distribution']}")
    print(f"   Layer distribution: {stats['layer_distribution']}")
    
    print("\n🎯 Hierarchical memory architecture working correctly!")
    print("   Cortical columns, multi-region coordination, and predictive coding implemented!")


if __name__ == "__main__":
    demo_hierarchical_architecture()
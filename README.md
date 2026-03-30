# NeuroMem-Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A neuromorphic memory management system for AI agents inspired by biological neural networks. Designed to improve memory efficiency and computational performance compared to traditional RAG systems.

## 🧠 Key Features

- **Biologically-Inspired Architecture**: Models human memory systems (sensory, working, episodic, semantic)
- **Associative Retrieval**: Connections between related memories for contextual recall
- **Active Forgetting**: Automatic pruning of irrelevant memories to optimize space
- **Memory Consolidation**: Transfers important information from working to long-term memory
- **Efficiency Focused**: Reduced token consumption and memory footprint compared to traditional approaches

## 🚀 Quick Start

### Installation

```bash
pip install neuromem-agents
```

### Basic Usage

```python
from neuromem.core import MemoryManager, MemoryType

# Initialize the memory manager
mem_manager = MemoryManager(capacity=1000)

# Encode information into memory
id1 = mem_manager.encode("The capital of France is Paris", MemoryType.SEMANTIC)
id2 = mem_manager.encode("I visited Paris last summer", MemoryType.EPISODIC)

# Create associations between memories
mem_manager.associate(id1, id2, strength=0.8)

# Retrieve related memories
results = mem_manager.retrieve("Paris", top_k=3)
for node in results:
    print(f"- {node.content}")

# Get system statistics
stats = mem_manager.get_statistics()
print(stats)
```

### Comparison with Traditional RAG

```python
from neuromem.experiments import ComparisonEngine

# Run comparative analysis
engine = ComparisonEngine()
test_data = [
    {'content': 'Sample document content...', 'query': 'Sample query...'}
]
results = engine.run_comparison_test(test_data)
print(results['summary'])
```

## 🏗️ Architecture

### Memory Types
- **Sensory Memory**: Instantaneous storage (milliseconds)
- **Working Memory**: Active processing (seconds to minutes) 
- **Episodic Memory**: Personal experiences and events
- **Semantic Memory**: General world knowledge

### Core Components
- `MemoryManager`: Main interface for memory operations
- `SpikingNeuralNetwork`: Neural dynamics simulation
- `MemoryNode`: Individual memory units with metadata
- `ComparisonEngine`: Benchmarking tools

## 📊 Performance Benefits

Our neuromorphic approach typically provides:
- 2-3x reduction in token consumption
- Faster retrieval times for related information
- More efficient memory utilization
- Natural forgetting of irrelevant information

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by biological neural network research
- Built for the AI agent community
# NeuroMem-Agents: Neuromorphic Memory Management System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A biologically-inspired memory management system for AI agents that mimics human memory architecture to improve contextual understanding and associative retrieval.

## 🧠 Key Features

- **Biological Inspiration**: Models human memory types (sensory, working, episodic, semantic)
- **Associative Networks**: Connections between related memories enable contextual recall
- **Spreading Activation**: Related memories are automatically activated during retrieval
- **Adaptive Forgetting**: Irrelevant memories are pruned to maintain efficiency
- **Contextual Tagging**: Rich metadata enables precise filtering and categorization

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

### Running Experiments and Visualizations

The project includes comprehensive benchmarking tools:

- `benchmark_test.py`: Basic performance comparison
- `advanced_benchmark.py`: Complex scenario evaluation
- `extended_conversation_test.py`: 25+ interaction conversation simulation
- `visualize_results.py`: Generate performance comparison charts

To run all tests and generate visualizations:

```bash
cd neuromem-agents
source venv/bin/activate  # if using virtual environment
pip3 install matplotlib numpy seaborn  # for visualizations
python3 extended_conversation_test.py  # Extended conversation test
python3 visualize_results.py          # Generate visualizations
```

Generated visualization files:
- `benchmark_visualization_fixed.png`: Performance comparison chart
- `extended_conversation_results_fixed.png`: Extended test results
- `simple_comparison_fixed.png`: Simple comparison chart

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

## 📊 Performance Characteristics

Our neuromorphic approach demonstrates:
- Superior performance in complex, interconnected knowledge domains
- Better contextual understanding for related information retrieval
- Improved handling of extended conversation scenarios
- Trade-offs in raw efficiency for enhanced cognitive capabilities

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by biological neural network research
- Built for the AI agent community

---

# NeuroMem-Agents：神经形态记忆管理系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

一个生物学启发的AI代理记忆管理系统，模仿人类记忆架构以改善情境理解和关联检索。

## 🧠 主要特性

- **生物学启发**：建模人类记忆类型（感觉记忆、工作记忆、情节记忆、语义记忆）
- **关联网络**：相关记忆间的连接实现情境召回
- **扩散激活**：检索时自动激活相关记忆
- **自适应遗忘**：修剪不相关信息以保持效率
- **情境标记**：丰富的元数据实现精确过滤和分类

## 🚀 快速开始

### 安装

```bash
pip install neuromem-agents
```

### 基础使用

```python
from neuromem.core import MemoryManager, MemoryType

# 初始化记忆管理器
mem_manager = MemoryManager(capacity=1000)

# 编码信息到记忆中
id1 = mem_manager.encode("法国的首都是巴黎", MemoryType.SEMANTIC)
id2 = mem_manager.encode("我去年夏天参观了巴黎", MemoryType.EPISODIC)

# 在记忆间创建关联
mem_manager.associate(id1, id2, strength=0.8)

# 检索相关记忆
results = mem_manager.retrieve("巴黎", top_k=3)
for node in results:
    print(f"- {node.content}")

# 获取系统统计信息
stats = mem_manager.get_statistics()
print(stats)
```

### 与传统RAG的对比

```python
from neuromem.experiments import ComparisonEngine

# 运行对比分析
engine = ComparisonEngine()
test_data = [
    {'content': '示例文档内容...', 'query': '示例查询...'}
]
results = engine.run_comparison_test(test_data)
print(results['summary'])
```

### 运行实验和可视化

项目包含全面的基准测试工具：

- `benchmark_test.py`: 基础性能对比
- `advanced_benchmark.py`: 复杂场景评估
- `extended_conversation_test.py`: 25+次交互对话模拟
- `visualize_results.py`: 生成性能对比图表

运行所有测试并生成可视化：

```bash
cd neuromem-agents
source venv/bin/activate  # 如果使用虚拟环境
pip3 install matplotlib numpy seaborn  # 用于可视化
python3 extended_conversation_test.py  # 扩展对话测试
python3 visualize_results.py          # 生成可视化
```

生成的可视化文件：
- `benchmark_visualization_fixed.png`: 性能对比图表
- `extended_conversation_results_fixed.png`: 扩展测试结果
- `simple_comparison_fixed.png`: 简单对比图表

## 🏗️ 架构

### 记忆类型
- **感觉记忆**：即时存储（毫秒级）
- **工作记忆**：主动处理（秒到分钟级）
- **情节记忆**：个人经验和事件
- **语义记忆**：一般世界知识

### 核心组件
- `MemoryManager`: 记忆操作的主接口
- `SpikingNeuralNetwork`: 神经动力学模拟
- `MemoryNode`: 带元数据的单个记忆单元
- `ComparisonEngine`: 基准测试工具

## 📊 性能特征

我们的神经形态方法展示：
- 在复杂、互联的知识领域表现卓越
- 在相关息检索的情境理解方面更优
- 在扩展对话场景中的处理能力改善
- 在原始效率与增强的认知能力之间的权衡

## 🤝 贡献

欢迎贡献！详情请参见我们的[贡献指南](CONTRIBUTING.md)。

## 📄 许可证

该项目基于MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 🙏 致谢

- 受生物神经网络研究启发
- 为AI代理社区构建
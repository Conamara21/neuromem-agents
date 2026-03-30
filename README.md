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
- **Persistent Storage**: Memory data is saved to and loaded from database for continuity
- **Memory Consolidation**: High-value memories are transferred from working to long-term storage

## 🧠 Biological Design Details

Our system closely mimics human memory architecture:

### Hippocampus-Cortex Analogy
- **Working Memory Buffer**: Analogous to the hippocampus - high-speed but limited capacity storage for recent information
- **Long-term Memory**: Analogous to the cortex - vast storage capacity but slower access
- **Consolidation Process**: During idle periods (simulating sleep), frequently accessed memories in the working buffer are transferred to long-term storage

### Sleep-like Consolidation Mechanism
- **Idle State Detection**: The system identifies when it's not actively processing requests
- **Background Processing**: During idle periods, consolidates high-value memories from working to long-term storage
- **Frequency-based Selection**: Memories accessed more than a threshold (default: 3 times) are moved to long-term storage
- **Compression**: Information is processed and integrated during transfer to optimize long-term storage

### Memory Types and Their Biological Counterparts
- **Sensory Memory**: Mimics instantaneous sensory storage (milliseconds)
- **Working Memory**: Like the hippocampus, handles active processing with limited capacity
- **Episodic Memory**: Stores personal experiences and contextual events
- **Semantic Memory**: General knowledge and facts, like cortical storage

### Adaptive Forgetting
- **Decay Modeling**: Memories fade over time if not accessed
- **Importance Weighting**: Frequently accessed memories are retained longer
- **Resource Optimization**: Low-value memories are automatically pruned to maintain system efficiency

### Associative Networks
- **Synaptic Connections**: Memories are connected like neural pathways in the brain
- **Hebbian Learning**: "Cells that fire together, wire together" - connections strengthen with repeated activation
- **Spreading Activation**: When retrieving a memory, related memories are automatically activated

## 🚀 Quick Start

### Installation

```bash
pip install neuromem-agents
```

### Basic Usage

```python
from neuromem.core import MemoryManager, MemoryType

# Initialize the memory manager with persistence
mem_manager = MemoryManager(capacity=1000, db_path='my_memory.db')

# Encode information into memory
id1 = mem_manager.encode("The capital of France is Paris", MemoryType.SEMANTIC)
id2 = mem_manager.encode("I visited Paris last summer", MemoryType.EPISODIC)

# Create associations between memories
mem_manager.associate(id1, id2, strength=0.8)

# Retrieve related memories
results = mem_manager.retrieve("Paris", top_k=3)
for node in results:
    print(f"- {node.content}")

# Perform memory consolidation (like sleep/dreaming)
mem_manager.consolidate()

# Save to persistent storage
mem_manager.save_to_db()

# Get system statistics
stats = mem_manager.get_statistics()
print(stats)
```

### Loading from Persistent Storage

```python
from neuromem.core import MemoryManager

# Create a new instance
restored_manager = MemoryManager(capacity=1000, db_path='my_memory.db')

# Load from persistent storage
restored_manager.load_from_db()

# Continue using the restored memory system
results = restored_manager.retrieve("Paris", top_k=3)
print(f"Retrieved {len(results)} memories from persistent storage")
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
- `MemoryManager`: Main interface for memory operations with persistence
- `SpikingNeuralNetwork`: Neural dynamics simulation
- `MemoryNode`: Individual memory units with metadata
- `MemoryDatabase`: Persistent storage using SQLite
- `ComparisonEngine`: Benchmarking tools

## 📊 Performance Characteristics

Our neuromorphic approach demonstrates:
- Superior performance in complex, interconnected knowledge domains
- Better contextual understanding for related information retrieval
- Improved handling of extended conversation scenarios
- Trade-offs in raw efficiency for enhanced cognitive capabilities
- Continuity through persistent storage and memory consolidation

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
- **持久化存储**：记忆数据保存到数据库以保证连续性
- **记忆巩固**：高价值记忆从工作记忆转移到长期存储

## 🧠 类人脑设计细节

我们的系统紧密模仿人类记忆架构：

### 海马体-皮质类比
- **工作记忆缓冲区**：类似于海马体 - 高速但容量有限的近期信息存储
- **长期记忆**：类似于皮质 - 巨大存储容量但访问较慢
- **巩固过程**：在空闲期间（模拟睡眠），工作缓冲区中频繁访问的记忆被转移到长期存储

### 睡眠样巩固机制
- **空闲状态检测**：系统识别何时未积极处理请求
- **后台处理**：空闲期间，将高价值记忆从工作记忆转移到长期存储
- **频率选择**：访问次数超过阈值（默认：3次）的记忆被移动到长期存储
- **压缩**：传输过程中处理和整合信息以优化长期存储

### 记忆类型及其生物对应物
- **感觉记忆**：模仿瞬间感官存储（毫秒级）
- **工作记忆**：类似海马体，处理活跃任务，容量有限
- **情节记忆**：存储个人经历和情境事件
- **语义记忆**：一般知识和事实，类似皮质存储

### 自适应遗忘
- **衰减建模**：如果不访问，记忆会随时间消退
- **重要性加权**：频繁访问的记忆保留更久
- **资源优化**：自动修剪低价值记忆以保持系统效率

### 关联网络
- **突触连接**：记忆像大脑中的神经通路一样相互连接
- **赫布学习**："一起激发的细胞，一起连接" - 重复激活会加强连接
- **扩散激活**：检索记忆时，相关记忆会自动激活

## 🚀 快速开始

### 安装

```bash
pip install neuromem-agents
```

### 基础使用

```python
from neuromem.core import MemoryManager, MemoryType

# 初始化带持久化的记忆管理器
mem_manager = MemoryManager(capacity=1000, db_path='my_memory.db')

# 编码信息到记忆中
id1 = mem_manager.encode("法国的首都是巴黎", MemoryType.SEMANTIC)
id2 = mem_manager.encode("我去年夏天参观了巴黎", MemoryType.EPISODIC)

# 在记忆间创建关联
mem_manager.associate(id1, id2, strength=0.8)

# 检索相关记忆
results = mem_manager.retrieve("巴黎", top_k=3)
for node in results:
    print(f"- {node.content}")

# 执行记忆巩固（类似睡眠/梦境）
mem_manager.consolidate()

# 保存到持久化存储
mem_manager.save_to_db()

# 获取系统统计信息
stats = mem_manager.get_statistics()
print(stats)
```

### 从持久化存储加载

```python
from neuromem.core import MemoryManager

# 创建新实例
restored_manager = MemoryManager(capacity=1000, db_path='my_memory.db')

# 从持久化存储加载
restored_manager.load_from_db()

# 继续使用恢复的记忆系统
results = restored_manager.retrieve("巴黎", top_k=3)
print(f"从持久化存储检索到 {len(results)} 个记忆")
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
- `MemoryManager`: 记忆操作的主接口（含持久化）
- `SpikingNeuralNetwork`: 神经动力学模拟
- `MemoryNode`: 带元数据的单个记忆单元
- `MemoryDatabase`: 使用SQLite的持久化存储
- `ComparisonEngine`: 基准测试工具

## 📊 性能特征

我们的神经形态方法展示：
- 在复杂、互联的知识领域表现卓越
- 在相关息检索的情境理解方面更优
- 在扩展对话场景中的处理能力改善
- 在原始效率与增强的认知能力之间的权衡
- 通过持久化存储和记忆巩固实现连续性

## 🤝 贡献

欢迎贡献！详情请参见我们的[贡献指南](CONTRIBUTING.md)。

## 📄 许可证

该项目基于MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 🙏 致谢

- 受生物神经网络研究启发
- 为AI代理社区构建
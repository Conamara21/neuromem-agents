# NeuroMem 脑启发结构思维导图

此文件由 `scripts/render_brain_region_mindmap.py` 生成，建议编辑生成脚本而不是手改本文件。

```mermaid
mindmap
  root((NeuroMem 脑启发记忆结构))
    海马体 Hippocampus
      工作记忆缓冲 + 记忆巩固
      module neuromem/core/enhanced_memory_manager.py
      key working_memory_buffer, consolidate()
    大脑皮质 Cortex
      长期记忆存储 + 知识沉淀
      module neuromem/core/enhanced_memory_manager.py
      key long_term_memory
    边缘系统 Limbic System
      重要性标注 + 检索偏置
      module neuromem/core/enhanced_memory_manager.py
      key importance_score, attention_weight
    神经振荡 Neural Oscillations
      脉冲放电 + 节律同步
      module neuromem/core/enhanced_memory_manager.py
      key SpikingNeuralNetwork
    突触可塑性 Synaptic Plasticity
      时序依赖的连接更新
      module neuromem/core/enhanced_memory_manager.py
      key STDPMechanism
    前额叶皮质 Prefrontal Cortex
      执行控制 + 注意力门控
      module neuromem/core/enhanced_memory_manager.py
      key AttentionGate
    小脑 Cerebellum
      元学习调参 + 协调优化
      module neuromem/core/enhanced_memory_manager.py
      key MetaLearningController
    颞叶 Temporal Lobe
      语言模式预测 + 状态外推
      module neuromem/core/hierarchical_memory.py
      key PredictionEngine
    顶叶 Parietal Lobe
      跨脑区协调 + 注意定向
      module neuromem/core/hierarchical_memory.py
      key coordinate_regions()
    预测编码 Predictive Coding
      预测误差驱动存储
      module neuromem/core/hierarchical_memory.py
      key _evaluate_storage_necessity()
```

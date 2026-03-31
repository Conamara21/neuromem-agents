# NeuroMem 架构思维导图

这个文件包含两种 Mermaid 图：

- `mindmap`：适合快速回忆整体设计
- `flowchart`：适合看写入、检索、巩固、遗忘的主流程

如果你的 Markdown 预览器不支持 `mindmap`，可以直接看下面的 `flowchart`，或者把代码粘到 Mermaid Live Editor。

## 1. 思维导图

```mermaid
mindmap
  root((NeuroMem 记忆结构))
    核心主线
      MemoryManager
        memory_nodes
        connections
        access_frequency
        working_memory_buffer
        long_term_memory
        current_context
      MemoryNode
        content
        embedding
        memory_type
        timestamp
        tags
        importance_score
        connectivity_strength
        decay_rate
      MemoryType
        SENSORY
        WORKING
        EPISODIC
        SEMANTIC
    写入路径
      encode
        生成 node_id
        生成 embedding
        创建 MemoryNode
        写入 memory_nodes
        初始化 access_frequency
        建立索引
        WORKING 写入短期缓冲
        持久化到 SQLite
    检索路径
      retrieve
        候选生成
          content_index 精确命中
          inverted_index 词项召回
          tag 匹配
          recent_accessed_nodes 最近激活种子
          association expansion 关联扩展
        候选打分
          cosine similarity
          访问频率加权
          关联加权
        输出
          top_k 返回
          更新 access_frequency
          更新 recent_accessed_nodes
    记忆演化
      consolidate
        WORKING 高频记忆
        转入 long_term_memory
        类型升级为 SEMANTIC
      forget
        时间衰减
        访问保护
        删除低价值节点
    索引系统
      content_index
      inverted_index
      node_index_terms
      token_document_frequency
      embedding_norms
    Embedding
      TextEmbedder 接口
      LexicalHashingEmbedder 默认
      TfidfEmbedder benchmark 用
    Persistence
      MemoryDatabase
        memory_nodes 表
        connections 表
        access_frequencies 表
        working_memory_buffer 表
        long_term_memory 表
      WAL
      batch save
      load_from_db 后重建索引
    扩展分支
      EnhancedMemoryManager
        STDPMechanism
        MetaLearningController
        AttentionGate
      HierarchicalMemoryManager
        BrainRegion
        MemoryLayer
        CorticalColumn
        PredictionEngine
        MultiRegionCoordinator
      EfficiencyOptimizedMemoryManager
        SparseActivationManager
        ProgressiveRefinementEngine
        QuantumInspiredOptimizer
    当前最成熟主干
      MemoryManager
      Shared Embeddings
      Persistence
      候选集检索
      严谨 benchmark 已验证
```

## 2. 结构流转图

```mermaid
flowchart TD
    A[输入内容 / Query] --> B{进入系统}

    B -->|写入| C[encode]
    C --> C1[生成 node_id]
    C1 --> C2[生成 embedding]
    C2 --> C3[创建 MemoryNode]
    C3 --> C4[写入 memory_nodes]
    C4 --> C5[更新 access_frequency]
    C5 --> C6[建立 content/tag/token 索引]
    C6 --> C7{memory_type == WORKING?}
    C7 -->|是| C8[写入 working_memory_buffer]
    C7 -->|否| C9[跳过短期缓冲]
    C8 --> C10[写入 SQLite]
    C9 --> C10

    B -->|检索| D[retrieve]
    D --> D1[生成 query embedding]
    D1 --> D2[候选集生成]
    D2 --> D21[content_index 精确命中]
    D2 --> D22[inverted_index 词项召回]
    D2 --> D23[tag/context 匹配]
    D2 --> D24[recent_accessed_nodes 种子]
    D2 --> D25[connections 关联扩展]
    D21 --> D3[候选集合并]
    D22 --> D3
    D23 --> D3
    D24 --> D3
    D25 --> D3
    D3 --> D4[cosine similarity]
    D4 --> D5[频率加权 + 关联加权]
    D5 --> D6[top_k 排序输出]
    D6 --> D7[更新 access_frequency]
    D7 --> D8[记录 recent_accessed_nodes]

    C10 --> E[持久化层 MemoryDatabase]
    D7 --> E

    E --> E1[memory_nodes]
    E --> E2[connections]
    E --> E3[access_frequencies]
    E --> E4[working_memory_buffer]
    E --> E5[long_term_memory]

    F[consolidate] --> F1[扫描 working_memory_buffer]
    F1 --> F2{访问次数 > 阈值?}
    F2 -->|是| F3[写入 long_term_memory]
    F3 --> F4[类型升级为 SEMANTIC]
    F2 -->|否| F5[保留]

    G[forget] --> G1[按 age 和 access 计算 forget_score]
    G1 --> G2{低于阈值?}
    G2 -->|是| G3[删除节点 + 索引 + 连接]
    G2 -->|否| G4[保留]

    H[扩展分支] --> H1[Enhanced: STDP / Meta-learning / Attention]
    H --> H2[Hierarchical: BrainRegion / Layer / Prediction]
    H --> H3[Efficiency: Sparse / Progressive / Quantum]
```

## 3. 一句话理解

NeuroMem 不是“纯向量库检索”，而是：

**分类型记忆节点 + 关联图 + 短期/长期记忆分层 + 访问驱动巩固与遗忘 + 上下文扩散检索**

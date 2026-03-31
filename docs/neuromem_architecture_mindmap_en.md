# NeuroMem Architecture Diagrams

This file contains two Mermaid diagrams:

- `mindmap`: a compact overview of the system design
- `flowchart`: the main write, retrieve, consolidate, and forget flows

If your Markdown renderer does not support `mindmap`, use the `flowchart` below or paste the code into Mermaid Live Editor.

## 1. Mindmap

```mermaid
mindmap
  root((NeuroMem Architecture))
    Core Spine
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
    Write Path
      encode
        generate node_id
        generate embedding
        create MemoryNode
        write to memory_nodes
        initialize access_frequency
        build indexes
        WORKING enters short-term buffer
        persist to SQLite
    Retrieval Path
      retrieve
        candidate generation
          exact hit via content_index
          token recall via inverted_index
          tag match
          recent_accessed_nodes seeds
          association expansion
        candidate scoring
          cosine similarity
          access-frequency boost
          association boost
        output
          return top_k
          update access_frequency
          update recent_accessed_nodes
    Memory Evolution
      consolidate
        high-frequency WORKING memory
        move into long_term_memory
        upgrade type to SEMANTIC
      forget
        time decay
        access protection
        delete low-value nodes
    Index System
      content_index
      inverted_index
      node_index_terms
      token_document_frequency
      embedding_norms
    Embeddings
      TextEmbedder interface
      LexicalHashingEmbedder default
      TfidfEmbedder for benchmark
    Persistence
      MemoryDatabase
        memory_nodes table
        connections table
        access_frequencies table
        working_memory_buffer table
        long_term_memory table
      WAL
      batch save
      rebuild indexes after load_from_db
    Extension Branches
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
    Most Mature Production Path
      MemoryManager
      Shared Embeddings
      Persistence
      candidate-pruned retrieval
      validated by rigorous benchmark
```

## 2. Flowchart

```mermaid
flowchart TD
    A[Input Content or Query] --> B{Enter System}

    B -->|Write| C[encode]
    C --> C1[Generate node_id]
    C1 --> C2[Generate embedding]
    C2 --> C3[Create MemoryNode]
    C3 --> C4[Write to memory_nodes]
    C4 --> C5[Update access_frequency]
    C5 --> C6[Build content, tag, and token indexes]
    C6 --> C7{memory_type == WORKING?}
    C7 -->|Yes| C8[Write to working_memory_buffer]
    C7 -->|No| C9[Skip short-term buffer]
    C8 --> C10[Persist to SQLite]
    C9 --> C10

    B -->|Retrieve| D[retrieve]
    D --> D1[Generate query embedding]
    D1 --> D2[Candidate generation]
    D2 --> D21[Exact hit via content_index]
    D2 --> D22[Token recall via inverted_index]
    D2 --> D23[tag and context match]
    D2 --> D24[recent_accessed_nodes seeds]
    D2 --> D25[connections association expansion]
    D21 --> D3[Merge candidate pool]
    D22 --> D3
    D23 --> D3
    D24 --> D3
    D25 --> D3
    D3 --> D4[cosine similarity]
    D4 --> D5[frequency boost plus association boost]
    D5 --> D6[top_k ranked output]
    D6 --> D7[Update access_frequency]
    D7 --> D8[Record recent_accessed_nodes]

    C10 --> E[Persistence Layer MemoryDatabase]
    D7 --> E

    E --> E1[memory_nodes]
    E --> E2[connections]
    E --> E3[access_frequencies]
    E --> E4[working_memory_buffer]
    E --> E5[long_term_memory]

    F[consolidate] --> F1[Scan working_memory_buffer]
    F1 --> F2{access count > threshold?}
    F2 -->|Yes| F3[Write to long_term_memory]
    F3 --> F4[Upgrade type to SEMANTIC]
    F2 -->|No| F5[Keep node]

    G[forget] --> G1[Compute forget_score from age and access]
    G1 --> G2{below threshold?}
    G2 -->|Yes| G3[Delete node, indexes, and links]
    G2 -->|No| G4[Keep node]

    H[Extension Branches] --> H1[Enhanced: STDP, Meta-learning, Attention]
    H --> H2[Hierarchical: BrainRegion, Layer, Prediction]
    H --> H3[Efficiency: Sparse, Progressive, Quantum]
```

## 3. One-line Summary

NeuroMem is not just a vector store. It is:

**typed memory nodes + an associative graph + short-term and long-term layering + access-driven consolidation and forgetting + context-expanding retrieval**


# Changelog

All notable changes to the NeuroMem-Agents project will be documented in this file.

## [Unreleased] - YYYY-MM-DD

### Added
- Enhanced Memory Manager with neural plasticity features
- STDP (Spike-Timing Dependent Plasticity) implementation
- Meta-learning controller for adaptive parameters
- Attention gating mechanism for selective processing
- Demo script for neural plasticity features (`demo_plasticity.py`)
- Feature comparison documentation
- Updated README with enhanced functionality

### Changed
- Refactored `MemoryManager` to support enhanced features
- Updated persistence layer to handle temporal connection data
- Improved README with detailed neural plasticity explanations
- Enhanced documentation with Chinese translations

## [1.1.0] - 2026-03-30

### Added
- Neural Plasticity Features:
  - STDP (Spike-Timing Dependent Plasticity) mechanism
    - Connection strengths adjust based on activation timing
    - Simulates biological long-term potentiation/depression
    - Uses temporal correlation for connection updates
  - Meta-Learning Controller
    - Dynamic learning rate adjustment
    - Adaptive consolidation thresholds
    - Performance-based parameter tuning
  - Attention Gate Mechanism
    - Selective processing of relevant information
    - Attention-weighted retrieval
    - Focus strength adjustment
  - EnhancedMemoryManager class with all plasticity features
  - Backward compatibility with original MemoryManager

### Changed
- Updated connection storage to include timestamps
- Modified association mechanism to incorporate STDP
- Enhanced retrieval with attention gating
- Improved memory consolidation with meta-learning

## [1.0.0] - 2026-03-30

### Added
- Core Memory Management System
- Biological Memory Types (Sensory, Working, Episodic, Semantic)
- Associative Networks with spreading activation
- Memory Consolidation (hippocampus-cortex analogy)
- Adaptive Forgetting mechanism
- Persistent Storage with SQLite
- Traditional RAG Baseline for comparison
- Comprehensive Benchmarking Tools
- Experimental Framework
- Visualization Tools
- Multi-language Documentation

### Features
- Bio-inspired memory architecture
- Contextual retrieval with associative networks
- Sleep-like consolidation process
- Cross-session persistence
- Performance benchmarking
- Easy-to-use API
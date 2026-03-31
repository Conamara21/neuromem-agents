# Running NeuroMem-Agents Tests

This guide explains how to run the various tests and benchmarks included in the NeuroMem-Agents project.

## Prerequisites

Make sure you have the required libraries installed:

```bash
cd /path/to/neuromem-agents
source venv/bin/activate  # Activate the virtual environment
pip3 install matplotlib numpy seaborn
```

## Available Tests

### 1. Simple Benchmark Test
Tests basic performance with simple queries:

```bash
python3 benchmark_test.py
```

### 2. Complex Scenario Test
Tests performance with interconnected knowledge domains:

```bash
python3 advanced_benchmark.py
```

### 3. Extended Conversation Test (25+ interactions)
Simulates extended dialogue scenarios:

```bash
python3 extended_conversation_test.py
```

### 4. Interactive Demo
Shows key features of the NeuroMem system:

```bash
python3 demo.py
```

### 5. Generate Visualizations
Creates charts showing performance comparisons:

```bash
python3 visualize_results.py
```

## Results Files

After running tests, you'll find results in:

- `benchmark_visualization.png` - Performance comparison charts
- `extended_conversation_results.png` - Extended test charts  
- `detailed_analysis.json` - Raw numerical data
- `COMPREHENSIVE_RESULTS.md` - Complete results summary

## Virtual Environment

Always activate the virtual environment before running tests:

```bash
cd /path/to/neuromem-agents
source venv/bin/activate
```

## Running All Tests Sequentially

To run all tests in sequence:

```bash
source venv/bin/activate
echo "Running simple benchmark..."
python3 benchmark_test.py
echo "Running complex scenario test..."
python3 advanced_benchmark.py
echo "Running extended conversation test..."
python3 extended_conversation_test.py
echo "Running demo..."
python3 demo.py
echo "Generating visualizations..."
python3 visualize_results.py
echo "All tests completed!"
```

## Interpreting Results

- **Lower ratios (< 1.0)** in performance comparisons favor NeuroMem
- **Higher percentages** in recall tests favor the respective system
- **Visualizations** provide intuitive comparison of the two approaches
- **Complex scenarios** tend to favor NeuroMem's associative capabilities
- **Simple scenarios** tend to favor Traditional RAG's efficiency

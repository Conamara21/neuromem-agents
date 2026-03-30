#!/bin/bash

# Installation script for NeuroMem-Agents

echo "Setting up NeuroMem-Agents..."
echo "==============================="

# Check if Python 3.8+ is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3.8+ is required but not found."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_VERSION" \< "3.8" ]; then
    echo "Error: Python 3.8+ is required. Found $PYTHON_VERSION."
    exit 1
fi

echo "Python version: $PYTHON_VERSION ✓"

# Install dependencies
echo "Installing dependencies..."
pip3 install -e .

echo "Installation completed!"
echo ""
echo "To run experiments:"
echo "python -m neuromem.experiments.comparison_engine"
echo ""
echo "To use the library in your own projects:"
echo "from neuromem.core import MemoryManager, MemoryType"
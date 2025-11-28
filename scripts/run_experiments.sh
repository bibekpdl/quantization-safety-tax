#!/bin/bash
# Run experiments for Quantization Safety Study
# Author: Bibek Poudel

set -e

echo "=========================================="
echo "Quantization Safety Study - Run Experiments"
echo "=========================================="

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running."
    echo "Please start Ollama first: ollama serve"
    exit 1
fi

echo "✓ Ollama is running"
echo ""

# Create directories
mkdir -p data/results
mkdir -p logs
mkdir -p figures

# Phase 1: Data Collection
echo "=========================================="
echo "Phase 1: Data Collection"
echo "=========================================="
python src/data_collection.py

# Phase 2: Classification
echo ""
echo "=========================================="
echo "Phase 2: Response Classification"
echo "=========================================="
python src/classification.py

# Phase 3: Analysis
echo ""
echo "=========================================="
echo "Phase 3: Statistical Analysis"
echo "=========================================="
python src/analysis.py

# Phase 4: Visualization
echo ""
echo "=========================================="
echo "Phase 4: Generate Figures"
echo "=========================================="
python src/visualization.py

echo ""
echo "=========================================="
echo "Experiments Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - data/results/ (JSON files)"
echo "  - figures/ (PNG and PDF figures)"
echo ""

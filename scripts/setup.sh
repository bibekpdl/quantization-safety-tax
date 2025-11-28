#!/bin/bash
# Setup script for Quantization Safety Study
# Author: Bibek Poudel

set -e

echo "=========================================="
echo "Quantization Safety Study - Setup"
echo "=========================================="

# Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama is not installed."
    echo "Please install Ollama from https://ollama.ai/"
    exit 1
fi

echo "✓ Ollama is installed"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 5
fi

echo "✓ Ollama is running"

# Create directories
mkdir -p data/results
mkdir -p logs
mkdir -p figures

echo "✓ Directories created"

# Pull models (this will take a while)
echo ""
echo "Pulling models... This may take 30-60 minutes depending on your connection."
echo ""

# Llama 3.2
echo "Pulling Llama 3.2 models..."
ollama pull llama3.2:3b-instruct-fp16 || true
ollama pull llama3.2:3b-instruct-q8_0 || true
ollama pull llama3.2:3b-instruct-q4_K_M || true
ollama pull llama3.2:3b-instruct-q2_K || true

# Gemma 2
echo "Pulling Gemma 2 models..."
ollama pull gemma2:2b-instruct-fp16 || true
ollama pull gemma2:2b-instruct-q8_0 || true
ollama pull gemma2:2b-instruct-q4_K_M || true
ollama pull gemma2:2b-instruct-q2_K || true

# Gemma 3
echo "Pulling Gemma 3 models..."
ollama pull gemma3:4b-it-fp16 || true
ollama pull gemma3:4b-it-q8_0 || true
ollama pull gemma3:4b-it-q4_K_M || true

# Qwen 2.5
echo "Pulling Qwen 2.5 models..."
ollama pull qwen2.5:3b-instruct-fp16 || true
ollama pull qwen2.5:3b-instruct-q8_0 || true
ollama pull qwen2.5:3b-instruct-q4_K_M || true
ollama pull qwen2.5:3b-instruct-q2_K || true

# Phi-3
echo "Pulling Phi-3 models..."
ollama pull phi3:3.8b-mini-4k-instruct-fp16 || true
ollama pull phi3:3.8b-mini-4k-instruct-q8_0 || true
ollama pull phi3:3.8b-mini-4k-instruct-q4_K_M || true
ollama pull phi3:3.8b-mini-4k-instruct-q2_K || true

# Granite 3.1
echo "Pulling Granite 3.1 models..."
ollama pull granite3.1-dense:2b-instruct-fp16 || true
ollama pull granite3.1-dense:2b-instruct-q8_0 || true
ollama pull granite3.1-dense:2b-instruct-q4_K_M || true
ollama pull granite3.1-dense:2b-instruct-q2_K || true

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run experiments: ./scripts/run_experiments.sh"
echo "   Or run individual scripts in src/"
echo ""

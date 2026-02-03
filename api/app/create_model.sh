#!/bin/bash

# Script to create custom Ollama model using CLI
# This is more reliable than using the Python API

echo "Creating custom Ollama model: llama31_datanator"
echo "================================================"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed or not in PATH"
    exit 1
fi

# Check if base model exists
echo "Checking for base model llama3.1..."
if ollama list | grep -q "llama3.1"; then
    echo "✓ Base model llama3.1 found"
else
    echo "Base model not found. Pulling llama3.1..."
    ollama pull llama3.1
fi

# Create the custom model from modelfile
echo ""
echo "Creating custom model from ollama_modfile..."
ollama create llama31_datanator -f ollama_modfile

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Model created successfully!"
    echo ""
    echo "Verify with: ollama list"
    ollama list | grep llama31_datanator
else
    echo ""
    echo "✗ Failed to create model"
    exit 1
fi

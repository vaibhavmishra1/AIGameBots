#!/bin/bash

# AttentionV1 API Server Startup Script

echo "Starting AttentionV1 API Server..."
echo "=================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import torch, fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    python3 -m pip install torch fastapi uvicorn requests
fi

# Check if model checkpoint exists
MODEL_PATH="/Users/vaibhav/Desktop/AIGameBots/training/attention_v1/training/attention_v1/checkpoints/av1_best.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model checkpoint not found at $MODEL_PATH"
    echo "Make sure the trained model is available"
fi

# Start the server
echo "Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

cd "/Users/vaibhav/Desktop/AIGameBots/training/attention_v1"
python3 api_server.py

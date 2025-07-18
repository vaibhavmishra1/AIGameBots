#!/bin/bash

# AI Game Agents API Server Startup Script

echo "Starting AI Game Agents API Server..."
echo "Unity Integration Enabled"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn, torch, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install -r requirements.txt
fi

# Check if models directory exists
MODELS_DIR="/Users/vaibhavmishra/Desktop/btx-game-aicode/AIGameAgents/training/ann/models"
if [ ! -d "$MODELS_DIR" ]; then
    echo "Warning: Models directory not found at $MODELS_DIR"
    echo "Please ensure you have trained models before starting the server."
    echo "You can train models using: python3 train.py"
fi

# Start the server
echo "Starting server on http://localhost:8000"
echo "API documentation will be available at http://localhost:8000/docs"
echo "Unity endpoint: POST http://localhost:8000/predict"
echo "Press Ctrl+C to stop the server"
echo ""

python3 api_server.py 
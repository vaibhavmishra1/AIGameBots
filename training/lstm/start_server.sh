#!/bin/bash

# LSTM AI Game Agents API Server Startup Script

echo "Starting LSTM AI Game Agents API Server..."
echo "Unity Integration Enabled with Session Management"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn, torch, numpy, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install fastapi uvicorn torch numpy matplotlib pydantic
fi

# Check if LSTM model checkpoint exists
CHECKPOINT_PATH="/Users/vaibhavmishra/Desktop/Desktop/btx-game-aicode/AIGameAgents/training/lstm/checkpoints/lstm_agent/best_model.pt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Warning: LSTM model checkpoint not found at $CHECKPOINT_PATH"
    echo "Please ensure you have trained the LSTM model before starting the server."
    echo "You can train the model using: python3 train.py"
    exit 1
fi

# Check if normalization stats exist
NORM_STATS="/Users/vaibhavmishra/Desktop/Desktop/btx-game-aicode/AIGameAgents/training/lstm/checkpoints/lstm_agent/normalization_stats.npz"
if [ ! -f "$NORM_STATS" ]; then
    echo "Warning: Normalization stats not found at $NORM_STATS"
    echo "The model may not work correctly without proper normalization stats."
fi

# Start the server
echo "Starting LSTM server on http://localhost:8001"
echo "API documentation will be available at http://localhost:8001/docs"
echo "Unity endpoint: POST http://localhost:8001/predict"
echo "Session management endpoint: POST http://localhost:8001/sessions/create"
echo "Press Ctrl+C to stop the server"
echo ""

python3 api_server.py
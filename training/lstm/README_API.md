# LSTM AI Game Agents API Server

This FastAPI server provides a REST API for serving trained LSTM models for sequential game action prediction with uncertainty estimation.

## Features

- **Sequential Model Serving**: Serve trained LSTM models with memory/state management
- **Session Management**: Maintain LSTM hidden states across requests for temporal consistency
- **Uncertainty Estimation**: Get uncertainty estimates for predictions
- **Unity Integration**: Direct compatibility with Unity game agents
- **Real-time Visualization**: Live plotting of Unity states
- **Performance Benchmarking**: Built-in performance testing
- **Auto-documentation**: Interactive API documentation with Swagger UI

## Model Architecture

The LSTM model is significantly different from the ANN model:

### Input/Output
- **Input**: 20-dimensional feature vectors (vs 500 for ANN)
- **Output**: 4 continuous actions: `[move_x, move_z, look_x, look_z]`
- **Sequence**: Uses history of 5 previous frames
- **Memory**: Maintains LSTM hidden state between predictions

### Architecture Details
- **Model Type**: LSTM with Uncertainty Estimation
- **Hidden Dimension**: 512
- **LSTM Layers**: 4 layers
- **Attention**: Multi-head attention mechanism
- **Residual Connections**: For better gradient flow
- **Activation**: GELU activation functions

## Installation

1. Install dependencies:
```bash
pip install fastapi uvicorn torch numpy matplotlib pydantic
```

2. Ensure you have a trained LSTM model:
```
training/lstm/checkpoints/lstm_agent/
├── best_model.pt
├── config.json
└── normalization_stats.npz
```

3. Train the model if needed:
```bash
cd training/lstm
python train.py
```

## Running the Server

### Using the startup script:
```bash
cd training/lstm
chmod +x start_server.sh
./start_server.sh
```

### Or run directly:
```bash
cd training/lstm
python api_server.py
```

The server will start on `http://localhost:8001` (different port from ANN server)

## API Documentation

Once running, access:
- **Interactive API docs**: http://localhost:8001/docs
- **Alternative API docs**: http://localhost:8001/redoc
- **Health check**: http://localhost:8001/

## API Endpoints

### Health Check
```http
GET /
```
Returns server status, model information, and active session count.

### Session Management

#### Create Session
```http
POST /sessions/create
```
Creates a new session with isolated LSTM state.

**Response:**
```json
{
  "session_id": "uuid-string",
  "message": "Session created successfully"
}
```

#### List Sessions
```http
GET /sessions
```
Lists all active sessions.

#### Reset Session
```http
POST /sessions/{session_id}/reset
```
Resets the LSTM hidden state for a session.

#### Delete Session
```http
DELETE /sessions/{session_id}
```
Deletes a session and frees memory.

### Prediction Endpoints

#### Unity Game Agent Prediction
```http
POST /predict
```
Main endpoint for Unity integration. Converts 20 Unity states into a [20x20] feature sequence for LSTM processing.

**Input Processing:**
- Takes 20 Unity states (complete game state objects)
- Each state is converted to 20 features  
- Creates a sequence of shape [20 timesteps, 20 features]
- LSTM processes the full temporal sequence

**Request Body:**
```json
{
  "states": [
    {
      "agentPosition": {"x": 0.01, "y": 0.0005, "z": -0.005},
      "agentRotation": {"x": 0, "y": 0.5, "z": 0},
      "agentForward": {"x": 0, "y": 0, "z": 1},
      "health": 1.0,
      "weapon": 0,
      "targetPosition": {"x": 0.015, "y": 0.0005, "z": -0.002},
      "targetRotation": {"x": 0, "y": 0.25, "z": 0},
      "targetForward": {"x": 1, "y": 0, "z": 0},
      "directionToTarget": {"x": 0.84, "y": 0, "z": 0.54},
      "cross": {"x": 0, "y": -0.54, "z": 0},
      "distance": 0.006,
      "dotProduct": 0.54,
      "islos": true
    }
    // ... 19 more states (total of 20 required)
  ],
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "actions": [0.2, -0.1, 1.5, 0.0],
  "uncertainties": [0.1, 0.05, 0.2, 0.08],
  "session_id": "session-uuid"
}
```

**Action Mapping:**
- Index 0: Movement X direction [-1, 1]
- Index 1: Movement Z direction [-1, 1]  
- Index 2: Look rotation X [-10, 10]
- Index 3: Look rotation Z [-10, 10]

#### Raw Features Prediction
```http
POST /predict/features
```
Make predictions using 20-dimensional feature vectors.

**Request Body:**
```json
{
  "features": [0.01, 0.0005, -0.005, 0.5, 0, 0, 1, 1.0, 0, 1, ...],  // 20 values
  "session_id": "optional-session-id",
  "return_uncertainty": true
}
```

#### Sequence Prediction
```http
POST /predict/sequence
```
Make predictions using feature sequences with optional previous actions.

**Request Body:**
```json
{
  "feature_sequences": [
    [0.01, 0.0005, ...],  // Frame 1 (20 features)
    [0.011, 0.0005, ...], // Frame 2 (20 features)
    [0.012, 0.0005, ...], // Frame 3 (20 features)
    [0.013, 0.0005, ...], // Frame 4 (20 features)
    [0.014, 0.0005, ...]  // Frame 5 (20 features)
  ],
  "prev_actions": [
    [0.1, 0.0, 0.5, 0.0], // Action after frame 1
    [0.2, 0.0, 1.0, 0.0], // Action after frame 2
    [0.1, 0.0, 0.5, 0.0], // Action after frame 3
    [0.0, 0.0, 0.0, 0.0]  // Action after frame 4
  ],
  "session_id": "optional-session-id",
  "return_uncertainty": true
}
```

### Model Information
```http
GET /model/info
```
Get detailed information about the loaded LSTM model.

### Performance Benchmarking
```http
POST /benchmark?num_iterations=1000
```
Benchmark inference speed and throughput.

## Unity Integration

### Key Differences from ANN
1. **State Management**: LSTM maintains memory between requests
2. **Feature Processing**: Converts 20 Unity states → 20x20 feature sequence (ANN used 20 states → 500 features)
3. **Temporal Modeling**: Processes sequences of 20 timesteps (ANN was single-frame)
4. **Action Space**: Outputs 4 continuous actions (not 10 discrete binary actions)
5. **Sessions**: Requires session management for proper temporal modeling

### Unity C# Integration

```csharp
[System.Serializable]
public class LSTMAPIResponse
{
    public float[] actions;
    public float[] uncertainties;
    public string session_id;
}

// Store session ID for consistent predictions
private string sessionId;

// Make prediction
var payload = new StatePayload { states = stateHistory };
var json = JsonUtility.ToJson(payload);

var content = new StringContent(json, Encoding.UTF8, "application/json");
var response = await httpClient.PostAsync("http://localhost:8001/predict", content);

var responseContent = await response.Content.ReadAsStringAsync();
var apiResponse = JsonUtility.FromJson<LSTMAPIResponse>(responseContent);

// Store session ID for next request
sessionId = apiResponse.session_id;

// Use continuous actions
float moveX = apiResponse.actions[0];      // [-1, 1]
float moveZ = apiResponse.actions[1];      // [-1, 1] 
float lookX = apiResponse.actions[2];      // [-10, 10]
float lookZ = apiResponse.actions[3];      // [-10, 10]
```

## Testing

### Unity Integration Test
```bash
python test_unity_client.py
```

### Direct Feature Test
```bash
python test_client.py
```

### Test Features
- Session creation and management
- Unity state conversion
- Sequential predictions
- Uncertainty estimation
- Performance benchmarking
- Error handling

## Example Usage with Python

```python
import requests
import numpy as np

# Create client and session
client = requests.Session()
base_url = "http://localhost:8001"

# Create session
response = client.post(f"{base_url}/sessions/create")
session_id = response.json()['session_id']

# Make sequential predictions
for i in range(10):
    features = np.random.randn(20).tolist()
    
    response = client.post(f"{base_url}/predict/features", json={
        "features": features,
        "session_id": session_id,
        "return_uncertainty": True
    })
    
    result = response.json()
    actions = result['actions']
    uncertainties = result['uncertainties']
    
    print(f"Step {i}: Actions={actions}, Uncertainties={uncertainties}")

# Clean up
client.delete(f"{base_url}/sessions/{session_id}")
```

## Performance Considerations

- **Session Memory**: Each session maintains LSTM state (~50MB per session)
- **Session Cleanup**: Automatic cleanup after 10 concurrent sessions
- **Batch Processing**: Use sequence prediction for multiple frames
- **Device**: Optimized for MPS (Apple Silicon) and CUDA
- **Throughput**: ~100-200 FPS on Apple M1/M2

## Troubleshooting

1. **Model not found**: Ensure `best_model.pt` exists in checkpoints directory
2. **Session timeout**: Sessions automatically clean up; create new ones as needed
3. **Memory issues**: Delete unused sessions or restart server
4. **Slow inference**: Check device setting (MPS/CUDA vs CPU)
5. **NaN predictions**: Ensure features are properly normalized

## Comparison with ANN API

| Feature | ANN API | LSTM API |
|---------|---------|----------|
| Port | 8000 | 8001 |
| Input Dimension | 500 | 20 |
| Output Dimension | 10 (binary) | 4 (continuous) |
| State Management | Stateless | Stateful (sessions) |
| Memory | None | LSTM hidden state |
| Actions | Discrete binary | Continuous values |
| Uncertainty | No | Yes |
| Sequence Processing | No | Yes |

The LSTM API is designed for more sophisticated temporal modeling and provides continuous action spaces with uncertainty estimation, making it suitable for more complex game AI behaviors.
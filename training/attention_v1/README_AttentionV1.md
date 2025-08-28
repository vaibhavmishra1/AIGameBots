# AttentionV1 Model Integration for Unity

This directory contains the implementation for integrating the attention-based Transformer model (attention_v1) with Unity game agents.

## Overview

The attention_v1 model is a sophisticated neural network that uses:
- **Temporal encoding**: Processes the agent's movement history over time
- **Spatial encoding**: Considers the current state of all agents in the scene
- **Attention mechanisms**: Learns to focus on relevant agents and historical states

The model predicts movement deltas (dx, dy) that guide agent navigation.

## Files

- `api_server.py` - FastAPI server that loads and serves the attention_v1 model
- `ILInput_Attention.cs` - Unity C# script for agent control using the attention model
- `test_api.py` - Test script to verify API server functionality
- `model.py` - PyTorch model architecture definition
- `train.py` - Training script (reference)
- `processed_dataloader.py` - Data loading utilities (reference)

## Setup and Usage

### 1. Start the API Server

```bash
cd /Users/vaibhav/Desktop/AIGameBots/training/attention_v1
python3 api_server.py
```

The server will:
- Load the trained model from `training/attention_v1/checkpoints/av1_best.pt`
- Start on `http://localhost:8000`
- Provide real-time prediction histograms

### 2. Unity Integration

#### Add the Script to Your Agent

1. Copy `ILInput_Attention.cs` to your Unity project
2. Attach it to your agent GameObject
3. Ensure your agent has the required components:
   - `Agent` component
   - `AIController` component (optional but recommended)

#### Configure the Script

In the Unity Inspector:

- **API Configuration**:
  - `Api Url`: `http://localhost:8000/predict/unity`
  - `Enable API Input`: Check to enable AI control
  - `API Call Interval`: 0.05 (50ms between API calls)

- **Debug**:
  - `Enable Debug Logs`: Check for detailed logging

- **Movement**:
  - `Target Reach Threshold`: 0.25 (units)
  - `Arrive Braking Distance`: 3.0 (units)
  - `Debug Log Movement`: Check for movement debugging

### 3. Feature Extraction

The system extracts 13 features per agent:

| Feature | Description |
|---------|-------------|
| team_index | Agent's team affiliation (0.0 for now) |
| rel_pos_x | Relative X position to target (scaled) |
| rel_pos_z | Relative Z position to target (scaled) |
| rotation | Agent's rotation (0-1 normalized) |
| move_dir_x | Movement direction X (-1 to 1) |
| move_dir_y | Movement direction Y (-1 to 1) |
| look_rot_delta_x | Look rotation delta X |
| look_rot_delta_y | Look rotation delta Y |
| attack | Attack state (0 or 1) |
| shrinking_key | Game-specific shrinking mechanic |
| delta_x | Position change X |
| delta_y | Position change Y |
| delta_rot | Rotation change |

### 4. API Endpoints

#### Health Check
```http
GET /
```
Returns server status and model loading state.

#### Model Info
```http
GET /model/info
```
Returns model metadata including training epoch and hyperparameters.

#### Reset Windows
```http
POST /window/reset
```
Clears temporal and spatial feature windows.

#### Make Prediction
```http
POST /predict/unity
Content-Type: application/json

{
  "temporal": {
    "agents": [
      {
        "team_index": 0.0,
        "rel_pos_x": 0.5,
        "rel_pos_z": -0.3,
        "rotation": 0.25,
        "move_dir_x": 0.1,
        "move_dir_y": 0.0,
        "look_rot_delta_x": 0.0,
        "look_rot_delta_y": 0.0,
        "attack": 0.0,
        "shrinking_key": 0.0,
        "delta_x": 0.0,
        "delta_y": 0.0,
        "delta_rot": 0.0
      }
    ]
  },
  "spatial": {
    "agents": [
      {
        "team_index": 0.0,
        "rel_pos_x": 0.5,
        "rel_pos_z": -0.3,
        "rotation": 0.25,
        "move_dir_x": 0.1,
        "move_dir_y": 0.0,
        "look_rot_delta_x": 0.0,
        "look_rot_delta_y": 0.0,
        "attack": 0.0,
        "shrinking_key": 0.0,
        "delta_x": 0.0,
        "delta_y": 0.0,
        "delta_rot": 0.0
      }
    ]
  }
}
```

**Response:**
```json
{
  "predictions": [0.016, 0.021]
}
```

### 5. Testing

Run the test suite to verify everything works:

```bash
cd /Users/vaibhav/Desktop/AIGameBots/training/attention_v1
python3 test_api.py
```

This will test:
- Health endpoint
- Model info endpoint
- Window reset
- Prediction with dummy data

## Architecture Details

### Model Structure

```python
TemporalSpatialTransformer(
  feature_dim=13,
  d_model=512,
  temp_layers=3,
  spat_layers=3,
  dropout=0.0,
  max_time=64
)
```

### Data Flow

1. **Unity** collects agent features every `featureUpdateInterval` (100ms)
2. **Unity** maintains temporal history (up to 64 timesteps)
3. **Unity** captures spatial snapshot of all agents
4. **Unity** sends data to API every `apiCallInterval` (50ms)
5. **API Server** processes features and normalizes them
6. **Model** makes prediction using attention mechanisms
7. **API Server** returns (delta_x, delta_y) predictions
8. **Unity** converts deltas to world-space movement targets
9. **Unity** drives agent movement toward targets

### Feature Processing

- **Temporal Window**: Maintains 64 timesteps of agent history
- **Spatial Snapshot**: Current state of all visible agents
- **Normalization**: Features are normalized using training statistics
- **Attention Pooling**: Model focuses on relevant spatial agents

## Troubleshooting

### Server Won't Start
- Ensure all dependencies are installed: `pip install fastapi uvicorn torch requests`
- Check that model checkpoint exists at the expected path
- Verify MPS/CUDA availability

### Unity Not Connecting
- Confirm server is running on port 8000
- Check firewall settings
- Verify API URL in Unity script

### Poor Agent Behavior
- Adjust `apiCallInterval` (faster = more responsive but more network traffic)
- Tune `targetReachThreshold` and `arriveBrakingDistance`
- Check feature extraction logic matches training data format

### Performance Issues
- Reduce `maxTemporalHistory` if memory constrained
- Increase `apiCallInterval` to reduce server load
- Monitor server logs for bottlenecks

## Dependencies

### Python Requirements
- torch>=1.9.0
- fastapi>=0.68.0
- uvicorn>=0.15.0
- requests>=2.25.0
- numpy>=1.19.0

### Unity Requirements
- Unity 2019.4+ (with .NET 4.x)
- TPSBR framework (Agent, AIController components)

## Model Training

The model was trained on processed game logs with:
- **Dataset**: 100,000+ samples with magnitude filtering
- **Epochs**: 50
- **Batch Size**: 1024
- **Learning Rate**: 1e-4 with AdamW optimizer
- **Best Validation Loss**: 0.001369

Training data includes low-motion filtering to focus on meaningful actions.

## Future Improvements

1. **Enhanced Spatial Processing**: Include line-of-sight and occlusion features
2. **Multi-Head Predictions**: Add look rotation and attack predictions
3. **Real-time Adaptation**: Implement online learning capabilities
4. **Multi-Agent Coordination**: Support for team-based behaviors
5. **Performance Optimization**: Quantization and model compression

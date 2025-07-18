# AI Game Agents API Server

This FastAPI server provides a REST API for serving trained neural network models for game action prediction.

## Features

- **Model Serving**: Serve trained models for both regression and classification tasks
- **Batch Predictions**: Make predictions for multiple inputs at once
- **Model Management**: Load/unload models dynamically
- **Model Evaluation**: Evaluate models on test data
- **Health Monitoring**: Check API status and loaded models
- **Auto-documentation**: Interactive API documentation with Swagger UI

## Supported Tasks

### Regression Tasks
- `moveDirection_x`: Predict horizontal movement direction
- `moveDirection_y`: Predict vertical movement direction  
- `lookRotationDelta_x`: Predict horizontal look rotation
- `lookRotationDelta_y`: Predict vertical look rotation

### Classification Tasks
- `Attack`: Predict attack action (binary)
- `Reload`: Predict reload action (binary)
- `thrust`: Predict thrust action (binary)
- `crouch`: Predict crouch action (binary)
- `sprint`: Predict sprint action (binary)
- `slide`: Predict slide action (binary)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have trained models in the models directory:
```
training/ann/models/
├── moveDirection_x_best_model.pth
├── moveDirection_y_best_model.pth
├── lookRotationDelta_x_best_model.pth
├── lookRotationDelta_y_best_model.pth
├── Attack_best_model.pth
├── Reload_best_model.pth
├── thrust_best_model.pth
├── crouch_best_model.pth
├── sprint_best_model.pth
└── slide_best_model.pth
```

## Running the Server

```bash
python api_server.py
```

The server will start on `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative API docs**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/

## API Endpoints

### Health Check
```http
GET /
```
Returns server status and loaded models.

### List Models
```http
GET /models
```
Returns information about all available models.

### Get Tasks
```http
GET /tasks
```
Returns information about available tasks.

### Unity Game Agent Prediction
```http
POST /predict
```
Make predictions for Unity game agent actions. This endpoint auto-detects whether the request is in Unity format or single prediction format.

**Unity Request Body:**
```json
{
  "states": [
    {
      "agentPosition": {"x": 10.5, "y": 0.5, "z": -5.2},
      "agentRotation": {"x": 0, "y": 180, "z": 0},
      "agentForward": {"x": 0, "y": 0, "z": 1},
      "health": 100,
      "weapon": 0,
      "targetPosition": {"x": 15.3, "y": 0.5, "z": -2.1},
      "targetRotation": {"x": 0, "y": 90, "z": 0},
      "targetForward": {"x": 1, "y": 0, "z": 0},
      "directionToTarget": {"x": 0.84, "y": 0, "z": 0.54},
      "cross": {"x": 0, "y": -0.54, "z": 0},
      "distance": 6.1,
      "dotProduct": 0.54,
      "islos": true
    }
    // ... 19 more states (total of 20 required)
  ]
}
```

**Unity Response:**
```json
{
  "predictions": [0, 1, 0, 0, 1, 0, 1]  // 7 action predictions
}
```

**Action Mapping:**
- Index 0: Move Forward (1) or not (0)
- Index 1: Move Backward (1) or not (0)
- Index 2: Move Left (1) or not (0)
- Index 3: Move Right (1) or not (0)
- Index 4: Turn Left (1) or not (0)
- Index 5: Turn Right (1) or not (0)
- Index 6: Attack/Shoot (1) or not (0)

### Single Prediction (Original Format)
```http
POST /predict
```
Make a prediction for a single input.

**Request Body:**
```json
{
  "features": [0.1, 0.2, 0.3, ...],  // 500-dimensional feature vector
  "task": "moveDirection_x"
}
```

**Response:**
```json
{
  "task": "moveDirection_x",
  "prediction": 0.75,
  "probability": null,  // Only for classification tasks
  "confidence": null    // Only for classification tasks
}
```

### Batch Prediction
```http
POST /predict/batch
```
Make predictions for multiple inputs.

**Request Body:**
```json
{
  "features_list": [
    [0.1, 0.2, 0.3, ...],  // 500-dimensional feature vector
    [0.4, 0.5, 0.6, ...],  // 500-dimensional feature vector
    ...
  ],
  "task": "Attack"
}
```

**Response:**
```json
{
  "task": "Attack",
  "predictions": [0.8, 0.2, 0.9],
  "probabilities": [0.8, 0.2, 0.9]  // Only for classification tasks
}
```

### Model Information
```http
GET /models/{task}/info
```
Get detailed information about a specific model.

### Load Model
```http
POST /models/{task}/load?model_type=best
```
Load a specific model into memory.

### Unload Model
```http
DELETE /models/{task}/unload
```
Unload a specific model from memory.

### Evaluate Model
```http
POST /evaluate/{task}?num_samples=1000
```
Evaluate a loaded model on test data.

## Unity Integration

The API server is designed to work with Unity game agents. The Unity integration expects:

1. **Exactly 20 states**: The Unity agent maintains a history of the last 20 game states
2. **State structure**: Each state contains agent position, rotation, health, target information, and spatial relationships
3. **Feature conversion**: The 20 states are automatically converted into a 500-dimensional feature vector (20 states × 25 features per state)
4. **Action predictions**: The API returns 7 binary predictions for movement and combat actions

### Unity C# Example

```csharp
// Create the payload
var payload = new StatePayload { states = stateHistory };
var json = JsonUtility.ToJson(payload);

// Send to API
var content = new StringContent(json, Encoding.UTF8, "application/json");
var response = await httpClient.PostAsync("http://localhost:8000/predict", content);

// Parse response
var responseContent = await response.Content.ReadAsStringAsync();
var apiResponse = JsonUtility.FromJson<APIResponse>(responseContent);
int[] predictions = apiResponse.predictions;
```

### Testing Unity Integration

A dedicated test client is provided for Unity integration:

```bash
python test_unity_client.py
```

This will:
1. Create sample Unity states
2. Test the prediction endpoint
3. Verify response format
4. Test error handling

## Example Usage with Python

```python
import requests
import numpy as np

# Generate sample features (replace with real game features)
features = np.random.randn(500).tolist()

# Make a prediction
response = requests.post("http://localhost:8000/predict", json={
    "features": features,
    "task": "moveDirection_x"
})

prediction = response.json()
print(f"Prediction: {prediction['prediction']}")

# Make batch predictions
batch_features = [np.random.randn(500).tolist() for _ in range(5)]
response = requests.post("http://localhost:8000/predict/batch", json={
    "features_list": batch_features,
    "task": "Attack"
})

predictions = response.json()
print(f"Batch predictions: {predictions['predictions']}")
```

## Example Usage with curl

```bash
# Health check
curl http://localhost:8000/

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, 0.3, ...],
    "task": "moveDirection_x"
  }'

# List models
curl http://localhost:8000/models

# Evaluate a model
curl -X POST "http://localhost:8000/evaluate/moveDirection_x?num_samples=100"
```

## Model File Format

Models are saved as PyTorch checkpoints with the following structure:

```python
{
    'epoch': int,                    # Training epoch
    'model_state_dict': dict,        # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'loss': float,                   # Training loss
    'task_type': str,                # 'classification' or 'regression'
    'model_config': dict             # Model configuration
}
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid input)
- `404`: Model not found or not loaded
- `500`: Internal server error

Error responses include a `detail` field with error information.

## Performance Considerations

- Models are loaded into memory on startup for faster inference
- Batch predictions are more efficient than multiple single predictions
- Use GPU acceleration if available (MPS for Apple Silicon, CUDA for NVIDIA)
- Consider model quantization for production deployment

## Security Notes

- The API currently allows CORS from all origins (configure for production)
- Input validation is performed on all endpoints
- Consider adding authentication for production use

## Troubleshooting

1. **Model not found**: Ensure trained models exist in the models directory
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Invalid features**: Ensure input features are exactly 500-dimensional
4. **Model loading errors**: Check model file integrity and PyTorch version compatibility 
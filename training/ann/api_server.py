from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional, Union
import uvicorn

# Import our model and dataset classes
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_model, get_loss_function
from dataset import NPYDataset

app = FastAPI(
    title="AI Game Agents API",
    description="API for serving trained neural network models for game action prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MODELS_DIR = "/Users/vaibhavmishra/Desktop/btx-game-aicode/AIGameAgents/training/ann/models"
LOADED_MODELS = {}

# Task definitions
REGRESSION_TASKS = ["moveDirection_x", "moveDirection_y", "lookRotationDelta_x", "lookRotationDelta_y"]
CLASSIFICATION_TASKS = ["Attack", "Reload", "thrust", "crouch", "sprint", "slide"]
ALL_TASKS = REGRESSION_TASKS + CLASSIFICATION_TASKS

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    features: List[float]
    task: str

class PredictionResponse(BaseModel):
    task: str
    prediction: float
    probability: Optional[float] = None
    confidence: Optional[float] = None

class ModelInfo(BaseModel):
    task: str
    task_type: str
    model_path: str
    loss: float
    epoch: int
    is_loaded: bool

class BatchPredictionRequest(BaseModel):
    features_list: List[List[float]]
    task: str

class BatchPredictionResponse(BaseModel):
    task: str
    predictions: List[float]
    probabilities: Optional[List[float]] = None

class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]
    available_tasks: List[str]
    device: str

class UnityState(BaseModel):
    """Unity state structure matching the C# State class."""
    agentPosition: Dict[str, float]  # x, y, z
    agentRotation: Dict[str, float]  # x, y, z
    agentForward: Dict[str, float]   # x, y, z
    health: float
    weapon: float
    targetPosition: Dict[str, float]  # x, y, z
    targetRotation: Dict[str, float]  # x, y, z
    targetForward: Dict[str, float]   # x, y, z
    directionToTarget: Dict[str, float]  # x, y, z
    cross: Dict[str, float]  # x, y, z
    distance: float
    dotProduct: float
    islos: bool

class UnityStatePayload(BaseModel):
    """Unity payload structure."""
    states: List[UnityState]

class UnityAPIResponse(BaseModel):
    """Unity response structure."""
    predictions: List[float]

def get_task_type(task: str) -> str:
    """Determine if a task is classification or regression."""
    if task in REGRESSION_TASKS:
        return "regression"
    elif task in CLASSIFICATION_TASKS:
        return "classification"
    else:
        raise ValueError(f"Unknown task: {task}")

def load_model(task: str, model_type: str = 'best') -> tuple[torch.nn.Module, Any]:
    """Load a trained model for a specific task."""
    model_path = os.path.join(MODELS_DIR, f"{task}_{model_type}_model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Create model with same configuration
    model_config = checkpoint['model_config']
    model = create_model(
        'standard',
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        dropout_rate=model_config['dropout_rate'],
        task_type=model_config['task_type']
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    return model, checkpoint

def preprocess_features(features: List[float]) -> torch.Tensor:
    """Preprocess input features."""
    if len(features) != 500:
        raise ValueError(f"Expected 500 features, got {len(features)}")
    
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return features_tensor

def convert_unity_state_to_features(state: UnityState) -> List[float]:
    """Convert a Unity state to a 25-dimensional feature vector."""
    features = []
    
    # Agent features (normalized as in Unity)
    features.append(state.agentPosition['x'])  # Already divided by 1000 in Unity
    features.append(state.agentPosition['y'])
    features.append(state.agentPosition['z'])
    features.append(state.agentRotation['y'])  # Already divided by 360 in Unity
    features.append(state.agentForward['x'])
    features.append(state.agentForward['y'])
    features.append(state.agentForward['z'])
    features.append(state.health)  # Already divided by 100 in Unity
    features.append(state.weapon)
    features.append(1.0 if state.islos else 0.0)
    
    # Target features
    features.append(state.targetPosition['x'])
    features.append(state.targetPosition['y'])
    features.append(state.targetPosition['z'])
    features.append(state.targetRotation['y'])
    features.append(state.targetForward['x'])
    features.append(state.targetForward['y'])
    features.append(state.targetForward['z'])
    
    # Relationship features
    features.append(state.directionToTarget['x'])
    features.append(state.directionToTarget['y'])
    features.append(state.directionToTarget['z'])
    features.append(state.cross['x'])
    features.append(state.cross['y'])
    features.append(state.cross['z'])
    features.append(state.distance)  # Already divided by 1000 in Unity
    features.append(state.dotProduct)
    
    return features

def convert_unity_states_to_features(states: List[UnityState]) -> List[float]:
    """Convert 20 Unity states to a 500-dimensional feature vector."""
    all_features = []
    
    # Convert each state to features and concatenate
    for state in states:
        state_features = convert_unity_state_to_features(state)
        all_features.extend(state_features)
    
    # Should have exactly 500 features (20 states * 25 features)
    if len(all_features) != 500:
        raise ValueError(f"Expected 500 features, got {len(all_features)}")
    
    return all_features

@app.on_event("startup")
async def startup_event():
    """Load all available models on startup."""
    print("Loading available models...")
    for task in ALL_TASKS:
        try:
            model, checkpoint = load_model(task, 'best')
            LOADED_MODELS[task] = {
                'model': model,
                'checkpoint': checkpoint,
                'task_type': get_task_type(task)
            }
            print(f"Loaded model for {task}")
        except Exception as e:
            print(f"Could not load model for {task}: {e}")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        loaded_models=list(LOADED_MODELS.keys()),
        available_tasks=ALL_TASKS,
        device=str(DEVICE)
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models and their status."""
    models_info = []
    
    for task in ALL_TASKS:
        model_path = os.path.join(MODELS_DIR, f"{task}_best_model.pth")
        is_loaded = task in LOADED_MODELS
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=DEVICE)
                models_info.append(ModelInfo(
                    task=task,
                    task_type=get_task_type(task),
                    model_path=model_path,
                    loss=checkpoint['loss'],
                    epoch=checkpoint['epoch'],
                    is_loaded=is_loaded
                ))
            except Exception as e:
                models_info.append(ModelInfo(
                    task=task,
                    task_type=get_task_type(task),
                    model_path=model_path,
                    loss=0.0,
                    epoch=0,
                    is_loaded=False
                ))
    
    return models_info

@app.post("/predict")
async def predict(request: dict):
    """Make a prediction - handles both single predictions and Unity format."""
    # Check if this is a Unity request by looking for 'states' field
    if 'states' in request:
        # Convert to UnityStatePayload
        unity_request = UnityStatePayload(**request)
        return await predict_unity(unity_request)
    else:
        # Convert to PredictionRequest
        single_request = PredictionRequest(**request)
        return await predict_single(single_request)

@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Make a prediction for a single input."""
    if request.task not in LOADED_MODELS:
        raise HTTPException(status_code=404, detail=f"Model for task '{request.task}' not loaded")
    
    try:
        # Preprocess features
        features_tensor = preprocess_features(request.features)
        
        # Get model and make prediction
        model_info = LOADED_MODELS[request.task]
        model = model_info['model']
        task_type = model_info['task_type']
        
        with torch.no_grad():
            prediction = model(features_tensor)
            pred_value = prediction.item()
        
        # Prepare response
        response = PredictionResponse(
            task=request.task,
            prediction=pred_value
        )
        
        # Add probability for classification tasks
        if task_type == 'classification':
            response.probability = pred_value
            response.confidence = max(pred_value, 1 - pred_value)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions for multiple inputs."""
    if request.task not in LOADED_MODELS:
        raise HTTPException(status_code=404, detail=f"Model for task '{request.task}' not loaded")
    
    try:
        # Validate all feature vectors
        for i, features in enumerate(request.features_list):
            if len(features) != 500:
                raise ValueError(f"Features at index {i} has {len(features)} values, expected 500")
        
        # Preprocess features
        features_tensor = torch.tensor(request.features_list, dtype=torch.float32).to(DEVICE)
        
        # Get model and make predictions
        model_info = LOADED_MODELS[request.task]
        model = model_info['model']
        task_type = model_info['task_type']
        
        with torch.no_grad():
            predictions = model(features_tensor)
            pred_values = predictions.squeeze().tolist()
        
        # Prepare response
        response = BatchPredictionResponse(
            task=request.task,
            predictions=pred_values
        )
        
        # Add probabilities for classification tasks
        if task_type == 'classification':
            response.probabilities = pred_values
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/models/{task}/load")
async def load_model_endpoint(task: str, model_type: str = 'best'):
    """Load a specific model."""
    if task not in ALL_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")
    
    try:
        model, checkpoint = load_model(task, model_type)
        LOADED_MODELS[task] = {
            'model': model,
            'checkpoint': checkpoint,
            'task_type': get_task_type(task)
        }
        return {"message": f"Model for task '{task}' loaded successfully", "loss": checkpoint['loss']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.delete("/models/{task}/unload")
async def unload_model(task: str):
    """Unload a specific model."""
    if task in LOADED_MODELS:
        del LOADED_MODELS[task]
        return {"message": f"Model for task '{task}' unloaded successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model for task '{task}' not loaded")

@app.get("/models/{task}/info")
async def get_model_info(task: str):
    """Get detailed information about a specific model."""
    if task not in ALL_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")
    
    model_path = os.path.join(MODELS_DIR, f"{task}_best_model.pth")
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found for task '{task}'")
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        return {
            "task": task,
            "task_type": get_task_type(task),
            "model_path": model_path,
            "loss": checkpoint['loss'],
            "epoch": checkpoint['epoch'],
            "is_loaded": task in LOADED_MODELS,
            "model_config": checkpoint['model_config']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read model info: {str(e)}")

@app.get("/tasks")
async def get_tasks():
    """Get information about all available tasks."""
    return {
        "regression_tasks": REGRESSION_TASKS,
        "classification_tasks": CLASSIFICATION_TASKS,
        "all_tasks": ALL_TASKS
    }

@app.post("/evaluate/{task}")
async def evaluate_model(task: str, num_samples: int = 1000):
    """Evaluate a loaded model on a subset of the dataset."""
    if task not in LOADED_MODELS:
        raise HTTPException(status_code=404, detail=f"Model for task '{task}' not loaded")
    
    try:
        # Create dataset
        dataset = NPYDataset(
            data_path="/Users/vaibhavmishra/Desktop/btx-game-aicode/clash_squad_partitioned_chunked/", 
            task=task
        )
        
        # Get model and loss function
        model_info = LOADED_MODELS[task]
        model = model_info['model']
        task_type = model_info['task_type']
        criterion = get_loss_function(task_type)
        
        # Evaluate on subset
        model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                features, target = dataset[i]
                features = features.unsqueeze(0).to(DEVICE)
                target = target.unsqueeze(0).to(DEVICE)
                
                prediction = model(features)
                loss = criterion(prediction, target)
                total_loss += loss.item()
                
                predictions.append(prediction.item())
                targets.append(target.item())
        
        avg_loss = total_loss / min(num_samples, len(dataset))
        
        # Calculate additional metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        if task_type == 'regression':
            mae = np.mean(np.abs(predictions - targets))
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            
            return {
                "task": task,
                "task_type": task_type,
                "num_samples": min(num_samples, len(dataset)),
                "average_loss": avg_loss,
                "mae": mae,
                "mse": mse,
                "rmse": rmse
            }
        else:
            # For classification, calculate accuracy
            binary_predictions = (predictions >= 0.5).astype(int)
            accuracy = np.mean(binary_predictions == targets)
            
            return {
                "task": task,
                "task_type": task_type,
                "num_samples": min(num_samples, len(dataset)),
                "average_loss": avg_loss,
                "accuracy": accuracy
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/predict/unity", response_model=UnityAPIResponse)
async def predict_unity(request: UnityStatePayload):
    """Make predictions for Unity game agent actions."""
    try:
        # Validate we have exactly 20 states
        if len(request.states) != 20:
            raise ValueError(f"Expected 20 states, got {len(request.states)}")
        
        # Convert Unity states to features
        features = convert_unity_states_to_features(request.states)
        features_tensor = preprocess_features(features)
        
        # Initialize predictions array (4 float values)
        predictions = [0.0] * 7
        
        # Log loaded models for debugging
        print(f"Loaded models: {list(LOADED_MODELS.keys())}")
        
        # Make predictions for relevant tasks
        # Movement predictions (Actions 0-3: forward, backward, left, right)
        if 'moveDirection_y' in LOADED_MODELS:
            model_info = LOADED_MODELS['moveDirection_y']
            model = model_info['model']
            with torch.no_grad():
                move_y = model(features_tensor).item()
                print(f"moveDirection_y prediction: {move_y}")
                # Map to forward/backward
                predictions[0] = move_y
        
        if 'moveDirection_x' in LOADED_MODELS:
            model_info = LOADED_MODELS['moveDirection_x']
            model = model_info['model']
            with torch.no_grad():
                move_x = model(features_tensor).item()
                print(f"moveDirection_x prediction: {move_x}")
                # Map to left/right
                predictions[1] = move_x
        
        # Turning predictions (Actions 4-5: turn left, turn right)
        if 'lookRotationDelta_x' in LOADED_MODELS:
            model_info = LOADED_MODELS['lookRotationDelta_x']
            model = model_info['model']
            with torch.no_grad():
                look_x = model(features_tensor).item()
                predictions[2] = look_x
        # Attack prediction (Action 6: shoot)
        # if 'Attack' in LOADED_MODELS:
        #     model_info = LOADED_MODELS['Attack']
        #     model = model_info['model']
        #     with torch.no_grad():
        #         attack_prob = model(features_tensor).item()
        #         print(f"Attack probability: {attack_prob}")
        #         # Use threshold for binary decision
        #         predictions[3] = attack_prob
        
        print(f"Final predictions: {predictions}")
        return UnityAPIResponse(predictions=predictions)
        
    except Exception as e:
        print(f"Unity prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unity prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
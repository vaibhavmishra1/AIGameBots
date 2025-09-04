from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import os
from typing import List, Optional, Dict, Any
from collections import deque
import threading
import uvicorn
import h5py
import argparse
from datetime import datetime
import json
from fastapi.encoders import jsonable_encoder
import json
from fastapi.encoders import jsonable_encoder
import time
# Ensure local imports work
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_temporal_and_spatial import TemporalSpatialTransformer

def log_debug(msg: str):
    """Debug logging with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [DEBUG] {msg}")

def log_error(msg: str):
    """Error logging with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [ERROR] {msg}")

def log_info(msg: str):
    """Info logging with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [INFO] {msg}")

app = FastAPI(
    title="AI Game Agents API (attention_v1)",
    description="Serve an attention-based Transformer model for Unity agent movement prediction",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def get_shrinking_area_centers():
    shrinking_area_centers = {}
    shrinking_area_centers[1] = {"x" : -365.63, "z" : -139.57}
    shrinking_area_centers[2] = {"x" : -368.84, "z" : 7.72}
    shrinking_area_centers[3] = {"x" : 94.97, "z" : 152.24}
    shrinking_area_centers[4] = {"x" : 363.16, "z" : -107.33}
    shrinking_area_centers[5] = {"x" : -132.66, "z" : -183.18}
    shrinking_area_centers[6] = {"x" : 198.90, "z" : -212.00}
    shrinking_area_centers[7] = {"x" : 196.30, "z" : 532.90}
    return shrinking_area_centers

# Globals
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "/Users/vaibhav/Desktop/AIGameBots/experiments/hawk/training/attention_v1/checkpoints/hawk_temporal_and_spatial_best.pt"
DATASET_PATH = "/Users/vaibhav/Desktop/processed_game_logs_attention_1.h5"

SINGLE_MODEL: Optional[torch.nn.Module] = None
MODEL_META: Dict[str, Any] = {}
shrinking_area_centers = get_shrinking_area_centers()
# Feature normalization stats (to be loaded from dataset)
FEATURE_STATS = {}

PRED1_HISTORY: "deque[float]" = deque(maxlen=1000)
PRED2_HISTORY: "deque[float]" = deque(maxlen=1000)
PRED_HISTORY_LOCK = threading.Lock()


# Schemas
class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


class AgentFeatures(BaseModel):
    """Features for a single agent"""
    agent_id: int
    game_time: float
    team_index: float
    pos_x: float
    pos_z: float
    rotation: float
    move_dir_x: float
    move_dir_y: float
    look_rot_delta_x: float
    look_rot_delta_y: float
    attack: float


class AgentTimeStep(BaseModel):
    """A single timestep containing a list of agents"""
    agents: List[AgentFeatures]

class UnityStatePayload(BaseModel):
    """Unity sends complete temporal history (all agents over time)"""
    temporal_history: List[AgentTimeStep]  # List of timesteps, each containing a list of agents


class UnityDualOutputResponse(BaseModel):
    predictions: List[float]  # length 10: [dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5]


def load_feature_stats():
    """Load feature normalization statistics from the dataset"""
    global FEATURE_STATS
    try:
        with h5py.File(DATASET_PATH, 'r') as f:
            if 'feature_stats' in f:
                stats_group = f['feature_stats']
                FEATURE_STATS = {
                    'temporal_mean': np.array(stats_group['temporal_mean']),
                    'temporal_std': np.array(stats_group['temporal_std']),
                    'spatial_mean': np.array(stats_group['spatial_mean']),
                    'spatial_std': np.array(stats_group['spatial_std']),
                }
                print(f"Loaded feature stats from {DATASET_PATH}")
            else:
                print("No feature stats found in dataset, using identity normalization")
                # Identity normalization (no scaling)
                FEATURE_STATS = {
                    'temporal_mean': np.zeros(13),
                    'temporal_std': np.ones(13),
                    'spatial_mean': np.zeros(13),
                    'spatial_std': np.ones(13),
                }
    except Exception as e:
        print(f"Error loading feature stats: {e}")
        # Fallback to identity normalization
        FEATURE_STATS = {
            'temporal_mean': np.zeros(13),
            'temporal_std': np.ones(13),
            'spatial_mean': np.zeros(13),
            'spatial_std': np.ones(13),
        }


def load_single_model() -> None:
    global SINGLE_MODEL, MODEL_META

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Load model with parameters from checkpoint
    args = checkpoint.get("args", {})
    model = TemporalSpatialTransformer(
        feature_dim=6,
        d_model=512,
        temp_layers=4,
        temp_heads=8,
        spat_layers=4,
        spat_heads=8,
        dropout=0,
        max_time=20,
        max_agents=10
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    SINGLE_MODEL = model
    MODEL_META = {
        "epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val"),
        "hyperparameters": args,
        "model_path": MODEL_PATH,
    }




def calculate_expected_shrinking_area_center( data):
    """
    Calculate the expected shrinking area center based on agent positions.

    Args:
        data: numpy array or list-like, shape (timesteps, agents, features)

    Returns:
        tuple: (x, z, key) of the closest shrinking area center. If unavailable, returns (0.0, 0.0, 0)
    """
    total_x, total_z, count = 0.0, 0.0, 0

    for timestep in data:
        for agent_data in timestep:
            # agent_data[4] = pos_x, agent_data[6] = pos_z
            # Guard against short feature vectors
            if agent_data is None:
                continue
            
            pos_x = agent_data[3]
            pos_z = agent_data[4]
            if pos_x != 0.0 and pos_z != 0.0:
                total_x += pos_x
                total_z += pos_z
                count += 1
                

    if count == 0:
        return (0.0, 0.0, 0)

    avg_x = total_x / count
    avg_z = total_z / count

    min_dist = float('inf')
    closest_key = None
    for key, center in shrinking_area_centers.items():
        dx = avg_x - center['x']
        dz = avg_z - center['z']
        dist = dx * dx + dz * dz
        if dist < min_dist:
            min_dist = dist
            closest_key = key

    if closest_key is not None:
        center = shrinking_area_centers[closest_key]
        return (center['x'], center['z'], closest_key)
    else:
        return (0.0, 0.0, 0)

def preprocess_data(data: np.ndarray) -> np.ndarray:
    """Preprocess the data"""
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    timesteps_total, num_agents, feature_dim = data.shape
    if timesteps_total < 1 or num_agents < 1:
        empty_temporal = np.zeros((0, 6), dtype=np.float32)
        empty_spatial = np.zeros((max(num_agents, 0), 6), dtype=np.float32)
        return empty_temporal, empty_spatial
    
    (zonecenterx, zonecenterz, shr_key) = calculate_expected_shrinking_area_center(data)
    print("shrinking area center: ", zonecenterx, zonecenterz, shr_key)
    shrinking_key_norm = float(shr_key) / 7.0 if shr_key != 0 else 0.0
    feature_count = 6
    temporal_limit = min(16, timesteps_total) 
    new_temporal_data = np.zeros((temporal_limit, feature_count), dtype=np.float32)
    new_spatial_data = np.zeros((num_agents, feature_count), dtype=np.float32)
    spatial_time_index = min(16, timesteps_total - 1) 
    for i in range(timesteps_total):
        for j in range(num_agents):
            feature = data[i][j]
            if np.all(feature == 0):
                    # If last timestep and agent 0 is all zeros, still keep actions as zeros
                continue
            prev_feature = data[i][j] if i == 0 else data[i - 1][j]

            team_index = feature[2]
            team_index = (team_index) / 2

            posx = feature[3]
            posz = feature[4]

            deltax = posx - prev_feature[3]
            deltay = posz - prev_feature[4]

            rel_pos_x = (posx - zonecenterx) 
            rel_pos_x = rel_pos_x if abs(rel_pos_x) < 150 else 150 * np.sign(rel_pos_x)
            rel_pos_x = rel_pos_x / 150

            rel_pos_z = (posz - zonecenterz) 
            rel_pos_z = rel_pos_z if abs(rel_pos_z) < 100 else 100 * np.sign(rel_pos_z)
            rel_pos_z = rel_pos_z / 100

            shr_key = shrinking_key_norm

            deltax = deltax if abs(deltax) < 1 else 1 * np.sign(deltax)
            
            deltay = deltay if abs(deltay) < 1 else 1 * np.sign(deltay)
            
            
            feat_vec = np.array([
                team_index, #0
                rel_pos_x,
                rel_pos_z,
                shr_key,
                deltax,
                deltay,
            ], dtype=np.float32)

            if j == 0 and i < temporal_limit:
                new_temporal_data[i] = feat_vec
            if i == spatial_time_index  and j < 10:
                new_spatial_data[j] = feat_vec
    return new_temporal_data, new_spatial_data



@app.on_event("startup")
async def on_startup() -> None:
    log_info("Loading feature normalization statistics...")
    load_feature_stats()
    
    log_info("Loading AI model...")
    load_single_model()

    if SINGLE_MODEL is not None:
        log_info("✅ Model and feature stats loaded successfully!")
        log_info(f"📊 Feature dimensions: 9 (updated from old 13)")
    else:
        log_error("❌ Failed to load model!")


@app.get("/", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy", device=str(DEVICE), model_loaded=SINGLE_MODEL is not None)


@app.get("/model/info")
async def model_info() -> Dict[str, Any]:
    if SINGLE_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return MODEL_META



@app.post("/predict/unity", response_model=UnityDualOutputResponse)
async def predict_unity(request: Request, payload: UnityStatePayload) -> UnityDualOutputResponse:

    if SINGLE_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert payload to raw dict for debugging
        payload_dict = jsonable_encoder(payload)
        log_debug(f"Converting payload to numpy array...")

        # Extract the temporal history
        temporal_history = payload_dict.get('temporal_history', [])
        if not temporal_history:
            raise HTTPException(status_code=422, detail="temporal_history is empty")

        # Get dimensions
        num_timesteps = len(temporal_history)
        num_agents = max(len(timestep.get('agents', [])) for timestep in temporal_history) if temporal_history else 0
        feature_dim = 11  # [agent_id, game_time, team_index, pos_x, pos_z, rotation, move_dir_x, move_dir_y, look_rot_delta_x, look_rot_delta_y, attack]

        log_debug(f"Dimensions: timesteps={num_timesteps}, agents={num_agents}, features={feature_dim}")

        # Initialize numpy array with zeros
        features_array = np.zeros((num_timesteps, num_agents, feature_dim))

        # Fill the array
        for t, timestep in enumerate(temporal_history):
            agents = timestep.get('agents', [])
            for a, agent in enumerate(agents):
                if agent:
                    features = [
                        agent.get('agent_id', 0), #0
                        agent.get('game_time', 0), #1
                        agent.get('team_index', 0), #2
                        agent.get('pos_x', 0), #3
                        agent.get('pos_z', 0), #4
                        agent.get('rotation', 0), #5
                        agent.get('move_dir_x', 0), #6
                        agent.get('move_dir_y', 0), #7
                        agent.get('look_rot_delta_x', 0), #8
                        agent.get('look_rot_delta_y', 0), #9
                        agent.get('attack', 0) #10
                    ]
                    features_array[t, a] = features

        log_debug(f"Converted to numpy array with shape: {features_array.shape}")
        log_debug(f"Sample values from first timestep: {features_array[0, 0]}")
        #(zonecenterx, zonecenterz, shr_key) = calculate_expected_shrinking_area_center(features_array)
        temporal_data, spatial_data = preprocess_data(features_array)
        print(temporal_data.shape, spatial_data.shape)
        print(temporal_data[0], spatial_data[0])
        temporal_data = torch.tensor(temporal_data, dtype=torch.float32, device=DEVICE)
        spatial_data = torch.tensor(spatial_data, dtype=torch.float32, device=DEVICE)
        temporal_tensor = temporal_data.unsqueeze(0)  # (1, T, 9)
        spatial_tensor = spatial_data.unsqueeze(0)    # (1, A, 9)

        # Forward pass
        with torch.no_grad():
            output = SINGLE_MODEL(temporal_tensor, spatial_tensor)  # (1, 10) - 5 pairs of (dx, dy)
            preds = output.squeeze(0).cpu().tolist()  # [dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5]

        # Return all 10 predictions (5 pairs of dx, dy) for web environment
        predictions = preds  # All 10 values: [dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5]
        print("predictions: ", predictions)
        return UnityDualOutputResponse(predictions=predictions)
    except Exception as e:
        log_error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Error processing request: {str(e)}")


    except Exception as e:
        log_error(f"Prediction error: {str(e)}")
        log_error(f"Exception type: {type(e).__name__}")
        import traceback
        log_error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the AI Agent API server')
    parser.add_argument('--port', type=int, default=8001, help='Port to run the server on')
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)

from fastapi import FastAPI, HTTPException
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

# Ensure local imports work
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import build_model

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

# Globals
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "/Users/vaibhav/Desktop/AIGameBots/training/attention_v1/training/attention_v1/checkpoints/av1_best.pt"
DATASET_PATH = "/Users/vaibhav/Desktop/processed_game_logs_attention_1.h5"

SINGLE_MODEL: Optional[torch.nn.Module] = None
MODEL_META: Dict[str, Any] = {}

# Feature normalization stats (to be loaded from dataset)
FEATURE_STATS = {}

# Sliding windows for temporal and spatial features
TEMPORAL_WINDOW: "deque[List[float]]" = deque(maxlen=64)  # 64 timesteps max
SPATIAL_AGENTS: List[List[float]] = []  # Current spatial snapshot

PRED1_HISTORY: "deque[float]" = deque(maxlen=1000)
PRED2_HISTORY: "deque[float]" = deque(maxlen=1000)
PRED_HISTORY_LOCK = threading.Lock()


# Schemas
class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


class AgentFeatures(BaseModel):
    """Features for a single agent (used in spatial snapshot)"""
    team_index: float
    rel_pos_x: float
    rel_pos_z: float
    rotation: float
    move_dir_x: float
    move_dir_y: float
    look_rot_delta_x: float
    look_rot_delta_y: float
    attack: float
    shrinking_key: float
    delta_x: float
    delta_y: float
    delta_rot: float


class UnityTemporalState(BaseModel):
    """Temporal features for the agent's own history"""
    agents: List[AgentFeatures]  # List of agent states over time (most recent first)


class UnitySpatialState(BaseModel):
    """Spatial snapshot of all agents at current timestep"""
    agents: List[AgentFeatures]  # All agents at current timestep


class UnityStatePayload(BaseModel):
    """Unity sends both temporal and spatial features"""
    temporal: UnityTemporalState
    spatial: UnitySpatialState


class UnityDualOutputResponse(BaseModel):
    predictions: List[float]  # length 2: [delta_x, delta_y]


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


def normalize_features(features: np.ndarray, feature_type: str) -> np.ndarray:
    """Normalize features using pre-computed statistics"""
    if feature_type == 'temporal':
        mean = FEATURE_STATS['temporal_mean']
        std = FEATURE_STATS['temporal_std']
    elif feature_type == 'spatial':
        mean = FEATURE_STATS['spatial_mean']
        std = FEATURE_STATS['spatial_std']
    else:
        return features

    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    return (features - mean) / std


def load_single_model() -> None:
    global SINGLE_MODEL, MODEL_META

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Load model with parameters from checkpoint
    args = checkpoint.get("args", {})
    model = build_model(
        feature_dim=13,  # Fixed for this dataset
        d_model=args.get("d_model", 512),
        temp_layers=args.get("temp_layers", 3),
        temp_heads=args.get("temp_heads", 8),
        spat_layers=args.get("spat_layers", 3),
        spat_heads=args.get("spat_heads", 8),
        dropout=args.get("dropout", 0.0),
        max_time=64,  # Fixed temporal window size
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


def agent_features_to_list(agent: AgentFeatures) -> List[float]:
    """Convert AgentFeatures to list in the expected order"""
    return [
        agent.team_index,
        agent.rel_pos_x,
        agent.rel_pos_z,
        agent.rotation,
        agent.move_dir_x,
        agent.move_dir_y,
        agent.look_rot_delta_x,
        agent.look_rot_delta_y,
        agent.attack,
        agent.shrinking_key,
        agent.delta_x,
        agent.delta_y,
        agent.delta_rot,
    ]


def process_temporal_features(temporal_state: UnityTemporalState) -> torch.Tensor:
    """Process temporal features into the expected format (T, F)"""
    global TEMPORAL_WINDOW

    # Add new temporal features to the window
    current_features = []
    for agent in temporal_state.agents:
        current_features = agent_features_to_list(agent)
        break  # Only take the first agent (the controlled agent's history)

    if current_features:
        TEMPORAL_WINDOW.append(current_features)

    # Convert to tensor and normalize
    if len(TEMPORAL_WINDOW) > 0:
        temporal_array = np.array(list(TEMPORAL_WINDOW))  # (T, 13)
        temporal_array = normalize_features(temporal_array, 'temporal')
        return torch.tensor(temporal_array, dtype=torch.float32, device=DEVICE)
    else:
        # Return zeros if no data
        return torch.zeros(1, 13, dtype=torch.float32, device=DEVICE)


def process_spatial_features(spatial_state: UnitySpatialState) -> torch.Tensor:
    """Process spatial features into the expected format (A, F)"""
    global SPATIAL_AGENTS

    # Update spatial snapshot
    SPATIAL_AGENTS = []
    for agent in spatial_state.agents:
        SPATIAL_AGENTS.append(agent_features_to_list(agent))

    # Convert to tensor and normalize
    if len(SPATIAL_AGENTS) > 0:
        spatial_array = np.array(SPATIAL_AGENTS)  # (A, 13)
        spatial_array = normalize_features(spatial_array, 'spatial')
        return torch.tensor(spatial_array, dtype=torch.float32, device=DEVICE)
    else:
        # Return zero agent if no data
        return torch.zeros(1, 13, dtype=torch.float32, device=DEVICE)


@app.on_event("startup")
async def on_startup() -> None:
    load_feature_stats()
    load_single_model()


@app.get("/", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy", device=str(DEVICE), model_loaded=SINGLE_MODEL is not None)


@app.get("/model/info")
async def model_info() -> Dict[str, Any]:
    if SINGLE_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return MODEL_META


@app.post("/window/reset")
async def reset_windows() -> Dict[str, Any]:
    TEMPORAL_WINDOW.clear()
    SPATIAL_AGENTS.clear()
    return {"message": "temporal and spatial windows cleared"}


@app.post("/predict/unity", response_model=UnityDualOutputResponse)
async def predict_unity(payload: UnityStatePayload) -> UnityDualOutputResponse:
    if SINGLE_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Process temporal features (agent's own history)
        temporal_tensor = process_temporal_features(payload.temporal)  # (T, 13)

        # Process spatial features (current snapshot of all agents)
        spatial_tensor = process_spatial_features(payload.spatial)  # (A, 13)

        # Ensure minimum dimensions
        if temporal_tensor.size(0) == 0:
            temporal_tensor = torch.zeros(1, 13, dtype=torch.float32, device=DEVICE)
        if spatial_tensor.size(0) == 0:
            spatial_tensor = torch.zeros(1, 13, dtype=torch.float32, device=DEVICE)

        # Add batch dimension
        temporal_tensor = temporal_tensor.unsqueeze(0)  # (1, T, 13)
        spatial_tensor = spatial_tensor.unsqueeze(0)    # (1, A, 13)

        # Forward pass
        with torch.no_grad():
            output = SINGLE_MODEL(temporal_tensor, spatial_tensor)  # (1, 3)
            preds = output.squeeze(0).cpu().tolist()  # [dx, dy, drot]

        # Only return dx, dy (first 2 predictions)
        predictions = preds[:2]

        # Record predictions for live histogram
        try:
            with PRED_HISTORY_LOCK:
                PRED1_HISTORY.append(float(predictions[0]))
                PRED2_HISTORY.append(float(predictions[1]))
        except Exception:
            pass

        return UnityDualOutputResponse(predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")


def _histogram_plotter(refresh_seconds: float = 0.1, bins: int = 40) -> None:
    """Continuously plot histograms of prediction heads in real time."""
    import time
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Matplotlib not available for live histogram: {e}")
        while True:
            time.sleep(1.0)

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Predictions Histogram (live)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

    while True:
        with PRED_HISTORY_LOCK:
            h1 = list(PRED1_HISTORY)
            h2 = list(PRED2_HISTORY)

        ax.clear()
        if h1:
            ax.hist(h1, bins=bins, alpha=0.6, label="delta_x")
        if h2:
            ax.hist(h2, bins=bins, alpha=0.6, label="delta_y")
        if h1 or h2:
            ax.legend(loc="best")
        ax.set_title("Movement Predictions Histogram (live)")
        ax.set_xlabel("Delta Value")
        ax.set_ylabel("Count")

        plt.pause(0.001)
        time.sleep(refresh_seconds)


if __name__ == "__main__":
    # Run uvicorn in a background thread and keep the main thread for GUI plotting
    import time

    def _run_uvicorn():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    server_thread = threading.Thread(target=_run_uvicorn, daemon=True)
    server_thread.start()

    # Allow server to start
    time.sleep(1.5)
    print("Uvicorn server running in background thread (http://0.0.0.0:8000).\n"
          "Launching live predictions histogram on main thread ...")

    _histogram_plotter(refresh_seconds=0.1, bins=40)

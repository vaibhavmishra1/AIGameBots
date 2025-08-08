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

# Ensure local imports work
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_model

app = FastAPI(
    title="AI Game Agents API (ann2)",
    description="Serve a single dual-head regression model (2 outputs) for Unity",
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
MODEL_PATH = \
    "/Users/vaibhavmishra/Desktop/Desktop/btx-game-aicode/AIGameAgents/training/ann2/models/best_model.pth"

SINGLE_MODEL: Optional[torch.nn.Module] = None
MODEL_META: Dict[str, Any] = {}


# Schemas
class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


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


class UnityStepPayload(BaseModel):
    """Unity sends a single game state per request."""
    state: UnityState


class UnityDualOutputResponse(BaseModel):
    predictions: List[float]  # length 2


def load_single_model() -> None:
    global SINGLE_MODEL, MODEL_META

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Restore hyperparameters if available
    hparams = checkpoint.get("hyperparameters", {})
    input_dim = hparams.get("input_dim", 360)
    hidden_dims = hparams.get("hidden_dims", [512, 256, 128, 64])
    dropout_rate = hparams.get("dropout_rate", 0.3)

    model = create_model(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)
    model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[arg-type]
    model.to(DEVICE)
    model.eval()
    
    SINGLE_MODEL = model
    MODEL_META = {
        "epoch": checkpoint.get("epoch"),
        "train_loss": checkpoint.get("train_loss"),
        "val_loss": checkpoint.get("val_loss"),
        "hyperparameters": hparams,
        "model_path": MODEL_PATH,
    }


FEATURE_WINDOW: "deque[List[float]]" = deque(maxlen=20)
PRED1_HISTORY: "deque[float]" = deque(maxlen=1000)
PRED2_HISTORY: "deque[float]" = deque(maxlen=1000)
PRED_HISTORY_LOCK = threading.Lock()


def extract_18d_features_from_state(state: UnityState) -> List[float]:
    """Derive 18-dim feature vector from a UnityState.

    You can change this mapping as needed; server maintains a 20-step window.
    """
    features: List[float] = []

    # Agent
    features.append(state.agentPosition["x"]/1000)  # 1
    features.append(state.agentPosition["z"]/1000)  # 2
    features.append(state.agentRotation["y"]/360)  # 3
    features.append(state.agentForward["x"])   # 4
    features.append(state.agentForward["z"])   # 5
    features.append(state.health/100)               # 6
    features.append(0)               # 7

    features.append(state.targetPosition["x"]/1000)  # 8
    features.append(state.targetPosition["z"]/1000)  # 9
    features.append(state.targetRotation["y"]/360)  # 10
    features.append(state.targetForward["x"])   # 11
    features.append(state.targetForward["z"])   # 12
    features.append(state.directionToTarget["x"]) # 13
    features.append(state.directionToTarget["z"]) # 14
    features.append(state.cross["x"])  # 15
    features.append(state.cross["z"])  # 16
    features.append((1 - state.distance)/1000)  # 17
    features.append(state.dotProduct)  # 18
    
    return features


def validate_step_and_prepare_tensor(step_features: List[float]) -> torch.Tensor:
    if len(step_features) != 18:
        raise ValueError(f"Expected 18-dim feature, got {len(step_features)}")

    # Append the new step into the sliding window
    FEATURE_WINDOW.append(list(map(float, step_features)))

    # Build a 20x18 window, left-pad with zeros if not yet full
    num_missing = 20 - len(FEATURE_WINDOW)
    if num_missing > 0:
        pad = [[0.0] * 18 for _ in range(num_missing)]
        window = pad + list(FEATURE_WINDOW)
    else:
        window = list(FEATURE_WINDOW)

    flat: List[float] = [v for row in window for v in row]
    tensor = torch.tensor(flat, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, 360)
    return tensor


@app.on_event("startup")
async def on_startup() -> None:
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
async def reset_window() -> Dict[str, Any]:
    FEATURE_WINDOW.clear()
    return {"message": "feature window cleared", "window_length": len(FEATURE_WINDOW)}


@app.post("/predict/unity", response_model=UnityDualOutputResponse)
async def predict_unity(payload: UnityStepPayload) -> UnityDualOutputResponse:
    if SINGLE_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        step_features = extract_18d_features_from_state(payload.state)
        x = validate_step_and_prepare_tensor(step_features)
        with torch.no_grad():
            y = SINGLE_MODEL(x)  # (1, 2)
            preds = y.squeeze(0).tolist()  # [y1, y2]
        if not isinstance(preds, list) or len(preds) != 2:
            raise RuntimeError("Model did not return 2 outputs")

        # Record predictions for live histogram visualisation
        try:
            with PRED_HISTORY_LOCK:
                PRED1_HISTORY.append(float(preds[0]))
                PRED2_HISTORY.append(float(preds[1]))
        except Exception:
            pass

        return UnityDualOutputResponse(predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")


def _histogram_plotter(refresh_seconds: float = 0.1, bins: int = 40) -> None:
    """Continuously plot histograms of prediction heads in real time.

    Must run on the main thread for macOS GUI backends.
    """
    import time
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Matplotlib not available for live histogram: {e}")
        # Fall back to sleeping loop so process stays alive
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
            ax.hist(h1, bins=bins, alpha=0.6, label="head1")
        if h2:
            ax.hist(h2, bins=bins, alpha=0.6, label="head2")
        if h1 or h2:
            ax.legend(loc="best")
        ax.set_title("Predictions Histogram (live)")
        ax.set_xlabel("Value")
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
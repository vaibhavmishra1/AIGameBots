from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional, Union
import uvicorn
import threading
import queue
import uuid
from pathlib import Path

# Import our inference engine
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import AgentInference

app = FastAPI(
    title="LSTM AI Game Agents API",
    description="API for serving trained LSTM models for sequential game action prediction",
    version="2.0.0"
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
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
CHECKPOINT_PATH = "/Users/vaibhavmishra/Desktop/Desktop/btx-game-aicode/AIGameAgents/training/lstm/checkpoints/lstm_agent/best_model.pt"
INFERENCE_ENGINE = None

# Session management for LSTM states
SESSIONS = {}  # session_id -> inference_engine instance
SESSION_TIMEOUT = 300  # 5 minutes

# Pydantic models for request/response
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
    session_id: Optional[str] = None

class LSTMPredictionRequest(BaseModel):
    """Request for LSTM prediction with features."""
    features: List[float]  # 20-dimensional feature vector
    session_id: Optional[str] = None
    return_uncertainty: Optional[bool] = False

class LSTMSequencePredictionRequest(BaseModel):
    """Request for LSTM prediction with sequence of features."""
    feature_sequences: List[List[float]]  # [seq_len, 20] feature sequences
    prev_actions: Optional[List[List[float]]] = None  # [seq_len-1, 4] previous actions
    session_id: Optional[str] = None
    return_uncertainty: Optional[bool] = False

class LSTMAPIResponse(BaseModel):
    """LSTM response structure."""
    predictions: List[float]  # 4 actions: [move_x, move_z, look_x, look_z]
    uncertainties: Optional[List[float]] = None
    session_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    device: str
    active_sessions: int

class SessionResponse(BaseModel):
    session_id: str
    message: str

# Queue for visualizer
STATE_QUEUE: "queue.Queue[UnityState]" = queue.Queue(maxsize=10)

def convert_unity_state_to_features(state: UnityState) -> List[float]:
    """
    Convert a Unity state to a 20-dimensional feature vector.
    
    Feature breakdown:
    [0-9]: Agent features (10 total)
    [10-16]: Target features (7 total)  
    [17-19]: Relationship features (3 total)
    Total: 20 features per state
    """
    features = []
    
    # Agent features (10 features: positions, rotation, forward, health, weapon, los)
    features.append(state.agentPosition['x'])    
    features.append(state.agentPosition['z'])  
    features.append(state.agentRotation['y'])  
    features.append(state.agentForward['x'])   
    features.append(state.agentForward['z'])  
    features.append(state.health)               
    features.append(state.weapon)               
    features.append(state.targetPosition['x'])   
    features.append(state.targetPosition['z'])  
    features.append(state.targetRotation['y'])  
    features.append(state.targetForward['x'])   
    features.append(state.targetForward['z'])   
    features.append(state.directionToTarget['x'])
    features.append(state.directionToTarget['z']) 
    features.append(state.cross['x'])            
    features.append(state.cross['z'])            
    features.append(1 - state.distance)
    features.append(state.dotProduct)
    features.append( 1 - (state.targetPosition['x'] - state.agentPosition['x']))
    features.append( 1 - (state.targetPosition['z'] - state.agentPosition['z']))
    
    assert len(features) == 20, f"Expected 20 features, got {len(features)}"
    return features

def convert_unity_states_to_sequence(states: List[UnityState]) -> List[List[float]]:
    """Convert Unity states to a sequence of 20-dimensional feature vectors."""
    sequence = []
    
    # Convert each state to features
    for state in states:
        state_features = convert_unity_state_to_features(state)
        sequence.append(state_features)
    
    return sequence

def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create new one."""
    if session_id and session_id in SESSIONS:
        return session_id
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    SESSIONS[new_session_id] = AgentInference(CHECKPOINT_PATH, device=DEVICE)
    
    print(f"Created new session: {new_session_id}")
    return new_session_id

def cleanup_sessions():
    """Clean up old sessions (this could be enhanced with proper timeout logic)."""
    # For now, just keep the last 10 sessions
    if len(SESSIONS) > 10:
        oldest_sessions = list(SESSIONS.keys())[:-10]
        for session_id in oldest_sessions:
            del SESSIONS[session_id]
            print(f"Cleaned up session: {session_id}")

# ---------------------------------------------------------------------------
# Live visualizer for incoming UnityState data (adapted from ANN version)
# ---------------------------------------------------------------------------

def _unity_state_plotter():
    """Visualizer for LSTM predictions."""
    import matplotlib.pyplot as plt
    import time

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))

    last_state: Optional[UnityState] = None

    while True:
        try:
            while True:
                last_state = STATE_QUEUE.get_nowait()
        except queue.Empty:
            pass

        if last_state is not None:
            # Extract information
            a_pos = last_state.agentPosition
            t_pos = last_state.targetPosition
            a_fwd = last_state.agentForward
            t_fwd = last_state.targetForward
            health = last_state.health * 100
            weapon = last_state.weapon

            ax_x = a_pos['x']
            az_z = a_pos['z']
            tx = t_pos['x']
            tz = t_pos['z']

            ax.clear()
            ax.set_xlim(-0.2, 0.2)
            ax.set_ylim(-0.2, 0.2)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.set_title("LSTM Agent - Unity State Visualization")

            # Agent visuals
            ax.scatter(0, 0, c="blue", label="Agent")
            ax.arrow(0, 0, a_fwd['x'], a_fwd['z'], head_width=0.01, fc="blue", ec="blue")
            ax.text(0.1, 0.1, f"Health: {health:.1f}%", color="blue")

            # Target visuals
            if tx != 0 or tz != 0:
                ax.scatter(tx - ax_x, tz - az_z, c="red", label="Target")
                ax.arrow(tx - ax_x, tz - az_z, t_fwd['x'], t_fwd['z'], head_width=0.01, fc="red", ec="red")
                ax.text(tx - ax_x + 0.1, tz - az_z + 0.1, f"Dist: {last_state.distance:.2f}", color="red")

            ax.legend(loc="upper right")

        plt.pause(0.05)

        if last_state is None:
            time.sleep(0.05)

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Load the LSTM model on startup."""
    global INFERENCE_ENGINE
    print("Loading LSTM model...")
    try:
        INFERENCE_ENGINE = AgentInference(CHECKPOINT_PATH, device=DEVICE)
        print("LSTM model loaded successfully")
    except Exception as e:
        print(f"Failed to load LSTM model: {e}")
        INFERENCE_ENGINE = None

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if INFERENCE_ENGINE is not None else "unhealthy",
        model_loaded=INFERENCE_ENGINE is not None,
        model_type="LSTM with Uncertainty",
        device=str(DEVICE),
        active_sessions=len(SESSIONS)
    )

@app.post("/sessions/create", response_model=SessionResponse)
async def create_session():
    """Create a new session for stateful LSTM predictions."""
    session_id = get_or_create_session()
    cleanup_sessions()
    return SessionResponse(
        session_id=session_id,
        message="Session created successfully"
    )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session."""
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

@app.post("/sessions/{session_id}/reset")
async def reset_session(session_id: str):
    """Reset a session's LSTM hidden state."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    SESSIONS[session_id].reset()
    return {"message": f"Session {session_id} reset successfully"}

@app.post("/predict")
async def predict_unity(request: UnityStatePayload):
    """Make predictions for Unity game agent actions using LSTM."""
    try:
        if INFERENCE_ENGINE is None:
            raise HTTPException(status_code=503, detail="LSTM model not loaded")

        # Get or create session
        session_id = get_or_create_session(request.session_id)
        engine = SESSIONS[session_id]

        # Convert Unity states to feature sequence
        if len(request.states) != 20:
            raise ValueError(f"Expected 20 states, got {len(request.states)}")
        
        # Convert all 20 Unity states to a sequence of features [20, 20]
        feature_sequence = convert_unity_states_to_sequence(request.states)
        feature_sequence = np.array(feature_sequence)  # Shape: [20, 20]
        
        print(f"Feature sequence shape: {feature_sequence.shape}")
        
        # Make prediction using the full sequence
        result = engine.predict(
            feature_sequence,  # [20, 20] - 20 timesteps of 20 features each
            return_uncertainty=True
        )
        
        actions = result['actions'].tolist()
        uncertainties = result.get('uncertainties', [0.0] * 4)
        if uncertainties is not None:
            uncertainties = uncertainties.tolist()

        print(f"LSTM Predictions - Session {session_id[:8]}...")
        print(f"  predictions: {actions}")
        print(f"  Uncertainties: {uncertainties}")

        # Enqueue the newest state for visualization
        try:
            STATE_QUEUE.put_nowait(request.states[-1])
        except queue.Full:
            try:
                STATE_QUEUE.get_nowait()
                STATE_QUEUE.put_nowait(request.states[-1])
            except queue.Empty:
                pass

        return LSTMAPIResponse(
            predictions=actions,
            uncertainties=uncertainties,
            session_id=session_id
        )
        
    except Exception as e:
        print(f"LSTM prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LSTM prediction error: {str(e)}")

@app.post("/predict/features", response_model=LSTMAPIResponse)
async def predict_features(request: LSTMPredictionRequest):
    """Make predictions using raw 20-dimensional features."""
    try:
        if len(request.features) != 20:
            raise ValueError(f"Expected 20 features, got {len(request.features)}")

        # Get or create session
        session_id = get_or_create_session(request.session_id)
        engine = SESSIONS[session_id]

        # Make prediction
        result = engine.predict(
            np.array(request.features),
            return_uncertainty=request.return_uncertainty
        )
        
        actions = result['actions'].tolist()
        uncertainties = result.get('uncertainties')
        if uncertainties is not None:
            uncertainties = uncertainties.tolist()

        return LSTMAPIResponse(
            predictions=actions,
            uncertainties=uncertainties,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature prediction error: {str(e)}")

@app.post("/predict/sequence", response_model=LSTMAPIResponse)
async def predict_sequence(request: LSTMSequencePredictionRequest):
    """Make predictions using feature sequences."""
    try:
        if INFERENCE_ENGINE is None:
            raise HTTPException(status_code=503, detail="LSTM model not loaded")

        # Validate input
        if not request.feature_sequences:
            raise ValueError("Feature sequences cannot be empty")
        
        for seq in request.feature_sequences:
            if len(seq) != 20:
                raise ValueError(f"Each feature sequence must have 20 features, got {len(seq)}")

        # Get or create session
        session_id = get_or_create_session(request.session_id)
        engine = SESSIONS[session_id]

        # Prepare data
        features_array = np.array(request.feature_sequences)
        prev_actions_array = None
        if request.prev_actions:
            prev_actions_array = np.array(request.prev_actions)

        # Make batch prediction
        result = engine.predict_batch(
            features_array.reshape(1, *features_array.shape),  # Add batch dimension
            prev_actions_array.reshape(1, *prev_actions_array.shape) if prev_actions_array is not None else None,
            return_uncertainty=request.return_uncertainty
        )
        
        actions = result['actions'][0].tolist()  # Remove batch dimension
        uncertainties = result.get('uncertainties')
        if uncertainties is not None:
            uncertainties = uncertainties[0].tolist()

        return LSTMAPIResponse(
            predictions=actions,
            uncertainties=uncertainties,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sequence prediction error: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded LSTM model."""
    if INFERENCE_ENGINE is None:
        raise HTTPException(status_code=503, detail="LSTM model not loaded")
    
    config = INFERENCE_ENGINE.config
    
    return {
        "model_type": config.get('model_type', 'unknown'),
        "feature_dim": config.get('feature_dim', 20),
        "action_dim": config.get('action_dim', 4),
        "hidden_dim": config.get('hidden_dim', 256),
        "num_lstm_layers": config.get('num_lstm_layers', 2),
        "use_attention": config.get('use_attention', False),
        "use_residual": config.get('use_residual', False),
        "use_history": config.get('use_history', 5),
        "normalize": config.get('normalize', True),
        "device": str(INFERENCE_ENGINE.device),
        "checkpoint_path": CHECKPOINT_PATH
    }

@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {
        "active_sessions": list(SESSIONS.keys()),
        "session_count": len(SESSIONS)
    }

@app.post("/benchmark")
async def benchmark_model(num_iterations: int = 1000):
    """Benchmark the LSTM model performance."""
    if INFERENCE_ENGINE is None:
        raise HTTPException(status_code=503, detail="LSTM model not loaded")
    
    try:
        results = INFERENCE_ENGINE.benchmark(num_iterations)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark error: {str(e)}")

if __name__ == "__main__":
    """Run uvicorn in a background thread, then launch the live visualizer on the
    main thread so that all matplotlib GUI operations occur in the main thread."""
    
    import time

    def _run_uvicorn():
        uvicorn.run(app, host="0.0.0.0", port=8000)  # Different port from ANN

    # Start server in a daemon thread
    server_thread = threading.Thread(target=_run_uvicorn, daemon=True)
    server_thread.start()

    # Small delay to ensure the server is listening
    time.sleep(2)

    print("LSTM API server running on http://0.0.0.0:8001")
    print("API documentation available at http://localhost:8001/docs")
    print("LSTM Unity endpoint: POST http://localhost:8001/predict")
    print("Launching Unity state visualizer on main thread...")

    _unity_state_plotter()
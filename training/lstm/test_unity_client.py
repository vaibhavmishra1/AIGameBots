import requests
import json
import numpy as np
from typing import List, Dict

class LSTMUnityGameAgentAPIClient:
    """Client for testing LSTM Unity game agent API integration."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session_id = None
    
    def create_sample_state(self) -> Dict:
        """Create a sample Unity state with realistic values."""
        return {
            "agentPosition": {"x": np.random.randn() * 0.01, "y": 0.0005, "z": np.random.randn() * 0.01},  # Scaled down
            "agentRotation": {"x": 0, "y": np.random.rand(), "z": 0},  # Normalized rotation
            "agentForward": {"x": np.random.randn(), "y": 0, "z": np.random.randn()},
            "health": np.random.rand(),  # Normalized health
            "weapon": 0,
            "targetPosition": {"x": np.random.randn() * 0.01, "y": 0.0005, "z": np.random.randn() * 0.01},
            "targetRotation": {"x": 0, "y": np.random.rand(), "z": 0},
            "targetForward": {"x": np.random.randn(), "y": 0, "z": np.random.randn()},
            "directionToTarget": {"x": np.random.randn(), "y": 0, "z": np.random.randn()},
            "cross": {"x": np.random.randn(), "y": np.random.randn(), "z": np.random.randn()},
            "distance": np.random.rand() * 0.05,  # Scaled down distance
            "dotProduct": np.random.rand() * 2 - 1,  # Range [-1, 1]
            "islos": np.random.rand() > 0.5
        }
    
    def create_states_payload(self, num_states: int = 20) -> Dict:
        """Create a Unity states payload with the specified number of states."""
        states = []
        for _ in range(num_states):
            states.append(self.create_sample_state())
        
        payload = {"states": states}
        if self.session_id:
            payload["session_id"] = self.session_id
        
        return payload
    
    def create_session(self) -> Dict:
        """Create a new session."""
        response = requests.post(f"{self.base_url}/sessions/create")
        return response.json()
    
    def predict_unity(self, states_payload: Dict) -> Dict:
        """Send Unity states to the API and get predictions."""
        response = requests.post(f"{self.base_url}/predict", json=states_payload)
        return response.json()
    
    def predict_features(self, features: List[float], return_uncertainty: bool = False) -> Dict:
        """Send raw features to the API."""
        payload = {
            "features": features,
            "return_uncertainty": return_uncertainty
        }
        if self.session_id:
            payload["session_id"] = self.session_id
        
        response = requests.post(f"{self.base_url}/predict/features", json=payload)
        return response.json()
    
    def predict_sequence(self, feature_sequences: List[List[float]], 
                        prev_actions: List[List[float]] = None,
                        return_uncertainty: bool = False) -> Dict:
        """Send feature sequences to the API."""
        payload = {
            "feature_sequences": feature_sequences,
            "return_uncertainty": return_uncertainty
        }
        if prev_actions:
            payload["prev_actions"] = prev_actions
        if self.session_id:
            payload["session_id"] = self.session_id
        
        response = requests.post(f"{self.base_url}/predict/sequence", json=payload)
        return response.json()
    
    def health_check(self) -> Dict:
        """Check API health."""
        response = requests.get(f"{self.base_url}/")
        return response.json()
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        response = requests.get(f"{self.base_url}/model/info")
        return response.json()
    
    def list_sessions(self) -> Dict:
        """List active sessions."""
        response = requests.get(f"{self.base_url}/sessions")
        return response.json()
    
    def reset_session(self, session_id: str) -> Dict:
        """Reset a session."""
        response = requests.post(f"{self.base_url}/sessions/{session_id}/reset")
        return response.json()
    
    def delete_session(self, session_id: str) -> Dict:
        """Delete a session."""
        response = requests.delete(f"{self.base_url}/sessions/{session_id}")
        return response.json()
    
    def benchmark(self, num_iterations: int = 100) -> Dict:
        """Benchmark the model."""
        response = requests.post(f"{self.base_url}/benchmark", params={"num_iterations": num_iterations})
        return response.json()


def main():
    """Test the LSTM Unity API integration."""
    client = LSTMUnityGameAgentAPIClient()
    
    print("=== LSTM Unity Game Agent API Test ===\n")
    
    # Health check
    print("1. Health Check:")
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Model loaded: {health['model_loaded']}")
        print(f"Model type: {health['model_type']}")
        print(f"Device: {health['device']}")
        print(f"Active sessions: {health['active_sessions']}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Get model info
    print("\n2. Model Information:")
    try:
        info = client.get_model_info()
        print(f"Feature dim: {info['feature_dim']}")
        print(f"Action dim: {info['action_dim']}")
        print(f"Hidden dim: {info['hidden_dim']}")
        print(f"LSTM layers: {info['num_lstm_layers']}")
        print(f"Use attention: {info['use_attention']}")
        print(f"Use history: {info['use_history']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Create session
    print("\n3. Creating Session:")
    try:
        session_resp = client.create_session()
        client.session_id = session_resp['session_id']
        print(f"Session created: {client.session_id}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test Unity prediction
    print("\n4. Testing Unity Prediction (20 states → 20x20 features):")
    try:
        # Create a sample payload with 20 states
        payload = client.create_states_payload(20)
        print(f"Created payload with {len(payload['states'])} Unity states")
        print("Each state will be converted to 20 features, creating a [20x20] sequence")
        
        # Make prediction
        response = client.predict_unity(payload)
        print(f"API Response received")
        
        if 'actions' in response:
            actions = response['actions']
            uncertainties = response.get('uncertainties', [])
            print(f"\nLSTM Predictions (from 20-state sequence):")
            print(f"  Movement X: {actions[0]:.4f} (±{uncertainties[0]:.4f})" if uncertainties else f"  Movement X: {actions[0]:.4f}")
            print(f"  Movement Z: {actions[1]:.4f} (±{uncertainties[1]:.4f})" if uncertainties else f"  Movement Z: {actions[1]:.4f}")
            print(f"  Look X: {actions[2]:.4f} (±{uncertainties[2]:.4f})" if uncertainties else f"  Look X: {actions[2]:.4f}")
            print(f"  Look Z: {actions[3]:.4f} (±{uncertainties[3]:.4f})" if uncertainties else f"  Look Z: {actions[3]:.4f}")
            print(f"  Session ID: {response.get('session_id', 'N/A')}")
        else:
            print(f"Error in response: {response}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test raw features prediction
    print("\n5. Testing Raw Features Prediction:")
    try:
        # Create sample 20-dimensional features
        features = np.random.randn(20).tolist()
        response = client.predict_features(features, return_uncertainty=True)
        
        actions = response['actions']
        uncertainties = response.get('uncertainties', [])
        print(f"Raw Features Prediction:")
        print(f"  Actions: {[f'{a:.4f}' for a in actions]}")
        if uncertainties:
            print(f"  Uncertainties: {[f'{u:.4f}' for u in uncertainties]}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test sequence prediction
    print("\n6. Testing Sequence Prediction:")
    try:
        # Create a sequence of features (5 timesteps)
        sequence_length = 5
        feature_sequences = [np.random.randn(20).tolist() for _ in range(sequence_length)]
        prev_actions = [np.random.randn(4).tolist() for _ in range(sequence_length - 1)]
        
        response = client.predict_sequence(
            feature_sequences, 
            prev_actions=prev_actions,
            return_uncertainty=True
        )
        
        actions = response['actions']
        uncertainties = response.get('uncertainties', [])
        print(f"Sequence Prediction (length {sequence_length}):")
        print(f"  Actions: {[f'{a:.4f}' for a in actions]}")
        if uncertainties:
            print(f"  Uncertainties: {[f'{u:.4f}' for u in uncertainties]}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # List sessions
    print("\n7. Active Sessions:")
    try:
        sessions = client.list_sessions()
        print(f"Active sessions: {sessions['session_count']}")
        for session in sessions['active_sessions']:
            print(f"  - {session}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test session reset
    print("\n8. Testing Session Reset:")
    try:
        if client.session_id:
            reset_resp = client.reset_session(client.session_id)
            print(f"Session reset: {reset_resp['message']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Benchmark
    print("\n9. Performance Benchmark:")
    try:
        benchmark_results = client.benchmark(num_iterations=50)
        single_pred = benchmark_results['single_prediction']
        print(f"Single prediction: {single_pred['mean_ms']:.2f} ± {single_pred['std_ms']:.2f} ms")
        print(f"FPS: {single_pred['fps']:.1f}")
        
        print("Batch predictions:")
        for batch_size, stats in benchmark_results['batch_predictions'].items():
            print(f"  Batch {batch_size}: {stats['per_sample']*1000:.2f} ms per sample")
    except Exception as e:
        print(f"Error: {e}")
    
    # Clean up - delete session
    print("\n10. Cleanup:")
    try:
        if client.session_id:
            delete_resp = client.delete_session(client.session_id)
            print(f"Session deleted: {delete_resp['message']}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()
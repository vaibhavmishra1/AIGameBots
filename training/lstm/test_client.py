import requests
import json
import numpy as np
from typing import List

class LSTMGameAgentAPIClient:
    """Client for interacting with the LSTM Game Agent API."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session_id = None
    
    def health_check(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/")
        return response.json()
    
    def get_model_info(self):
        """Get model information."""
        response = requests.get(f"{self.base_url}/model/info")
        return response.json()
    
    def create_session(self):
        """Create a new session."""
        response = requests.post(f"{self.base_url}/sessions/create")
        result = response.json()
        self.session_id = result['session_id']
        return result
    
    def predict_features(self, features: List[float], return_uncertainty: bool = False):
        """Make a prediction using raw features."""
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
                        return_uncertainty: bool = False):
        """Make a prediction using feature sequences."""
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
    
    def list_sessions(self):
        """List active sessions."""
        response = requests.get(f"{self.base_url}/sessions")
        return response.json()
    
    def reset_session(self, session_id: str = None):
        """Reset a session."""
        session_id = session_id or self.session_id
        if not session_id:
            raise ValueError("No session ID provided")
        
        response = requests.post(f"{self.base_url}/sessions/{session_id}/reset")
        return response.json()
    
    def delete_session(self, session_id: str = None):
        """Delete a session."""
        session_id = session_id or self.session_id
        if not session_id:
            raise ValueError("No session ID provided")
        
        response = requests.delete(f"{self.base_url}/sessions/{session_id}")
        if session_id == self.session_id:
            self.session_id = None
        return response.json()
    
    def benchmark(self, num_iterations: int = 100):
        """Benchmark the model."""
        response = requests.post(f"{self.base_url}/benchmark", 
                                params={"num_iterations": num_iterations})
        return response.json()


def generate_sample_features() -> List[float]:
    """Generate sample 20-dimensional feature vector."""
    # Generate realistic game features
    features = []
    
    # Agent position (normalized)
    features.extend(np.random.randn(3) * 0.01)  # x, y, z position
    
    # Agent rotation (normalized)
    features.append(np.random.rand())  # y rotation [0, 1]
    
    # Agent forward vector
    features.extend(np.random.randn(3))  # forward x, y, z
    
    # Agent health and weapon
    features.append(np.random.rand())  # health [0, 1]
    features.append(0.0)  # weapon
    
    # Line of sight
    features.append(np.random.choice([0.0, 1.0]))  # islos
    
    # Target position
    features.extend(np.random.randn(3) * 0.01)  # target x, y, z
    
    # Target rotation
    features.append(np.random.rand())  # target y rotation
    
    # Target forward
    features.extend(np.random.randn(3))  # target forward x, y, z
    
    # Direction to target
    features.extend(np.random.randn(2))  # direction x, z (no y)
    
    # Distance and dot product
    features.append(np.random.rand() * 0.05)  # distance [0, 0.05]
    features.append(np.random.rand() * 2 - 1)  # dot product [-1, 1]
    
    return features


def main():
    """Test the LSTM API client."""
    client = LSTMGameAgentAPIClient()
    
    print("=== LSTM Game Agent API Test Client ===\n")
    
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
    
    print("\n2. Model Information:")
    try:
        info = client.get_model_info()
        print(f"Feature dimension: {info['feature_dim']}")
        print(f"Action dimension: {info['action_dim']}")
        print(f"Hidden dimension: {info['hidden_dim']}")
        print(f"LSTM layers: {info['num_lstm_layers']}")
        print(f"Use attention: {info['use_attention']}")
        print(f"Use residual: {info['use_residual']}")
        print(f"Use history: {info['use_history']}")
        print(f"Normalize: {info['normalize']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Create session
    print("\n3. Creating Session:")
    try:
        session_resp = client.create_session()
        print(f"Session ID: {client.session_id}")
        print(f"Message: {session_resp['message']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test single feature prediction
    print("\n4. Testing Single Feature Prediction:")
    try:
        features = generate_sample_features()
        prediction = client.predict_features(features, return_uncertainty=True)
        
        actions = prediction['actions']
        uncertainties = prediction.get('uncertainties', [])
        
        print(f"Input features (first 5): {features[:5]}")
        print(f"Predicted actions:")
        action_names = ['move_x', 'move_z', 'look_x', 'look_z']
        for i, name in enumerate(action_names):
            if uncertainties:
                print(f"  {name}: {actions[i]:.4f} (± {uncertainties[i]:.4f})")
            else:
                print(f"  {name}: {actions[i]:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test sequence prediction
    print("\n5. Testing Sequence Prediction:")
    try:
        # Create a sequence of features
        sequence_length = 5
        feature_sequences = [generate_sample_features() for _ in range(sequence_length)]
        prev_actions = [[0.0, 0.0, 0.0, 0.0] for _ in range(sequence_length - 1)]
        
        prediction = client.predict_sequence(
            feature_sequences,
            prev_actions=prev_actions,
            return_uncertainty=True
        )
        
        actions = prediction['actions']
        uncertainties = prediction.get('uncertainties', [])
        
        print(f"Sequence length: {sequence_length}")
        print(f"Predicted actions:")
        for i, name in enumerate(action_names):
            if uncertainties:
                print(f"  {name}: {actions[i]:.4f} (± {uncertainties[i]:.4f})")
            else:
                print(f"  {name}: {actions[i]:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test multiple predictions with same session (to see state evolution)
    print("\n6. Testing Sequential Predictions (Session State):")
    try:
        print("Making 3 consecutive predictions to see LSTM state evolution:")
        for i in range(3):
            features = generate_sample_features()
            prediction = client.predict_features(features, return_uncertainty=False)
            actions = prediction['actions']
            print(f"  Prediction {i+1}: {[f'{a:.3f}' for a in actions]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Reset session
    print("\n7. Testing Session Reset:")
    try:
        reset_resp = client.reset_session()
        print(f"Reset message: {reset_resp['message']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # List sessions
    print("\n8. Listing Active Sessions:")
    try:
        sessions = client.list_sessions()
        print(f"Active session count: {sessions['session_count']}")
        print(f"Session IDs: {sessions['active_sessions'][:3]}...")  # Show first 3
    except Exception as e:
        print(f"Error: {e}")
    
    # Benchmark
    print("\n9. Performance Benchmark:")
    try:
        print("Running benchmark (this may take a moment)...")
        results = client.benchmark(num_iterations=50)
        
        single_pred = results['single_prediction']
        print(f"Single prediction: {single_pred['mean_ms']:.2f} ± {single_pred['std_ms']:.2f} ms")
        print(f"Throughput: {single_pred['fps']:.1f} FPS")
        
        print("\nBatch predictions:")
        for batch_size, stats in results['batch_predictions'].items():
            print(f"  Batch size {batch_size}: {stats['per_sample']*1000:.2f} ms per sample")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Clean up
    print("\n10. Cleanup:")
    try:
        delete_resp = client.delete_session()
        print(f"Delete message: {delete_resp['message']}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
import requests
import json
import numpy as np
from typing import List, Dict

class UnityGameAgentAPIClient:
    """Client for testing Unity game agent API integration."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def create_sample_state(self) -> Dict:
        """Create a sample Unity state with realistic values."""
        return {
            "agentPosition": {"x": np.random.randn() * 10, "y": 0.5, "z": np.random.randn() * 10},
            "agentRotation": {"x": 0, "y": np.random.rand() * 360, "z": 0},
            "agentForward": {"x": np.random.randn(), "y": 0, "z": np.random.randn()},
            "health": np.random.rand() * 100,
            "weapon": 0,
            "targetPosition": {"x": np.random.randn() * 10, "y": 0.5, "z": np.random.randn() * 10},
            "targetRotation": {"x": 0, "y": np.random.rand() * 360, "z": 0},
            "targetForward": {"x": np.random.randn(), "y": 0, "z": np.random.randn()},
            "directionToTarget": {"x": np.random.randn(), "y": 0, "z": np.random.randn()},
            "cross": {"x": np.random.randn(), "y": np.random.randn(), "z": np.random.randn()},
            "distance": np.random.rand() * 50,
            "dotProduct": np.random.rand() * 2 - 1,  # Range [-1, 1]
            "islos": np.random.rand() > 0.5
        }
    
    def create_states_payload(self, num_states: int = 20) -> Dict:
        """Create a Unity states payload with the specified number of states."""
        states = []
        for _ in range(num_states):
            states.append(self.create_sample_state())
        
        return {"states": states}
    
    def predict_unity(self, states_payload: Dict) -> Dict:
        """Send Unity states to the API and get predictions."""
        response = requests.post(f"{self.base_url}/predict", json=states_payload)
        return response.json()
    
    def health_check(self) -> Dict:
        """Check API health."""
        response = requests.get(f"{self.base_url}/")
        return response.json()


def main():
    """Test the Unity API integration."""
    client = UnityGameAgentAPIClient()
    
    print("=== Unity Game Agent API Test ===\n")
    
    # Health check
    print("1. Health Check:")
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Loaded models: {health['loaded_models']}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Test Unity prediction
    print("\n2. Testing Unity Prediction:")
    try:
        # Create a sample payload with 20 states
        payload = client.create_states_payload(20)
        print(f"Created payload with {len(payload['states'])} states")
        
        # Make prediction
        response = client.predict_unity(payload)
        print(f"API Response: {response}")
        
        if 'predictions' in response:
            predictions = response['predictions']
            print(f"\nPredictions breakdown:")
            print(f"  Movement - Forward: {predictions[0]}, Backward: {predictions[1]}")
            print(f"  Movement - Left: {predictions[2]}, Right: {predictions[3]}")
            print(f"  Turning - Left: {predictions[4]}, Right: {predictions[5]}")
            print(f"  Attack - Shoot: {predictions[6]}")
        else:
            print(f"Error in response: {response}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with varying number of states (should fail)
    print("\n3. Testing with incorrect number of states (should fail):")
    try:
        payload = client.create_states_payload(10)  # Wrong number of states
        response = client.predict_unity(payload)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Expected error: {e}")
    
    # Test individual state creation
    print("\n4. Sample state structure:")
    sample_state = client.create_sample_state()
    print(json.dumps(sample_state, indent=2))


if __name__ == "__main__":
    main() 
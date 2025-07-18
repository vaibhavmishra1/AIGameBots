import requests
import json
import numpy as np
from typing import List

class GameAgentAPIClient:
    """Client for interacting with the Game Agent API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/")
        return response.json()
    
    def list_models(self):
        """List all available models."""
        response = requests.get(f"{self.base_url}/models")
        return response.json()
    
    def get_tasks(self):
        """Get information about available tasks."""
        response = requests.get(f"{self.base_url}/tasks")
        return response.json()
    
    def predict(self, features: List[float], task: str):
        """Make a single prediction."""
        payload = {
            "features": features,
            "task": task
        }
        response = requests.post(f"{self.base_url}/predict", json=payload)
        return response.json()
    
    def predict_batch(self, features_list: List[List[float]], task: str):
        """Make batch predictions."""
        payload = {
            "features_list": features_list,
            "task": task
        }
        response = requests.post(f"{self.base_url}/predict/batch", json=payload)
        return response.json()
    
    def get_model_info(self, task: str):
        """Get information about a specific model."""
        response = requests.get(f"{self.base_url}/models/{task}/info")
        return response.json()
    
    def evaluate_model(self, task: str, num_samples: int = 1000):
        """Evaluate a model."""
        response = requests.post(f"{self.base_url}/evaluate/{task}?num_samples={num_samples}")
        return response.json()
    
    def load_model(self, task: str, model_type: str = 'best'):
        """Load a specific model."""
        response = requests.post(f"{self.base_url}/models/{task}/load?model_type={model_type}")
        return response.json()
    
    def unload_model(self, task: str):
        """Unload a specific model."""
        response = requests.delete(f"{self.base_url}/models/{task}/unload")
        return response.json()


def generate_sample_features() -> List[float]:
    """Generate sample 500-dimensional feature vector."""
    # Generate random features (you would replace this with real game features)
    return np.random.randn(500).tolist()


def main():
    """Test the API client."""
    client = GameAgentAPIClient()
    
    print("=== Game Agent API Test Client ===\n")
    
    # Health check
    print("1. Health Check:")
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Device: {health['device']}")
        print(f"Loaded models: {health['loaded_models']}")
        print(f"Available tasks: {health['available_tasks']}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print("\n2. Available Tasks:")
    try:
        tasks = client.get_tasks()
        print(f"Regression tasks: {tasks['regression_tasks']}")
        print(f"Classification tasks: {tasks['classification_tasks']}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n3. Model Information:")
    try:
        models = client.list_models()
        for model in models:
            print(f"- {model['task']} ({model['task_type']}): Loaded={model['is_loaded']}, Loss={model['loss']:.4f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test predictions
    print("\n4. Testing Predictions:")
    
    # Test regression task
    regression_task = "moveDirection_x"
    if regression_task in [model['task'] for model in models]:
        try:
            features = generate_sample_features()
            prediction = client.predict(features, regression_task)
            print(f"Regression prediction ({regression_task}): {prediction['prediction']:.4f}")
        except Exception as e:
            print(f"Error predicting {regression_task}: {e}")
    
    # Test classification task
    classification_task = "Attack"
    if classification_task in [model['task'] for model in models]:
        try:
            features = generate_sample_features()
            prediction = client.predict(features, classification_task)
            print(f"Classification prediction ({classification_task}): {prediction['prediction']:.4f}")
            print(f"Probability: {prediction['probability']:.4f}")
            print(f"Confidence: {prediction['confidence']:.4f}")
        except Exception as e:
            print(f"Error predicting {classification_task}: {e}")
    
    # Test batch predictions
    print("\n5. Testing Batch Predictions:")
    try:
        batch_features = [generate_sample_features() for _ in range(3)]
        batch_prediction = client.predict_batch(batch_features, regression_task)
        print(f"Batch predictions ({regression_task}): {batch_prediction['predictions']}")
    except Exception as e:
        print(f"Error in batch prediction: {e}")
    
    # Test model evaluation
    print("\n6. Testing Model Evaluation:")
    try:
        evaluation = client.evaluate_model(regression_task, num_samples=100)
        print(f"Evaluation results for {regression_task}:")
        for key, value in evaluation.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error in evaluation: {e}")


if __name__ == "__main__":
    main() 
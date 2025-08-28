#!/usr/bin/env python3
"""
Test script for the attention_v1 API server.
Tests model loading and prediction functionality.
"""

import requests
import json
import time
import numpy as np
from typing import Dict, Any


def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            data = response.json()
            print("✓ Health check passed")
            print(f"  Status: {data['status']}")
            print(f"  Device: {data['device']}")
            print(f"  Model loaded: {data['model_loaded']}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def test_model_info():
    """Test the model info endpoint"""
    try:
        response = requests.get("http://localhost:8000/model/info")
        if response.status_code == 200:
            data = response.json()
            print("✓ Model info retrieved")
            print(f"  Epoch: {data.get('epoch', 'N/A')}")
            print(f"  Best val loss: {data.get('best_val_loss', 'N/A'):.6f}")
            print(f"  Model path: {data.get('hyperparameters', {}).get('model_path', 'N/A')}")
            return True
        else:
            print(f"✗ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Model info error: {e}")
        return False


def create_test_payload() -> Dict[str, Any]:
    """Create a test payload with dummy data"""
    # Create dummy agent features
    agent_features = {
        "team_index": 0.0,
        "rel_pos_x": 0.5,
        "rel_pos_z": -0.3,
        "rotation": 0.25,
        "move_dir_x": 0.1,
        "move_dir_y": 0.0,
        "look_rot_delta_x": 0.0,
        "look_rot_delta_y": 0.0,
        "attack": 0.0,
        "shrinking_key": 0.0,
        "delta_x": 0.0,
        "delta_y": 0.0,
        "delta_rot": 0.0
    }

    # Create temporal state (agent's history - let's say 5 timesteps)
    temporal_agents = [agent_features] * 5

    # Create spatial state (current snapshot - agent + target)
    spatial_agents = [agent_features, agent_features]  # Agent and target

    payload = {
        "temporal": {
            "agents": temporal_agents
        },
        "spatial": {
            "agents": spatial_agents
        }
    }

    return payload


def test_prediction():
    """Test the prediction endpoint"""
    try:
        payload = create_test_payload()

        response = requests.post(
            "http://localhost:8000/predict/unity",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            data = response.json()
            predictions = data.get("predictions", [])

            if len(predictions) >= 2:
                print("✓ Prediction successful")
                print(f"  Delta X: {predictions[0]:.6f}")
                print(f"  Delta Y: {predictions[1]:.6f}")
                print(f"  Raw response: {json.dumps(data, indent=2)}")
                return True
            else:
                print(f"✗ Invalid predictions format: {predictions}")
                return False
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return False


def test_window_reset():
    """Test the window reset endpoint"""
    try:
        response = requests.post("http://localhost:8000/window/reset")
        if response.status_code == 200:
            print("✓ Window reset successful")
            return True
        else:
            print(f"✗ Window reset failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Window reset error: {e}")
        return False


def main():
    """Run all tests"""
    print("Testing attention_v1 API server...")
    print("=" * 50)

    # Wait a moment for server to start if needed
    time.sleep(1)

    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Window Reset", test_window_reset),
        ("Prediction", test_prediction),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nTesting: {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! The API server is working correctly.")
    else:
        print("❌ Some tests failed. Please check the server logs.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

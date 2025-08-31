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
        response = requests.get("http://localhost:8001/")
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
        response = requests.get("http://localhost:8001/model/info")
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
        "agent_id": 1,
        "game_time": 10.5,
        "team_index": 0.0,
        "pos_x": 100.5,
        "pos_z": -30.3,
        "rotation": 45.0,
        "move_dir_x": 0.1,
        "move_dir_y": 0.0,
        "look_rot_delta_x": 0.0,
        "look_rot_delta_y": 0.0,
        "attack": 0.0
    }

    target_features = {
        "agent_id": 2,
        "game_time": 10.5,
        "team_index": 1.0,
        "pos_x": 120.5,
        "pos_z": -40.3,
        "rotation": 90.0,
        "move_dir_x": -0.1,
        "move_dir_y": 0.0,
        "look_rot_delta_x": 0.0,
        "look_rot_delta_y": 0.0,
        "attack": 0.0
    }

    # Create timesteps (each timestep has a list of agents)
    timesteps = []
    for i in range(20):  # 5 timesteps
        # Slightly modify positions for each timestep to simulate movement
        agent_copy = agent_features.copy()
        target_copy = target_features.copy()
        agent_copy["pos_x"] += i * 1.0
        target_copy["pos_x"] -= i * 1.0
        agent_copy["game_time"] = 10.5 + i * 0.1
        target_copy["game_time"] = 10.5 + i * 0.1
        timesteps.append({
            "agents": [agent_copy, target_copy]
        })

    payload = {
        "temporal_history": timesteps
    }

    return payload


def test_prediction():
    """Test the prediction endpoint"""
    try:
        payload = create_test_payload()

        # Print the payload structure
        print("\nTest payload structure:")
        print(f"Number of timesteps: {len(payload['temporal_history'])}")
        print(f"Agents per timestep: {len(payload['temporal_history'][0]['agents'])}")
        print(f"First timestep first agent: {json.dumps(payload['temporal_history'][0]['agents'][0], indent=2)}")

        response = requests.post(
            "http://localhost:8001/predict/unity",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        print(f"\nResponse status code: {response.status_code}")
        
        try:
            print(f"Response content: {response.content.decode()}")
        except:
            print(f"Raw response content: {response.content}")

        if response.status_code == 200:
            data = response.json()
            predictions = data.get("predictions", [])

            if len(predictions) >= 2:
                print("\n✓ Prediction successful")
                print(f"  Delta X: {predictions[0]:.6f}")
                print(f"  Delta Y: {predictions[1]:.6f}")
                return True
            else:
                print(f"\n✗ Invalid predictions format: {predictions}")
                return False
        else:
            print(f"\n✗ Prediction failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"\n✗ Prediction error: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return False


def test_window_reset():
    """Test the window reset endpoint"""
    try:
        response = requests.post("http://localhost:8001/window/reset")
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

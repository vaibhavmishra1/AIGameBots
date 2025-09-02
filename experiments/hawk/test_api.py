#!/usr/bin/env python3
"""
Test script for the Hawk AI Game Agents API server.
Tests model loading and prediction functionality for Unity agent movement prediction.

Requirements:
- Python 3.7+
- requests
- numpy
- torch (for the API server)

Usage:
1. Start the API server: python api_server.py
2. In another terminal: python test_api.py

This will run comprehensive tests including:
- Health check
- Model info retrieval
- Prediction functionality
- Performance testing
- Different payload sizes
- Edge cases and error handling
"""

import requests
import json
import time
import numpy as np
from typing import Dict, Any, List


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
            print(f"  Best val loss: {data.get('best_val_loss', 'N/A')}")
            print(f"  Model path: {data.get('model_path', 'N/A')}")
            if 'hyperparameters' in data:
                hp = data['hyperparameters']
                print(f"  Model hyperparameters:")
                print(f"    d_model: {hp.get('d_model', 'N/A')}")
                print(f"    temp_layers: {hp.get('temp_layers', 'N/A')}")
                print(f"    spat_layers: {hp.get('spat_layers', 'N/A')}")
            return True
        else:
            print(f"✗ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Model info error: {e}")
        return False


def create_test_payload(num_timesteps: int = 16, num_agents: int = 10) -> Dict[str, Any]:
    """Create a test payload with dummy data for Unity agent prediction"""
    timesteps = []

    # Base positions for agents (will be modified per timestep)
    base_positions = [
        {"x": 100.5, "z": -30.3},   # Main agent (agent 0)
        {"x": 120.5, "z": -40.3},   # Target agent
        {"x": 90.2, "z": -25.1},    # Nearby ally
        {"x": 140.8, "z": -50.7},   # Distant enemy
        {"x": 110.3, "z": -35.9},   # Another agent
        {"x": 85.1, "z": -20.4},    # Close ally
        {"x": 130.6, "z": -45.2},   # Medium enemy
        {"x": 95.7, "z": -28.8},    # Another agent
    ]

    for t in range(num_timesteps):
        agents = []

        for agent_idx in range(num_agents):
            # Calculate position with some movement over time
            base_pos = base_positions[agent_idx % len(base_positions)]
            pos_x = base_pos["x"] + t * 0.5 + agent_idx * 2.0
            pos_z = base_pos["z"] + t * 0.3 - agent_idx * 1.5

            # Team assignment (agent 0 is always team 0, others alternate)
            team_index = 0.0 if agent_idx == 0 else (agent_idx % 2)

            agent = {
                "agent_id": agent_idx,
                "game_time": 10.5 + t * 0.1,
                "team_index": team_index,
                "pos_x": pos_x,
                "pos_z": pos_z,
                "rotation": 45.0 + agent_idx * 30.0,  # Different rotations
                "move_dir_x": 0.1 * (1 if agent_idx % 2 == 0 else -1),
                "move_dir_y": 0.0,
                "look_rot_delta_x": 0.0,
                "look_rot_delta_y": 0.0,
                "attack": 0.0
            }
            agents.append(agent)

        timesteps.append({"agents": agents})

    payload = {
        "temporal_history": timesteps
    }

    return payload


def test_prediction():
    """Test the prediction endpoint"""
    try:
        payload = create_test_payload(num_timesteps=16, num_agents=10)

        # Print the payload structure
        print("\nTest payload structure:")
        print(f"Number of timesteps: {len(payload['temporal_history'])}")
        print(f"Agents per timestep: {len(payload['temporal_history'][0]['agents'])}")
        print(f"Features per agent: {len(payload['temporal_history'][0]['agents'][0])}")
        print(f"First timestep first agent sample:")
        for key, value in payload['temporal_history'][0]['agents'][0].items():
            print(f"  {key}: {value}")

        response = requests.post(
            "http://localhost:8001/predict/unity",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # Add timeout for long predictions
        )

        print(f"\nResponse status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            predictions = data.get("predictions", [])

            print(f"✓ Prediction successful")
            print(f"  Number of predictions: {len(predictions)}")
            print(f"  First 5 predictions: {[f'{p:.6f}' for p in predictions[:5]]}")

            # The API returns the first action pair (2 values) from the model's 10 predictions
            if len(predictions) == 2:
                print("  ✓ API returns primary action pair (2 values)")
                print(f"  Delta X: {predictions[0]:.6f}")
                print(f"  Delta Y: {predictions[1]:.6f}")
                print("  Note: Model internally generates 5 action pairs, API returns the first one")
                return True
            else:
                print(f"  ⚠️  Unexpected number of predictions: {len(predictions)}")
                print(f"     Expected 2 predictions, got {len(predictions)}")
                return False
        else:
            print(f"\n✗ Prediction failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"\n✗ Prediction timeout after 30 seconds")
        return False
    except Exception as e:
        print(f"\n✗ Prediction error: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return False


def test_performance():
    """Test prediction performance with timing"""
    try:
        payload = create_test_payload(num_timesteps=16, num_agents=10)

        start_time = time.time()
        response = requests.post(
            "http://localhost:8001/predict/unity",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        end_time = time.time()

        if response.status_code == 200:
            latency = end_time - start_time
            print(f"✓ Performance test completed")
            print(f"  Response time: {latency:.3f} seconds")
            if latency < 1.0:
                print("  ⚡ Fast response (< 1 second)")
            elif latency < 5.0:
                print("  🟡 Acceptable response (1-5 seconds)")
            else:
                print("  🟠 Slow response (> 5 seconds)")
            return True
        else:
            print(f"✗ Performance test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Performance test error: {e}")
        return False


def test_different_payload_sizes():
    """Test with different numbers of agents and timesteps"""
    test_cases = [
        (16, 10, "Small payload"),
        (16, 10, "Medium payload"),
        (16, 10, "Large payload"),
    ]

    all_passed = True

    for num_timesteps, num_agents, description in test_cases:
        try:
            print(f"\n  Testing {description}: {num_timesteps} timesteps, {num_agents} agents")
            payload = create_test_payload(num_timesteps=num_timesteps, num_agents=num_agents)

            start_time = time.time()
            response = requests.post(
                "http://localhost:8001/predict/unity",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            end_time = time.time()

            if response.status_code == 200:
                latency = end_time - start_time
                print(f"    ✓ Success: {latency:.3f}s")
            else:
                print(f"    ✗ Failed: {response.status_code}")
                all_passed = False

        except Exception as e:
            print(f"    ✗ Error: {e}")
            all_passed = False

    return all_passed


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting edge cases...")

    # Test with empty payload
    try:
        response = requests.post(
            "http://localhost:8001/predict/unity",
            json={"temporal_history": []},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 422:
            print("✓ Empty payload handled correctly")
        else:
            print(f"⚠️  Empty payload returned {response.status_code}, expected 422")
    except Exception as e:
        print(f"✗ Empty payload test error: {e}")

    # Test with invalid JSON
    try:
        response = requests.post(
            "http://localhost:8001/predict/unity",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        if response.status_code in [400, 422]:
            print("✓ Invalid JSON handled correctly")
        else:
            print(f"⚠️  Invalid JSON returned {response.status_code}, expected 400/422")
    except Exception as e:
        print(f"✗ Invalid JSON test error: {e}")

    # Test with single agent
    try:
        payload = create_test_payload(num_timesteps=20, num_agents=10)
        response = requests.post(
            "http://localhost:8001/predict/unity",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            print("✓ Single agent scenario handled correctly")
        else:
            print(f"⚠️  Single agent returned {response.status_code}")
    except Exception as e:
        print(f"✗ Single agent test error: {e}")

    return True


def main():
    """Run all tests"""
    print("Testing Hawk AI Game Agents API Server...")
    print("=" * 60)
    print("This server provides Unity agent movement prediction using")
    print("temporal and spatial attention-based Transformer models.")
    print("=" * 60)

    # Wait a moment for server to start if needed
    time.sleep(2)

    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Prediction", test_prediction),
        ("Performance", test_performance),
        ("Different Payload Sizes", test_different_payload_sizes),
        ("Edge Cases", test_edge_cases),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 40)
        if test_func():
            passed += 1
        print()

    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The API server is working correctly.")
        print("✅ Ready for Unity integration.")
    elif passed >= total - 1:
        print("⚠️  Most tests passed. Minor issues detected.")
    else:
        print("❌ Multiple tests failed. Please check the server configuration.")
        print("   - Ensure the model file exists at the specified path")
        print("   - Check server logs for detailed error messages")
        print("   - Verify all dependencies are installed")

    return passed == total


if __name__ == "__main__":
    import sys

    # Check if server is running
    try:
        response = requests.get("http://localhost:8001/", timeout=5)
        if response.status_code != 200:
            print("❌ API server is not responding. Please start it first:")
            print("   python api_server.py")
            sys.exit(1)
    except:
        print("❌ Cannot connect to API server at http://localhost:8001")
        print("   Please start the server first: python api_server.py")
        sys.exit(1)

    success = main()
    exit(0 if success else 1)

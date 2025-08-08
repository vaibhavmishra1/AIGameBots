import argparse
import time
from typing import List, Dict

import numpy as np
import requests


def generate_random_state(rng: np.random.Generator) -> Dict:
    # Generate a plausible random UnityState payload
    def vec3(scale=1.0):
        v = rng.normal(0.0, scale, size=(3,)).astype(np.float32)
        return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}

    state = {
        "agentPosition": vec3(0.1),
        "agentRotation": {"x": 0.0, "y": float(rng.uniform(-1.0, 1.0)), "z": 0.0},
        "agentForward": vec3(1.0),
        "health": float(np.clip(rng.uniform(0.0, 1.0), 0.0, 1.0)),
        "weapon": float(rng.integers(0, 5)),
        "targetPosition": vec3(0.1),
        "targetRotation": {"x": 0.0, "y": float(rng.uniform(-1.0, 1.0)), "z": 0.0},
        "targetForward": vec3(1.0),
        "directionToTarget": vec3(1.0),
        "cross": vec3(1.0),
        "distance": float(np.abs(rng.normal(0.0, 0.1))),
        "dotProduct": float(np.clip(rng.uniform(-1.0, 1.0), -1.0, 1.0)),
        "islos": bool(rng.integers(0, 2)),
    }
    return state


def load_states_from_npy(path: str) -> List[Dict]:
    """Load (N x 18) numeric features and wrap into UnityState-shaped dicts.

    This lets you reuse existing numeric datasets by mapping columns to fields.
    Adjust mapping if needed.
    """
    arr = np.load(path)
    if arr.ndim == 1 and arr.size == 360:
        arr = arr.reshape(20, 18)
    if arr.ndim != 2 or arr.shape[1] != 18:
        raise ValueError(f"Expected (N, 18), got {arr.shape}")

    def row_to_state(row: np.ndarray) -> Dict:
        # Minimal mapping to satisfy server's 18-d extraction
        return {
            "agentPosition": {"x": float(row[0]), "y": float(row[1]), "z": float(row[2])},
            "agentRotation": {"x": 0.0, "y": float(row[3]), "z": 0.0},
            "agentForward": {"x": float(row[4]), "y": float(row[5]), "z": float(row[6])},
            "health": float(row[7]),
            "weapon": float(row[8]),
            "islos": bool(row[9] > 0.5),
            "targetPosition": {"x": float(row[10]), "y": float(row[11]), "z": float(row[12])},
            "targetRotation": {"x": 0.0, "y": float(row[13]), "z": 0.0},
            "targetForward": {"x": 0.0, "y": 0.0, "z": 0.0},
            "directionToTarget": {"x": float(row[14]), "y": float(row[15]), "z": float(row[16])},
            "cross": {"x": 0.0, "y": 0.0, "z": 0.0},
            "distance": float(row[17]),
            "dotProduct": 0.0,
        }

    return [row_to_state(arr[i].astype(np.float32)) for i in range(arr.shape[0])]


def main():
    parser = argparse.ArgumentParser(description="Streaming test client for ann2 dual-regression API (single 18-d step)")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API server")
    parser.add_argument("--endpoint", default="/predict/unity", help="Endpoint path to test")
    parser.add_argument("--npy", default=None, help="Path to .npy file with rows of 18 features (N x 18 or flat 360)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generated features")
    parser.add_argument("--repeat", type=int, default=40, help="Number of steps to send if generating randomly")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep seconds between requests")
    parser.add_argument("--reset", action="store_true", help="Reset the server-side feature window before streaming")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    post_url = base + args.endpoint
    reset_url = base + "/window/reset"

    if args.reset:
        try:
            r = requests.post(reset_url, timeout=5)
            r.raise_for_status()
            print(f"Reset window: {r.json()}")
        except Exception as e:
            print(f"Failed to reset window: {e}")

    if args.npy:
        states = load_states_from_npy(args.npy)
    else:
        rng = np.random.default_rng(args.seed)
        states = [generate_random_state(rng) for _ in range(args.repeat)]

    print(f"POST {post_url}")
    for i, state in enumerate(states, 1):
        payload = {"state": state}
        try:
            response = requests.post(post_url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"[{i}/{len(states)}] preds = {data.get('predictions')}")
        except Exception as e:
            print(f"Request {i} failed: {e}")
        if i < len(states):
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()



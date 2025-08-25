import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_actions(h5_path: str, group_name: str = "processed", max_samples: int = 0) -> np.ndarray:
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    all_actions = []
    with h5py.File(h5_path, "r") as f:
        if group_name not in f:
            raise KeyError(f"Group '{group_name}' not found in {h5_path}")
        grp = f[group_name]
        keys = sorted(list(grp.keys()))
        if max_samples and max_samples > 0:
            keys = keys[:max_samples]
        for key in tqdm(keys, desc="Loading actions"):
            acts = grp[key]["actions"][()]
            all_actions.append(np.asarray(acts, dtype=np.float32))

    return np.asarray(all_actions, dtype=np.float32)  # (N, 3)


def plot_hist(actions: np.ndarray, out_path: str = "actions_hist.png") -> None:
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError(f"Expected actions of shape (N, 3), got {actions.shape}")

    titles = ["delta_x (-3..3)", "delta_y (-3..3)", "delta_rot (-30..30)"]
    ranges = [(-3, 3), (-3, 3), (-30, 30)]
    bins = [100, 100, 120]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        vals = actions[:, i]
        axs[i].hist(vals, bins=bins[i], range=ranges[i], color="#1f77b4", alpha=0.85)
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("value")
        axs[i].set_ylabel("count")
        axs[i].grid(True, linestyle=":", alpha=0.4)
        print(f"{titles[i]} -> min={vals.min():.4f} max={vals.max():.4f} mean={vals.mean():.4f} std={vals.std():.4f}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved histogram to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot histogram of actions from processed HDF5")
    parser.add_argument("--h5_path", type=str, default="/Users/vaibhav/Desktop/processed_game_logs.h5")
    parser.add_argument("--group", type=str, default="processed")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional cap on number of samples (0 = all)")
    parser.add_argument("--out", type=str, default="actions_hist.png")
    args = parser.parse_args()

    actions = load_actions(args.h5_path, args.group, args.max_samples)
    print(f"Loaded actions: {actions.shape}")
    plot_hist(actions, args.out)


if __name__ == "__main__":
    main()



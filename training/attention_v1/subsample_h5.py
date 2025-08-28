import argparse
import os
import random

import h5py
from tqdm import tqdm


def subsample_h5(src_path: str, out_path: str, group_name: str, num_samples: int, seed: int = 42, min_xy_sq: float = 0.2) -> None:
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Source HDF5 not found: {src_path}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with h5py.File(src_path, "r") as src:
        if group_name not in src:
            raise KeyError(f"Group '{group_name}' not found in {src_path}")
        src_group = src[group_name]
        keys = list(src_group.keys())
        random.seed(seed)
        random.shuffle(keys)
        total = len(keys)
        if total == 0:
            raise RuntimeError("No samples found in source group.")

        random.seed(seed)
        # Filter keys by movement magnitude threshold: dx^2 + dy^2 > min_xy_sq
        filtered_keys = []
        for key in tqdm(keys, desc="Filtering samples"):
            try:
                acts = src_group[key]["actions"][()]
                dx = float(acts[0])
                dy = float(acts[1])
                if (dx * dx + dy * dy) > 0.1 and (dx * dx + dy * dy) < 0.2:
                    filtered_keys.append(key)
                if len(filtered_keys) >= num_samples:
                    break
            except Exception:
                # Skip malformed samples
                continue

        available = len(filtered_keys)
        if available == 0:
            raise RuntimeError("No samples satisfy the movement threshold.")
        random.seed(seed)
        if num_samples >= available:
            chosen = filtered_keys
        else:
            chosen = random.sample(filtered_keys, num_samples)

        if os.path.isfile(out_path):
            os.remove(out_path)

        with h5py.File(out_path, "w") as dst:
            dst_group = dst.create_group(group_name)
            for key in tqdm(chosen, desc=f"Writing {os.path.basename(out_path)}"):
                # Copy the entire sample group as-is
                src_sample = src_group[key]
                dst_group.copy(src_sample, key)

    print(f"Created {out_path} with {len(chosen)} samples (from {total}, after filter {available}).")


def parse_counts(counts_str: str):
    parts = counts_str.split(",")
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def main():
    parser = argparse.ArgumentParser(description="Subsample a processed HDF5 into smaller files")
    parser.add_argument("--src", type=str, default="/Users/vaibhav/Desktop/processed_game_logs_attention_1.h5")
    parser.add_argument("--out_dir", type=str, default="/Users/vaibhav/Desktop")
    parser.add_argument("--group", type=str, default="processed")
    parser.add_argument("--counts", type=str, default="1000,10000,100000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_xy_sq", type=float, default=0.2, help="Keep samples with dx^2+dy^2 > threshold")
    args = parser.parse_args()

    counts = parse_counts(args.counts)
    base = os.path.splitext(os.path.basename(args.src))[0]

    for n in counts:
        out_path = os.path.join(args.out_dir, f"{base}_sub_magnitude_0_2_{n}.h5")
        subsample_h5(args.src, out_path, args.group, n, seed=args.seed, min_xy_sq=args.min_xy_sq)


if __name__ == "__main__":
    main()


"""
python3 subsample_h5.py \
  --src "/Users/vaibhav/Desktop/processed_game_logs_attention_1.h5" \
  --out_dir "/Users/vaibhav/Desktop" \
  --group processed \
  --counts "1000,10000,100000" \
  --min_xy_sq 0.1
  """
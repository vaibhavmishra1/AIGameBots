import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessedH5Dataset(Dataset):
    """
    Dataset to read processed features/actions from a consolidated HDF5 file
    created by export_processed_to_hdf5.
    """

    def __init__(self, h5_path, group_name="processed", return_numpy=False):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"Processed HDF5 not found: {h5_path}")
        self.h5_path = h5_path
        self.group_name = group_name
        self.return_numpy = return_numpy

        with h5py.File(self.h5_path, 'r') as f:
            if self.group_name not in f:
                raise KeyError(f"Group '{self.group_name}' not found in {self.h5_path}")
            grp = f[self.group_name]
            # Store sample keys deterministically
            self.sample_keys = sorted(list(grp.keys()))

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        key = self.sample_keys[idx]
        with h5py.File(self.h5_path, 'r') as f:
            grp = f[self.group_name][key]
            # Read temporal/spatial/actions written by export_processed_to_hdf5
            temporal = grp['temporal'][()]
            spatial = grp['spatial'][()]
            actions = grp['actions'][()]

        if self.return_numpy:
            return temporal, spatial, actions
        return (
            torch.from_numpy(np.asarray(temporal, dtype=np.float32)),
            torch.from_numpy(np.asarray(spatial, dtype=np.float32)),
            torch.from_numpy(np.asarray(actions, dtype=np.float32)),
        )


def visualize_distributions(h5_path: str, group_name: str = "processed", sample_cap: int = 100000, out_dir: str = "training/attention_v1") -> None:
    import matplotlib.pyplot as plt

    ds = ProcessedH5Dataset(h5_path, group_name=group_name, return_numpy=True)
    total = len(ds)
    if total == 0:
        print("Dataset is empty; nothing to visualize.")
        return

    feature_names = [
        'team_index', 'rel_pos_x', 'rel_pos_z', 'rotation',
        'move_dir_x', 'move_dir_y', 'look_rot_delta_x', 'look_rot_delta_y',
        'attack', 'shrinking_key', 'delta_x', 'delta_y', 'delta_rot'
    ]

    temporal_values = [[] for _ in range(13)]
    spatial_values = [[] for _ in range(13)]
    action_values = [[] for _ in range(3)]

    indices = np.random.choice(total, min(sample_cap, total), replace=False)
    skipped = 0
    taken = 0
    for i in indices:
        temporal, spatial, actions = ds[i]
        # Low-motion filter (same thresholds as dataset.py)
        try:
            dx, dy, drot = float(actions[0]), float(actions[1]), float(actions[2])
            if abs(dx) < 0.05 and abs(dy) < 0.05 and abs(drot) < 0.5:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        taken += 1
        if temporal.size > 0:
            tf = temporal.reshape(-1, temporal.shape[-1])
            for j in range(13):
                temporal_values[j].extend(tf[:, j])
        if spatial.size > 0:
            sf = spatial.reshape(-1, spatial.shape[-1])
            for j in range(13):
                spatial_values[j].extend(sf[:, j])
        for k in range(3):
            action_values[k].append(actions[k])

    print(f"Collected {taken} samples, skipped {skipped} (low-motion filter)")

    # Plot helpers
    def plot_feature_grid(values_by_feat, title, out_name):
        fig, axs = plt.subplots(5, 3, figsize=(15, 20))
        fig.suptitle(title)
        axs = axs.flatten()
        for i in range(13):
            vals = np.array(values_by_feat[i])
            vals = vals[~np.isinf(vals) & ~np.isnan(vals)]
            if vals.size == 0:
                axs[i].set_title(f"{feature_names[i]} (no data)")
                axs[i].axis('off')
                continue
            hist, bins = np.histogram(vals, bins=50)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            axs[i].plot(bin_centers, hist, '-', linewidth=2)
            axs[i].set_title(feature_names[i])
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Frequency')
            axs[i].grid(True)
            print(f"{title} | {feature_names[i]}: min={vals.min():.3f} max={vals.max():.3f} mean={vals.mean():.3f} std={vals.std():.3f}")
        for i in range(13, len(axs)):
            fig.delaxes(axs[i])
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_name)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

    plot_feature_grid(temporal_values, 'Processed Temporal Feature Distributions', 'temporal_feature_distributions_processed.png')
    plot_feature_grid(spatial_values, 'Processed Spatial Feature Distributions', 'spatial_feature_distributions_processed.png')

    # Actions
    act_titles = ['delta_x', 'delta_y', 'delta_rot']
    act_ranges = [(-3, 3), (-3, 3), (-30, 30)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        vals = np.array(action_values[i])
        vals = vals[~np.isinf(vals) & ~np.isnan(vals)]
        if vals.size == 0:
            axs[i].set_title(f"{act_titles[i]} (no data)")
            axs[i].axis('off')
            continue
        axs[i].hist(vals, bins=100, range=act_ranges[i], color='#1f77b4', alpha=0.85)
        axs[i].set_title(act_titles[i])
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Count')
        axs[i].grid(True, linestyle=':', alpha=0.4)
        print(f"Actions | {act_titles[i]}: min={vals.min():.3f} max={vals.max():.3f} mean={vals.mean():.3f} std={vals.std():.3f}")
    os.makedirs(out_dir, exist_ok=True)
    out_act = os.path.join(out_dir, 'action_distributions_processed.png')
    plt.tight_layout()
    plt.savefig(out_act)
    plt.close()
    print(f"Saved {out_act}")


def main():
    h5_path = "/Users/vaibhav/Desktop/processed_game_logs_attention_1.h5"
    agents_dataset = ProcessedH5Dataset(h5_path, group_name="processed")
    (temporal, spatial, actions )= agents_dataset[0]
    print(temporal.shape)
    print(spatial.shape)
    print(actions.shape)
    # visualize_distributions(h5_path, group_name="processed", sample_cap=100000)


if __name__ == "__main__":
    main()



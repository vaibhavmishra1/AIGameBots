import torch 
import os
from torch.utils.data import Dataset
import h5py
import time 
import numpy as np
from tqdm import tqdm
class AgentControlDataset(Dataset):
    def __init__(self, dataset_path, slice_fn=None, return_numpy=False, debug=False):
        self.dataset_path = dataset_path
        # Ensure deterministic ordering of files
        self.all_h5_files = sorted(
            [os.path.join(self.dataset_path, filename) for filename in os.listdir(self.dataset_path)]
        )
        self.total_samples = self.read_all_hdf5_file()
        self._files = {}
        self._pid = os.getpid()
        self.slice_fn = slice_fn
        self.return_numpy = return_numpy
        self.debug = debug
        self.shrinking_area_centers = self.get_shrinking_area_centers()
        self.min_feature_values = [float('inf')] * 13
        self.max_feature_values = [float('-inf')] * 13

    def update_min_max_feature_values(self, feature_vector):
        """Update the running minimum and maximum values for each feature.
        
        Args:
            feature_vector: numpy array of shape (13,) containing the current feature values
        """
        for i in range(len(feature_vector)):
            if not np.isnan(feature_vector[i]) and not np.isinf(feature_vector[i]):
                self.min_feature_values[i] = min(self.min_feature_values[i], feature_vector[i])
                self.max_feature_values[i] = max(self.max_feature_values[i], feature_vector[i])
    def get_min_max_feature_values(self):
        """Plot temporal, spatial feature distributions and action histograms.

        - Temporal: per-timestep features for agent 0 (shape (T-1, 13))
        - Spatial: per-agent features at T-2 (shape (A, 13))
        - Actions: 3-dim label
        """
        import matplotlib.pyplot as plt
        import numpy as np

        feature_names = [
            'team_index', 'rel_pos_x', 'rel_pos_z', 'rotation',
            'move_dir_x', 'move_dir_y', 'look_rot_delta_x', 'look_rot_delta_y',
            'attack', 'shrinking_key', 'delta_x', 'delta_y', 'delta_rot'
        ]

        temporal_values = [[] for _ in range(13)]
        spatial_values = [[] for _ in range(13)]
        action_values = [[] for _ in range(3)]

        print("Collecting temporal/spatial/action values from random samples...")
        total_samples = len(self)
        sample_cap = min(100000, total_samples)
        if sample_cap == 0:
            print("Dataset is empty; skipping plots.")
            return
        random_indices = np.random.choice(total_samples, sample_cap, replace=False)

        skipped = 0
        taken = 0
        for idx in tqdm(random_indices):
            temporal, spatial, actions = self[idx]
            if isinstance(temporal, torch.Tensor):
                temporal = temporal.numpy()
            if isinstance(spatial, torch.Tensor):
                spatial = spatial.numpy()
            if isinstance(actions, torch.Tensor):
                actions = actions.numpy()

            # Apply low-motion rejection: |dx|<0.05 & |dy|<0.05 & |drot|<0.5
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
            if actions.size > 0:
                for k in range(3):
                    action_values[k].append(actions[k])
        print(f"Collected {taken} samples, skipped {skipped} (low-motion filter)")

        # Helper to plot 13 features into a 5x3 grid
        def plot_feature_grid(values_by_feat, title, out_path):
            fig, axs = plt.subplots(5, 3, figsize=(15, 20))
            fig.suptitle(title)
            axs = axs.flatten()
            for i in range(13):
                vals = np.array(values_by_feat[i])
                vals = vals[~np.isinf(vals) & ~np.isnan(vals)]
                if vals.size == 0:
                    axs[i].set_title(f'{feature_names[i]} (no data)')
                    axs[i].axis('off')
                    continue
                hist, bins = np.histogram(vals, bins=50)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                axs[i].plot(bin_centers, hist, '-', linewidth=2)
                axs[i].set_title(f'{feature_names[i]}')
                axs[i].set_xlabel('Value')
                axs[i].set_ylabel('Frequency')
                axs[i].grid(True)
                print(f"{title} | {feature_names[i]}: min={vals.min():.3f} max={vals.max():.3f} mean={vals.mean():.3f} std={vals.std():.3f}")
            for i in range(13, len(axs)):
                fig.delaxes(axs[i])
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            print(f"Saved {out_path}")

        plot_feature_grid(temporal_values, 'Temporal Feature Distributions', 'temporal_feature_distributions.png')
        plot_feature_grid(spatial_values, 'Spatial Feature Distributions', 'spatial_feature_distributions.png')

        # Actions histogram
        act_titles = ['delta_x', 'delta_y', 'delta_rot']
        act_ranges = [(-3, 3), (-3, 3), (-30, 30)]
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for i in range(3):
            vals = np.array(action_values[i])
            vals = vals[~np.isinf(vals) & ~np.isnan(vals)]
            if vals.size == 0:
                axs[i].set_title(f'{act_titles[i]} (no data)')
                axs[i].axis('off')
                continue
            axs[i].hist(vals, bins=100, range=act_ranges[i], color='#1f77b4', alpha=0.85)
            axs[i].set_title(act_titles[i])
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Count')
            axs[i].grid(True, linestyle=':', alpha=0.4)
            print(f"Actions | {act_titles[i]}: min={vals.min():.3f} max={vals.max():.3f} mean={vals.mean():.3f} std={vals.std():.3f}")
        plt.tight_layout()
        plt.savefig('action_distributions.png')
        plt.close()
        print("Saved action_distributions.png")
        
    def get_shrinking_area_centers(self):
        shrinking_area_centers = {}
        shrinking_area_centers[1] = {"x" : -365.63, "z" : -139.57}
        shrinking_area_centers[2] = {"x" : -368.84, "z" : 7.72}
        shrinking_area_centers[3] = {"x" : 94.97, "z" : 152.24}
        shrinking_area_centers[4] = {"x" : 363.16, "z" : -107.33}
        shrinking_area_centers[5] = {"x" : -132.66, "z" : -183.18}
        shrinking_area_centers[6] = {"x" : 198.90, "z" : -212.00}
        shrinking_area_centers[7] = {"x" : 196.30, "z" : 532.90}
        return shrinking_area_centers

    def __len__(self):
        return len(self.total_samples)

    def _get_file(self, file_name):
        # Ensure we don't share file handles across DataLoader worker processes
        current_pid = os.getpid()
        if current_pid != self._pid:
            # Process changed (e.g., after fork); close old handles and reset cache
            for f in self._files.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._files = {}
            self._pid = current_pid
        f = self._files.get(file_name)
        if f is None or not f.id.valid:
            try:
                f = h5py.File(file_name, 'r', swmr=True)
            except Exception:
                f = h5py.File(file_name, 'r')
            self._files[file_name] = f
        return f

    def calculate_expected_shrinking_area_center(self, data):
        """
        Calculate the expected shrinking area center based on agent positions.

        Args:
            data: numpy array or list-like, shape (timesteps, agents, features)

        Returns:
            tuple: (x, z, key) of the closest shrinking area center. If unavailable, returns (0.0, 0.0, 0)
        """
        total_x, total_z, count = 0.0, 0.0, 0

        for timestep in data:
            for agent_data in timestep:
                # agent_data[4] = pos_x, agent_data[6] = pos_z
                # Guard against short feature vectors
                if agent_data is None:
                    continue
                try:
                    has_min_len = len(agent_data) > 6
                except Exception:
                    has_min_len = False
                if not has_min_len:
                    continue
                pos_x = agent_data[4]
                pos_z = agent_data[6]
                if pos_x != 0.0 and pos_z != 0.0:
                    total_x += pos_x
                    total_z += pos_z
                    count += 1

        if count == 0:
            return (0.0, 0.0, 0)

        avg_x = total_x / count
        avg_z = total_z / count

        min_dist = float('inf')
        closest_key = None
        for key, center in self.shrinking_area_centers.items():
            dx = avg_x - center['x']
            dz = avg_z - center['z']
            dist = dx * dx + dz * dz
            if dist < min_dist:
                min_dist = dist
                closest_key = key

        if closest_key is not None:
            center = self.shrinking_area_centers[closest_key]
            return (center['x'], center['z'], closest_key)
        else:
            return (0.0, 0.0, 0)


    def preprocess_data(self, data):
        if self.debug:
            print("preprocessing data --- ", data.shape)

        # Determine dynamic shapes: use all but the last timestep for features (to compute deltas)
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim != 3:
            # Unexpected shape; return empty matrices with consistent output signature
            if self.debug:
                print("Unexpected data.ndim:", data.ndim)
            empty_temporal = np.zeros((0, 13), dtype=np.float32)
            empty_spatial = np.zeros((0, 13), dtype=np.float32)
            zero_actions = np.zeros((3,), dtype=np.float32)
            return empty_temporal, empty_spatial, zero_actions

        timesteps_total, num_agents, feature_dim = data.shape
        if timesteps_total < 1 or num_agents < 1:
            empty_temporal = np.zeros((0, 13), dtype=np.float32)
            empty_spatial = np.zeros((max(num_agents, 0), 13), dtype=np.float32)
            zero_actions = np.zeros((3,), dtype=np.float32)
            return empty_temporal, empty_spatial, zero_actions

        (zonecenterx, zonecenterz, shr_key) = self.calculate_expected_shrinking_area_center(data)
        # Normalize shrinking key ONCE; do not mutate in the loops
        shrinking_key_norm = float(shr_key) / 7.0 if shr_key != 0 else 0.0

        # We now construct two matrices:
        # 1) temporal features for agent 0 across ALL timesteps: (T, F)
        # 2) spatial features for ALL agents at the LAST timestep: (A, F)
        feature_count = 13
        target_timesteps = max(0, timesteps_total - 1)
        new_temporal_data = np.zeros((target_timesteps, feature_count), dtype=np.float32)
        new_spatial_data = np.zeros((num_agents, feature_count), dtype=np.float32)
        actions = np.zeros((3,), dtype=np.float32)

        for i in range(timesteps_total):
            is_last = (i == timesteps_total - 1)
            spatial_time_index = max(0, timesteps_total - 2)
            if is_last:
                # No target slot for the last timestep; only used to compute deltas if needed
                pass

            for j in range(num_agents):
                feature = data[i][j]
                if np.all(feature == 0):
                    # If last timestep and agent 0 is all zeros, still keep actions as zeros
                    continue

                prev_feature = data[i][j] if i == 0 else data[i - 1][j]

                # Guard feature length; require indices up to 34
                if feature_dim <= 34:
                    # Not enough features; skip to avoid index errors
                    continue

                team_index = feature[3]
                posx = feature[4]
                posz = feature[6]
                rot_deg = feature[7]
                move_direction_x = feature[30]
                move_direction_y = feature[31]
                lookRotationDelta_x = feature[32]
                lookRotationDelta_y = feature[33]
                attack = feature[34]

                deltax = posx - prev_feature[4]
                deltay = posz - prev_feature[6]
                delta_rot = rot_deg - prev_feature[7]
                rel_pos_x = (posx - zonecenterx) 
                rel_pos_x = rel_pos_x if abs(rel_pos_x) < 50 else 50 * np.sign(rel_pos_x)
                rel_pos_x = rel_pos_x / 50
                rel_pos_z = (posz - zonecenterz) 
                rel_pos_z = rel_pos_z if abs(rel_pos_z) < 50 else 50 * np.sign(rel_pos_z)
                rel_pos_z = rel_pos_z / 50
                rot = rot_deg / 360
                team_index = (team_index) / 2
                lookRotationDelta_x = lookRotationDelta_x if abs(lookRotationDelta_x) < 3 else 3 * np.sign(lookRotationDelta_x)
                lookRotationDelta_x = lookRotationDelta_x / 3
                lookRotationDelta_y = lookRotationDelta_y if abs(lookRotationDelta_y) < 3 else 3 * np.sign(lookRotationDelta_y)
                lookRotationDelta_y = lookRotationDelta_y / 3
                # Use precomputed normalized shrinking key
                shr_key = shrinking_key_norm
                deltax = deltax if abs(deltax) < 1 else 1 * np.sign(deltax)
                deltax = deltax / 1
                deltay = deltay if abs(deltay) < 1 else 1 * np.sign(deltay)
                deltay = deltay / 1
                delta_rot = delta_rot if abs(delta_rot) < 30 else 30 * np.sign(delta_rot)
                delta_rot = delta_rot / 30
                
                feat_vec = np.array([
                    team_index,
                    rel_pos_x,
                    rel_pos_z,
                    rot,
                    move_direction_x,
                    move_direction_y,
                    lookRotationDelta_x,
                    lookRotationDelta_y,
                    attack,
                    shr_key,
                    deltax,
                    deltay,
                    delta_rot,
                ], dtype=np.float32)
                self.update_min_max_feature_values(feat_vec)
                # Fill temporal matrix for agent 0 for timesteps [0..T-2]
                if j == 0 and i < target_timesteps:
                    new_temporal_data[i] = feat_vec
                # Fill spatial matrix for all agents at timestep T-2 (second last)
                if i == spatial_time_index:
                    new_spatial_data[j] = feat_vec
                    # Actions will be computed after the loop using T-1 and T-2 for agent 0

        # Compute actions from the last (T-1) and previous (T-2) timesteps for agent 0
        if timesteps_total >= 2:
            last = data[timesteps_total - 1][0]
            prev = data[timesteps_total - 2][0]
            ax = last[4] - prev[4]
            az = last[6] - prev[6]
            arot = last[7] - prev[7]
            ax = ax if abs(ax) < 1 else 1 * np.sign(ax)
            az = az if abs(az) < 1 else 1 * np.sign(az)
            arot = arot if abs(arot) < 30 else 30 * np.sign(arot)
            actions[0] = ax
            actions[1] = az
            actions[2] = arot
                        

        return new_temporal_data, new_spatial_data, actions
        
    def __getitem__(self, idx):
        file_name, group_name, dataset_name = self.total_samples[idx]
        f = self._get_file(file_name)
        dset = f[group_name][dataset_name]

        if self.slice_fn is not None:
            data = self.slice_fn(dset)
        else:
            data = dset[()]  # load the dataset array

        temporal, spatial, actions = self.preprocess_data(data)

        if self.return_numpy:
            return temporal, spatial, actions
        return (
            torch.from_numpy(np.asarray(temporal, dtype=np.float32)),
            torch.from_numpy(np.asarray(spatial, dtype=np.float32)),
            torch.from_numpy(np.asarray(actions, dtype=np.float32)),
        )

    def read_all_hdf5_file(self):
        total_samples = []
        for h5_file in self.all_h5_files:
            if not h5_file.endswith('.h5'):
                continue
            try:
                with h5py.File(h5_file, 'r') as f:
                    for group_name in f:
                        group = f[group_name]
                        for dataset_name in group:
                            total_samples.append((h5_file, group_name, dataset_name))
            except:
                print(f"Error reading file: {h5_file}")
            
        
        return total_samples

    def __del__(self):
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass


def export_processed_to_hdf5(dataset, output_h5_path, group_name="processed", compression="gzip", compression_level=4):
    """
    Export processed features and actions from an AgentControlDataset into a single HDF5 file.
    Each sample is stored under a subgroup of `group_name` named 'sample_XXXXXXXX' with datasets
    'features' and 'actions'.
    If the target group already exists, it is replaced.
    """
    parent_dir = os.path.dirname(output_h5_path)
    if parent_dir and not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    with h5py.File(output_h5_path, 'a') as out_f:
        if group_name in out_f:
            del out_f[group_name]
        grp = out_f.create_group(group_name)

        skipped = 0
        written = 0
        for index in tqdm(range(len(dataset)), desc="Exporting samples to HDF5"):
            temporal, spatial, actions = dataset[index]
            if isinstance(temporal, torch.Tensor):
                temporal_np = temporal.cpu().numpy()
            else:
                temporal_np = np.asarray(temporal, dtype=np.float32)
            if isinstance(spatial, torch.Tensor):
                spatial_np = spatial.cpu().numpy()
            else:
                spatial_np = np.asarray(spatial, dtype=np.float32)
            if isinstance(actions, torch.Tensor):
                actions_np = actions.cpu().numpy()
            else:
                actions_np = np.asarray(actions, dtype=np.float32)

            # Reject low-motion samples: |dx|<0.05 and |dy|<0.05 and |drot|<0.5
            try:
                if (
                    abs(float(actions_np[0])) < 0.05 and
                    abs(float(actions_np[1])) < 0.05 and
                    abs(float(actions_np[2])) < 0.5
                ):
                    skipped += 1
                    continue
            except Exception:
                # If actions are malformed, skip
                skipped += 1
                continue

            sample_group = grp.create_group(f"sample_{index:08d}")
            sample_group.create_dataset("temporal", data=temporal_np, compression=compression, compression_opts=compression_level)
            sample_group.create_dataset("spatial", data=spatial_np, compression=compression, compression_opts=compression_level)
            sample_group.create_dataset("actions", data=actions_np, compression=compression, compression_opts=compression_level)
            written += 1
        print(f"Exported {written} samples, skipped {skipped} (low-motion filter)")
def main():
    dataset = AgentControlDataset("/Users/vaibhav/Desktop/game_logs_hdf5")
    print(len(dataset))
    temporal, spatial, actions = dataset[10]
    print(temporal.shape, spatial.shape, actions.shape)
    # Print numpy arrays neatly, with all values shown and in float format
    np.set_printoptions(precision=6, suppress=True, linewidth=200, threshold=np.inf, floatmode='fixed')
    print("Temporal:\n", np.asarray(temporal, dtype=np.float32))
    print("Spatial:\n", np.asarray(spatial, dtype=np.float32))
    print("Actions:\n", np.asarray(actions, dtype=np.float32))
    start =  time.time()
    output_h5 = "/Users/vaibhav/Desktop/processed_game_logs_attention_1.h5"
    export_processed_to_hdf5(dataset, output_h5, group_name="processed")
    #dataset.get_min_max_feature_values()
    end = time.time()



if __name__ == "__main__":
    main()

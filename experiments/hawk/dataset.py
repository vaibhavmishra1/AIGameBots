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
            empty_temporal = np.zeros((0, 6), dtype=np.float32)
            empty_spatial = np.zeros((0, 6), dtype=np.float32)
            zero_actions = np.zeros((2,), dtype=np.float32)
            return empty_temporal, empty_spatial, zero_actions

        timesteps_total, num_agents, feature_dim = data.shape
        if timesteps_total < 1 or num_agents < 1:
            empty_temporal = np.zeros((0, 6), dtype=np.float32)
            empty_spatial = np.zeros((max(num_agents, 0), 6), dtype=np.float32)
            zero_actions = np.zeros((2,), dtype=np.float32)
            return empty_temporal, empty_spatial, zero_actions

        (zonecenterx, zonecenterz, shr_key) = self.calculate_expected_shrinking_area_center(data)
        # Normalize shrinking key ONCE; do not mutate in the loops
        shrinking_key_norm = float(shr_key) / 7.0 if shr_key != 0 else 0.0

        # We now construct two matrices:
        # 1) temporal features for agent 0 across first 16 timesteps: (16, F)
        # 2) spatial features for ALL agents at timestep 16: (A, F)
        # 3) actions as deltas for next 5 timesteps after 16: (10,) - 5 timesteps * 2 features
        feature_count = 6
        temporal_limit = min(16, timesteps_total)  # Use first 16 timesteps or all if less
        new_temporal_data = np.zeros((temporal_limit, feature_count), dtype=np.float32)
        new_spatial_data = np.zeros((num_agents, feature_count), dtype=np.float32)
        actions = np.zeros((10,), dtype=np.float32)  # 5 timesteps * 2 features (dx, dy)
        spatial_time_index = min(16, timesteps_total - 1)  # Use timestep 16 or last if shorter
        for i in range(timesteps_total):
            for j in range(num_agents):
                feature = data[i][j]
                if np.all(feature == 0):
                    continue

                prev_feature = data[i][j] if i == 0 else data[i - 1][j]


                team_index = feature[3]
                team_index = (team_index) / 2

                posx = feature[4]
                posz = feature[6]
                
                deltax = posx - prev_feature[4]
                deltay = posz - prev_feature[6]

                rel_pos_x = (posx - zonecenterx) 
                rel_pos_x = rel_pos_x if abs(rel_pos_x) < 150 else 150 * np.sign(rel_pos_x)
                rel_pos_x = rel_pos_x / 150

                rel_pos_z = (posz - zonecenterz) 
                rel_pos_z = rel_pos_z if abs(rel_pos_z) < 100 else 100 * np.sign(rel_pos_z)
                rel_pos_z = rel_pos_z / 100

                
                # Use precomputed normalized shrinking key
                shr_key = shrinking_key_norm

                deltax = deltax if abs(deltax) < 1 else 1 * np.sign(deltax)
                
                deltay = deltay if abs(deltay) < 1 else 1 * np.sign(deltay)
                
                feat_vec = np.array([
                    team_index, #0
                    rel_pos_x,
                    rel_pos_z,
                    shr_key,
                    deltax, #10
                    deltay, #11
                ], dtype=np.float32)
                
                # Fill temporal matrix for agent 0 for first 16 timesteps
                if j == 0 and i < temporal_limit:
                    new_temporal_data[i] = feat_vec
                # Fill spatial matrix for all agents at timestep 16
                if i == spatial_time_index:
                    new_spatial_data[j] = feat_vec

        # Compute actions as deltas for next 5 timesteps after timestep 16
        if timesteps_total > 16:
            for t in range(min(5, timesteps_total - 16 - 1)):  # -1 because we need pairs of timesteps
                curr = data[16 + t + 1][0]  # Next timestep
                prev = data[16 + t][0]      # Current timestep
                
                # Calculate deltas
                ax = curr[4] - prev[4]  # delta x
                az = curr[6] - prev[6]  # delta z
                
                # Clip deltas
                ax = ax if abs(ax) < 1 else 1 * np.sign(ax)
                az = az if abs(az) < 1 else 1 * np.sign(az)
                
                # Store in actions array - alternating dx, dy
                actions[t * 2] = ax
                actions[t * 2 + 1] = az

                        

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
            if(written > 1000000):
                break
            if(index % 10000 == 0):
                print(f"written {written} samples")
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

            # Calculate average of absolute dx and dy over all 5 timesteps
            avg_dx = np.mean(np.abs(actions_np[::2]))  # Take absolute mean of dx values
            avg_dy = np.mean(np.abs(actions_np[1::2]))  # Take absolute mean of dy values
            avg_movement = avg_dx * avg_dx + avg_dy * avg_dy
            if avg_movement > 0.02 and avg_movement < 0.3:
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
    output_h5 = "dataset_exp_hawk_0p02_0p3_1000000.h5"
    export_processed_to_hdf5(dataset, output_h5, group_name="processed")
    #dataset.get_min_max_feature_values()
    end = time.time()



if __name__ == "__main__":
    main()

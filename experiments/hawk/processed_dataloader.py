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

    def __init__(self, h5_path, experiment_type = "hawk_temporal_and_spatial",group_name="processed", return_numpy=False):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"Processed HDF5 not found: {h5_path}")
        self.h5_path = h5_path
        self.group_name = group_name
        self.return_numpy = return_numpy
        self.experiment_type = experiment_type

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

        if self.experiment_type == "hawk_temporal_and_spatial":
            return (
            torch.from_numpy(np.asarray(temporal, dtype=np.float32)),
            torch.from_numpy(np.asarray(spatial, dtype=np.float32)),
            torch.from_numpy(np.asarray(actions, dtype=np.float32)),
            )

        elif self.experiment_type == "hawk_temporal_only":
            return (
            torch.from_numpy(np.asarray(temporal, dtype=np.float32)),
            torch.from_numpy(np.asarray(actions, dtype=np.float32)),
            )
        elif self.experiment_type == "hawk_spatial_only":
            return (
            torch.from_numpy(np.asarray(spatial, dtype=np.float32)),
            torch.from_numpy(np.asarray(actions, dtype=np.float32)),
            )
        elif self.experiment_type == "hawk_only_main_agent_current_features":
            temporal = temporal[-1]
            return (
            torch.from_numpy(np.asarray(temporal, dtype=np.float32)),
            torch.from_numpy(np.asarray(actions, dtype=np.float32)),
            )
        
        


def main():
    h5_path = "/Users/vaibhav/Desktop/AIGameBots/experiments/hawk/dataset_exp_hawk_0p02_0p3_100000.h5"
    experiment_type = "hawk_only_main_agent_current_features"
    if experiment_type == "hawk_temporal_and_spatial":
        agents_dataset = ProcessedH5Dataset(h5_path, group_name="processed", experiment_type=experiment_type)
        (temporal, spatial, actions )= agents_dataset[0]
        print(temporal.shape)
        print(spatial.shape)
        print(actions.shape)
    elif experiment_type == "hawk_temporal_only":
        agents_dataset = ProcessedH5Dataset(h5_path, group_name="processed", experiment_type=experiment_type)
        (temporal, actions )= agents_dataset[0]
        print(temporal.shape)
        print(actions.shape)
    elif experiment_type == "hawk_spatial_only":
        agents_dataset = ProcessedH5Dataset(h5_path, group_name="processed", experiment_type=experiment_type)
        (spatial, actions )= agents_dataset[0]
        print(spatial.shape)
        print(actions.shape)
    elif experiment_type == "hawk_only_main_agent_current_features":
        agents_dataset = ProcessedH5Dataset(h5_path, group_name="processed", experiment_type=experiment_type)
        (temporal, actions )= agents_dataset[0]
        print(temporal.shape)
        print(actions.shape)

if __name__ == "__main__":
    main()



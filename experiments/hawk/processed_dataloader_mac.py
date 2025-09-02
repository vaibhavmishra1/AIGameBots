import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessedH5DatasetMac(Dataset):
    """
    Dataset to read processed features/actions from a consolidated HDF5 file
    created by export_processed_to_hdf5.
    
    This version is optimized for macOS multiprocessing by avoiding persistent file handles.
    """

    def __init__(self, h5_path, experiment_type="hawk_temporal_and_spatial", group_name="processed", return_numpy=False):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"Processed HDF5 not found: {h5_path}")
        self.h5_path = h5_path
        self.group_name = group_name
        self.return_numpy = return_numpy
        self.experiment_type = experiment_type

        # Get sample keys once during initialization
        with h5py.File(self.h5_path, 'r') as f:
            if self.group_name not in f:
                raise KeyError(f"Group '{self.group_name}' not found in {self.h5_path}")
            # Store sample keys deterministically
            self.sample_keys = sorted(list(f[self.group_name].keys()))

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        key = self.sample_keys[idx]
        # Open file for each access to support multiprocessing
        with h5py.File(self.h5_path, 'r') as f:
            sample_grp = f[self.group_name][key]
            # Read temporal/spatial/actions written by export_processed_to_hdf5
            temporal = sample_grp['temporal'][()]
            spatial = sample_grp['spatial'][()]
            actions = sample_grp['actions'][()]

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
    experiment_type = "hawk_temporal_and_spatial"
    if experiment_type == "hawk_temporal_and_spatial":
        agents_dataset = ProcessedH5DatasetMac(h5_path, group_name="processed", experiment_type=experiment_type)
        (temporal, spatial, actions) = agents_dataset[0]
        print(temporal.shape)
        print(spatial.shape)
        print(actions.shape)
    elif experiment_type == "hawk_temporal_only":
        agents_dataset = ProcessedH5DatasetMac(h5_path, group_name="processed", experiment_type=experiment_type)
        (temporal, actions) = agents_dataset[0]
        print(temporal.shape)
        print(actions.shape)
    elif experiment_type == "hawk_spatial_only":
        agents_dataset = ProcessedH5DatasetMac(h5_path, group_name="processed", experiment_type=experiment_type)
        (spatial, actions) = agents_dataset[0]
        print(spatial.shape)
        print(actions.shape)
    elif experiment_type == "hawk_only_main_agent_current_features":
        agents_dataset = ProcessedH5DatasetMac(h5_path, group_name="processed", experiment_type=experiment_type)
        (temporal, actions) = agents_dataset[0]
        print(temporal.shape)
        print(actions.shape)


if __name__ == "__main__":
    main()

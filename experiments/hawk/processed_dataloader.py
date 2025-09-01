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

        # Keep file open for the lifetime of the dataset to avoid repeated open/close
        self.h5_file = h5py.File(self.h5_path, 'r')
        if self.group_name not in self.h5_file:
            raise KeyError(f"Group '{self.group_name}' not found in {self.h5_path}")
        self.grp = self.h5_file[self.group_name]
        # Store sample keys deterministically
        self.sample_keys = sorted(list(self.grp.keys()))

    def __del__(self):
        # Clean up the file handle
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        key = self.sample_keys[idx]
        # Use the persistent file handle instead of opening/closing repeatedly
        sample_grp = self.grp[key]
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



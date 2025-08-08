import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple, Any

import os
class NPYDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 transform: Optional[Any] = None,
                 target_transform: Optional[Any] = None):
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        features_path = os.path.join(data_path, "features")
        self.features_files = [os.path.join(features_path, file) for file in os.listdir(features_path)]
        

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the sample data
        feature_path = self.features_files[idx]
        action_path = feature_path.replace("features", "actions")
        sample = np.load(feature_path)
        sample = sample.reshape(-1)  # Flatten 20x18 into 360
        actions = np.load(action_path)
        # Convert to torch tensor
        features = torch.from_numpy(sample).float()
        actions = torch.from_numpy(actions[[0,1]]).float()
        return features, actions

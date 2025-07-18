import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple, Any

import os
class NPYDataset(Dataset):

    def __init__(self, 
                 data_path: str, task: str,
                 transform: Optional[Any] = None,
                 target_transform: Optional[Any] = None):
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        features_path = os.path.join(data_path, "features")
        self.features_files = [os.path.join(features_path, file) for file in os.listdir(features_path)]
        self.features_files = self.features_files[:100000]
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Get the sample data
        feature_path = self.features_files[idx]

        action_path = feature_path.replace("features", "actions")
        sample = np.load(feature_path)
        actions = np.load(action_path)
        # Convert to torch tensor
        features = torch.from_numpy(sample).float()
        actions = torch.from_numpy(actions).float()
        # For now, we'll use the same data as target (you can modify this based on your needs)
        if self.task == "moveDirection_x":
            target = actions[0]
        elif self.task == "moveDirection_y":
            target = actions[1]
        elif self.task == "lookRotationDelta_x":
            target = actions[2]
        elif self.task == "lookRotationDelta_y":
            target = actions[3]
        elif self.task == "Attack":
            target = actions[4]
        elif self.task == "Reload":
            target = actions[5]
        elif self.task == "thrust":
            target = actions[6]
        elif self.task == "crouch":
            target = actions[7]
        elif self.task == "sprint":
            target = actions[8]
        elif self.task == "slide":
            target = actions[9]
        else:
            raise ValueError(f"Unknown task: {self.task}")
        return features, target


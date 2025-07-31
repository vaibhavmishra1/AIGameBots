import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import random
from typing import Tuple, Optional, List
import glob
from concurrent.futures import ThreadPoolExecutor
import pickle


class AgentDataset(Dataset):
    """Dataset for agent imitation learning with features and actions."""
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 20,
        predict_next: bool = True,
        use_history: int = 5,
        action_indices: List[int] = [0, 1, 2, 3],
        normalize: bool = True,
        cache_data: bool = False,
        max_samples: Optional[int] = None,
        train_split: float = 0.8,
        is_train: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the directory containing features and actions subdirectories
            sequence_length: Length of input sequences (default: 20)
            predict_next: If True, predict next action; if False, predict current action
            use_history: Number of previous frames to use as input (default: 5)
            action_indices: Indices of actions to use (default: [0,1,2,3] for movement and rotation)
            normalize: Whether to normalize features and actions
            cache_data: Whether to cache loaded data in memory
            max_samples: Maximum number of samples to load (None for all)
            train_split: Fraction of data to use for training
            is_train: Whether this is training or validation set
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features"
        self.actions_dir = self.data_dir / "actions"
        self.sequence_length = sequence_length
        self.predict_next = predict_next
        self.use_history = use_history
        self.action_indices = action_indices
        self.normalize = normalize
        self.cache_data = cache_data
        self.max_samples = max_samples
        self.train_split = train_split
        self.is_train = is_train
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Get all feature files
        self.feature_files = sorted(glob.glob(str(self.features_dir / "*.npy")))
        
        if max_samples and max_samples < len(self.feature_files):
            random.shuffle(self.feature_files)
            self.feature_files = self.feature_files[:max_samples]
        
        # Split into train/val
        split_idx = int(len(self.feature_files) * train_split)
        if is_train:
            self.feature_files = self.feature_files[:split_idx]
        else:
            self.feature_files = self.feature_files[split_idx:]
        
        print(f"{'Train' if is_train else 'Validation'} dataset: {len(self.feature_files)} files")
        
        # Cache for loaded data
        self.cache = {} if cache_data else None
        
        # Calculate normalization statistics if needed
        if normalize:
            self._calculate_normalization_stats()
            
        # Stats save path
        self.stats_save_path = None
    
    def _calculate_normalization_stats(self):
        """Calculate mean and std for normalization."""
        print("Calculating normalization statistics...")
        
        # Sample a subset of files for statistics
        sample_files = random.sample(self.feature_files, min(1000, len(self.feature_files)))
        
        features_list = []
        actions_list = []
        
        for file_path in sample_files:
            features = np.load(file_path)
            action_file = file_path.replace("/features/", "/actions/").replace("_features_", "_actions_")
            actions = np.load(action_file)
            
            features_list.append(features)
            actions_list.append(actions[:, self.action_indices])
        
        # Calculate statistics
        all_features = np.concatenate(features_list, axis=0)
        all_actions = np.concatenate(actions_list, axis=0)
        
        self.feature_mean = all_features.mean(axis=0)
        self.feature_std = all_features.std(axis=0) + 1e-8
        
        self.action_mean = all_actions.mean(axis=0)
        self.action_std = all_actions.std(axis=0) + 1e-8
        
        # Handle action ranges based on your specifications
        # movedirection_x, movedirection_z: [-1, 1]
        # lookrotation_x, lookrotation_z: [-10, 10]
        self.action_ranges = np.array([
            [-1, 1],   # movedirection_x
            [-1, 1],   # movedirection_z
            [-10, 10], # lookrotation_x
            [-10, 10]  # lookrotation_z
        ])
        
        print("Normalization statistics calculated.")
    
    def save_normalization_stats(self, save_path: str):
        """Save normalization statistics to file."""
        if self.normalize:
            np.savez(
                save_path,
                feature_mean=self.feature_mean,
                feature_std=self.feature_std,
                action_ranges=self.action_ranges
            )
            print(f"Normalization statistics saved to {save_path}")
    
    def _load_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load features and actions for a given index."""
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
        
        feature_file = self.feature_files[idx]
        features = np.load(feature_file)
        
        # Load corresponding actions
        action_file = feature_file.replace("/features/", "/actions/").replace("_features_", "_actions_")
        actions = np.load(action_file)
        
        # Select only the action indices we care about
        actions = actions[:, self.action_indices]
        
        if self.cache_data:
            self.cache[idx] = (features, actions)
        
        return features, actions
    
    def __len__(self) -> int:
        # Each file provides (sequence_length - use_history) samples
        return len(self.feature_files) * (self.sequence_length - self.use_history)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine which file and position within file
        file_idx = idx // (self.sequence_length - self.use_history)
        pos_idx = idx % (self.sequence_length - self.use_history)
        
        features, actions = self._load_data(file_idx)
        
        # Extract the sequence window
        start_idx = pos_idx
        end_idx = start_idx + self.use_history
        
        # Input features: [use_history, feature_dim]
        input_features = features[start_idx:end_idx]
        
        # Previous actions: [use_history-1, action_dim]
        # We use actions from start_idx to end_idx-1
        if self.use_history > 1:
            prev_actions = actions[start_idx:end_idx-1]
        else:
            prev_actions = np.zeros((0, len(self.action_indices)))
        
        # Target action
        if self.predict_next:
            target_action = actions[end_idx]  # Next action
        else:
            target_action = actions[end_idx-1]  # Current action
        
        # Normalize if needed
        if self.normalize:
            input_features = (input_features - self.feature_mean) / self.feature_std
            
            # Normalize actions to [-1, 1] range
            for i in range(len(self.action_indices)):
                action_range = self.action_ranges[i]
                prev_actions[:, i] = 2 * (prev_actions[:, i] - action_range[0]) / (action_range[1] - action_range[0]) - 1
                target_action[i] = 2 * (target_action[i] - action_range[0]) / (action_range[1] - action_range[0]) - 1
        
        # Convert to tensors
        input_features = torch.FloatTensor(input_features)
        prev_actions = torch.FloatTensor(prev_actions)
        target_action = torch.FloatTensor(target_action)
        
        return (input_features, prev_actions), target_action
    
    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions back to original ranges."""
        if not self.normalize:
            return actions
        
        actions = actions.clone()
        
        # Convert from [-1, 1] back to original ranges
        for i in range(len(self.action_indices)):
            action_range = self.action_ranges[i]
            actions[..., i] = (actions[..., i] + 1) * (action_range[1] - action_range[0]) / 2 + action_range[0]
        
        return actions


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    # Create train dataset
    train_dataset = AgentDataset(
        data_dir=data_dir,
        is_train=True,
        **dataset_kwargs
    )
    
    # Create validation dataset
    val_dataset = AgentDataset(
        data_dir=data_dir,
        is_train=False,
        **dataset_kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    data_dir = "/Users/vaibhavmishra/Desktop/Desktop/btx-game-aicode/clash_squad_partitioned_chunked"
    
    dataset = AgentDataset(
        data_dir=data_dir,
        sequence_length=20,
        use_history=5,
        max_samples=100,
        normalize=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    (features, prev_actions), target_action = dataset[0]
    print(f"Input features shape: {features.shape}")
    print(f"Previous actions shape: {prev_actions.shape}")
    print(f"Target action shape: {target_action.shape}")
    
    # Test dataloader
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        max_samples=1000
    )
    
    for batch in train_loader:
        (features_batch, prev_actions_batch), target_batch = batch
        print(f"Batch features shape: {features_batch.shape}")
        print(f"Batch prev actions shape: {prev_actions_batch.shape}")
        print(f"Batch targets shape: {target_batch.shape}")
        break

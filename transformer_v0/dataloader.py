import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Tuple, Optional, Dict, Any
import sys
import os
import atexit

from torchvision import transforms
from PIL import Image

# Import shared config from the original project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Counter-Strike_Behavioural_Cloning'))
from config import *  # noqa: F401,F403


class CSGODataset(Dataset):
    """
    PyTorch Dataset for Counter-Strike: Global Offensive behavioral cloning data.
    Similar to the TensorFlow DataGenerator but implemented in PyTorch.
    """

    def __init__(
        self,
        data_list: List[str],
        folder_name: str = '/root/AIGameBots/Counter-Strike_Behavioural_Cloning/dataset_dm_expert_dust2/',
        n_jitter: int = 20,
        is_mirror: bool = False,
        transform: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            data_list: List of data IDs in format 'filenum-framenum'
            folder_name: Path to the folder containing HDF5 files
            n_jitter: Number of frames to randomly offset by
            is_mirror: Whether to apply mirroring augmentation
            transform: Whether to apply data augmentation
        """
        self.data_list = data_list
        self.folder_name = folder_name
        self.n_jitter = n_jitter
        self.is_mirror = is_mirror
        self.transform = transform
        # Pre-compute file numbers and paths referenced by this dataset
        self._file_numbers = sorted({int(str_id.split('-')[0]) for str_id in self.data_list})
        self._file_paths: Dict[int, str] = {
            file_num: os.path.join(self.folder_name, f'hdf5_dm_july2021_expert_{file_num}.hdf5')
            for file_num in self._file_numbers
        }

        # Open and keep HDF5 file handles alive; worker processes will reopen via __setstate__
        self._h5_files: Dict[int, h5py.File] = {}
        self._worker_pid: Optional[int] = None
        self._open_all_h5_files()
        atexit.register(self._close_all_h5_files)

        # ViT preprocessing transforms (matches IMAGENET1K_V1: resize 256, crop 224, normalize)
        self.vit_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # Converts PIL to tensor and scales to [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            index: Index of the sample

        Returns:
            Tuple of (X, y) where X is the input tensor and y is the target tensor
        """
        ID = self.data_list[index]
        ID = ID.split('-')
        file_num = int(ID[0])
        frame_num = int(ID[1]) + np.random.randint(0, self.n_jitter)
        frame_num = np.minimum(frame_num, 999 - N_TIMESTEPS)
        frame_num = np.maximum(frame_num, 0)

        # Ensure files are opened for this process and get the file handle
        self._ensure_files_open()
        h5file = self._h5_files.get(file_num)
        if h5file is None:
            # Fallback safeguard: open on-demand if missing (should not happen)
            h5file = h5py.File(self._file_paths[file_num], 'r')
            self._h5_files[file_num] = h5file

        # Initialize arrays
        x_data = np.empty((N_TIMESTEPS, *csgo_img_dimension, 3), dtype=np.uint8)
        y_data = np.empty((N_TIMESTEPS, n_keys + n_clicks + n_mouse_x + n_mouse_y), dtype=np.float32)
        rewards = np.empty((N_TIMESTEPS,), dtype=np.float32)

        # Load data for each timestep
        for j in range(N_TIMESTEPS):
            current_frame = frame_num + j

            # Load image data
            x_data[j] = h5file[f'frame_{current_frame}_x'][:]

            # Load label data
            y_data[j] = h5file[f'frame_{current_frame}_y'][:]

            # Load helper data for reward calculation
            helper_i = h5file[f'frame_{current_frame}_helperarr'][:]
            kill_i = helper_i[0]
            dead_i = helper_i[1]
            shoot_i = y_data[j, n_keys:n_keys+1]  # left click

            # Calculate reward
            reward_i = kill_i - 0.5 * dead_i - 0.01 * shoot_i
            rewards[j] = reward_i

            # Apply mouse discretization fixes (similar to TensorFlow implementation)
            self._fix_mouse_discretization(y_data[j])

        # Add reward and advantage to y_data (similar to TensorFlow implementation)
        # y_data shape becomes (N_TIMESTEPS, n_keys + n_clicks + n_mouse_x + n_mouse_y + 2)
        y_with_reward = np.zeros((N_TIMESTEPS, y_data.shape[1] + 2), dtype=np.float32)
        y_with_reward[:, :-2] = y_data
        y_with_reward[:, -2] = rewards  # reward
        y_with_reward[:, -1] = 0.0      # placeholder for advantage

        # Apply data augmentation
        if self.transform:
            x_data, y_with_reward = self._apply_augmentations(x_data, y_with_reward)

        # Apply ViT preprocessing to each frame (expects uint8 numpy -> PIL -> tensor)
        x_transformed = torch.empty((N_TIMESTEPS, 3, 224, 224), dtype=torch.float32)
        for j in range(N_TIMESTEPS):
            # Convert numpy uint8 frame to PIL Image
            frame_pil = Image.fromarray(x_data[j])
            # Apply transforms
            x_transformed[j] = self.vit_transform(frame_pil)

        # Convert to tensors
        # y_tensor and aux_tensor remain as before
        y_tensor = torch.from_numpy(y_with_reward).float()

        # Build auxiliary input consisting of previous timestep's action one-hots
        # Use AUGMENTED actions (exclude reward/advantage columns)
        action_dim = n_keys + n_clicks + n_mouse_x + n_mouse_y
        prev_actions = np.zeros((N_TIMESTEPS, action_dim), dtype=np.float32)
        prev_actions[1:] = y_with_reward[:-1, :action_dim]
        aux_tensor = torch.from_numpy(prev_actions).float()

        return x_transformed, y_tensor, aux_tensor

    def _open_all_h5_files(self) -> None:
        """
        Open all referenced HDF5 files for this dataset instance.
        """
        # If already opened for this PID, do nothing
        current_pid = os.getpid()
        if getattr(self, '_worker_pid', None) == current_pid and getattr(self, '_h5_files', None):
            return
        # Close any existing before reopening
        self._close_all_h5_files()
        self._h5_files = {}
        for file_num, path in self._file_paths.items():
            if file_num == 90 or file_num == 96:
                continue
            self._h5_files[file_num] = h5py.File(path, 'r')
        self._worker_pid = current_pid

    def _close_all_h5_files(self) -> None:
        """
        Close all opened HDF5 files for this dataset instance.
        """
        if getattr(self, '_h5_files', None):
            for f in list(self._h5_files.values()):
                try:
                    f.close()
                except Exception:
                    pass
        self._h5_files = {}

    def _ensure_files_open(self) -> None:
        """Ensure HDF5 file handles are open for the current worker process."""
        if self._h5_files is None or len(self._h5_files) == 0 or self._worker_pid != os.getpid():
            self._open_all_h5_files()

    def __getstate__(self) -> Dict[str, Any]:
        """
        Make dataset picklable by removing non-picklable HDF5 handles.
        Workers will reopen files in __setstate__.
        """
        state = self.__dict__.copy()
        # Do not pickle file handles or PID
        state['_h5_files'] = None
        state['_worker_pid'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Reopen files in the worker process
        self._open_all_h5_files()

    def __del__(self):
        # Best-effort cleanup
        try:
            self._close_all_h5_files()
        except Exception:
            pass

    def _fix_mouse_discretization(self, y_frame: np.ndarray) -> None:
        """
        Apply the same mouse discretization fixes as in the TensorFlow implementation.

        Args:
            y_frame: Single frame of y data
        """
        # Fix mouse x discretization
        if y_frame[n_keys + n_clicks] == 1:
            y_frame[n_keys + n_clicks] = 0
            y_frame[n_keys + n_clicks + 2] = 1
        elif y_frame[n_keys + n_clicks + 1] == 1:
            y_frame[n_keys + n_clicks + 1] = 0
            y_frame[n_keys + n_clicks + 2] = 1
        elif y_frame[n_keys + n_clicks + n_mouse_x - 1] == 1:
            y_frame[n_keys + n_clicks + n_mouse_x - 1] = 0
            y_frame[n_keys + n_clicks + n_mouse_x - 3] = 1
        elif y_frame[n_keys + n_clicks + n_mouse_x - 2] == 1:
            y_frame[n_keys + n_clicks + n_mouse_x - 2] = 0
            y_frame[n_keys + n_clicks + n_mouse_x - 3] = 1

        # Fix mouse y discretization (similar to 20 aug update)
        mouse_y_start = n_keys + n_clicks + n_mouse_x
        if y_frame[mouse_y_start] == 1:
            y_frame[mouse_y_start] = 0
            y_frame[mouse_y_start + 2] = 1
        elif y_frame[mouse_y_start + 1] == 1:
            y_frame[mouse_y_start + 1] = 0
            y_frame[mouse_y_start + 2] = 1
        elif y_frame[mouse_y_start + n_mouse_y - 1] == 1:
            y_frame[mouse_y_start + n_mouse_y - 1] = 0
            y_frame[mouse_y_start + n_mouse_y - 3] = 1
        elif y_frame[mouse_y_start + n_mouse_y - 2] == 1:
            y_frame[mouse_y_start + n_mouse_y - 2] = 0
            y_frame[mouse_y_start + n_mouse_y - 3] = 1

    def _apply_augmentations(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentations similar to TensorFlow implementation.

        Args:
            x_data: Input image data
            y_data: Target data

        Returns:
            Tuple of augmented (x_data, y_data)
        """
        # Mirroring augmentation
        if self.is_mirror and np.random.rand() < 0.3:
            x_data = np.flip(x_data, axis=2)  # flip width dimension

            # Also need to flip mouse x movement
            mouse_x_start = n_keys + n_clicks
            mouse_x_end = n_keys + n_clicks + n_mouse_x
            y_data[:, mouse_x_start:mouse_x_end] = np.flip(y_data[:, mouse_x_start:mouse_x_end], axis=1)

            # Also flip 'a' and 'd' keys (indices 1 and 3)
            a_key = y_data[:, 1].copy()
            d_key = y_data[:, 3].copy()
            y_data[:, 1] = d_key
            y_data[:, 3] = a_key

        # Brightness augmentation
        if np.random.rand() < 0.5:
            bright = np.random.rand() * 0.6 + 0.7
            x_data = x_data * bright
            x_data = np.clip(x_data, 0, 255).astype(np.uint8)

        # Contrast augmentation
        if np.random.rand() < 0.5:
            contrast = np.random.rand() * 0.6 + 0.7
            x_data = np.clip(128 + contrast * x_data.astype(np.float32) - contrast * 128, 0, 255).astype(np.uint8)

        return x_data, y_data


class CSGODataLoader:
    """
    DataLoader wrapper for CSGO dataset with similar functionality to TensorFlow training loop.
    """

    def __init__(
        self,
        data_list: List[str],
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        folder_name: str = '/Users/vaibhav/Desktop/AIGameBots/Counter-Strike_Behavioural_Cloning/dataset_dm_expert_dust2/',
        n_jitter: int = 20,
        is_mirror: bool = False,
        transform: bool = True
    ):
        """
        Initialize the DataLoader.

        Args:
            data_list: List of data IDs in format 'filenum-framenum'
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for GPU training
            folder_name: Path to the folder containing HDF5 files
            n_jitter: Number of frames to randomly offset by
            is_mirror: Whether to apply mirroring augmentation
            transform: Whether to apply data augmentation
        """
        self.dataset = CSGODataset(
            data_list=data_list,
            folder_name=folder_name,
            n_jitter=n_jitter,
            is_mirror=is_mirror,
            transform=transform
        )

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self) -> int:
        return len(self.data_loader)

    def get_dataset_length(self) -> int:
        """Get the total number of samples in the dataset."""
        return len(self.dataset)


def create_data_lists(
    starting_num: int = 1,
    highest_num: int = 30,
    n_timesteps: int = N_TIMESTEPS,
    n_jitter: int = 20
) -> List[str]:
    """
    Create data lists similar to the TensorFlow implementation.

    Args:
        starting_num: Lowest file number to use
        highest_num: Highest file number to use
        n_timesteps: Number of timesteps
        n_jitter: Number of jitter frames

    Returns:
        List of data IDs in format 'filenum-framenum'
    """
    data_list_full = [
        f"{x1}-{x2}"
        for x1 in np.arange(starting_num, highest_num + 1)
        if x1 != 90 and x1 != 96
        for x2 in np.arange(0, 1000 - n_timesteps - n_jitter, n_timesteps)
    ]

    return  data_list_full


def create_data_loaders(
    batch_size: int = 32,
    starting_num: int = 1,
    highest_num: int = 30,
    folder_name: str = '/Users/vaibhav/Desktop/AIGameBots/Counter-Strike_Behavioural_Cloning/dataset_dm_expert_dust2/',
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    n_jitter: int = 20,
    is_mirror: bool = False,
    transform: bool = True
) -> CSGODataLoader:
    """
    Create multiple data loaders for training (similar to TensorFlow implementation).

    Args:
        batch_size: Batch size for training
        starting_num: Lowest file number to use
        highest_num: Highest file number to use
        folder_name: Path to the folder containing HDF5 files
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for GPU training
        n_jitter: Number of frames to randomly offset by
        is_mirror: Whether to apply mirroring augmentation
        transform: Whether to apply data augmentation

    Returns:
        Tuple of 5 data loaders (4 for subselection, 1 full)
    """
    data_list_full = create_data_lists(
        starting_num, highest_num, N_TIMESTEPS, n_jitter
    )
    np.random.shuffle(data_list_full)


    partition_full = {
        'train_full': data_list_full[:int(len(data_list_full) * 0.90)],
        'validation_full': data_list_full[int(len(data_list_full) * 0.90):]
    }
    training_loader_full = CSGODataLoader(
        data_list=partition_full['train_full'],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        folder_name=folder_name,
        n_jitter=n_jitter,
        is_mirror=is_mirror,
        transform=transform
    )

    validation_loader_full = CSGODataLoader(
        data_list=partition_full['validation_full'],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        folder_name=folder_name,
        n_jitter=n_jitter,
        is_mirror=is_mirror,
        transform=transform
    )
    
    return training_loader_full, validation_loader_full


if __name__ == "__main__":
    # Basic smoke test
    loaders = create_data_loaders(
        batch_size=1,
        starting_num=2,
        highest_num=2,
        num_workers=0,
        pin_memory=False,
        n_jitter=1,
        transform=False,
    )
    print('✓ transformer_v0.dataloader ready')
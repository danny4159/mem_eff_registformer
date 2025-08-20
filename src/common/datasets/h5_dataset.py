"""
Common Dataset Module

This module provides base dataset classes that can be inherited by specific model datasets.
"""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
from typing import Optional, Tuple
from torch.utils.data import DataLoader


class H5Dataset(Dataset):
    """
    Base dataset class for medical image datasets.
    
    This class provides common functionality for loading and preprocessing
    medical images from HDF5 files. Specific models can inherit from this
    class and override methods as needed.
    """
    
    def __init__(
        self,
        h5_file_path: str,
        data_group_1: str,
        data_group_2: str,
        data_group_3: Optional[str],
        data_group_4: Optional[str],
        data_group_5: Optional[str],
        is_3d: bool,
        padding_size: Optional[Tuple[int, int]],
        crop_size: Optional[Tuple[int, int]] = None,
        flip_prob: float = 0.0,
        rot_prob: float = 0.0,
        reverse: bool = False,
        norm_ZeroToOne: bool = False,
    ):
        """
        Initialize base dataset.
        
        Args:
            h5_file_path: Path to HDF5 file
            data_group_1: Name of first data group (source images)
            data_group_2: Name of second data group (target images)
            data_group_3: Name of third data group (optional)
            data_group_4: Name of fourth data group (optional)
            data_group_5: Name of fifth data group (optional)
            is_3d: Whether data is 3D
            padding_size: Size to pad images to
            crop_size: Size to crop images to
            flip_prob: Probability of horizontal flip
            rot_prob: Probability of rotation
            reverse: Whether to reverse source and target
            norm_ZeroToOne: Whether to normalize to [0, 1]
        """
        super().__init__()
        self.h5_file_path = h5_file_path
        self.data_group_1 = data_group_1
        self.data_group_2 = data_group_2
        self.data_group_3 = data_group_3
        self.data_group_4 = data_group_4
        self.data_group_5 = data_group_5
        self.is_3d = is_3d
        self.padding_size = padding_size
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob
        self.reverse = reverse
        self.norm_ZeroToOne = norm_ZeroToOne

        # Load patient keys and slice information
        with h5py.File(self.h5_file_path, "r") as f:
            self.patient_keys = list(f[self.data_group_1].keys())
            if not self.is_3d:
                self.slice_counts = [f[self.data_group_1][k].shape[-1] for k in self.patient_keys]
                self.cumulative_slice_counts = np.cumsum([0] + self.slice_counts)

    def __len__(self):
        """Return the total number of samples."""
        if self.is_3d:
            return len(self.patient_keys)
        else:
            return self.cumulative_slice_counts[-1]

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (A, B, C, D, E) where A is source, B is target
        """
        # Load data
        A, B, C, D, E = self._load_data(idx)
        
        # Convert to tensors
        A, B, C, D, E = self._to_tensors(A, B, C, D, E)
        
        # Apply preprocessing
        A, B, C, D, E = self._preprocess_data(A, B, C, D, E)
        
        # Return processed data
        return A, B, C, D, E
    
    def _load_data(self, idx):
        """Load raw data from HDF5 file."""
        if self.is_3d:
            patient_key = self.patient_keys[idx]
            with h5py.File(self.h5_file_path, "r") as f:
                A = f[self.data_group_1][patient_key][...]
                B = f[self.data_group_2][patient_key][...]
                C = f[self.data_group_3][patient_key][...] if self.data_group_3 else None
                D = f[self.data_group_4][patient_key][...] if self.data_group_4 else None
                E = f[self.data_group_5][patient_key][...] if self.data_group_5 else None
        else:
            patient_idx = np.searchsorted(self.cumulative_slice_counts, idx + 1) - 1
            slice_idx = idx - self.cumulative_slice_counts[patient_idx]
            patient_key = self.patient_keys[patient_idx]
            with h5py.File(self.h5_file_path, "r") as f:
                A = f[self.data_group_1][patient_key][..., slice_idx]
                B = f[self.data_group_2][patient_key][..., slice_idx]
                C = f[self.data_group_3][patient_key][..., slice_idx] if self.data_group_3 else None
                D = f[self.data_group_4][patient_key][..., slice_idx] if self.data_group_4 else None
                E = f[self.data_group_5][patient_key][..., slice_idx] if self.data_group_5 else None
        
        return A, B, C, D, E
    
    def _to_tensors(self, A, B, C, D, E):
        """Convert numpy arrays to PyTorch tensors."""
        def to_tensor(x):
            if x is None: 
                return None
            t = torch.from_numpy(x).unsqueeze(0).float() if x.ndim == 2 else torch.from_numpy(x).float()
            return t

        return to_tensor(A), to_tensor(B), to_tensor(C), to_tensor(D), to_tensor(E)
    
    def _preprocess_data(self, A, B, C, D, E):
        """Apply preprocessing to the data."""
        # Padding
        if self.padding_size:
            A, B, C, D, E = self._padding_height_width_depth(A, B, C, D, E, self.padding_size)

        # Random flip
        if self.flip_prob > 0 and torch.rand(1).item() < self.flip_prob:
            A = torch.flip(A, dims=[-1])
            B = torch.flip(B, dims=[-1])
            if C is not None: C = torch.flip(C, dims=[-1])
            if D is not None: D = torch.flip(D, dims=[-1])
            if E is not None: E = torch.flip(E, dims=[-1])

        # Random rotation
        if self.rot_prob > 0 and torch.rand(1).item() < self.rot_prob:
            k = torch.randint(1, 4, (1,)).item()
            A = torch.rot90(A, k, dims=[-2, -1])
            B = torch.rot90(B, k, dims=[-2, -1])
            if C is not None: C = torch.rot90(C, k, dims=[-2, -1])
            if D is not None: D = torch.rot90(D, k, dims=[-2, -1])
            if E is not None: E = torch.rot90(E, k, dims=[-2, -1])

        # Random crop
        if self.crop_size:
            A, B, C, D, E = self._random_crop(A, B, C, D, E, self.crop_size)

        # Normalization
        if self.norm_ZeroToOne:
            A = self._norm01(A)
            B = self._norm01(B)
            if C is not None: C = self._norm01(C)
            if D is not None: D = self._norm01(D)
            if E is not None: E = self._norm01(E)

        # Reverse order if needed
        if self.reverse:
            A, B = B, A

        return A, B, C, D, E

    def _norm01(self, x):
        """Normalize tensor to [0, 1] range."""
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def _padding_height_width_depth(self, A, B, C=None, D=None, E=None, target_size=(256, 256), pad_value=-1):
        """Pad images to target size."""
        if A is None or B is None:
            return A, B, C, D, E
            
        # Get current dimensions
        _, h, w = A.shape
        
        # Calculate padding
        pad_h = max(0, target_size[0] - h)
        pad_w = max(0, target_size[1] - w)
        
        # Apply padding
        if pad_h > 0 or pad_w > 0:
            A = torch.nn.functional.pad(A, (0, pad_w, 0, pad_h), value=pad_value)
            B = torch.nn.functional.pad(B, (0, pad_w, 0, pad_h), value=pad_value)
            if C is not None: C = torch.nn.functional.pad(C, (0, pad_w, 0, pad_h), value=pad_value)
            if D is not None: D = torch.nn.functional.pad(D, (0, pad_w, 0, pad_h), value=pad_value)
            if E is not None: E = torch.nn.functional.pad(E, (0, pad_w, 0, pad_h), value=pad_value)
        
        return A, B, C, D, E

    def _random_crop(self, A, B, C=None, D=None, E=None, target_size=(128, 128)):
        """Randomly crop images to target size."""
        if A is None or B is None:
            return A, B, C, D, E
            
        # Get current dimensions
        _, h, w = A.shape
        
        # Calculate crop boundaries
        if h > target_size[0]:
            top = torch.randint(0, h - target_size[0] + 1, (1,)).item()
        else:
            top = 0
            
        if w > target_size[1]:
            left = torch.randint(0, w - target_size[1] + 1, (1,)).item()
        else:
            left = 0
            
        bottom = min(top + target_size[0], h)
        right = min(left + target_size[1], w)
        
        # Apply crop
        A = A[:, top:bottom, left:right]
        B = B[:, top:bottom, left:right]
        if C is not None: C = C[:, top:bottom, left:right]
        if D is not None: D = D[:, top:bottom, left:right]
        if E is not None: E = E[:, top:bottom, left:right]
        
        return A, B, C, D, E

    def _even_crop_height_width(self, A, B, C=None, D=None, multiple=(16, 16)):
        """Crop images to be divisible by multiple."""
        if A is None or B is None:
            return A, B, C, D
            
        # Get current dimensions
        _, h, w = A.shape
        
        # Calculate new dimensions
        new_h = (h // multiple[0]) * multiple[0]
        new_w = (w // multiple[1]) * multiple[1]
        
        # Apply crop
        A = A[:, :new_h, :new_w]
        B = B[:, :new_h, :new_w]
        if C is not None: C = C[:, :new_h, :new_w]
        if D is not None: D = D[:, :new_h, :new_w]
        
        return A, B, C, D


class BaseDataModule:
    """
    Base data module class for creating data loaders.
    
    This class provides common functionality for creating train, validation,
    and test data loaders. Specific models can inherit from this class.
    """
    
    def __init__(self, config: dict):
        """
        Initialize base data module.
        
        Args:
            config: Configuration dictionary containing dataset parameters
        """
        self.config = config

    def train_dataloader(self):
        """Create training data loader."""
        raise NotImplementedError("Subclasses must implement train_dataloader")

    def val_dataloader(self):
        """Create validation data loader."""
        raise NotImplementedError("Subclasses must implement val_dataloader")

    def test_dataloader(self):
        """Create test data loader."""
        raise NotImplementedError("Subclasses must implement test_dataloader") 
    
    def get_slice_inferer(self):
        """Get SliceInferer for validation and test (if supported)."""
        # Check if SliceInferer should be used
        use_slice_inferer = self.config.get('use_slice_inferer', False)
        
        if use_slice_inferer:
            # Import here to avoid circular imports
            from monai.inferers import SliceInferer
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            slice_inferer = SliceInferer(
                roi_size=self.config.get('roi_size', (128, 128)),
                sw_batch_size=self.config.get('sw_batch_size', 4),
                spatial_dim=2,
                device=device,
                padding_mode="replicate",
            )
            return slice_inferer
        else:
            return None 
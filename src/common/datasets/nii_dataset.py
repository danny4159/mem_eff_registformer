"""
NII Dataset Module for PadainSynthesis

This module provides NII file-based dataset classes that handle medical images
with masking, preprocessing, and MONAI transforms.
"""

import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    EnsureTyped,
    Compose,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    Lambdad,
)
from monai.inferers import SliceInferer


def create_normalization_transform():
    """Create shared normalization transform for both MR and CT."""
    return Compose([
        # CT: HU clipping and normalization
        ScaleIntensityRanged(
            keys="ct",
            a_min=-1000.0, a_max=2000.0,
            b_min=-1.0, b_max=1.0,
            clip=True
        ),
        # MR: Simple 0-1 then -1 to 1 normalization
        ScaleIntensityd(keys="mr"),  # [0~1 Norm]
        Lambdad(keys=["mr"], func=lambda x: x * 2 - 1),  # 0~1 → -1~1
        EnsureTyped(keys=["mr", "ct", "mask"]),
    ])


def apply_masking(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Apply masking (mask=1: keep original, mask=0: set to -1)."""
    mask = data["mask"]
    data["mr"] = torch.where(mask == 1, data["mr"], torch.tensor(-1.0))
    data["ct"] = torch.where(mask == 1, data["ct"], torch.tensor(-1.0))
    return data


class NiiDataset(Dataset):
    """
    NII file-based dataset for medical images with masking and preprocessing.
    
    This dataset loads MR and CT images from NII files, applies masking,
    and performs preprocessing with MONAI transforms.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        is_3d: bool = False,
        flip_prob: float = 0.0,
        rot_prob: float = 0.0,
        reverse: bool = False,
        roi_size: Tuple[int, int] = (256, 256),
        sw_batch_size: int = 4,
    ):
        """
        Initialize NII dataset.
        
        Args:
            data_root: Root directory containing train/val/test folders
            split: Dataset split ("train", "val", "test")
            is_3d: Whether to use 3D data (False for 2D slices)
            flip_prob: Probability of horizontal flip
            rot_prob: Probability of rotation
            reverse: Whether to reverse source and target
            roi_size: ROI size for SliceInferer
            sw_batch_size: Sliding window batch size
        """
        self.data_root = Path(data_root)
        self.split = split
        self.is_3d = is_3d
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob
        self.reverse = reverse
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        
        # Get patient directories
        split_dir = self.data_root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        self.patient_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        self.patient_keys = [d.name for d in self.patient_dirs]
        
        # For 2D slice mode, calculate slice counts
        if not self.is_3d:
            self.slice_counts = []
            self.cumulative_slice_counts = [0]
            for patient_dir in self.patient_dirs:
                mr_path = patient_dir / "mr.nii.gz"
                if mr_path.exists():
                    img = nib.load(str(mr_path))
                    slice_count = img.shape[-1]  # Last dimension is depth
                    self.slice_counts.append(slice_count)
                    self.cumulative_slice_counts.append(
                        self.cumulative_slice_counts[-1] + slice_count
                    )
                else:
                    self.slice_counts.append(0)
                    self.cumulative_slice_counts.append(
                        self.cumulative_slice_counts[-1]
                    )
        
        # Setup transforms
        self.normalize_transform = create_normalization_transform()
        self.augmentation_transform = Compose([
            RandFlipd(keys=["mr", "ct", "mask"], prob=self.flip_prob, spatial_axis=1),
            RandRotate90d(keys=["mr", "ct", "mask"], prob=self.rot_prob, max_k=3),
        ])
    
    def __len__(self):
        """Return the total number of samples."""
        if self.is_3d:
            return len(self.patient_keys)
        else:
            return self.cumulative_slice_counts[-1]
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if self.is_3d:
            patient_dir = self.patient_dirs[idx]
            slice_idx = None
        else:
            patient_idx = np.searchsorted(self.cumulative_slice_counts, idx + 1) - 1
            slice_idx = idx - self.cumulative_slice_counts[patient_idx]
            patient_dir = self.patient_dirs[patient_idx]
        
        # Load and process data
        data = self._load_patient_data(patient_dir, slice_idx)
        data = self.normalize_transform(data)
        data = apply_masking(data)
        
        # Apply augmentation for training
        if self.split == "train":
            data = self.augmentation_transform(data)
        
        # Reverse if needed
        if self.reverse:
            data["mr"], data["ct"] = data["ct"], data["mr"]
        
        return data
    
    def _load_patient_data(self, patient_dir: Path, slice_idx: Optional[int] = None):
        """Load patient data from NII files using MONAI transforms."""
        mr_path = patient_dir / "mr.nii.gz"
        ct_path = patient_dir / "ct.nii.gz"
        mask_path = patient_dir / "mask.nii.gz"
        
        if not all(p.exists() for p in [mr_path, ct_path, mask_path]):
            raise FileNotFoundError(f"Missing files in {patient_dir}")
        
        # Create file dict for MONAI transforms
        file_dict = {
            "mr": str(mr_path),
            "ct": str(ct_path),
            "mask": str(mask_path)
        }
        
        # Define loading transforms
        load_transform = Compose([
            LoadImaged(keys=["mr", "ct", "mask"]),
            EnsureChannelFirstd(keys=["mr", "ct", "mask"]),
        ])
        
        # Load data using MONAI transforms
        data = load_transform(file_dict)
        
        # Extract slice if needed
        if slice_idx is not None:
            data["mr"] = data["mr"][..., slice_idx]
            data["ct"] = data["ct"][..., slice_idx]
            data["mask"] = data["mask"][..., slice_idx]
        
        return data


class NiiGridPatchDataset:
    """
    Efficient NII GridPatchDataset for training with 2D slices from 3D volumes.
    
    This implementation pre-loads and caches 2D slices for faster training.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        crop_size: Tuple[int, int] = (128, 128),
        flip_prob: float = 0.0,
        rot_prob: float = 0.0,
        reverse: bool = False,
    ):
        """Initialize efficient NII GridPatchDataset."""
        self.data_root = Path(data_root)
        self.split = split
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob
        self.reverse = reverse
        
        # Get patient directories
        split_dir = self.data_root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        self.patient_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        # Pre-load and cache all slices for efficiency
        print(f"Loading {split} data slices...")
        self._preload_slices()
        
        # Setup augmentation transforms
        self.augmentation_transform = Compose([
            RandFlipd(keys=["mr", "ct", "mask"], prob=self.flip_prob, spatial_axis=1),
            RandRotate90d(keys=["mr", "ct", "mask"], prob=self.rot_prob, max_k=3),
        ])
        
        print(f"Loaded {len(self.slices)} slices from {len(self.patient_dirs)} patients")
    
    def _preload_slices(self):
        """Pre-load all 2D slices from 3D volumes for efficiency."""
        self.slices = []
        
        # Define transforms for loading and preprocessing
        load_transform = Compose([
            LoadImaged(keys=["mr", "ct", "mask"]),
            EnsureChannelFirstd(keys=["mr", "ct", "mask"]),
        ])
        
        # Use shared normalization transform
        normalize_transform = create_normalization_transform()
        
        # Crop transform
        crop_transform = RandSpatialCropd(
            keys=["mr", "ct", "mask"], 
            roi_size=[self.crop_size[0], self.crop_size[1]], 
            random_center=True, 
            random_size=False
        )
        
        for patient_dir in self.patient_dirs:
            mr_path = patient_dir / "mr.nii.gz"
            ct_path = patient_dir / "ct.nii.gz"
            mask_path = patient_dir / "mask.nii.gz"
            
            if not all(p.exists() for p in [mr_path, ct_path, mask_path]):
                continue
            
            # Load 3D volume
            file_dict = {
                "mr": str(mr_path),
                "ct": str(ct_path),
                "mask": str(mask_path)
            }
            
            volume_data = load_transform(file_dict)
            volume_data = normalize_transform(volume_data)
            
            # Extract 2D slices from 3D volume
            _, _, _, depth = volume_data["mr"].shape  # (C, H, W, D)
            
            for slice_idx in range(depth):
                # Extract slice
                slice_data = {
                    "mr": volume_data["mr"][:, :, :, slice_idx],      # (C, H, W)
                    "ct": volume_data["ct"][:, :, :, slice_idx],      # (C, H, W)
                    "mask": volume_data["mask"][:, :, :, slice_idx]   # (C, H, W)
                }
                
                # Apply crop
                slice_data = crop_transform(slice_data)
                
                # Apply masking
                slice_data = apply_masking(slice_data)
                
                self.slices.append(slice_data)
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.slices)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        # Get pre-loaded slice
        data = self.slices[idx].copy()
        
        # Apply augmentation for training
        if self.split == "train":
            data = self.augmentation_transform(data)
        
        # Reverse if needed
        if self.reverse:
            data["mr"], data["ct"] = data["ct"], data["mr"]
        
        return data


class NiiDataModule:
    """
    NII data module for creating data loaders with SliceInferer support.
    
    This class provides functionality for creating train, validation,
    and test data loaders with proper MONAI integration.
    """
    
    def __init__(self, config: dict):
        """Initialize NII data module."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup SliceInferer for validation and test
        self.slice_inferer = SliceInferer(
            roi_size=self.config.get('roi_size', (256, 256)),
            sw_batch_size=self.config.get('sw_batch_size', 4),
            spatial_dim=2,
            device=self.device,
            padding_mode="replicate",
        )
    
    def train_dataloader(self):
        """Create training data loader."""
        # Check if we should use try dataset for training
        split_to_use = "train"
        if self.config.get('use_try_dataset', False):
            split_to_use = "try"
            print(f"INFO: Using '{split_to_use}' data for training (try dataset)")
        
        # For training, use GridPatchDataset for 2D slices from 3D volumes
        dataset = NiiGridPatchDataset(
            data_root=self.config['data_root'],
            split=split_to_use,
            crop_size=self.config.get('crop_size', (128, 128)),
            flip_prob=self.config.get('flip_prob', 0.0),
            rot_prob=self.config.get('rot_prob', 0.0),
            reverse=self.config.get('reverse', False),
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            pin_memory=self.config.get('pin_memory', False),
            shuffle=True,
        )
    
    def val_dataloader(self):
        """Create validation data loader."""
        # Check if we should use try dataset for validation
        split_to_use = "val"
        if self.config.get('use_try_dataset', False):
            split_to_use = "try"
            print(f"INFO: Using '{split_to_use}' data for validation (try dataset)")
        elif self.config.get('use_try_for_val_only', False):
            split_to_use = "try"
            print(f"INFO: Using '{split_to_use}' data for validation (val uses try, train uses train)")
        
        # For validation with SliceInferer, use 3D volumes
        dataset = NiiDataset(
            data_root=self.config['data_root'],
            split=split_to_use,
            is_3d=True,  # Use 3D volumes for validation
            flip_prob=0.0,  # No augmentation for validation
            rot_prob=0.0,
            reverse=self.config.get('reverse', False),
            roi_size=self.config.get('roi_size', (256, 256)),
            sw_batch_size=self.config.get('sw_batch_size', 4),
        )
        
        return DataLoader(
            dataset,
            batch_size=1,  # Single volume per batch for SliceInferer
            num_workers=self.config['num_workers'],
            pin_memory=self.config.get('pin_memory', False),
            shuffle=False,
        )
    
    def test_dataloader(self):
        """Create test data loader."""
        # For test with SliceInferer, use 3D volumes
        dataset = NiiDataset(
            data_root=self.config['data_root'],
            split="test",
            is_3d=True,  # Use 3D volumes for test
            flip_prob=0.0,  # No augmentation for test
            rot_prob=0.0,
            reverse=self.config.get('reverse', False),
            roi_size=self.config.get('roi_size', (256, 256)),
            sw_batch_size=self.config.get('sw_batch_size', 4),
        )
        
        return DataLoader(
            dataset,
            batch_size=1,  # Single volume per batch for SliceInferer
            num_workers=self.config['num_workers'],
            pin_memory=self.config.get('pin_memory', False),
            shuffle=False,
        )
    
    def get_slice_inferer(self):
        """Get SliceInferer for validation and test."""
        return self.slice_inferer


# Denormalization functions for CT
def denormalize_ct(normalized_ct: torch.Tensor) -> torch.Tensor:
    """
    Denormalize CT from [-1, 1] back to [-1000, 2000] range.
    
    Args:
        normalized_ct: CT tensor in [-1, 1] range
        
    Returns:
        Denormalized CT tensor in [-1000, 2000] range
    """
    # [-1, 1] → [0, 1]
    ct_01 = (normalized_ct + 1) / 2
    # [0, 1] → [-1000, 2000]
    ct_denorm = ct_01 * 3000 - 1000
    return ct_denorm


def denormalize_mr(normalized_mr: torch.Tensor) -> torch.Tensor:
    """
    Denormalize MR from [-1, 1] back to [0, 1] range.
    
    Args:
        normalized_mr: MR tensor in [-1, 1] range
        
    Returns:
        Denormalized MR tensor in [0, 1] range
    """
    # [-1, 1] → [0, 1]
    mr_denorm = (normalized_mr + 1) / 2
    return mr_denorm 
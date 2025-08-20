"""
Simple test script for NII Dataset

This script tests the NII dataset implementation without complex imports.
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    EnsureTyped,
    Compose,
    Lambdad,
)
from monai.inferers import SliceInferer
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Import our dataset classes directly
sys.path.append(str(Path(__file__).parent.parent.parent / "src" / "padain_synthesis"))
from nii_dataset import NiiDataset, NiiDataModule, denormalize_ct, denormalize_mr

def test_basic_functionality():
    """Test basic NII dataset functionality."""
    
    print("=== Testing Basic NII Dataset Functionality ===")
    
    # Test data root
    data_root = "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis"
    
    # Check if directories exist
    train_dir = Path(data_root) / "train"
    val_dir = Path(data_root) / "val"
    test_dir = Path(data_root) / "test"
    
    print(f"Train directory exists: {train_dir.exists()}")
    print(f"Val directory exists: {val_dir.exists()}")
    print(f"Test directory exists: {test_dir.exists()}")
    
    if train_dir.exists():
        patient_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        print(f"Number of training patients: {len(patient_dirs)}")
        
        if patient_dirs:
            # Check first patient
            first_patient = patient_dirs[0]
            print(f"First patient: {first_patient.name}")
            
            mr_path = first_patient / "mr.nii.gz"
            ct_path = first_patient / "ct.nii.gz"
            mask_path = first_patient / "mask.nii.gz"
            
            print(f"MR exists: {mr_path.exists()}")
            print(f"CT exists: {ct_path.exists()}")
            print(f"Mask exists: {mask_path.exists()}")
            
            if all(p.exists() for p in [mr_path, ct_path, mask_path]):
                # Load and check shapes
                mr_img = nib.load(str(mr_path))
                ct_img = nib.load(str(ct_path))
                mask_img = nib.load(str(mask_path))
                
                print(f"MR shape: {mr_img.shape}")
                print(f"CT shape: {ct_img.shape}")
                print(f"Mask shape: {mask_img.shape}")
                
                # Check data ranges
                mr_data = mr_img.get_fdata()
                ct_data = ct_img.get_fdata()
                mask_data = mask_img.get_fdata()
                
                print(f"MR range: [{mr_data.min():.1f}, {mr_data.max():.1f}]")
                print(f"CT range: [{ct_data.min():.1f}, {ct_data.max():.1f}]")
                print(f"Mask range: [{mask_data.min():.1f}, {mask_data.max():.1f}]")
                print(f"Mask unique values: {np.unique(mask_data)}")

def test_monai_transforms():
    """Test MONAI transforms separately."""
    
    print("\n=== Testing MONAI Transforms ===")
    
    # Create sample data (already loaded arrays, not file paths)
    # Add channel dimension manually
    sample_data = {
        "mr": np.random.rand(1, 256, 256, 64).astype(np.float32),  # (C, H, W, D)
        "ct": (np.random.rand(1, 256, 256, 64).astype(np.float32) * 3000 - 1000),  # CT-like range
        "mask": np.random.randint(0, 2, (1, 256, 256, 64)).astype(np.float32)
    }
    
    # Define transforms (without LoadImaged and EnsureChannelFirstd since we already have correct format)
    transform = Compose([
        ScaleIntensityRanged(
            keys="ct",
            a_min=-1000.0, a_max=2000.0,
            b_min=-1.0, b_max=1.0,
            clip=True
        ),
        ScaleIntensityd(keys="mr"),
        Lambdad(keys=["mr"], func=lambda x: x * 2 - 1),
        EnsureTyped(keys=["mr", "ct", "mask"]),
    ])
    
    # Apply transform
    transformed = transform(sample_data)
    
    print(f"Transformed MR shape: {transformed['mr'].shape}")
    print(f"Transformed CT shape: {transformed['ct'].shape}")
    print(f"Transformed Mask shape: {transformed['mask'].shape}")
    print(f"Transformed MR range: [{transformed['mr'].min():.3f}, {transformed['mr'].max():.3f}]")
    print(f"Transformed CT range: [{transformed['ct'].min():.3f}, {transformed['ct'].max():.3f}]")
    
    # Test masking
    print("\n--- Testing Masking ---")
    mask = transformed["mask"]
    mr_masked = transformed["mr"] * mask
    ct_masked = transformed["ct"] * mask
    
    print(f"Masked MR range: [{mr_masked.min():.3f}, {mr_masked.max():.3f}]")
    print(f"Masked CT range: [{ct_masked.min():.3f}, {ct_masked.max():.3f}]")
    print(f"Mask sum: {mask.sum()}")
    print(f"Masked MR non-zero: {(mr_masked != 0).sum()}")
    print(f"Masked CT non-zero: {(ct_masked != 0).sum()}")

def test_slice_inferer():
    """Test SliceInferer functionality."""
    
    print("\n=== Testing SliceInferer ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create SliceInferer
    slice_inferer = SliceInferer(
        roi_size=(128, 128),
        sw_batch_size=4,
        spatial_dim=2,
        device=device,
        padding_mode="replicate",
    )
    
    print(f"SliceInferer created successfully")
    print(f"ROI size: {slice_inferer.roi_size}")
    print(f"SW batch size: {slice_inferer.sw_batch_size}")

def test_denormalization():
    """Test denormalization functions."""
    
    print("\n=== Testing Denormalization ===")
    
    # Create normalized tensors
    normalized_ct = torch.rand(1, 1, 128, 128) * 2 - 1  # [-1, 1]
    normalized_mr = torch.rand(1, 1, 128, 128) * 2 - 1  # [-1, 1]
    
    # Denormalize
    ct_denorm = denormalize_ct(normalized_ct)
    mr_denorm = denormalize_mr(normalized_mr)
    
    print(f"CT normalized range: [{normalized_ct.min():.3f}, {normalized_ct.max():.3f}]")
    print(f"CT denormalized range: [{ct_denorm.min():.1f}, {ct_denorm.max():.1f}]")
    print(f"MR normalized range: [{normalized_mr.min():.3f}, {normalized_mr.max():.3f}]")
    print(f"MR denormalized range: [{mr_denorm.min():.3f}, {mr_denorm.max():.3f}]")

if __name__ == "__main__":
    test_basic_functionality()
    test_monai_transforms()
    test_slice_inferer()
    test_denormalization()
    print("\n=== All tests completed! ===") 
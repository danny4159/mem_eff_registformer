"""
Test script for NiiGridPatchDataset

This script tests the NiiGridPatchDataset implementation with GridPatchDataset.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import autorootcwd
from src.common.nii_dataset import NiiGridPatchDataset, NiiDataModule, denormalize_ct, denormalize_mr

def test_gridpatch_dataset():
    """Test NiiGridPatchDataset functionality."""
    
    print("=== Testing NiiGridPatchDataset ===")
    
    # Configuration
    data_root = "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis"
    
    # Test GridPatchDataset
    print("\n1. Testing GridPatchDataset")
    gridpatch_dataset = NiiGridPatchDataset(
        data_root=data_root,
        split="train",
        crop_size=(128, 128),
        flip_prob=0.0,
        rot_prob=0.0,
    )
    
    print(f"GridPatchDataset length: {len(gridpatch_dataset)}")
    
    # Get a sample
    sample = gridpatch_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"MR shape: {sample['mr'].shape}")
    print(f"CT shape: {sample['ct'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"MR range: [{sample['mr'].min():.3f}, {sample['mr'].max():.3f}]")
    print(f"CT range: [{sample['ct'].min():.3f}, {sample['ct'].max():.3f}]")
    print(f"Mask range: [{sample['mask'].min():.3f}, {sample['mask'].max():.3f}]")
    
    # Test multiple samples
    print("\n2. Testing multiple samples")
    for i in range(min(5, len(gridpatch_dataset))):
        sample = gridpatch_dataset[i]
        print(f"Sample {i}: MR shape {sample['mr'].shape}, CT shape {sample['ct'].shape}")
    
    return gridpatch_dataset

def test_data_module_with_gridpatch():
    """Test NiiDataModule with GridPatchDataset."""
    
    print("\n3. Testing DataModule with GridPatchDataset")
    
    # Configuration
    config = {
        'data_root': "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis",
        'batch_size': 2,
        'num_workers': 0,
        'padding_size': None,
        'crop_size': (128, 128),
        'flip_prob': 0.0,
        'rot_prob': 0.0,
        'reverse': False,
        'pin_memory': False,
        'roi_size': (128, 128),
        'sw_batch_size': 4,
    }
    
    data_module = NiiDataModule(config)
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")
    
    # Get a batch from train loader
    train_batch = next(iter(train_loader))
    print(f"Train batch keys: {list(train_batch.keys())}")
    print(f"Train batch MR shape: {train_batch['mr'].shape}")
    print(f"Train batch CT shape: {train_batch['ct'].shape}")
    
    # Get a batch from val loader
    val_batch = next(iter(val_loader))
    print(f"Val batch keys: {list(val_batch.keys())}")
    print(f"Val batch MR shape: {val_batch['mr'].shape}")
    print(f"Val batch CT shape: {val_batch['ct'].shape}")
    
    return data_module

def visualize_gridpatch_samples(dataset, num_samples=3):
    """Visualize sample images from the GridPatchDataset."""
    
    print(f"\n4. Visualizing {num_samples} GridPatch samples")
    
    # Create output directory
    output_dir = Path("scripts/unit_test/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize samples
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # Convert tensors to numpy for visualization
        mr = sample['mr'].squeeze().numpy()
        ct = sample['ct'].squeeze().numpy()
        mask = sample['mask'].squeeze().numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot MR
        im1 = axes[0].imshow(mr, cmap='gray')
        axes[0].set_title(f'MR (range: [{mr.min():.3f}, {mr.max():.3f}])')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot CT
        im2 = axes[1].imshow(ct, cmap='gray')
        axes[1].set_title(f'CT (range: [{ct.min():.3f}, {ct.max():.3f}])')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot Mask
        im3 = axes[2].imshow(mask, cmap='gray')
        axes[2].set_title(f'Mask (range: [{mask.min():.3f}, {mask.max():.3f}])')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(output_dir / f"gridpatch_sample_{i:02d}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved gridpatch_sample_{i:02d}.png")

def test_denormalization():
    """Test denormalization functions."""
    
    print("\n5. Testing Denormalization")
    
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
    # Run all tests
    gridpatch_dataset = test_gridpatch_dataset()
    data_module = test_data_module_with_gridpatch()
    visualize_gridpatch_samples(gridpatch_dataset)
    test_denormalization()
    
    print("\n=== All GridPatchDataset tests completed successfully! ===") 
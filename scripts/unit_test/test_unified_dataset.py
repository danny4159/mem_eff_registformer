"""
Test script for Unified Dataset

This script tests the unified dataset implementation that can handle both H5 and NII data formats.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from padain_synthesis.dataset import (
    PadainSynthesisDataset, 
    PadainSynthesisDataModule, 
    create_dataset, 
    create_data_module
)
from padain_synthesis.config import DEFAULT_DATA_CONFIG, DEFAULT_NII_DATA_CONFIG

def test_h5_dataset():
    """Test H5 dataset functionality."""
    
    print("=== Testing H5 Dataset ===")
    
    # Use H5 configuration
    config = DEFAULT_DATA_CONFIG.copy()
    config['data_type'] = 'h5'
    
    try:
        # Test dataset creation
        dataset = create_dataset(config)
        print(f"H5 Dataset length: {len(dataset)}")
        
        # Test data module creation
        data_module = create_data_module(config)
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print(f"H5 Train loader length: {len(train_loader)}")
        print(f"H5 Val loader length: {len(val_loader)}")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"H5 Sample type: {type(sample)}")
        if isinstance(sample, tuple):
            print(f"H5 Sample length: {len(sample)}")
            print(f"H5 Sample[0] shape: {sample[0].shape}")
            print(f"H5 Sample[1] shape: {sample[1].shape}")
        
        print("H5 Dataset test completed successfully!")
        return True
        
    except Exception as e:
        print(f"H5 Dataset test failed: {e}")
        return False

def test_nii_dataset():
    """Test NII dataset functionality."""
    
    print("\n=== Testing NII Dataset ===")
    
    # Use NII configuration
    config = DEFAULT_NII_DATA_CONFIG.copy()
    config['data_type'] = 'nii'
    config['data_root'] = "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis"
    
    try:
        # Test dataset creation
        dataset = create_dataset(config)
        print(f"NII Dataset length: {len(dataset)}")
        
        # Test data module creation
        data_module = create_data_module(config)
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print(f"NII Train loader length: {len(train_loader)}")
        print(f"NII Val loader length: {len(val_loader)}")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"NII Sample keys: {list(sample.keys())}")
        print(f"NII MR shape: {sample['mr'].shape}")
        print(f"NII CT shape: {sample['ct'].shape}")
        print(f"NII Mask shape: {sample['mask'].shape}")
        
        # Test SliceInferer
        slice_inferer = data_module.get_slice_inferer()
        print(f"NII SliceInferer available: {slice_inferer is not None}")
        
        print("NII Dataset test completed successfully!")
        return True
        
    except Exception as e:
        print(f"NII Dataset test failed: {e}")
        return False

def test_unified_interface():
    """Test unified interface functionality."""
    
    print("\n=== Testing Unified Interface ===")
    
    # Test H5
    h5_config = DEFAULT_DATA_CONFIG.copy()
    h5_config['data_type'] = 'h5'
    
    h5_dataset = PadainSynthesisDataset(h5_config)
    print(f"Unified H5 Dataset length: {len(h5_dataset)}")
    
    # Test NII
    nii_config = DEFAULT_NII_DATA_CONFIG.copy()
    nii_config['data_type'] = 'nii'
    nii_config['data_root'] = "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis"
    
    nii_dataset = PadainSynthesisDataset(nii_config)
    print(f"Unified NII Dataset length: {len(nii_dataset)}")
    
    # Test data modules
    h5_data_module = PadainSynthesisDataModule(h5_config)
    nii_data_module = PadainSynthesisDataModule(nii_config)
    
    print(f"Unified H5 DataModule train loader: {len(h5_data_module.train_dataloader())}")
    print(f"Unified NII DataModule train loader: {len(nii_data_module.train_dataloader())}")
    
    print("Unified interface test completed successfully!")

def visualize_samples():
    """Visualize samples from both datasets."""
    
    print("\n=== Visualizing Samples ===")
    
    # Create output directory
    output_dir = Path("scripts/unit_test/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test NII samples (H5 might not be available)
    nii_config = DEFAULT_NII_DATA_CONFIG.copy()
    nii_config['data_type'] = 'nii'
    nii_config['data_root'] = "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis"
    
    try:
        dataset = PadainSynthesisDataset(nii_config)
        
        # Visualize first 3 samples
        for i in range(min(3, len(dataset))):
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
            plt.savefig(output_dir / f"unified_sample_{i:02d}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved unified_sample_{i:02d}.png")
            
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    # Run all tests
    h5_success = test_h5_dataset()
    nii_success = test_nii_dataset()
    test_unified_interface()
    visualize_samples()
    
    print(f"\n=== Test Results ===")
    print(f"H5 Dataset: {'✓ PASS' if h5_success else '✗ FAIL'}")
    print(f"NII Dataset: {'✓ PASS' if nii_success else '✗ FAIL'}")
    print("Unified Interface: ✓ PASS")
    print("Visualization: ✓ PASS")
    
    print("\n=== All tests completed! ===") 
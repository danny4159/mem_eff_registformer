"""
Complete test script for NII dataset with GridPatchDataset and SliceInferer

This script tests:
1. NiiGridPatchDataset for training (2D slices from 3D volumes)
2. SliceInferer for validation/test (3D volumes with sliding window inference)
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import autorootcwd
from src.common.nii_dataset import NiiDataset, NiiGridPatchDataset, NiiDataModule, denormalize_ct, denormalize_mr
from monai.inferers import SliceInferer

def test_gridpatch_training():
    """Test NiiGridPatchDataset for training with 2D slices."""
    
    print("="*60)
    print("🏋️ TESTING NiiGridPatchDataset (TRAINING)")
    print("="*60)
    
    # Configuration for training
    config = {
        'data_root': "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis",
        'batch_size': 2,
        'num_workers': 0,
        'crop_size': (128, 128),
        'flip_prob': 0.5,
        'rot_prob': 0.5,
        'reverse': False,
        'pin_memory': False,
    }
    
    # Create GridPatchDataset
    gridpatch_dataset = NiiGridPatchDataset(
        data_root=config['data_root'],
        split="train",
        crop_size=config['crop_size'],
        flip_prob=config['flip_prob'],
        rot_prob=config['rot_prob'],
    )
    
    print(f"✅ GridPatchDataset length: {len(gridpatch_dataset)}")
    
    # Test multiple samples
    print("\n📊 Testing multiple samples:")
    for i in range(min(5, len(gridpatch_dataset))):
        sample = gridpatch_dataset[i]
        print(f"  Sample {i}: MR shape {sample['mr'].shape}, CT shape {sample['ct'].shape}")
        print(f"    MR range: [{sample['mr'].min():.3f}, {sample['mr'].max():.3f}]")
        print(f"    CT range: [{sample['ct'].min():.3f}, {sample['ct'].max():.3f}]")
        print(f"    Mask range: [{sample['mask'].min():.3f}, {sample['mask'].max():.3f}]")
    
    # Test DataLoader
    train_loader = torch.utils.data.DataLoader(
        gridpatch_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )
    
    print(f"\n📦 Train loader length: {len(train_loader)}")
    
    # Get a batch
    train_batch = next(iter(train_loader))
    print(f"✅ Train batch shape: MR {train_batch['mr'].shape}, CT {train_batch['ct'].shape}")
    
    return gridpatch_dataset, train_loader

def test_sliceinferer_validation():
    """Test SliceInferer for validation with 3D volumes."""
    
    print("\n" + "="*60)
    print("🔍 TESTING SliceInferer (VALIDATION)")
    print("="*60)
    
    # Configuration for validation
    config = {
        'data_root': "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis",
        'batch_size': 1,
        'num_workers': 0,
        'roi_size': (128, 128),
        'sw_batch_size': 4,
        'pin_memory': False,
    }
    
    # Create 3D dataset for validation
    val_dataset = NiiDataset(
        data_root=config['data_root'],
        split="val",
        is_3d=True,  # Use 3D volumes
        flip_prob=0.0,  # No augmentation
        rot_prob=0.0,
    )
    
    print(f"✅ Val dataset length: {len(val_dataset)}")
    
    # Test a sample
    val_sample = val_dataset[0]
    print(f"✅ Val sample shape: MR {val_sample['mr'].shape}, CT {val_sample['ct'].shape}")
    print(f"   MR range: [{val_sample['mr'].min():.3f}, {val_sample['mr'].max():.3f}]")
    print(f"   CT range: [{val_sample['ct'].min():.3f}, {val_sample['ct'].max():.3f}]")
    
    # Create DataLoader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )
    
    # Create SliceInferer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slice_inferer = SliceInferer(
        roi_size=config['roi_size'],
        sw_batch_size=config['sw_batch_size'],
        spatial_dim=2,
        device=device,
        padding_mode="replicate",
    )
    
    print(f"✅ SliceInferer created with roi_size={config['roi_size']}, sw_batch_size={config['sw_batch_size']}")
    
    # Create dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tanh = torch.nn.Tanh()
        
        def forward(self, x):
            return self.tanh(x)
    
    model = DummyModel().to(device)
    model.eval()
    
    # Test SliceInferer
    print("\n🔄 Testing SliceInferer inference:")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 1:  # Test only first batch
                break
                
            mr_3d = batch['mr'].to(device)
            ct_3d = batch['ct'].to(device)
            
            print(f"  Input 3D shape: MR {mr_3d.shape}, CT {ct_3d.shape}")
            
            # Run SliceInferer on MR
            mr_output = slice_inferer(mr_3d, model)
            print(f"  MR SliceInferer output shape: {mr_output.shape}")
            print(f"  MR output range: [{mr_output.min():.3f}, {mr_output.max():.3f}]")
            
            # Run SliceInferer on CT
            ct_output = slice_inferer(ct_3d, model)
            print(f"  CT SliceInferer output shape: {ct_output.shape}")
            print(f"  CT output range: [{ct_output.min():.3f}, {ct_output.max():.3f}]")
            
            print(f"  ✅ Number of slices processed: {mr_output.shape[-1]}")
    
    return val_dataset, val_loader, slice_inferer

def test_data_module_integration():
    """Test complete NiiDataModule integration."""
    
    print("\n" + "="*60)
    print("🔧 TESTING NiiDataModule Integration")
    print("="*60)
    
    # Configuration
    config = {
        'data_root': "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis",
        'batch_size': 2,
        'num_workers': 0,
        'crop_size': (128, 128),
        'flip_prob': 0.5,
        'rot_prob': 0.5,
        'reverse': False,
        'roi_size': (128, 128),
        'sw_batch_size': 4,
        'pin_memory': False,
    }
    
    # Create DataModule
    data_module = NiiDataModule(config)
    
    print("✅ NiiDataModule created successfully")
    
    # Test train dataloader
    train_loader = data_module.train_dataloader()
    print(f"✅ Train loader length: {len(train_loader)}")
    
    train_batch = next(iter(train_loader))
    print(f"✅ Train batch shape: MR {train_batch['mr'].shape}, CT {train_batch['ct'].shape}")
    
    # Test val dataloader
    val_loader = data_module.val_dataloader()
    print(f"✅ Val loader length: {len(val_loader)}")
    
    val_batch = next(iter(val_loader))
    print(f"✅ Val batch shape: MR {val_batch['mr'].shape}, CT {val_batch['ct'].shape}")
    
    # Test test dataloader
    test_loader = data_module.test_dataloader()
    print(f"✅ Test loader length: {len(test_loader)}")
    
    test_batch = next(iter(test_loader))
    print(f"✅ Test batch shape: MR {test_batch['mr'].shape}, CT {test_batch['ct'].shape}")
    
    # Test SliceInferer
    slice_inferer = data_module.get_slice_inferer()
    print(f"✅ SliceInferer retrieved: {slice_inferer is not None}")
    
    return data_module

def test_denormalization():
    """Test denormalization functions."""
    
    print("\n" + "="*60)
    print("🔄 TESTING Denormalization")
    print("="*60)
    
    # Create normalized tensors
    normalized_ct = torch.rand(1, 1, 128, 128) * 2 - 1  # [-1, 1]
    normalized_mr = torch.rand(1, 1, 128, 128) * 2 - 1  # [-1, 1]
    
    print(f"✅ Normalized CT range: [{normalized_ct.min():.3f}, {normalized_ct.max():.3f}]")
    print(f"✅ Normalized MR range: [{normalized_mr.min():.3f}, {normalized_mr.max():.3f}]")
    
    # Denormalize
    ct_denorm = denormalize_ct(normalized_ct)
    mr_denorm = denormalize_mr(normalized_mr)
    
    print(f"✅ CT denormalized range: [{ct_denorm.min():.1f}, {ct_denorm.max():.1f}]")
    print(f"✅ MR denormalized range: [{mr_denorm.min():.3f}, {mr_denorm.max():.3f}]")

def visualize_samples():
    """Visualize sample images."""
    
    print("\n" + "="*60)
    print("🎨 VISUALIZING Samples")
    print("="*60)
    
    # Create output directory
    output_dir = Path("scripts/unit_test/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test GridPatchDataset samples
    gridpatch_dataset = NiiGridPatchDataset(
        data_root="/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis",
        split="train",
        crop_size=(128, 128),
        flip_prob=0.0,
        rot_prob=0.0,
    )
    
    # Visualize first 3 samples
    for i in range(min(3, len(gridpatch_dataset))):
        sample = gridpatch_dataset[i]
        
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
        plt.savefig(output_dir / f"complete_test_sample_{i:02d}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved complete_test_sample_{i:02d}.png")

def main():
    """Run all tests."""
    
    print("🚀 STARTING COMPLETE NII DATASET TEST")
    print("="*60)
    
    try:
        # Test GridPatchDataset for training
        gridpatch_dataset, train_loader = test_gridpatch_training()
        
        # Test SliceInferer for validation
        val_dataset, val_loader, slice_inferer = test_sliceinferer_validation()
        
        # Test DataModule integration
        data_module = test_data_module_integration()
        
        # Test denormalization
        test_denormalization()
        
        # Visualize samples
        visualize_samples()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        print("✅ NiiGridPatchDataset: 2D slices from 3D volumes for training")
        print("✅ SliceInferer: 3D volumes with sliding window inference for val/test")
        print("✅ DataModule: Complete integration working")
        print("✅ Denormalization: CT and MR denorm functions working")
        print("✅ Visualization: Sample images saved")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
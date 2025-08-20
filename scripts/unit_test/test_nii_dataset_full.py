"""
Full test script for NII Dataset

This script tests the complete NII dataset implementation with real data.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Import our dataset classes directly
sys.path.append(str(Path(__file__).parent.parent.parent / "src" / "padain_synthesis"))
from nii_dataset import NiiDataset, NiiDataModule, denormalize_ct, denormalize_mr

def test_nii_dataset_full():
    """Test complete NII dataset functionality."""
    
    print("=== Testing Complete NII Dataset ===")
    
    # Configuration
    data_root = "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis"
    
    # Test training dataset (2D slices)
    print("\n1. Testing Training Dataset (2D slices)")
    train_dataset = NiiDataset(
        data_root=data_root,
        split="train",
        is_3d=False,
        crop_size=(128, 128),
        flip_prob=0.0,
        rot_prob=0.0,
    )
    
    print(f"Training dataset length: {len(train_dataset)}")
    
    # Get a sample
    sample = train_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"MR shape: {sample['mr'].shape}")
    print(f"CT shape: {sample['ct'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"MR range: [{sample['mr'].min():.3f}, {sample['mr'].max():.3f}]")
    print(f"CT range: [{sample['ct'].min():.3f}, {sample['ct'].max():.3f}]")
    print(f"Mask range: [{sample['mask'].min():.3f}, {sample['mask'].max():.3f}]")
    
    # Test validation dataset (3D volumes)
    print("\n2. Testing Validation Dataset (3D volumes)")
    val_dataset = NiiDataset(
        data_root=data_root,
        split="val",
        is_3d=True,
    )
    
    print(f"Validation dataset length: {len(val_dataset)}")
    
    # Get a sample
    sample_3d = val_dataset[0]
    print(f"3D Sample keys: {list(sample_3d.keys())}")
    print(f"3D MR shape: {sample_3d['mr'].shape}")
    print(f"3D CT shape: {sample_3d['ct'].shape}")
    print(f"3D Mask shape: {sample_3d['mask'].shape}")
    
    # Test denormalization
    print("\n3. Testing Denormalization")
    ct_denorm = denormalize_ct(sample['ct'])
    mr_denorm = denormalize_mr(sample['mr'])
    
    print(f"CT denormalized range: [{ct_denorm.min():.1f}, {ct_denorm.max():.1f}]")
    print(f"MR denormalized range: [{mr_denorm.min():.3f}, {mr_denorm.max():.3f}]")
    
    return train_dataset, val_dataset

def test_data_module():
    """Test NiiDataModule functionality."""
    
    print("\n4. Testing DataModule")
    
    # Configuration
    config = {
        'data_root': "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis",
        'batch_size': 1,
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
    
    # Test SliceInferer
    print("\n5. Testing SliceInferer")
    slice_inferer = data_module.get_slice_inferer()
    print(f"SliceInferer created: {slice_inferer is not None}")
    
    return data_module

def visualize_samples(train_dataset, num_samples=3):
    """Visualize sample images from the dataset."""
    
    print(f"\n6. Visualizing {num_samples} samples")
    
    # Create output directory
    output_dir = Path("scripts/unit_test/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize samples
    for i in range(min(num_samples, len(train_dataset))):
        sample = train_dataset[i]
        
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
        plt.savefig(output_dir / f"nii_sample_{i:02d}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved nii_sample_{i:02d}.png")

def test_slice_inferer_inference(data_module):
    """Test SliceInferer with a simple model."""
    
    print("\n7. Testing SliceInferer Inference")
    
    # Simple model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)
            
        def forward(self, x):
            return torch.tanh(self.conv(x))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    model.eval()
    
    # Get validation data
    val_loader = data_module.val_dataloader()
    val_batch = next(iter(val_loader))
    
    # Get SliceInferer
    slice_inferer = data_module.get_slice_inferer()
    
    # Test inference
    with torch.no_grad():
        input_data = val_batch['mr'].to(device)
        print(f"Input shape: {input_data.shape}")
        
        # Use SliceInferer for inference
        output = slice_inferer(input_data, model)
        print(f"Output shape: {output.shape}")
        
        print("SliceInferer inference successful!")

if __name__ == "__main__":
    # Run all tests
    train_dataset, val_dataset = test_nii_dataset_full()
    data_module = test_data_module()
    visualize_samples(train_dataset)
    test_slice_inferer_inference(data_module)
    
    print("\n=== All tests completed successfully! ===") 
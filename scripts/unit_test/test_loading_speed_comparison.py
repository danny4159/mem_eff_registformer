"""
Loading Speed Comparison Test

This script compares the loading speed between H5 and NII datasets
for training data to identify performance bottlenecks.
"""

import sys
import time
import torch
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.h5_dataset import H5_Dataset
from src.common.nii_dataset import NiiGridPatchDataset


def time_dataset_loading(dataset, name, num_samples=100):
    """
    Time dataset loading performance.
    
    Args:
        dataset: Dataset to test
        name: Name for logging
        num_samples: Number of samples to test
    
    Returns:
        Dictionary with timing results
    """
    print(f"\n🚀 Testing {name} Dataset Loading Speed")
    print("=" * 50)
    
    # Test individual sample loading
    sample_times = []
    print(f"Testing individual sample loading ({num_samples} samples)...")
    
    start_time = time.time()
    for i in range(min(num_samples, len(dataset))):
        sample_start = time.time()
        try:
            sample = dataset[i]
            sample_end = time.time()
            sample_times.append(sample_end - sample_start)
            
            if i % 20 == 0:
                print(f"  Sample {i}: {sample_times[-1]:.4f}s")
                
        except Exception as e:
            print(f"  Error loading sample {i}: {e}")
            break
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    if sample_times:
        avg_sample_time = np.mean(sample_times)
        min_sample_time = np.min(sample_times)
        max_sample_time = np.max(sample_times)
        std_sample_time = np.std(sample_times)
    else:
        avg_sample_time = min_sample_time = max_sample_time = std_sample_time = 0
    
    results = {
        'name': name,
        'total_samples': len(sample_times),
        'total_time': total_time,
        'avg_sample_time': avg_sample_time,
        'min_sample_time': min_sample_time,
        'max_sample_time': max_sample_time,
        'std_sample_time': std_sample_time,
        'samples_per_second': len(sample_times) / total_time if total_time > 0 else 0
    }
    
    print(f"\n📊 {name} Results:")
    print(f"  Total samples loaded: {results['total_samples']}")
    print(f"  Total time: {results['total_time']:.2f}s")
    print(f"  Average time per sample: {results['avg_sample_time']:.4f}s")
    print(f"  Min time per sample: {results['min_sample_time']:.4f}s")
    print(f"  Max time per sample: {results['max_sample_time']:.4f}s")
    print(f"  Std deviation: {results['std_sample_time']:.4f}s")
    print(f"  Samples per second: {results['samples_per_second']:.2f}")
    
    return results


def time_dataloader_performance(dataset, name, batch_size=2, num_workers=2, num_batches=10):
    """
    Time DataLoader performance.
    
    Args:
        dataset: Dataset to test
        name: Name for logging
        batch_size: Batch size for DataLoader
        num_workers: Number of workers
        num_batches: Number of batches to test
    
    Returns:
        Dictionary with timing results
    """
    print(f"\n🔄 Testing {name} DataLoader Performance")
    print("=" * 50)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )
    
    batch_times = []
    print(f"Testing DataLoader with batch_size={batch_size}, num_workers={num_workers}")
    
    start_time = time.time()
    try:
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            batch_start = time.time()
            # Simulate some processing
            if isinstance(batch, dict):
                # NII format
                mr_shape = batch["mr"].shape
                ct_shape = batch["ct"].shape
                print(f"  Batch {i}: MR {mr_shape}, CT {ct_shape}")
            else:
                # H5 format (tuple)
                mr_shape = batch[0].shape
                ct_shape = batch[1].shape
                print(f"  Batch {i}: MR {mr_shape}, CT {ct_shape}")
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
    except Exception as e:
        print(f"  Error in DataLoader: {e}")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    if batch_times:
        avg_batch_time = np.mean(batch_times)
        min_batch_time = np.min(batch_times)
        max_batch_time = np.max(batch_times)
    else:
        avg_batch_time = min_batch_time = max_batch_time = 0
    
    results = {
        'name': name,
        'total_batches': len(batch_times),
        'batch_size': batch_size,
        'num_workers': num_workers,
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'min_batch_time': min_batch_time,
        'max_batch_time': max_batch_time,
        'batches_per_second': len(batch_times) / total_time if total_time > 0 else 0
    }
    
    print(f"\n📊 {name} DataLoader Results:")
    print(f"  Total batches: {results['total_batches']}")
    print(f"  Total time: {results['total_time']:.2f}s")
    print(f"  Average time per batch: {results['avg_batch_time']:.4f}s")
    print(f"  Min time per batch: {results['min_batch_time']:.4f}s")
    print(f"  Max time per batch: {results['max_batch_time']:.4f}s")
    print(f"  Batches per second: {results['batches_per_second']:.2f}")
    
    return results


def main():
    """Main comparison test."""
    print("🔍 Dataset Loading Speed Comparison Test")
    print("=" * 60)
    
    # Configuration
    h5_config = {
        'h5_file_path': "./data/synthrad2023_mr-ct_pelvis/train/Ver3_OnlyOnePatient.h5",
        'data_group_1': "MR",
        'data_group_2': "CT",
        'data_group_3': None,
        'data_group_4': None,
        'data_group_5': None,
        'is_3d': False,
        'padding_size': None,
        'crop_size': (128, 128),  # Random crop
        'flip_prob': 0.0,
        'rot_prob': 0.0,
        'reverse': False,
        'norm_ZeroToOne': False,
    }
    
    nii_config = {
        'data_root': "./data/synthrad2023_mr-ct_pelvis",
        'split': "train",
        'crop_size': (128, 128),  # GridPatch crop
        'flip_prob': 0.0,
        'rot_prob': 0.0,
        'reverse': False,
    }
    
    # Test parameters
    num_samples = 50  # Number of individual samples to test
    num_batches = 10  # Number of batches to test
    batch_size = 2
    num_workers = 2
    
    results = {}
    
    try:
        # Test H5 Dataset
        print("\n🗂️  Initializing H5 Dataset...")
        h5_dataset = H5_Dataset(**h5_config)
        print(f"H5 Dataset size: {len(h5_dataset)} samples")
        
        # Time H5 individual loading
        h5_individual_results = time_dataset_loading(h5_dataset, "H5", num_samples)
        results['h5_individual'] = h5_individual_results
        
        # Time H5 DataLoader
        h5_dataloader_results = time_dataloader_performance(
            h5_dataset, "H5", batch_size, num_workers, num_batches
        )
        results['h5_dataloader'] = h5_dataloader_results
        
    except Exception as e:
        print(f"❌ Error testing H5 dataset: {e}")
        results['h5_individual'] = None
        results['h5_dataloader'] = None
    
    try:
        # Test NII Dataset
        print("\n🗂️  Initializing NII GridPatch Dataset...")
        nii_dataset = NiiGridPatchDataset(**nii_config)
        print(f"NII Dataset size: {len(nii_dataset)} samples")
        
        # Time NII individual loading
        nii_individual_results = time_dataset_loading(nii_dataset, "NII GridPatch", num_samples)
        results['nii_individual'] = nii_individual_results
        
        # Time NII DataLoader
        nii_dataloader_results = time_dataloader_performance(
            nii_dataset, "NII GridPatch", batch_size, num_workers, num_batches
        )
        results['nii_dataloader'] = nii_dataloader_results
        
    except Exception as e:
        print(f"❌ Error testing NII dataset: {e}")
        results['nii_individual'] = None
        results['nii_dataloader'] = None
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("📋 SUMMARY COMPARISON")
    print("=" * 60)
    
    if results['h5_individual'] and results['nii_individual']:
        h5_speed = results['h5_individual']['samples_per_second']
        nii_speed = results['nii_individual']['samples_per_second']
        
        print(f"\n🔍 Individual Sample Loading:")
        print(f"  H5 Dataset:        {h5_speed:.2f} samples/sec")
        print(f"  NII GridPatch:     {nii_speed:.2f} samples/sec")
        
        if h5_speed > 0 and nii_speed > 0:
            if h5_speed > nii_speed:
                speedup = h5_speed / nii_speed
                print(f"  → H5 is {speedup:.2f}x faster")
            else:
                speedup = nii_speed / h5_speed
                print(f"  → NII is {speedup:.2f}x faster")
    
    if results['h5_dataloader'] and results['nii_dataloader']:
        h5_batch_speed = results['h5_dataloader']['batches_per_second']
        nii_batch_speed = results['nii_dataloader']['batches_per_second']
        
        print(f"\n🔄 DataLoader Performance:")
        print(f"  H5 Dataset:        {h5_batch_speed:.2f} batches/sec")
        print(f"  NII GridPatch:     {nii_batch_speed:.2f} batches/sec")
        
        if h5_batch_speed > 0 and nii_batch_speed > 0:
            if h5_batch_speed > nii_batch_speed:
                speedup = h5_batch_speed / nii_batch_speed
                print(f"  → H5 is {speedup:.2f}x faster")
            else:
                speedup = nii_batch_speed / h5_batch_speed
                print(f"  → NII is {speedup:.2f}x faster")
    
    # Memory usage comparison
    print(f"\n💾 Memory Usage Comparison:")
    if results['h5_individual']:
        print(f"  H5: Loads individual slices on-demand")
    if results['nii_individual']:
        print(f"  NII GridPatch: Pre-loads all {len(nii_dataset)} slices in memory")
    
    print(f"\n✅ Test completed!")
    
    return results


if __name__ == "__main__":
    results = main() 
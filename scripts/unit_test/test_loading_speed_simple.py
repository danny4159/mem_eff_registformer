"""
Simple Loading Speed Comparison Test

This script compares the loading speed between H5 and NII datasets
focusing on individual sample loading to avoid DataLoader complications.
"""

import sys
import time
import torch
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.h5_dataset import H5_Dataset
from src.common.nii_dataset import NiiGridPatchDataset


def simple_speed_test(dataset, name, num_samples=20):
    """
    Simple speed test for dataset loading.
    
    Args:
        dataset: Dataset to test
        name: Name for logging
        num_samples: Number of samples to test
    """
    print(f"\n🚀 Testing {name}")
    print("=" * 40)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Test individual sample loading
    times = []
    
    print(f"Testing {num_samples} samples...")
    for i in range(min(num_samples, len(dataset))):
        start = time.time()
        try:
            sample = dataset[i]
            end = time.time()
            elapsed = end - start
            times.append(elapsed)
            
            # Print sample info
            if isinstance(sample, dict):
                # NII format
                mr_shape = sample["mr"].shape if "mr" in sample else "N/A"
                ct_shape = sample["ct"].shape if "ct" in sample else "N/A"
                print(f"  Sample {i}: {elapsed:.4f}s - MR: {mr_shape}, CT: {ct_shape}")
            else:
                # H5 format (tuple)
                mr_shape = sample[0].shape if sample[0] is not None else "N/A"
                ct_shape = sample[1].shape if sample[1] is not None else "N/A"
                print(f"  Sample {i}: {elapsed:.4f}s - MR: {mr_shape}, CT: {ct_shape}")
                
        except Exception as e:
            print(f"  Sample {i}: ERROR - {e}")
            break
    
    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        total_time = np.sum(times)
        samples_per_sec = len(times) / total_time if total_time > 0 else 0
        
        print(f"\n📊 {name} Results:")
        print(f"  Samples tested: {len(times)}")
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Min time: {min_time:.4f}s")
        print(f"  Max time: {max_time:.4f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Speed: {samples_per_sec:.2f} samples/sec")
        
        return {
            'name': name,
            'samples_tested': len(times),
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'total_time': total_time,
            'samples_per_sec': samples_per_sec
        }
    else:
        print(f"❌ No samples loaded successfully")
        return None


def main():
    """Main comparison test."""
    print("🔍 Simple Dataset Loading Speed Comparison")
    print("=" * 50)
    
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
        'crop_size': (128, 128),
        'flip_prob': 0.0,
        'rot_prob': 0.0,
        'reverse': False,
        'norm_ZeroToOne': False,
    }
    
    nii_config = {
        'data_root': "./data/synthrad2023_mr-ct_pelvis",
        'split': "train",
        'crop_size': (128, 128),
        'flip_prob': 0.0,
        'rot_prob': 0.0,
        'reverse': False,
    }
    
    num_samples = 10  # Test fewer samples for speed
    
    results = {}
    
    # Test H5 Dataset
    try:
        print("\n🗂️  Testing H5 Dataset...")
        h5_dataset = H5_Dataset(**h5_config)
        h5_results = simple_speed_test(h5_dataset, "H5 Dataset", num_samples)
        results['h5'] = h5_results
    except Exception as e:
        print(f"❌ Error with H5 dataset: {e}")
        results['h5'] = None
    
    # Test NII Dataset (with timeout monitoring)
    try:
        print("\n🗂️  Testing NII GridPatch Dataset...")
        print("⏳ Initializing NII dataset (this may take a while)...")
        
        init_start = time.time()
        nii_dataset = NiiGridPatchDataset(**nii_config)
        init_end = time.time()
        
        print(f"✅ NII dataset initialized in {init_end - init_start:.2f}s")
        nii_results = simple_speed_test(nii_dataset, "NII GridPatch", num_samples)
        results['nii'] = nii_results
        
    except Exception as e:
        print(f"❌ Error with NII dataset: {e}")
        results['nii'] = None
    
    # Comparison
    print("\n" + "=" * 50)
    print("📋 COMPARISON SUMMARY")
    print("=" * 50)
    
    if results['h5'] and results['nii']:
        h5_speed = results['h5']['samples_per_sec']
        nii_speed = results['nii']['samples_per_sec']
        
        print(f"\n⚡ Loading Speed:")
        print(f"  H5 Dataset:      {h5_speed:.2f} samples/sec")
        print(f"  NII GridPatch:   {nii_speed:.2f} samples/sec")
        
        if h5_speed > 0 and nii_speed > 0:
            if h5_speed > nii_speed:
                ratio = h5_speed / nii_speed
                print(f"  → H5 is {ratio:.2f}x faster than NII")
            else:
                ratio = nii_speed / h5_speed
                print(f"  → NII is {ratio:.2f}x faster than H5")
        
        print(f"\n📊 Average Load Time:")
        print(f"  H5 Dataset:      {results['h5']['avg_time']:.4f}s per sample")
        print(f"  NII GridPatch:   {results['nii']['avg_time']:.4f}s per sample")
        
    elif results['h5']:
        print(f"\n✅ H5 Dataset: {results['h5']['samples_per_sec']:.2f} samples/sec")
        print("❌ NII Dataset: Failed to test")
        
    elif results['nii']:
        print("❌ H5 Dataset: Failed to test")
        print(f"✅ NII Dataset: {results['nii']['samples_per_sec']:.2f} samples/sec")
        
    else:
        print("❌ Both datasets failed to test")
    
    print(f"\n🎯 Key Insights:")
    print(f"  • H5: On-demand loading from compressed file")
    print(f"  • NII: Pre-loaded slices in memory (faster access, more RAM)")
    
    return results


if __name__ == "__main__":
    results = main() 
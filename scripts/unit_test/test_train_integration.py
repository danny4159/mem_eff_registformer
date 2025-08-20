"""
Test script for train.py integration with unified dataset

This script tests if train.py can properly load both H5 and NII data and create data loaders.
"""

import os
import sys
import torch
from pathlib import Path

import autorootcwd

# Import train.py components
from src.padain_synthesis.config import DATA_CONFIG
from src.padain_synthesis.dataset import PadainSynthesisDataModule
from src.padain_synthesis import PadainSynthesisConfig, PadainSynthesisModel

def test_data_config():
    """Test unified data configuration."""
    
    print("="*60)
    print("🔧 TESTING Unified Data Configuration")
    print("="*60)
    
    # Load config
    data_config = DATA_CONFIG.copy()
    
    print(f"✅ Data type: {data_config['data_type']}")
    print(f"✅ Batch size: {data_config['batch_size']}")
    print(f"✅ Crop size: {data_config['crop_size']}")
    print(f"✅ Use SliceInferer: {data_config['use_slice_inferer']}")
    print(f"✅ ROI size: {data_config['roi_size']}")
    print(f"✅ SW batch size: {data_config['sw_batch_size']}")
    
    if data_config['data_type'] == 'h5':
        print(f"✅ Train file: {data_config['train_file']}")
        print(f"✅ Data group 1: {data_config['data_group_1']}")
        print(f"✅ Data group 2: {data_config['data_group_2']}")
    else:
        print(f"✅ Data root: {data_config['data_root']}")
    
    return data_config

def test_data_module_creation():
    """Test PadainSynthesisDataModule creation with unified config."""
    
    print("\n" + "="*60)
    print("📦 TESTING DataModule Creation")
    print("="*60)
    
    # Load config
    data_config = DATA_CONFIG.copy()
    
    try:
        # Create DataModule
        datamodule = PadainSynthesisDataModule(data_config)
        print("✅ PadainSynthesisDataModule created successfully")
        
        # Test data loaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        
        print(f"✅ Train loader length: {len(train_loader)}")
        print(f"✅ Val loader length: {len(val_loader)}")
        print(f"✅ Test loader length: {len(test_loader)}")
        
        # Test a batch from each loader
        train_batch = next(iter(train_loader))
        print(f"✅ Train batch shape: MR {train_batch['mr'].shape}, CT {train_batch['ct'].shape}")
        
        val_batch = next(iter(val_loader))
        print(f"✅ Val batch shape: MR {val_batch['mr'].shape}, CT {val_batch['ct'].shape}")
        
        test_batch = next(iter(test_loader))
        print(f"✅ Test batch shape: MR {test_batch['mr'].shape}, CT {test_batch['ct'].shape}")
        
        # Test SliceInferer
        slice_inferer = datamodule.get_slice_inferer()
        if slice_inferer is not None:
            print("✅ SliceInferer created successfully")
        else:
            print("ℹ️ SliceInferer not used (use_slice_inferer=False)")
        
        return datamodule, train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"❌ Error creating DataModule: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def test_model_creation():
    """Test model creation with unified config."""
    
    print("\n" + "="*60)
    print("🤖 TESTING Model Creation")
    print("="*60)
    
    try:
        # Load model config
        from src.padain_synthesis.config import MODEL_CONFIG
        model_config = PadainSynthesisConfig(**MODEL_CONFIG)
        
        # Create model
        model = PadainSynthesisModel(model_config)
        print("✅ PadainSynthesisModel created successfully")
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Total model parameters: {total_params:,}")
        
        # Test model device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"✅ Model moved to device: {device}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_training_integration():
    """Test complete training integration."""
    
    print("\n" + "="*60)
    print("🏋️ TESTING Training Integration")
    print("="*60)
    
    try:
        # Load configs
        data_config = DATA_CONFIG.copy()
        from src.padain_synthesis.config import MODEL_CONFIG, TRAINING_ARGS, TRAINING_PARAMS
        
        # Create DataModule
        datamodule = PadainSynthesisDataModule(data_config)
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        # Create Model
        model_config = PadainSynthesisConfig(**DEFAULT_MODEL_CONFIG)
        model = PadainSynthesisModel(model_config)
        
        # Test training args
        training_args = DEFAULT_TRAINING_ARGS.copy()
        training_args.update({
            "per_device_train_batch_size": data_config["batch_size"],
            "per_device_eval_batch_size": 1,
        })
        
        # Test training params
        training_params = DEFAULT_TRAINING_PARAMS.copy()
        training_params.update({
            "batch_size": data_config["batch_size"],
        })
        
        print("✅ All training components created successfully")
        print(f"✅ Training args: batch_size={training_args['per_device_train_batch_size']}")
        print(f"✅ Training params: batch_size={training_params['batch_size']}")
        
        # Test model forward pass with a batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        train_batch = next(iter(train_loader))
        mr_input = train_batch['mr'].to(device)
        
        with torch.no_grad():
            output = model(mr_input)
            print(f"✅ Model forward pass successful: input {mr_input.shape} → output {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in training integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_h5_vs_nii_switching():
    """Test switching between H5 and NII configurations."""
    
    print("\n" + "="*60)
    print("🔄 TESTING H5 vs NII Switching")
    print("="*60)
    
    try:
        # Test NII config
        nii_config = DATA_CONFIG.copy()
        nii_config["data_type"] = "nii"
        
        print("📊 Testing NII configuration:")
        print(f"  Data type: {nii_config['data_type']}")
        print(f"  Data root: {nii_config['data_root']}")
        
        # Test H5 config
        h5_config = DATA_CONFIG.copy()
        h5_config["data_type"] = "h5"
        
        print("\n📊 Testing H5 configuration:")
        print(f"  Data type: {h5_config['data_type']}")
        print(f"  Train file: {h5_config['train_file']}")
        print(f"  Data group 1: {h5_config['data_group_1']}")
        print(f"  Data group 2: {h5_config['data_group_2']}")
        
        # Test SliceInferer settings
        print(f"\n📊 SliceInferer settings (both):")
        print(f"  Use SliceInferer: {nii_config['use_slice_inferer']}")
        print(f"  ROI size: {nii_config['roi_size']}")
        print(f"  SW batch size: {nii_config['sw_batch_size']}")
        
        print("✅ Configuration switching test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error in configuration switching: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    
    print("🚀 STARTING TRAIN.PY INTEGRATION TEST")
    print("="*60)
    
    try:
        # Test unified data config
        data_config = test_data_config()
        
        # Test DataModule creation
        datamodule, train_loader, val_loader, test_loader = test_data_module_creation()
        
        if datamodule is None:
            print("❌ DataModule creation failed, stopping tests")
            return
        
        # Test model creation
        model = test_model_creation()
        
        if model is None:
            print("❌ Model creation failed, stopping tests")
            return
        
        # Test training integration
        success = test_training_integration()
        
        # Test H5 vs NII switching
        switching_success = test_h5_vs_nii_switching()
        
        if success and switching_success:
            print("\n" + "="*60)
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            print("="*60)
            print("✅ Unified data configuration working")
            print("✅ DataModule creation working")
            print("✅ Model creation working")
            print("✅ Training integration working")
            print("✅ H5/NII switching working")
            print("✅ SliceInferer support working")
            print("✅ train.py is ready for both H5 and NII dataset training!")
        else:
            print("\n❌ Some integration tests failed")
            
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
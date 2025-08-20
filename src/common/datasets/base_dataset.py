"""
Base Dataset Module

This module provides a base dataset interface that can handle both H5 and NII data formats.
It automatically selects the appropriate dataset implementation based on configuration.
Can be used by any network architecture.
"""

from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from .h5_dataset import H5Dataset, BaseDataModule
from .nii_dataset import NiiDataset, NiiGridPatchDataset, NiiDataModule


class Dataset:
    """
    Base dataset class that can handle both H5 and NII data formats.
    
    This class automatically selects the appropriate dataset implementation
    based on the configuration provided. Can be used by any network architecture.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base dataset.
        
        Args:
            config: Configuration dictionary containing dataset parameters
        """
        self.config = config
        self.data_type = config.get('data_type', 'h5')  # 'h5' or 'nii'
        
        if self.data_type == 'h5':
            self._init_h5_dataset()
        elif self.data_type == 'nii':
            self._init_nii_dataset()
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def _init_h5_dataset(self):
        """Initialize H5 dataset."""
        # Extract H5-specific parameters
        h5_config = {
            'h5_file_path': self.config.get('train_file'),
            'data_group_1': self.config.get('data_group_1', 'MR'),
            'data_group_2': self.config.get('data_group_2', 'CT'),
            'data_group_3': self.config.get('data_group_3'),
            'data_group_4': self.config.get('data_group_4'),
            'data_group_5': self.config.get('data_group_5'),
            'is_3d': self.config.get('is_3d', False),
            'padding_size': self.config.get('padding_size'),
            'crop_size': self.config.get('crop_size'),
            'flip_prob': self.config.get('flip_prob', 0.0),
            'rot_prob': self.config.get('rot_prob', 0.0),
            'reverse': self.config.get('reverse', False),
            'norm_ZeroToOne': self.config.get('norm_ZeroToOne', False),
        }
        
        self.dataset = H5Dataset(**h5_config)
    
    def _init_nii_dataset(self):
        """Initialize NII dataset."""
        # Check if we should use try dataset for both train and val
        if self.config.get('use_try_dataset', False):
            # Create a modified config that uses 'try' split for both train and val
            modified_config = self.config.copy()
            modified_config['use_try_dataset'] = True
            self.data_module = NiiDataModule(modified_config)
        else:
            self.data_module = NiiDataModule(self.config)
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        return self.dataset[idx]


class DataModule:
    """
    Base data module that can handle both H5 and NII data formats.
    
    This class automatically selects the appropriate data module implementation
    based on the configuration provided. Can be used by any network architecture.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base data module.
        
        Args:
            config: Configuration dictionary containing dataset parameters
        """
        self.config = config
        self.data_type = config.get('data_type', 'h5')  # 'h5' or 'nii'
        
        if self.data_type == 'h5':
            self._init_h5_data_module()
        elif self.data_type == 'nii':
            self._init_nii_data_module()
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def _init_h5_data_module(self):
        """Initialize H5 data module."""
        # Create separate configs for train/val/test
        train_config = self._create_h5_config('train')
        val_config = self._create_h5_config('val')
        test_config = self._create_h5_config('test')
        
        self.train_dataset = H5Dataset(**train_config)
        self.val_dataset = H5Dataset(**val_config)
        self.test_dataset = H5Dataset(**test_config)
    
    def _init_nii_data_module(self):
        """Initialize NII data module."""
        self.data_module = NiiDataModule(self.config)
    
    def _create_h5_config(self, split: str) -> Dict[str, Any]:
        """Create H5 configuration for a specific split."""
        file_key = f'{split}_file'
        file_path = self.config.get(file_key)
        
        if not file_path:
            raise ValueError(f"Missing {file_key} in configuration")
        
        return {
            'h5_file_path': file_path,
            'data_group_1': self.config.get('data_group_1', 'MR'),
            'data_group_2': self.config.get('data_group_2', 'CT'),
            'data_group_3': self.config.get('data_group_3'),
            'data_group_4': self.config.get('data_group_4'),
            'data_group_5': self.config.get('data_group_5'),
            'is_3d': self.config.get('is_3d', False),
            'padding_size': self.config.get('padding_size'),
            'crop_size': self.config.get('crop_size'),
            'flip_prob': self.config.get('flip_prob', 0.0) if split == 'train' else 0.0,
            'rot_prob': self.config.get('rot_prob', 0.0) if split == 'train' else 0.0,
            'reverse': self.config.get('reverse', False),
            'norm_ZeroToOne': self.config.get('norm_ZeroToOne', False),
        }
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self.data_type == 'h5':
            return DataLoader(
                self.train_dataset,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=self.config.get('pin_memory', False),
                shuffle=True,
            )
        else:
            return self.data_module.train_dataloader()
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self.data_type == 'h5':
            return DataLoader(
                self.val_dataset,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=self.config.get('pin_memory', False),
                shuffle=False,
            )
        else:
            return self.data_module.val_dataloader()
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self.data_type == 'h5':
            return DataLoader(
                self.test_dataset,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=self.config.get('pin_memory', False),
                shuffle=False,
            )
        else:
            return self.data_module.test_dataloader()
    
    def get_slice_inferer(self):
        """Get SliceInferer (only available for NII data)."""
        if self.data_type == 'nii':
            return self.data_module.get_slice_inferer()
        else:
            return None


# Convenience functions for backward compatibility and easy usage
def create_dataset(config: Dict[str, Any]) -> Dataset:
    """Create a base dataset based on configuration."""
    return Dataset(config)


def create_data_module(config: Dict[str, Any]) -> DataModule:
    """Create a base data module based on configuration."""
    return DataModule(config) 
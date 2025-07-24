"""
PadainSynthesis Dataset Module

This module provides PadainSynthesis-specific dataset classes that inherit from base classes.
"""

from torch.utils.data import DataLoader
from src.common.dataset import BaseDataset, BaseDataModule


class PadainSynthesisDataset(BaseDataset):
    """
    PadainSynthesis-specific dataset class.
    
    This class inherits from BaseDataset and provides PadainSynthesis-specific
    functionality. Currently, it uses all the base functionality without modification.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize PadainSynthesis dataset."""
        super().__init__(*args, **kwargs)


class PadainSynthesisDataModule(BaseDataModule):
    """
    PadainSynthesis-specific data module class.
    
    This class inherits from BaseDataModule and provides PadainSynthesis-specific
    data loader creation functionality.
    """
    
    def __init__(self, config: dict):
        """Initialize PadainSynthesis data module."""
        super().__init__(config)

    def train_dataloader(self):
        """Create training data loader for PadainSynthesis."""
        dataset = PadainSynthesisDataset(
            h5_file_path=self.config['train_file'],
            data_group_1=self.config['data_group_1'],
            data_group_2=self.config['data_group_2'],
            data_group_3=self.config.get('data_group_3'),
            data_group_4=self.config.get('data_group_4'),
            data_group_5=self.config.get('data_group_5'),
            is_3d=self.config['is_3d'],
            padding_size=self.config.get('padding_size'),
            crop_size=self.config.get('crop_size'),
            flip_prob=self.config.get('flip_prob', 0.0),
            rot_prob=self.config.get('rot_prob', 0.0),
            reverse=self.config.get('reverse', False),
            norm_ZeroToOne=self.config.get('norm_ZeroToOne', False),
        )
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            pin_memory=self.config.get('pin_memory', False),
            shuffle=True,
        )

    def val_dataloader(self):
        """Create validation data loader for PadainSynthesis."""
        dataset = PadainSynthesisDataset(
            h5_file_path=self.config['val_file'],
            data_group_1=self.config['data_group_1'],
            data_group_2=self.config['data_group_2'],
            data_group_3=self.config.get('data_group_3'),
            data_group_4=self.config.get('data_group_4'),
            data_group_5=self.config.get('data_group_5'),
            is_3d=self.config['is_3d'],
            padding_size=self.config.get('padding_size'),
            flip_prob=self.config.get('flip_prob', 0.0),
            rot_prob=self.config.get('rot_prob', 0.0),
            reverse=self.config.get('reverse', False),
            norm_ZeroToOne=self.config.get('norm_ZeroToOne', False),
        )
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'],
            pin_memory=self.config.get('pin_memory', False),
            shuffle=False,
        )

    def test_dataloader(self):
        """Create test data loader for PadainSynthesis."""
        dataset = PadainSynthesisDataset(
            h5_file_path=self.config['test_file'],
            data_group_1=self.config['data_group_1'],
            data_group_2=self.config['data_group_2'],
            data_group_3=self.config.get('data_group_3'),
            data_group_4=self.config.get('data_group_4'),
            data_group_5=self.config.get('data_group_5'),
            is_3d=self.config['is_3d'],
            padding_size=self.config.get('padding_size'),
            flip_prob=0.0,  # No augmentation for test
            rot_prob=0.0,   # No augmentation for test
            reverse=self.config.get('reverse', False),
            norm_ZeroToOne=self.config.get('norm_ZeroToOne', False),
        )
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'],
            pin_memory=self.config.get('pin_memory', False),
            shuffle=False,
        ) 
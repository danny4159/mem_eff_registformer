"""
Common Datasets Package

This package contains all dataset implementations used across different networks.
"""

# Base dataset (handles both H5 and NII automatically)
from .base_dataset import (
    Dataset,
    DataModule,
    create_dataset,
    create_data_module
)

# H5-specific datasets
from .h5_dataset import (
    H5Dataset,
    BaseDataModule
)

# NII-specific datasets
from .nii_dataset import (
    NiiDataset,
    NiiGridPatchDataset,
    NiiDataModule,
    denormalize_ct,
    denormalize_mr
)

__all__ = [
    # Base dataset classes
    "Dataset",
    "DataModule", 
    "create_dataset",
    "create_data_module",
    
    # H5 dataset classes
    "H5Dataset",
    "BaseDataModule",
    
    # NII dataset classes
    "NiiDataset",
    "NiiGridPatchDataset",
    "NiiDataModule",
    "denormalize_ct",
    "denormalize_mr",
] 
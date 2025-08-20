"""Base Data Configuration

This file contains common data configuration that is shared across different networks.
Network-specific data configurations can override these base settings.
"""

# Base data configuration - common across all networks
BASE_DATA_CONFIG = {
    # ============================================================================
    # Common settings (for both H5 and NII)
    # ============================================================================
    "data_type": "nii",        # "h5" or "nii"
    
    # Data processing settings
    "batch_size": 16,           # Batch size for training
    "num_workers": 8,          # Number of data loading workers
    "crop_size": (128, 128),   # Size for random cropping
    "flip_prob": 0.5,          # Probability of horizontal flipping
    "rot_prob": 0.5,           # Probability of 90-degree rotation
    "reverse": False,          # Reverse source and target
    "pin_memory": False,       # Pin memory for GPU transfer
    
    # ============================================================================
    # H5-specific settings (used when data_type="h5")
    # ============================================================================
    # Default H5 file paths (can be overridden per network)
    "train_file": "./data/synthrad2023_mr-ct_pelvis/train/Ver3_OnlyOnePatient.h5",
    "val_file": "./data/synthrad2023_mr-ct_pelvis/val/Ver3_OnlyOnePatient.h5",
    "test_file": "./data/synthrad2023_mr-ct_pelvis/test/Ver3_OnlyOnePatient.h5",
    
    # H5 file data group names
    "data_group_1": "MR",      # Source modality
    "data_group_2": "CT",      # Target modality
    "data_group_3": None,      # Additional modality 1
    "data_group_4": None,      # Additional modality 2
    "data_group_5": None,      # Additional modality 3
    
    # H5-specific data processing
    "is_3d": False,            # Process 3D volumes or 2D slices
    "padding_size": None,      # Target size for padding
    "norm_ZeroToOne": False,   # Normalize to [0, 1] range
    "resize_size": None,       # Target size for resizing
    
    # ============================================================================
    # NII-specific settings (used when data_type="nii")
    # ============================================================================
    # Default data root directory (can be overridden per network)
    "data_root": "./data/synthrad2023_mr-ct_pelvis",
    
    # ============================================================================
    # SliceInferer settings (for both H5 and NII)
    # ============================================================================
    "use_slice_inferer": True, # Use SliceInferer for validation/test
    "roi_size": (256, 256),    # ROI size for SliceInferer
    "sw_batch_size": 8,        # Sliding window batch size
}

# Network-specific data config overrides
NETWORK_DATA_OVERRIDES = {
    "padain_synthesis": {
        # PadainSynthesis might use specific crop sizes or data paths
        # Add specific overrides here if needed
    },
    # Add other networks here as needed
    # "other_network": {
    #     "crop_size": (64, 64),
    #     "batch_size": 32,
    # },
}

def get_data_config(network_name: str = None, **overrides):
    """
    Get data configuration for a specific network.
    
    Args:
        network_name (str): Name of the network (optional)
        **overrides: Additional overrides for data configuration
    
    Returns:
        dict: Data configuration with network-specific and custom overrides applied
    """
    # Start with base configuration
    data_config = BASE_DATA_CONFIG.copy()
    
    # Apply network-specific overrides if network name is provided
    if network_name and network_name in NETWORK_DATA_OVERRIDES:
        data_config.update(NETWORK_DATA_OVERRIDES[network_name])
    
    # Apply custom overrides
    data_config.update(overrides)
    
    return data_config 
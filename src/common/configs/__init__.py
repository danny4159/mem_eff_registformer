"""Common Configuration Package

This package contains shared configuration settings that can be used across different networks.
"""

from .base_training_args import BASE_TRAINING_ARGS, NETWORK_SPECIFIC_OVERRIDES, get_training_args
from .base_data_config import BASE_DATA_CONFIG, NETWORK_DATA_OVERRIDES, get_data_config

__all__ = [
    'BASE_TRAINING_ARGS',
    'NETWORK_SPECIFIC_OVERRIDES', 
    'get_training_args',
    'BASE_DATA_CONFIG',
    'NETWORK_DATA_OVERRIDES',
    'get_data_config',
] 
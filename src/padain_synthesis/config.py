"""PadainSynthesis configuration settings"""

from src.common.configs import get_training_args, get_data_config

# ============================================================================
# Network-specific configurations (MODEL_CONFIG and TRAINING_PARAMS)
# ============================================================================

# Model configuration - specific to PadainSynthesis
MODEL_CONFIG = {
    # Generator settings
    "input_nc": 1,               # Number of input channels
    "feat_ch": 512,              # Number of feature channels
    "output_nc": 1,              # Number of output channels
    "demodulate": True,           # Use demodulation in AdaIN layers
    "is_3d": False,              # 3D model operation
    
    # Discriminator settings
    "use_discriminator": True,   # Use discriminator
    "discriminator_ndf": 64,     # Number of discriminator filters
    "discriminator_n_layers": 3, # Number of discriminator layers
    "gan_type": "lsgan",         # GAN type
    
    # PatchSampleF settings
    "use_mlp": False,            # Use MLP in PatchSampleF
    "init_type": "normal",       # Initialization type
    "init_gain": 0.02,           # Initialization gain
    "nc": 256,                   # Number of channels
    "input_nc_patch": 256,       # Input channels for PatchSampleF
}

# Training parameters - specific to PadainSynthesis
TRAINING_PARAMS = {
    "lambda_ctx": 1.0,
    "lambda_nce": 1.0,
    "lambda_mind": 0.0,
    "lambda_l1": 0.0,
    "lambda_gan": 1.0,
    "nce_on_vgg": False,
    "nce_layers": [0, 2, 4, 6],
    "use_misalign_simul": False,
    "is_3d": False,
    "eval_on_align": False,
    "flip_equivariance": False,
}

# ============================================================================
# Get common configurations with PadainSynthesis-specific overrides
# ============================================================================

# Get training arguments from common config with PadainSynthesis-specific settings
TRAINING_ARGS = get_training_args(
    network_name="padain_synthesis",
    run_name="AllDataset_MaskSolve"  # Default run name, can be overridden
)

# Get data configuration from common config
# Note: No PadainSynthesis-specific data config overrides needed currently
DATA_CONFIG = get_data_config(network_name="padain_synthesis")

# ============================================================================
# Legacy compatibility - keep the old variable names for backward compatibility
# ============================================================================
# These are kept for any existing code that might still use the old names
# In the future, these can be removed once all code is updated to use the new structure

 
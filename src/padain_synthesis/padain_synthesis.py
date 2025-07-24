"""
PadainSynthesis Model Module

This module provides the main PadainSynthesis model implementation with Hugging Face integration.
It includes the configuration class and model class for the PadainSynthesis architecture.
"""

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoModel, AutoConfig

# Import model components
from src.models.padain_synthesis_module import ProposedSynthesisModule
from src.models.components.patch_sample_f import PatchSampleF
from src.models.components.discriminator import NLayerDiscriminator


class PadainSynthesisConfig(PretrainedConfig):
    """
    Configuration class for PadainSynthesis model.
    
    This class defines all the hyperparameters and settings needed to initialize
    the PadainSynthesis model, including generator, discriminator, and PatchSampleF settings.
    """
    
    model_type = "padain_synthesis"
    
    def __init__(self, 
                 input_nc: int,
                 feat_ch: int,
                 output_nc: int,
                 demodulate: bool,
                 is_3d: bool,
                 use_discriminator: bool,
                 discriminator_ndf: int,
                 discriminator_n_layers: int,
                 gan_type: str,
                 use_mlp: bool = False,
                 init_type: str = "normal",
                 init_gain: float = 0.02,
                 nc: int = 256,
                 input_nc_patch: int = 256,
                 **kwargs):
        """
        Initialize PadainSynthesis configuration.
        
        Args:
            input_nc: Number of input channels
            feat_ch: Number of feature channels
            output_nc: Number of output channels
            demodulate: Whether to use demodulation in generator
            is_3d: Whether the model operates on 3D data
            use_discriminator: Whether to use discriminator for GAN training
            discriminator_ndf: Number of discriminator features
            discriminator_n_layers: Number of discriminator layers
            gan_type: Type of GAN loss ('lsgan', 'vanilla', etc.)
            use_mlp: Whether to use MLP in PatchSampleF
            init_type: Weight initialization type
            init_gain: Weight initialization gain
            nc: Number of channels for PatchSampleF
            input_nc_patch: Number of input channels for PatchSampleF
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        # Generator settings
        self.input_nc = input_nc
        self.feat_ch = feat_ch
        self.output_nc = output_nc
        self.demodulate = demodulate
        self.is_3d = is_3d
        
        # Discriminator settings
        self.use_discriminator = use_discriminator
        self.discriminator_ndf = discriminator_ndf
        self.discriminator_n_layers = discriminator_n_layers
        self.gan_type = gan_type
        
        # PatchSampleF settings
        self.use_mlp = use_mlp
        self.init_type = init_type
        self.init_gain = init_gain
        self.nc = nc
        self.input_nc_patch = input_nc_patch


class PadainSynthesisModel(PreTrainedModel):
    """
    PadainSynthesis Model
    
    A complete image synthesis model that combines a generator, discriminator (optional),
    and PatchSampleF for feature sampling. This model is designed for medical image
    synthesis tasks and integrates with Hugging Face's ecosystem.
    """
    
    config_class = PadainSynthesisConfig
    main_input_name = "merged_input"

    def __init__(self, config: PadainSynthesisConfig):
        """
        Initialize PadainSynthesis model.
        
        Args:
            config: Configuration object containing all model parameters
        """
        super().__init__(config)
        
        # Initialize main generator
        self.model = ProposedSynthesisModule(**config.to_dict())
        
        # Initialize PatchSampleF for feature sampling
        self.netF_A = PatchSampleF(
            use_mlp=config.use_mlp,
            init_type=config.init_type,
            init_gain=config.init_gain,
            nc=config.nc,
            input_nc=config.input_nc_patch
        )
        
        # Initialize discriminator if enabled
        if config.use_discriminator:
            self.netD_A = NLayerDiscriminator(
                input_nc=config.output_nc,
                ndf=config.discriminator_ndf,
                n_layers=config.discriminator_n_layers,
                gan_type=config.gan_type
            )
        else:
            self.netD_A = None

    def forward(self, merged_input, **kwargs):
        """
        Forward pass through the model.
        
        Args:
            merged_input: Concatenated input tensors [B, 2*C, H, W]
            **kwargs: Additional arguments including:
                - layers: List of layer indices to extract features from
                - encode_only: Whether to only encode without decoding
                
        Returns:
            BaseModelOutput with last_hidden_state containing the generated image
        """
        layers = kwargs.get('layers', [])
        encode_only = kwargs.get('encode_only', False)
        
        # Generate output through the main model
        output = self.model(merged_input, layers=layers, encode_only=encode_only)
        
        return BaseModelOutput(last_hidden_state=output)
    
    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the model and configuration to disk.
        
        This method overrides the default Hugging Face save_pretrained to avoid
        issues with non-standard configuration parameters.
        
        Args:
            save_directory: Directory to save the model
            **kwargs: Additional save parameters
        """
        import os
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save configuration
        config_path = os.path.join(save_directory, "config.json")
        config_dict = self.config.to_dict()
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)


# Register model with Hugging Face AutoModel system
AutoConfig.register("padain_synthesis", PadainSynthesisConfig)
AutoModel.register(PadainSynthesisConfig, PadainSynthesisModel) 
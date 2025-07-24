"""
PadainSynthesis Trainer Module

This module provides a custom Hugging Face Trainer for the PadainSynthesis model,
supporting GAN training with multiple loss functions including contextual loss,
PatchNCE loss, MIND loss, and L1 loss.
"""

import torch
import torch.nn as nn
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import EvalLoopOutput
from transformers.modeling_outputs import BaseModelOutput
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import wandb

from .padain_synthesis import PadainSynthesisModel, PadainSynthesisConfig
from .dataset import PadainSynthesisDataset
from src.common.losses import GANLoss, Contextual_Loss, VGG_Model, PatchNCELoss, MINDLoss
from .metrics import PadainSynthesisMetrics


def padain_synthesis_data_collator(features):
    """
    Custom data collator for PadainSynthesis that converts tuple format to dict format.
    
    Args:
        features: List of tuples (input_image, reference_image, C, D, E)
        
    Returns:
        Dictionary with batched tensors
    """
    # Separate components
    input_images, reference_images, Cs, Ds, Es = zip(*features)
    
    # Stack tensors
    input_batch = torch.stack(input_images)
    reference_batch = torch.stack(reference_images)
    
    # Handle optional components
    C_batch = None
    if any(c is not None for c in Cs):
        C_batch = torch.stack([c for c in Cs if c is not None])
    
    D_batch = None
    if any(d is not None for d in Ds):
        D_batch = torch.stack([d for d in Ds if d is not None])
    
    E_batch = None
    if any(e is not None for e in Es):
        E_batch = torch.stack([e for e in Es if e is not None])
    
    # Return as dictionary for Transformers compatibility
    return {
        'input_image': input_batch,
        'reference_image': reference_batch,
        'C': C_batch,
        'D': D_batch,
        'E': E_batch
    }


class PadainSynthesisTrainer(Trainer):
    """
    Custom Trainer for PadainSynthesis model with GAN training support.
    
    This trainer extends Hugging Face's Trainer to support:
    - GAN training with Generator and Discriminator
    - Multiple loss functions (Contextual, PatchNCE, MIND, L1)
    - Custom evaluation metrics
    - Wandb logging integration
    """
    
    def __init__(self, 
                 model: PadainSynthesisModel,
                 args: TrainingArguments,
                 train_dataset: Optional[PadainSynthesisDataset] = None,
                 eval_dataset: Optional[PadainSynthesisDataset] = None,
                 params: Optional[Dict[str, Any]] = None,
                 data_collator=None,
                 **kwargs):
        """Initialize the PadainSynthesis trainer with loss functions and metrics."""
        
        # Remove duplicate data_collator from kwargs
        kwargs.pop('data_collator', None)
        
        # Initialize parent trainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            **kwargs
        )
        
        # Store training parameters
        self.params = params or {}
        
        # Initialize loss functions and metrics
        self._setup_loss_functions()
        self._setup_metrics()
    
    def _setup_loss_functions(self):
        """Initialize all loss functions based on training parameters."""
        
        # Contextual Loss (Style Loss)
        if self.params.get('lambda_ctx', 0) != 0:
            style_layers = {
                "conv_2_2": 1.0,
                "conv_3_2": 1.0,
                "conv_4_2": 1.0,
                "conv_4_4": 1.0
            }
            self.contextual_loss = Contextual_Loss(style_layers)
            if hasattr(self.contextual_loss, 'vgg_pred'):
                self.contextual_loss.vgg_pred = self.contextual_loss.vgg_pred.cuda(0)
        else:
            self.contextual_loss = None
        
        # GAN Loss (only if discriminator exists)
        if hasattr(self.model, 'netD_A') and self.model.netD_A is not None:
            self.gan_loss = GANLoss(gan_type=self.params.get('gan_type', 'lsgan'))
        else:
            self.gan_loss = None
        
        # PatchNCE Loss
        if self.params.get('lambda_nce', 0) != 0:
            self.patch_nce_loss = PatchNCELoss(
                False, 
                nce_T=0.07, 
                batch_size=self.params.get('batch_size', 4)
            )
        else:
            self.patch_nce_loss = None
        
        # MIND Loss
        if self.params.get('lambda_mind', 0) != 0:
            self.mind_loss = MINDLoss()
        else:
            self.mind_loss = None
        
        # L1 Loss
        if self.params.get('lambda_l1', 0) != 0:
            self.l1_loss = torch.nn.L1Loss()
        else:
            self.l1_loss = None
        
        # VGG model for PatchNCE (if needed)
        if self.params.get('nce_on_vgg', False):
            self.vgg = VGG_Model(listen_list=["conv_4_2", "conv_5_4"])
        else:
            self.vgg = None
    
    def _setup_metrics(self):
        """Initialize evaluation metrics."""
        self.metrics = PadainSynthesisMetrics(
            eval_on_align=False,  # Always use general metrics for evaluation
            device=self.model.device
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the total loss for training step.
        
        Args:
            model: The model to compute loss for
            inputs: Dictionary containing 'input_image', 'reference_image', 'C', 'D', 'E'
            return_outputs: Whether to return model outputs
            **kwargs: Additional arguments
            
        Returns:
            Total loss or (loss, outputs) tuple
        """
        # Extract input tensors from dictionary
        input_image = inputs.get('input_image')
        reference_image = inputs.get('reference_image')
        C = inputs.get('C')
        D = inputs.get('D')
        E = inputs.get('E')
        
        if input_image is None or reference_image is None:
            raise ValueError("Both input_image and reference_image are required.")
        
        # Forward pass
        merged_input = torch.cat([input_image, reference_image], dim=1)
        outputs = model(merged_input)
        output_image = outputs.last_hidden_state
        
        # Compute losses
        if hasattr(model, 'netD_A') and model.netD_A is not None:
            # GAN training: Generator + Discriminator losses
            generator_loss = self._compute_generator_loss(input_image, reference_image, output_image)
            discriminator_loss = self._compute_discriminator_loss(reference_image, output_image.detach())
            total_loss = generator_loss + discriminator_loss
            
            # Store discriminator loss for logging
            if hasattr(self, 'current_loss_dict'):
                self.current_loss_dict['discriminator'] = discriminator_loss.detach().item()
        else:
            # Generator-only training
            total_loss = self._compute_generator_loss(input_image, reference_image, output_image)
        
        # Store loss dictionary for logging
        self.current_loss_dict = self._get_loss_dict()
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _compute_generator_loss(self, input_image, reference_image, output_image):
        """
        Compute generator loss combining multiple loss functions.
        
        Args:
            input_image: Source image
            reference_image: Target/reference image
            output_image: Generated image
            
        Returns:
            Total generator loss
        """
        total_loss = torch.tensor(0.0, device=input_image.device)
        loss_dict = {}
        
        # Handle misaligned simulated data
        if self.params.get('use_misalign_simul', False):
            reference_image = input_image
        
        # 1. GAN Loss
        if hasattr(self.model, 'netD_A') and self.model.netD_A is not None:
            gan_loss = self.model.netD_A.calc_gen_loss(output_image)
            gan_loss = gan_loss * self.params.get('lambda_gan', 1.0)
            total_loss += gan_loss
            loss_dict['gan'] = gan_loss.detach().item()
        
        # 2. Contextual Loss (Style Loss)
        if self.contextual_loss:
            style_loss = self.contextual_loss(reference_image, output_image)
            style_loss = style_loss * self.params.get('lambda_ctx', 1.0)
            total_loss += style_loss.squeeze()
            loss_dict['contextual'] = style_loss.detach().item()
        
        # 3. PatchNCE Loss
        if self.patch_nce_loss:
            nce_loss = self._compute_patch_nce_loss(input_image, reference_image, output_image)
            total_loss += nce_loss
            loss_dict['patch_nce'] = nce_loss.detach().item()
        
        # 4. MIND Loss
        if self.mind_loss:
            mind_loss = self.mind_loss(output_image, input_image) * self.params.get('lambda_mind', 1.0)
            total_loss += mind_loss
            loss_dict['mind'] = mind_loss.detach().item()
        
        # 5. L1 Loss
        if self.l1_loss:
            l1_loss = self.l1_loss(reference_image, output_image) * self.params.get('lambda_l1', 1.0)
            total_loss += l1_loss
            loss_dict['l1'] = l1_loss.detach().item()
        
        # Store total loss
        loss_dict['total'] = total_loss.detach().item()
        self.current_loss_dict = loss_dict
        
        return total_loss
    
    def _compute_patch_nce_loss(self, input_image, reference_image, output_image):
        """Compute PatchNCE loss using either VGG or generator features."""
        
        # Get PatchSampleF network
        netF_A = getattr(self.model, 'netF_A', None)
        if netF_A is None:
            raise AttributeError('PadainSynthesisModel must have netF_A attribute.')
        
        # Choose feature extraction method
        if self.params.get('nce_on_vgg', False) and self.vgg is not None:
            # Use VGG features
            input_rgb = input_image.repeat(1, 3, 1, 1)
            output_rgb = output_image.repeat(1, 3, 1, 1)
            self.vgg.to(input_image.device)
            
            feat_input = list(self.vgg(input_rgb).values())
            feat_output = list(self.vgg(output_rgb).values())
        else:
            # Use generator features
            feat_input = [reference_image]
            feat_output = [output_image]
        
        # Sample features using PatchSampleF
        feat_input_pool, sample_ids = netF_A(feat_input, 256, None)
        feat_output_pool, _ = netF_A(feat_output, 256, sample_ids)
        
        # Compute PatchNCE loss
        total_nce_loss = 0.0
        for f_input, f_output in zip(feat_input_pool, feat_output_pool):
            loss = self.patch_nce_loss(f_input, f_output) * self.params.get('lambda_nce', 1.0)
            total_nce_loss += loss.mean()
        
        return total_nce_loss / len(feat_output_pool)
    
    def _compute_discriminator_loss(self, reference_image, output_image):
        """Compute discriminator loss for GAN training."""
        if not hasattr(self.model, 'netD_A') or self.model.netD_A is None:
            return torch.tensor(0.0, device=reference_image.device)
        
        loss_D = self.model.netD_A.calc_dis_loss(output_image, reference_image)
        return loss_D * self.params.get('lambda_gan', 1.0)
    
    def _get_loss_dict(self):
        """Get current loss dictionary for logging."""
        return getattr(self, 'current_loss_dict', {})
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override logging to include custom loss components."""
        # Add loss components to logs
        if hasattr(self, 'current_loss_dict'):
            for key, value in self.current_loss_dict.items():
                logs[f"loss_{key}"] = value
        
        # Call parent logging
        super().log(logs, start_time)
        
        # Log to wandb if enabled
        if self.args.report_to and "wandb" in self.args.report_to:
            wandb.log(logs)
    
    def evaluation_step(self, model, inputs, prediction_loss_only=None, ignore_keys=None):
        """Custom evaluation step with metric updates."""
        model.eval()
        
        with torch.no_grad():
            try:
                # Extract input tensors from dictionary
                input_image = inputs.get('input_image')
                reference_image = inputs.get('reference_image')
                C = inputs.get('C')
                D = inputs.get('D')
                E = inputs.get('E')
                
                # Forward pass
                merged_input = torch.cat([input_image, reference_image], dim=1)
                outputs = model(merged_input)
                output_image = outputs.last_hidden_state
                
                # Always update metrics during evaluation, regardless of prediction_loss_only
                self.metrics.update_metrics(
                    real_a=input_image,      # input image
                    real_b=reference_image,  # reference image
                    fake_b=output_image      # output image
                )
                
                # Handle prediction-only mode (only affects loss computation)
                if prediction_loss_only:
                    return (None, outputs, None)
                
                # Compute loss
                loss = self.compute_loss(model, inputs)
                return (loss, None, None)
                
            except Exception as e:
                print(f"Error in evaluation_step: {e}")
                return (None, None, None)
    
    def evaluation_loop(self, dataloader, description: str, prediction_loss_only=None, 
                       ignore_keys=None, metric_key_prefix="eval"):
        """Custom evaluation loop with metric computation."""
        
        eval_losses = []
        
        # Force prediction_loss_only to False for metric computation
        if prediction_loss_only is None:
            prediction_loss_only = False
        
        # Process all batches
        for step, batch in enumerate(dataloader):
            # batch is now a dictionary from our custom data collator
            # Move each tensor to device
            inputs = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Evaluation step - always compute metrics, but conditionally compute loss
            loss, _, _ = self.evaluation_step(self.model, inputs, prediction_loss_only, ignore_keys)
            
            if loss is not None:
                eval_losses.append(loss.item())
        
        # Compute final metrics
        try:
            metrics_dict = self.metrics.compute_metrics()
        except Exception as e:
            print(f"Warning: Metrics computation failed: {e}. Using default values.")
            metrics_dict = {
                'gc': 0.0,
                'nmi': 0.0,
                'fid': 0.0,
                'kid': 0.0,
                'sharpness': 0.0
            }
        
        # Prepare metrics dictionary
        metrics = {}
        if eval_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.mean(eval_losses)
        
        metrics.update({f"{metric_key_prefix}_{key}": value 
                       for key, value in metrics_dict.items()})
        
        # Reset metrics for next evaluation
        self.metrics.reset_metrics()
        
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=len(eval_losses) if eval_losses else 0
        )

    def configure_optimizers(self):
        """Configure optimizers for generator and discriminator."""
        optimizers = []
        
        # Generator optimizer (main model + PatchSampleF)
        generator_params = (list(self.model.model.parameters()) + 
                          list(self.model.netF_A.parameters()))
        optimizer_G = torch.optim.Adam(generator_params, lr=self.args.learning_rate)
        optimizers.append(optimizer_G)
        
        # Discriminator optimizer (if exists)
        if hasattr(self.model, 'netD_A') and self.model.netD_A is not None:
            optimizer_D = torch.optim.Adam(self.model.netD_A.parameters(), 
                                         lr=self.args.learning_rate)
            optimizers.append(optimizer_D)
        
        return optimizers


def create_padain_synthesis_trainer(
    model: PadainSynthesisModel,
    train_dataset: PadainSynthesisDataset,
    training_args: Dict[str, Any],
    params: Dict[str, Any],
    eval_dataset: Optional[PadainSynthesisDataset] = None,
    data_collator=None,
    **kwargs
) -> PadainSynthesisTrainer:
    """
    Factory function to create a PadainSynthesisTrainer instance.
    
    Args:
        model: The PadainSynthesis model
        train_dataset: Training dataset
        training_args: Training arguments dictionary
        params: Model parameters dictionary
        eval_dataset: Evaluation dataset (optional)
        data_collator: Data collator (optional)
        **kwargs: Additional arguments
        
    Returns:
        Configured PadainSynthesisTrainer instance
    """
    
    # Validate required parameters
    if training_args is None:
        raise ValueError("training_args must be provided")
    if params is None:
        raise ValueError("params must be provided")
    
    # Create TrainingArguments
    args = TrainingArguments(**training_args)
    
    # Create and return trainer
    return PadainSynthesisTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        params=params,
        data_collator=data_collator,
        **kwargs
    )


def load_trained_model(checkpoint_path: str, config: Optional[PadainSynthesisConfig] = None):
    """
    Load a trained PadainSynthesis model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        config: Model configuration (optional)
        
    Returns:
        Loaded PadainSynthesisModel
    """
    if config is None:
        config = PadainSynthesisConfig()
    
    return PadainSynthesisModel.from_pretrained(checkpoint_path, config=config) 
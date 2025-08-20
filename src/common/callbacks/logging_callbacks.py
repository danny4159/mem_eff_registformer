"""로깅 관련 Callback들"""

from transformers.trainer_callback import TrainerCallback
import wandb
import numpy as np
import torch


class WandbImageCallback(TrainerCallback):
    """Wandb에 이미지를 업로드하는 Callback"""
    
    def __init__(self):
        super().__init__()
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return
        
        try:
            dataset = self.trainer.train_dataset
            
            # Check dataset type and handle accordingly
            if hasattr(dataset, 'patient_keys'):  # H5 dataset
                # H5 dataset logic (original)
                indices = []
                num_patients = len(dataset.patient_keys)
                for patient_idx in [0, 1]:
                    if patient_idx >= num_patients:
                        continue
                    base = dataset.cumulative_slice_counts[patient_idx]
                    num_slices = dataset.slice_counts[patient_idx]
                    for slice_idx in [30, 60]:
                        if slice_idx < num_slices:
                            indices.append(base + slice_idx)
                
                images = [dataset[i] for i in indices]
                for i, sample in enumerate(images):
                    # H5 format: tuple (A, B, C, D, E)
                    input_image, reference_image, C, D, E = sample
                    self._log_h5_images(input_image, reference_image, i, state.epoch)
                    
            else:  # NII dataset
                # NII dataset logic - just sample a few random indices
                dataset_len = len(dataset)
                indices = [30, 60, 100, 150] if dataset_len > 150 else [0, 1, 2, 3]
                indices = [i for i in indices if i < dataset_len]
                
                images = [dataset[i] for i in indices[:4]]  # Limit to 4 samples
                for i, sample in enumerate(images):
                    # NII format: dict {"mr": tensor, "ct": tensor, "mask": tensor}
                    self._log_nii_images(sample, i, state.epoch)
                    
        except Exception as e:
            # Continue without failing the training
            pass
    
    def _log_h5_images(self, input_image, reference_image, sample_idx, epoch):
        """Log H5 dataset images to wandb"""
        def normalize_for_wandb(img):
            img_normalized = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            return img_normalized
        
        real_a = normalize_for_wandb(input_image[0].cpu().numpy())
        real_b = normalize_for_wandb(reference_image[0].cpu().numpy())
        
        # Generate fake image
        fake_b = normalize_for_wandb(
            self.trainer.model(
                torch.cat([input_image, reference_image], dim=0).unsqueeze(0).to(self.trainer.model.device)
            ).last_hidden_state[0,0].detach().cpu().numpy()
        )
        
        wandb.log({
            f"epoch_{epoch}_sample_{sample_idx}_real_a": wandb.Image(real_a, caption="real_a"),
            f"epoch_{epoch}_sample_{sample_idx}_real_b": wandb.Image(real_b, caption="real_b"),
            f"epoch_{epoch}_sample_{sample_idx}_fake_b": wandb.Image(fake_b, caption="fake_b"),
        })
    
    def _log_nii_images(self, sample, sample_idx, epoch):
        """Log NII dataset images to wandb"""
        def normalize_for_wandb(img):
            img_normalized = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            return img_normalized
        
        mr_img = normalize_for_wandb(sample["mr"][0].cpu().numpy())  # Remove channel dim
        ct_img = normalize_for_wandb(sample["ct"][0].cpu().numpy())
        mask_img = (sample["mask"][0].cpu().numpy() * 255).astype(np.uint8)
        
        # Generate fake image using model
        # Correct input format: concatenate along channel dimension -> [B, 2, H, W]
        input_tensor = torch.cat([sample["mr"], sample["ct"]], dim=0).unsqueeze(0).to(self.trainer.model.device)
        with torch.no_grad():
            fake_ct = self.trainer.model(input_tensor).last_hidden_state[0,0].detach().cpu().numpy()
        fake_ct_img = normalize_for_wandb(fake_ct)
        
        wandb.log({
            f"epoch_{epoch}_sample_{sample_idx}_mr": wandb.Image(mr_img, caption="MR"),
            f"epoch_{epoch}_sample_{sample_idx}_ct_real": wandb.Image(ct_img, caption="CT Real"),
            f"epoch_{epoch}_sample_{sample_idx}_ct_fake": wandb.Image(fake_ct_img, caption="CT Fake"),
            f"epoch_{epoch}_sample_{sample_idx}_mask": wandb.Image(mask_img, caption="Mask"),
        })


class EpochProgressCallback(TrainerCallback):
    """에포크 진행상황을 출력하는 Callback"""
    
    def __init__(self):
        self.trainer = None
    
    def set_trainer(self, trainer):
        """Set trainer instance to access loss information"""
        self.trainer = trainer
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training and evaluation metrics with better formatting (only at epoch end)"""
        if logs is None:
            return
            
        # Only log at epoch end to avoid duplicating HF's step-by-step logging
        if 'epoch' in logs and state.log_history and len(state.log_history) > 1:
            epoch = int(logs['epoch'])
            
            # Training metrics - show individual losses instead of total
            if 'loss' in logs:
                print(f"\n{'='*80}")
                print(f"📈 EPOCH {epoch} TRAINING RESULTS")
                print(f"{'='*80}")
                
                # Get trainer instance to access loss_dict
                # Try different ways to get trainer instance
                trainer = kwargs.get('trainer') or getattr(self, 'trainer', None)
                if trainer and hasattr(trainer, 'current_loss_dict'):
                    loss_dict = trainer.current_loss_dict
                    if loss_dict:
                        # Individual losses
                        loss_parts = []
                        if 'total' in loss_dict:
                            loss_parts.append(f"Total: {loss_dict['total']:.4f}")
                        if 'gan' in loss_dict:
                            loss_parts.append(f"GAN: {loss_dict['gan']:.4f}")
                        if 'contextual' in loss_dict:
                            loss_parts.append(f"CTX: {loss_dict['contextual']:.4f}")
                        if 'patch_nce' in loss_dict:
                            loss_parts.append(f"NCE: {loss_dict['patch_nce']:.4f}")
                        if 'mind' in loss_dict and loss_dict['mind'] > 0:
                            loss_parts.append(f"MIND: {loss_dict['mind']:.4f}")
                        if 'l1' in loss_dict and loss_dict['l1'] > 0:
                            loss_parts.append(f"L1: {loss_dict['l1']:.4f}")
                        if 'discriminator' in loss_dict:
                            loss_parts.append(f"Disc: {loss_dict['discriminator']:.4f}")
                        
                        if loss_parts:
                            print(f"🔥 Training  → {' | '.join(loss_parts)}")
                        else:
                            # Fallback to total loss
                            print(f"🔥 Training  → Loss: {logs['loss']:.4f}")
                    else:
                        # Fallback to total loss
                        print(f"🔥 Training  → Loss: {logs['loss']:.4f}")
                else:
                    # Fallback to total loss
                    print(f"🔥 Training  → Loss: {logs['loss']:.4f}")
            
            # Evaluation metrics (eval_* keys) - only show clean format, no raw dict
            eval_metrics = {k: v for k, v in logs.items() if k.startswith('eval_') and k != 'eval_runtime' and k != 'eval_samples_per_second' and k != 'eval_steps_per_second'}
            if eval_metrics:
                print(f"📊 Evaluation → ", end="")
                
                # Image quality metrics
                quality_metrics = []
                if 'eval_fid_B' in eval_metrics:
                    quality_metrics.append(f"FID: {eval_metrics['eval_fid_B']:.2f}")
                if 'eval_kid_B' in eval_metrics:
                    quality_metrics.append(f"KID: {eval_metrics['eval_kid_B']:.4f}")
                if 'eval_sharpness_B' in eval_metrics:
                    quality_metrics.append(f"Sharpness: {eval_metrics['eval_sharpness_B']:.2f}")
                
                # Correlation metrics  
                corr_metrics = []
                if 'eval_gc_B' in eval_metrics:
                    corr_metrics.append(f"GC: {eval_metrics['eval_gc_B']:.3f}")
                if 'eval_nmi_B' in eval_metrics:
                    corr_metrics.append(f"NMI: {eval_metrics['eval_nmi_B']:.3f}")
                
                # Combine all metrics
                all_metrics = []
                if quality_metrics:
                    all_metrics.extend(quality_metrics)
                if corr_metrics:
                    all_metrics.extend(corr_metrics)
                
                print(" | ".join(all_metrics))
                
                # Runtime info
                if 'eval_runtime' in logs:
                    print(f"⏱️  Runtime    → {logs['eval_runtime']:.1f}s")
                
                print(f"{'='*80}\n")
                
                # Suppress the raw dict output by marking this log as handled
                # We do this by setting a flag that can be checked by other logging mechanisms
                logs['_formatted_output_shown'] = True
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        # This will be handled by on_log when metrics are logged
        pass 
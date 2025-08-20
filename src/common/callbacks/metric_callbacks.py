"""메트릭 관련 Callback들"""

from transformers.trainer_callback import TrainerCallback
import os
import glob
import torch


class FIDTopKCheckpointCallback(TrainerCallback):
    """FID 메트릭 기반으로 Top-K 체크포인트만 유지하는 Callback"""
    
    def __init__(self, output_dir, k=3, metric_key="fid_B"):
        super().__init__()
        self.output_dir = output_dir
        self.k = k
        self.metric_key = metric_key
        self.fid_ckpts = []  # List of (fid_score, epoch, weights_path) tuples
        self.pending_cleanup = None  # Store evaluation results for cleanup after save
        self.weights_dir = None  # Will be set in on_train_begin

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training - clean up existing checkpoints if needed"""
        self.weights_dir = args.output_dir
        
        # Find all existing epoch directories
        existing_epochs = []
        if os.path.exists(self.weights_dir):
            for item in os.listdir(self.weights_dir):
                if item.startswith('epoch') and os.path.isdir(os.path.join(self.weights_dir, item)):
                    try:
                        epoch_num = int(item.replace('epoch', ''))
                        epoch_path = os.path.join(self.weights_dir, item)
                        existing_epochs.append((epoch_num, epoch_path))
                    except ValueError:
                        continue
        
        if len(existing_epochs) > self.k:
            # Sort by epoch number (keep latest ones) - cleanup silently
            existing_epochs.sort(key=lambda x: x[0], reverse=True)
            
            to_keep = existing_epochs[:self.k]
            to_delete = existing_epochs[self.k:]
            
            # Delete old checkpoints silently
            for epoch_num, epoch_path in to_delete:
                if os.path.exists(epoch_path):
                    import shutil
                    shutil.rmtree(epoch_path)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after each evaluation - store results for cleanup after save"""
        if metrics is None:
            return
            
        fid = metrics.get(self.metric_key, None)
        if fid is None:
            return
        
        # Store evaluation results for cleanup after checkpoint is saved
        epoch = int(state.epoch)
        weights_ckpt_path = os.path.abspath(os.path.join(args.output_dir, f"epoch{epoch}"))
        
        # Store epoch evaluation info (removed verbose prints)
        
        # Store for cleanup after save
        self.pending_cleanup = (fid, epoch, weights_ckpt_path)
    
    def on_save(self, args, state, control, **kwargs):
        """Called after checkpoint is saved - perform cleanup now"""
        if self.pending_cleanup is None:
            return
            
        fid, epoch, weights_ckpt_path = self.pending_cleanup
        
        # Add to tracking list (weights path only)
        self.fid_ckpts.append((fid, epoch, weights_ckpt_path))
        
        # Sort by FID (lower is better)
        self.fid_ckpts = sorted(self.fid_ckpts, key=lambda x: x[0])
        
        # Delete checkpoints beyond top-k (silently)
        if len(self.fid_ckpts) > self.k:
            to_keep = self.fid_ckpts[:self.k]
            to_delete = self.fid_ckpts[self.k:]
            
            # Delete without printing
            for fid_score, ep, weights_path in to_delete:
                if weights_path and os.path.exists(weights_path):
                    import shutil
                    shutil.rmtree(weights_path)
            
            # Keep only top-k
            self.fid_ckpts = to_keep
        
        # Clear pending cleanup
        self.pending_cleanup = None
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training - show final checkpoint summary"""
        if self.fid_ckpts:
            print(f"\n" + "="*80)
            print(f"FINAL TOP-{self.k} CHECKPOINTS (by FID score)")
            print("="*80)
            for i, (fid_score, ep, weights_path) in enumerate(self.fid_ckpts):
                w_status = "✓" if (weights_path and os.path.exists(weights_path)) else "✗"
                print(f"  {i+1}. epoch{ep} - FID: {fid_score:.4f} [Weights: {w_status}]")
            print("="*80) 
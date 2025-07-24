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
        self.fid_ckpts = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        fid = metrics.get(self.metric_key, None)
        if fid is None:
            return
        ckpt_path = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        self.fid_ckpts.append((fid, ckpt_path))
        self.fid_ckpts = sorted(self.fid_ckpts, key=lambda x: x[0])
        for _, path in self.fid_ckpts[self.k:]:
            if os.path.exists(path):
                print(f"Deleting checkpoint: {path}")
                import shutil
                shutil.rmtree(path)
        self.fid_ckpts = self.fid_ckpts[:self.k] 
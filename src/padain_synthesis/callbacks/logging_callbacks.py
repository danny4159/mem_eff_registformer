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
            print("WandbImageCallback: trainer 인스턴스가 없습니다.")
            return
        
        # train_dataset에서 첫 번째/두 번째 환자의 30, 60번째 slice 추출
        dataset = self.trainer.train_dataset
        indices = []
        num_patients = len(dataset.patient_keys)
        for patient_idx in [0, 1]:  # Changed from [0] to [0,1]
            if patient_idx >= num_patients:
                continue
            base = dataset.cumulative_slice_counts[patient_idx]
            num_slices = dataset.slice_counts[patient_idx]
            for slice_idx in [30, 60]:
                if slice_idx < num_slices:
                    indices.append(base + slice_idx)
        
        images = [dataset[i] for i in indices]
        for i, sample in enumerate(images):
            # sample is now a tuple (A, B, C, D, E)
            input_image, reference_image, C, D, E = sample
            
            # [-1, 1] 범위를 [0, 255] 범위로 정규화
            def normalize_for_wandb(img):
                # [-1, 1] -> [0, 1] -> [0, 255]
                img_normalized = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                return img_normalized
            
            real_a = normalize_for_wandb(input_image[0].cpu().numpy())
            real_b = normalize_for_wandb(reference_image[0].cpu().numpy())
            fake_b = normalize_for_wandb(
                self.trainer.model(
                    torch.cat([input_image, reference_image], dim=0).unsqueeze(0).to(self.trainer.model.device)
                ).last_hidden_state[0,0].detach().cpu().numpy()
            )
            
            # wandb.Image로 업로드
            wandb.log({
                f"epoch_{state.epoch}_sample_{i}_real_a": wandb.Image(real_a, caption="real_a"),
                f"epoch_{state.epoch}_sample_{i}_real_b": wandb.Image(real_b, caption="real_b"),
                f"epoch_{state.epoch}_sample_{i}_fake_b": wandb.Image(fake_b, caption="fake_b"),
            })


class EpochProgressCallback(TrainerCallback):
    """에포크 진행상황을 출력하는 Callback"""
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {int(state.epoch)} finished. Step: {state.global_step}") 
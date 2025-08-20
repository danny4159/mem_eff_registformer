"""설정 저장 관련 Callback들"""

from transformers.trainer_callback import TrainerCallback
import os
import json


class ParamsSaveCallback(TrainerCallback):
    """훈련 파라미터를 체크포인트와 함께 저장하는 Callback"""
    
    def __init__(self, params):
        super().__init__()
        self.params = params

    def on_save(self, args, state, control, **kwargs):
        # Use epoch-based naming - save directly in epoch folder
        if state.epoch is not None:
            epoch = int(state.epoch)
            checkpoint_dir = os.path.join(args.output_dir, f"epoch{epoch}")
        else:
            # Fallback to step-based naming
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        # Ensure directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        params_path = os.path.join(checkpoint_dir, "params.json")
        with open(params_path, "w") as f:
            json.dump(self.params, f, indent=2)


class DataConfigSaveCallback(TrainerCallback):
    """데이터 설정을 체크포인트와 함께 저장하는 Callback"""
    
    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config

    def on_save(self, args, state, control, **kwargs):
        # Use epoch-based naming - save directly in epoch folder
        if state.epoch is not None:
            epoch = int(state.epoch)
            checkpoint_dir = os.path.join(args.output_dir, f"epoch{epoch}")
        else:
            # Fallback to step-based naming
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        # Ensure directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        data_config_path = os.path.join(checkpoint_dir, "data_config.json")
        with open(data_config_path, "w") as f:
            json.dump(self.data_config, f, indent=2) 
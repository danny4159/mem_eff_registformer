import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Set GPU device

import torch
from src.padain_synthesis.dataset import PadainSynthesisDataModule
from src.padain_synthesis.trainer import padain_synthesis_data_collator
from src.padain_synthesis import (
    PadainSynthesisConfig, 
    PadainSynthesisModel,
    DEFAULT_TRAINING_ARGS,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_PARAMS,
    DEFAULT_DATA_CONFIG
)
from src.padain_synthesis.trainer import (
    create_padain_synthesis_trainer,
)
from src.padain_synthesis.callbacks import (
    WandbImageCallback, 
    EpochProgressCallback,
    FIDTopKCheckpointCallback,
    ParamsSaveCallback,
    DataConfigSaveCallback
)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
from src.padain_synthesis.config import DEFAULT_DATA_CONFIG

# 기본 데이터 설정 사용 (필요시에만 오버라이드)
data_config = DEFAULT_DATA_CONFIG.copy()

# ============================================================================
# MODEL CONFIGURATION  
# ============================================================================
from src.padain_synthesis.config import DEFAULT_MODEL_CONFIG

# 기본 모델 설정 사용
model_config = PadainSynthesisConfig(**DEFAULT_MODEL_CONFIG)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# 기본 훈련 설정 사용 (배치 크기 동적 설정)
training_args = DEFAULT_TRAINING_ARGS.copy()
training_args.update({
    "per_device_train_batch_size": data_config["batch_size"],
    "per_device_eval_batch_size": 1,  # 평가는 항상 배치 크기 1
})

# Loss function weights and parameters (배치 크기 동적 설정)
training_params = DEFAULT_TRAINING_PARAMS.copy()
training_params.update({
    "batch_size": data_config["batch_size"],  # data_config에서 가져온 배치 크기
})

# ============================================================================
# TRAINING EXECUTION
# ============================================================================

# 1. Prepare data loaders
datamodule = PadainSynthesisDataModule(data_config)
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

# 2. Initialize model
model = PadainSynthesisModel(model_config)

# 3. Setup callbacks
wandb_callback = WandbImageCallback()
fid_callback = FIDTopKCheckpointCallback(output_dir="weights/padain_synthesis", k=3, metric_key="fid_B")
params_callback = ParamsSaveCallback(training_params)
data_config_callback = DataConfigSaveCallback(data_config)

# 4. Create trainer and start training
model_type = "padain_synthesis"  # ← 모델 이름(혹은 실험 타입)
run_name = training_args["run_name"]
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("weights", model_type, run_name, now)
os.makedirs(output_dir, exist_ok=True)
training_args["output_dir"] = output_dir

trainer = create_padain_synthesis_trainer(
    model=model,
    train_dataset=train_loader.dataset,
    training_args=training_args,
    params=training_params,
    eval_dataset=val_loader.dataset,
    data_collator=padain_synthesis_data_collator, # PadainSynthesisDataCollator 제거
    callbacks=[wandb_callback, EpochProgressCallback(), fid_callback, params_callback, data_config_callback],
)

wandb_callback.set_trainer(trainer)
trainer.train()

# ============================================================================
# GPU STATUS VERIFICATION
# ============================================================================
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"Model device: {next(model.parameters()).device}")
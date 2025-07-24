"""PadainSynthesis 기본 설정값들"""

# 기본 훈련 인수 (실험용 설정으로 변경)
DEFAULT_TRAINING_ARGS = {
    "num_train_epochs": 1,           # 실험용으로 1 에포크만
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "warmup_steps": 500,
    "logging_steps": 999999,          # 로깅 최소화
    "eval_steps": 1000,
    "save_steps": 1000,
    "save_strategy": "epoch",         # 실험용으로 epoch 단위 저장
    "save_total_limit": None,         # 실험용으로 제한 없음
    "load_best_model_at_end": True,   # 실험용으로 최고 모델 로드
    "metric_for_best_model": "fid_B", # 실험용 메트릭
    "greater_is_better": False,       # FID는 낮을수록 좋음
    "fp16": True,
    "dataloader_num_workers": 0,      # 디버깅용
    "remove_unused_columns": False,
    "report_to": "wandb",             # 실험용으로 wandb 사용
    "eval_strategy": "epoch",         # 실험용으로 epoch 단위 평가
    "run_name": "padain_synthesis_experiment",  # 실험명
    # batch_size 관련 설정은 제거 - data_config에서 동적 설정
}

# 기본 모델 설정
DEFAULT_MODEL_CONFIG = {
    # Generator settings
    "input_nc": 1,               # Number of input channels (1 for grayscale medical images)
    "feat_ch": 512,              # Number of feature channels
    "output_nc": 1,              # Number of output channels (1 for grayscale medical images)
    "demodulate": True,           # Whether to use demodulation in AdaIN layers
    "is_3d": False,              # Whether the model operates on 3D volumes
    
    # Discriminator settings
    "use_discriminator": True,   # Whether to use discriminator
    "discriminator_ndf": 64,     # Number of discriminator filters
    "discriminator_n_layers": 3, # Number of discriminator layers
    "gan_type": "lsgan",         # GAN type (lsgan, nsgan) - vanilla는 지원되지 않음
    
    # PatchSampleF settings
    "use_mlp": False,            # Whether to use MLP in PatchSampleF
    "init_type": "normal",       # Initialization type for PatchSampleF
    "init_gain": 0.02,           # Initialization gain for PatchSampleF
    "nc": 256,                   # Number of channels for PatchSampleF
    "input_nc_patch": 256,       # Input channels for PatchSampleF
}

# 기본 훈련 파라미터
DEFAULT_TRAINING_PARAMS = {
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
    # batch_size는 제거 - data_config에서 동적 설정
}

# 기본 데이터 설정
DEFAULT_DATA_CONFIG = {
    # File paths for train/validation/test datasets
    "train_file": "./data/synthrad2023_mr-ct_pelvis/train/Ver3_OnlyOnePatient.h5",
    "val_file": "./data/synthrad2023_mr-ct_pelvis/val/Ver3_OnlyOnePatient.h5",  # Ver3_OnlyOnePatient / Ver3_OnlyTwoPatients
    "test_file": "./data/synthrad2023_mr-ct_pelvis/test/Ver3_OnlyOnePatient.h5",
    
    # H5 file data group names (source and target modalities)
    "data_group_1": "MR",      # Source modality (input)
    "data_group_2": "CT",      # Target modality (output)
    "data_group_3": None,      # Additional modality 1 (optional)
    "data_group_4": None,      # Additional modality 2 (optional) 
    "data_group_5": None,      # Additional modality 3 (optional)
    
    # Data processing settings
    "is_3d": False,            # Whether to process 3D volumes (True) or 2D slices (False)
    "batch_size": 1,           # Batch size for training
    "num_workers": 0,          # Number of data loading workers (0 for debugging)
    "padding_size": None,      # Target size for padding (height, width) - None for no padding
    "crop_size": (96, 96),     # Size for random cropping (height, width) - training only
    "flip_prob": 0.0,          # Probability of horizontal flipping for data augmentation
    "rot_prob": 0.0,           # Probability of 90-degree rotation for data augmentation
    "reverse": False,          # Whether to reverse the order of source and target
    "norm_ZeroToOne": False,   # Whether to normalize data to [0, 1] range
    "resize_size": None,       # Target size for resizing (height, width) - None for no resizing
    "pin_memory": False,       # Whether to pin memory for faster GPU transfer
} 
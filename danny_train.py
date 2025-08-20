import os
import datetime
import sys
import logging
import torch
import importlib

# ============================================================================
# GPU SETUP
# ============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set GPU device

# ============================================================================
# NETWORK CONFIGURATION - Change this to use different networks
# ============================================================================
# NOTE: All network implementations must be located under src/ folder
#       Each network should have its own subfolder (e.g., src/padain_synthesis/, src/another_network/)
NETWORK_NAME = "padain_synthesis"

# ============================================================================
# DYNAMIC IMPORT BASED ON NETWORK NAME
# ============================================================================
def load_network_config(network_name: str):
    """
    Dynamically load network configuration based on network name.
    
    Args:
        network_name: Name of the network (must match folder name in src/)
        
    Returns:
        Tuple of (module, TRAINING_ARGS, MODEL_CONFIG, TRAINING_PARAMS, DATA_CONFIG)
    """
    import importlib
    from pathlib import Path
    
    # Check if network folder exists
    project_root = Path(__file__).parent
    network_path = project_root / "src" / network_name
    config_path = network_path / "config.py"
    
    if not network_path.exists():
        available_networks = [d.name for d in (project_root / "src").iterdir() 
                            if d.is_dir() and d.name != "common" and d.name != "models"]
        raise ValueError(f"Network '{network_name}' not found in src/. Available networks: {available_networks}")
    
    if not config_path.exists():
        raise ValueError(f"config.py not found in src/{network_name}/")
    
    # Import the network module
    try:
        # Import config
        config_module = importlib.import_module(f"src.{network_name}.config")
        
        # Import main module components
        main_module = importlib.import_module(f"src.{network_name}")
        
        # Get required configurations
        TRAINING_ARGS = getattr(config_module, 'TRAINING_ARGS')
        MODEL_CONFIG = getattr(config_module, 'MODEL_CONFIG')
        TRAINING_PARAMS = getattr(config_module, 'TRAINING_PARAMS')
        DATA_CONFIG = getattr(config_module, 'DATA_CONFIG')
        
        print(f"✅ Successfully loaded network: {network_name}")
        print(f"📁 Network path: {network_path}")
        
        return main_module, TRAINING_ARGS, MODEL_CONFIG, TRAINING_PARAMS, DATA_CONFIG
        
    except Exception as e:
        raise Exception(f"Failed to load {network_name}: {e}")

# Load network configuration
network_module, TRAINING_ARGS, MODEL_CONFIG, TRAINING_PARAMS, DATA_CONFIG = load_network_config(NETWORK_NAME)

# Extract required classes using Hugging Face standard naming convention
# Each network should have Config and Model classes in their modeling_*.py
try:
    model_module = importlib.import_module(f"src.{NETWORK_NAME}.modeling_{NETWORK_NAME}")
    
    # Standard naming: {NetworkName}Config and {NetworkName}Model
    config_class_name = f'{NETWORK_NAME.title().replace("_", "")}Config'
    model_class_name = f'{NETWORK_NAME.title().replace("_", "")}Model'
    
    ConfigClass = getattr(model_module, config_class_name, None)
    ModelClass = getattr(model_module, model_class_name, None)
    
    if ConfigClass is None or ModelClass is None:
        raise ImportError(f"Required classes {config_class_name}, {model_class_name} not found in {NETWORK_NAME}.modeling_{NETWORK_NAME}")
        
except ImportError as e:
    raise ImportError(f"Failed to import model classes from {NETWORK_NAME}: {e}")

# ============================================================================
# DYNAMIC IMPORTS - Import required components from the network module
# ============================================================================
def import_network_components(network_name: str):
    """Import required components from the network module."""
    import importlib
    
    try:
        # Import trainer module
        trainer_module = importlib.import_module(f"src.{network_name}.trainer")
        
        # Import callbacks module from common
        callbacks_module = importlib.import_module("src.common.callbacks")
        
        # Import DataModule directly from common datasets (no longer network-specific)
        datasets_module = importlib.import_module("src.common.datasets")
        
        # Get required components
        DataModule = getattr(datasets_module, 'DataModule')
        data_collator = getattr(trainer_module, f'{network_name}_data_collator')
        create_trainer = getattr(trainer_module, f'create_{network_name}_trainer')
        
        # Get callbacks
        WandbImageCallback = getattr(callbacks_module, 'WandbImageCallback')
        EpochProgressCallback = getattr(callbacks_module, 'EpochProgressCallback')
        FIDTopKCheckpointCallback = getattr(callbacks_module, 'FIDTopKCheckpointCallback')
        ParamsSaveCallback = getattr(callbacks_module, 'ParamsSaveCallback')
        DataConfigSaveCallback = getattr(callbacks_module, 'DataConfigSaveCallback')
        
        return (DataModule, data_collator, create_trainer, 
                WandbImageCallback, EpochProgressCallback, FIDTopKCheckpointCallback,
                ParamsSaveCallback, DataConfigSaveCallback)
        
    except Exception as e:
        print(f"❌ Error importing network '{network_name}': {e}")
        raise

# Import network components
(DataModule, data_collator, create_trainer, 
 WandbImageCallback, EpochProgressCallback, FIDTopKCheckpointCallback,
 ParamsSaveCallback, DataConfigSaveCallback) = import_network_components(NETWORK_NAME)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Use data configuration (override if needed)
data_config = DATA_CONFIG.copy()

# ============================================================================
# TRAINING/VALIDATION SPLIT CONFIGURATION
# ============================================================================
# NOTE: 'try' dataset contains only 1 NII.gz file for quick testing/debugging
# 
# use_try_dataset: Both training and validation use 'try' folder (1 file only)
# use_try_for_val_only: Training uses 'train' folder, validation uses 'try' folder (recommended for fast validation)
data_config["use_try_dataset"] = True
data_config["use_try_for_val_only"] = True

# ============================================================================
# MODEL CONFIGURATION  
# ============================================================================
# Use model configuration
model_config = ConfigClass(**MODEL_CONFIG)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Use training arguments (dynamic batch size)
training_args = TRAINING_ARGS.copy()

# Enhanced progress bar and logging settings
training_args.update({
    "disable_tqdm": False,              # Enable progress bar
    "logging_first_step": True,         # Log first step
    "logging_steps": 10,                # Log every 10 steps  
    "dataloader_drop_last": False,      # Don't drop incomplete batches
    "log_level": "error",               # Suppress most default logging including raw dict output
})

# Loss function weights and parameters (dynamic batch size)
training_params = TRAINING_PARAMS.copy()
training_params.update({
    "batch_size": data_config["batch_size"],  # Required by PatchNCE Loss for negative sampling
})

# ============================================================================
# TRAINING EXECUTION
# ============================================================================

# 1. Setup directory structure
model_type = NETWORK_NAME  # Use dynamic network name
run_name = training_args["run_name"]
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create unified training output directory structure with run_name
base_output_dir = os.path.join("training_output", model_type)
run_dir = os.path.join(base_output_dir, run_name)  # run_name 추가
timestamp_dir = os.path.join(run_dir, now)
weights_dir = os.path.join(timestamp_dir, "weights")
wandb_dir = os.path.join(timestamp_dir, "wandb")
outputs_dir = os.path.join(timestamp_dir, "outputs")

# Create directories
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(wandb_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

# ============================================================================
# LOGGING SETUP - 모든 터미널 출력을 파일로 저장
# ============================================================================
log_file = os.path.join(timestamp_dir, "training.log")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # 터미널에도 계속 출력
    ]
)

# stdout을 로그 파일에도 저장하도록 설정
class TeeOutput:
    def __init__(self, *files):
        self.files = files
    
    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()
    
    def flush(self):
        for file in self.files:
            file.flush()

# 원본 stdout 백업
original_stdout = sys.stdout
original_stderr = sys.stderr

# 로그 파일 열기
log_file_handle = open(log_file, 'w', encoding='utf-8')

# stdout과 stderr를 터미널과 파일 모두에 출력
sys.stdout = TeeOutput(original_stdout, log_file_handle)
sys.stderr = TeeOutput(original_stderr, log_file_handle)

print(f"🚀 Starting training with network: {NETWORK_NAME}")
print(f"📝 모든 터미널 출력이 다음 파일에 저장됩니다: {log_file}")
print("="*80)

# Update training arguments with new output directory
training_args["output_dir"] = weights_dir

# Set wandb environment variables for unified directory
os.environ["WANDB_DIR"] = timestamp_dir  # Set to timestamp_dir instead of wandb_dir
os.environ["WANDB_PROJECT"] = f"{model_type}_training"

# 2. Prepare data loaders
datamodule = DataModule(data_config)
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

# 3. Initialize model (using dynamic model class)
model = ModelClass(model_config)

# 4. Setup callbacks
wandb_callback = WandbImageCallback()
epoch_callback = EpochProgressCallback()
fid_callback = FIDTopKCheckpointCallback(output_dir=outputs_dir, k=2, metric_key="eval_fid_B")
params_callback = ParamsSaveCallback(training_params)
data_config_callback = DataConfigSaveCallback(data_config)

# 5. Create trainer and start training
slice_inferer = datamodule.get_slice_inferer()
trainer = create_trainer(
    model=model,
    train_dataset=train_loader.dataset,
    training_args=training_args,
    params=training_params,
    eval_dataset=val_loader.dataset,
    data_collator=data_collator,
    slice_inferer=slice_inferer,
    callbacks=[wandb_callback, epoch_callback, fid_callback, params_callback, data_config_callback],
)

wandb_callback.set_trainer(trainer)
epoch_callback.set_trainer(trainer)

# Print training information
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80)
print(f"📊 Network: {NETWORK_NAME}")
print(f"📈 Epochs: {training_args['num_train_epochs']}")
print(f"🔢 Batch Size: {data_config['batch_size']}")
print(f"📚 Train Dataset: {len(train_loader.dataset)} samples")
print(f"🔍 Val Dataset: {len(val_loader.dataset)} samples")
print(f"⚡ Device: GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"📁 Output Dir: {weights_dir}")
print("="*80)

trainer.train()

# ============================================================================
# CLEANUP - 로그 파일 정리
# ============================================================================
# stdout과 stderr를 원래대로 복원
sys.stdout = original_stdout
sys.stderr = original_stderr

# 로그 파일 닫기
log_file_handle.close()

print(f"\n✅ 학습 완료! 모든 로그가 저장되었습니다: {log_file}")

# ============================================================================
# OUTPUT DIRECTORY STRUCTURE
# ============================================================================
print("\n" + "="*80)
print("TRAINING OUTPUT DIRECTORY STRUCTURE")
print("="*80)
print(f"Base directory: {base_output_dir}")
print(f"└── {run_name}/")
print(f"    └── {now}/")
print(f"        ├── training.log          # 모든 터미널 출력 로그")
print(f"        ├── weights/              # Model checkpoints, configs, and training artifacts")
print(f"        ├── wandb/                # Wandb logs and artifacts (run-xxxxx folders)")
print(f"        └── outputs/              # Additional outputs (FID checkpoints, etc.)")
print("="*80)
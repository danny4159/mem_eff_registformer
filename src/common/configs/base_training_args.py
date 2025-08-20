"""Base Training Arguments Configuration

This file contains common training arguments that are shared across different networks.
Network-specific configurations can override these base settings.
"""

# Base training arguments - common across all networks
BASE_TRAINING_ARGS = {
    # Training schedule
    "num_train_epochs": 100,           # Number of training epochs
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "warmup_steps": 500,
    
    # Logging and evaluation
    "logging_strategy": "epoch",       # Log every epoch instead of steps
    "eval_strategy": "epoch",          # Evaluation strategy
    "save_strategy": "epoch",          # Save strategy
    "save_total_limit": 3,             # Keep only last 3 checkpoints to save space
    "load_best_model_at_end": True,    # Load best model at end
    "greater_is_better": False,        # Lower is better (for loss metrics like FID)
    
    # Performance optimization
    "fp16": True,
    "remove_unused_columns": False,
    "dataloader_num_workers": 8,       # Data loader workers
    "dataloader_pin_memory": True,     # Pin memory for faster GPU transfer
    
    # Monitoring and Logging
    "report_to": "wandb",              # Reporting tool
    "disable_tqdm": False,             # Enable progress bar
    "log_level": "warning",            # Reduce log verbosity to focus on important messages
    "logging_steps": 10,               # Log every 10 steps
    "logging_first_step": True,        # Log first step
    "logging_nan_inf_filter": True,    # Filter NaN/Inf values in logs
}

# Network-specific overrides can be defined here if needed
NETWORK_SPECIFIC_OVERRIDES = {
    "padain_synthesis": {
        "metric_for_best_model": "eval_fid_B",  # Best model metric for PadainSynthesis
    },
    # Add other networks here as needed
    # "other_network": {
    #     "metric_for_best_model": "eval_loss",
    # },
}

def get_training_args(network_name: str, run_name: str = None, output_dir: str = None, **overrides):
    """
    Get TrainingArguments instance for a specific network.
    
    Args:
        network_name (str): Name of the network
        run_name (str): Name of the training run
        output_dir (str): Output directory for training artifacts
        **overrides: Additional overrides for training arguments
    
    Returns:
        dict: Training arguments dictionary (to maintain compatibility)
    """
    # Start with base arguments
    training_args = BASE_TRAINING_ARGS.copy()
    
    # Apply network-specific overrides
    if network_name in NETWORK_SPECIFIC_OVERRIDES:
        training_args.update(NETWORK_SPECIFIC_OVERRIDES[network_name])
    
    # Set run name if provided
    if run_name:
        training_args["run_name"] = run_name
    
    # Set output directory if provided
    if output_dir:
        training_args["output_dir"] = output_dir
    
    # Apply custom overrides
    training_args.update(overrides)
    
    # Return dict to maintain compatibility with danny_train.py
    return training_args

def create_training_arguments(**kwargs):
    """
    Create TrainingArguments instance from dictionary.
    
    Args:
        **kwargs: Training arguments as keyword arguments
    
    Returns:
        TrainingArguments: Configured training arguments instance
    """
    from transformers import TrainingArguments
    return TrainingArguments(**kwargs) 
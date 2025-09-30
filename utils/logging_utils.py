"""
Logging Utilities for AML Multi-GNN Project
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import yaml


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "results/experiments",
    experiment_name: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for the project
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to save log files
        experiment_name: Name for the experiment (used in log filename)
    
    Returns:
        Configured logger instance
    """
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        log_filename = f"{experiment_name}_{timestamp}.log"
    else:
        log_filename = f"aml_multignn_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("AML_MultiGNN")
    logger.info(f"Logging initialized. Log file: {log_path}")
    
    return logger


def log_experiment_config(logger: logging.Logger, config: Dict[str, Any]):
    """
    Log experiment configuration
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("=" * 50)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 50)
    
    for section, params in config.items():
        logger.info(f"\n{section.upper()}:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("=" * 50)


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: Dict[str, float]
):
    """
    Log training progress
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        metrics: Dictionary of evaluation metrics
    """
    logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")


def log_model_info(logger: logging.Logger, model: Any):
    """
    Log model information
    
    Args:
        logger: Logger instance
        model: PyTorch model instance
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=" * 50)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 50)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")
    logger.info("=" * 50)


def log_data_info(logger: logging.Logger, data_info: Dict[str, Any]):
    """
    Log dataset information
    
    Args:
        logger: Logger instance
        data_info: Dictionary containing dataset statistics
    """
    logger.info("=" * 50)
    logger.info("DATASET INFORMATION")
    logger.info("=" * 50)
    
    for key, value in data_info.items():
        logger.info(f"{key}: {value}")
    
    logger.info("=" * 50)


def log_gpu_info(logger: logging.Logger):
    """
    Log GPU information
    
    Args:
        logger: Logger instance
    """
    import torch
    from .gpu_utils import get_gpu_memory_info, get_system_info
    
    logger.info("=" * 50)
    logger.info("GPU INFORMATION")
    logger.info("=" * 50)
    
    if torch.cuda.is_available():
        system_info = get_system_info()
        memory_info = get_gpu_memory_info()
        
        logger.info(f"GPU: {system_info['gpu_name']}")
        logger.info(f"CUDA Version: {system_info['cuda_version']}")
        logger.info(f"PyTorch Version: {system_info['pytorch_version']}")
        logger.info(f"GPU Memory Total: {memory_info['total']:.1f} GB")
        logger.info(f"GPU Memory Used: {memory_info['allocated']:.1f} GB")
        logger.info(f"GPU Memory Free: {memory_info['free']:.1f} GB")
    else:
        logger.warning("No GPU available")
    
    logger.info("=" * 50)


def save_config_to_yaml(config: Dict[str, Any], filepath: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the YAML file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to: {filepath}")


def load_config_from_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        filepath: Path to the YAML file
    
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


class ExperimentLogger:
    """
    Context manager for experiment logging
    """
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.config = config
        self.logger = None
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger = setup_logging(experiment_name=self.experiment_name)
        
        log_experiment_config(self.logger, self.config)
        log_gpu_info(self.logger)
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            end_time = datetime.now()
            duration = end_time - self.start_time
            self.logger.info(f"Experiment completed in {duration}")
            
            if exc_type is not None:
                self.logger.error(f"Experiment failed: {exc_val}")


if __name__ == "__main__":
    # Test logging utilities
    logger = setup_logging()
    logger.info("Testing logging utilities")
    
    # Test experiment logger
    config = {"model": {"hidden_dim": 64}, "training": {"epochs": 100}}
    
    with ExperimentLogger("test_experiment", config) as exp_logger:
        exp_logger.info("This is a test log message")

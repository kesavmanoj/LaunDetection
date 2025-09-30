"""
GPU Management Utilities for AML Multi-GNN Project
Optimized for Google Colab with Tesla T4 GPU
"""

import torch
import gc
import psutil
import os
from typing import Optional, Dict, Any


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, otherwise CPU)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    return device


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage information
    Returns memory usage in GB
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "cached": 0.0, "total": 0.0}
    
    allocated = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return {
        "allocated": allocated,
        "cached": cached,
        "total": total,
        "free": total - allocated
    }


def clear_gpu_memory():
    """
    Clear GPU memory cache
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU memory cleared")


def monitor_memory_usage(threshold: float = 0.8) -> bool:
    """
    Monitor GPU memory usage and return True if usage is below threshold
    
    Args:
        threshold: Memory usage threshold (0.8 = 80%)
    
    Returns:
        True if memory usage is acceptable, False otherwise
    """
    if not torch.cuda.is_available():
        return True
    
    memory_info = get_gpu_memory_info()
    usage_ratio = memory_info["allocated"] / memory_info["total"]
    
    if usage_ratio > threshold:
        print(f"Warning: GPU memory usage is {usage_ratio:.1%} (threshold: {threshold:.1%})")
        return False
    
    return True


def optimize_memory_settings():
    """
    Optimize PyTorch memory settings for Google Colab
    """
    if torch.cuda.is_available():
        # Enable memory efficient attention if available
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        print("Memory optimization settings applied")


def get_system_info() -> Dict[str, Any]:
    """
    Get system information including CPU, RAM, and GPU details
    """
    info = {
        "cpu_count": os.cpu_count(),
        "ram_gb": psutil.virtual_memory().total / 1e9,
        "ram_available_gb": psutil.virtual_memory().available / 1e9,
    }
    
    if torch.cuda.is_available():
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        })
    
    return info


def print_system_info():
    """
    Print comprehensive system information
    """
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    
    info = get_system_info()
    
    print(f"CPU Cores: {info['cpu_count']}")
    print(f"RAM Total: {info['ram_gb']:.1f} GB")
    print(f"RAM Available: {info['ram_available_gb']:.1f} GB")
    
    if torch.cuda.is_available():
        print(f"GPU: {info['gpu_name']}")
        print(f"GPU Memory: {info['gpu_memory_gb']:.1f} GB")
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"PyTorch Version: {info['pytorch_version']}")
        
        # Current memory usage
        memory_info = get_gpu_memory_info()
        print(f"GPU Memory Used: {memory_info['allocated']:.1f} GB / {memory_info['total']:.1f} GB")
    else:
        print("No GPU available")
    
    print("=" * 50)


def setup_mixed_precision():
    """
    Setup mixed precision training for memory efficiency
    """
    if torch.cuda.is_available():
        # Enable automatic mixed precision
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print("Mixed precision settings enabled")
        return True
    return False


def memory_cleanup():
    """
    Comprehensive memory cleanup
    """
    # Clear Python garbage collection
    gc.collect()
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("Memory cleanup completed")


if __name__ == "__main__":
    # Test GPU utilities
    print_system_info()
    device = get_device()
    optimize_memory_settings()
    setup_mixed_precision()

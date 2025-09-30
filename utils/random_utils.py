"""
Random Seed and Reproducibility Utilities for AML Multi-GNN Project
"""

import random
import numpy as np
import torch
import os
from typing import Optional


def set_random_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic algorithms
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed} (deterministic: {deterministic})")


def get_random_state():
    """
    Get current random state for all libraries
    
    Returns:
        Dictionary containing random states
    """
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }


def set_random_state(state_dict: dict):
    """
    Set random state from saved state
    
    Args:
        state_dict: Dictionary containing random states
    """
    random.setstate(state_dict['python'])
    np.random.set_state(state_dict['numpy'])
    torch.set_rng_state(state_dict['torch'])
    
    if torch.cuda.is_available() and state_dict['torch_cuda'] is not None:
        torch.cuda.set_rng_state(state_dict['torch_cuda'])


def ensure_reproducibility(seed: int = 42):
    """
    Ensure complete reproducibility across runs
    
    Args:
        seed: Random seed value
    """
    set_random_seed(seed, deterministic=True)
    
    # Additional reproducibility measures
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print("Reproducibility ensured with deterministic algorithms")


def create_reproducible_dataloader(dataset, batch_size: int, shuffle: bool = True, **kwargs):
    """
    Create a reproducible DataLoader
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        **kwargs: Additional DataLoader arguments
    
    Returns:
        Reproducible DataLoader
    """
    from torch.utils.data import DataLoader
    
    def worker_init_fn(worker_id):
        """Initialize worker with different seed for each worker"""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=worker_init_fn,
        **kwargs
    )


def test_reproducibility(seed: int = 42, num_tests: int = 5):
    """
    Test reproducibility by running the same operation multiple times
    
    Args:
        seed: Random seed value
        num_tests: Number of tests to run
    
    Returns:
        True if reproducible, False otherwise
    """
    results = []
    
    for i in range(num_tests):
        set_random_seed(seed, deterministic=True)
        
        # Generate some random numbers
        python_rand = random.random()
        numpy_rand = np.random.random()
        torch_rand = torch.rand(1).item()
        
        results.append((python_rand, numpy_rand, torch_rand))
    
    # Check if all results are identical
    first_result = results[0]
    is_reproducible = all(result == first_result for result in results)
    
    if is_reproducible:
        print("✓ Reproducibility test passed")
    else:
        print("✗ Reproducibility test failed")
        print("Results:", results)
    
    return is_reproducible


if __name__ == "__main__":
    # Test reproducibility
    print("Testing reproducibility...")
    test_reproducibility()
    
    # Test random state saving/loading
    print("\nTesting random state saving/loading...")
    set_random_seed(42)
    state = get_random_state()
    
    # Generate some numbers
    nums1 = [random.random() for _ in range(3)]
    
    # Generate more numbers
    nums2 = [random.random() for _ in range(3)]
    
    # Restore state and generate again
    set_random_state(state)
    nums3 = [random.random() for _ in range(3)]
    
    print(f"First generation: {nums1}")
    print(f"After restore: {nums3}")
    print(f"Match: {nums1 == nums3}")

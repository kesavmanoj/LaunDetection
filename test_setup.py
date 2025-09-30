"""
Test script to verify AML Multi-GNN setup
"""

import sys
import os
from pathlib import Path


def test_basic_imports():
    """
    Test basic package imports
    """
    print("Testing basic imports...")
    
    try:
        import pandas as pd
        print(f"✓ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas: {e}")
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
    
    try:
        import matplotlib.pyplot as plt
        print(f"✓ Matplotlib: {plt.matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
    
    try:
        import seaborn as sns
        print(f"✓ Seaborn: {sns.__version__}")
    except ImportError as e:
        print(f"✗ Seaborn: {e}")
    
    try:
        import sklearn
        print(f"✓ Scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn: {e}")
    
    try:
        import networkx as nx
        print(f"✓ NetworkX: {nx.__version__}")
    except ImportError as e:
        print(f"✗ NetworkX: {e}")


def test_pytorch():
    """
    Test PyTorch installation
    """
    print("\nTesting PyTorch...")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  GPU not available")
            
    except ImportError as e:
        print(f"✗ PyTorch: {e}")


def test_pytorch_geometric():
    """
    Test PyTorch Geometric installation
    """
    print("\nTesting PyTorch Geometric...")
    
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric: {torch_geometric.__version__}")
        
        # Test basic functionality
        from torch_geometric.data import Data
        print("✓ Basic PyTorch Geometric functionality works")
        
    except ImportError as e:
        print(f"⚠️  PyTorch Geometric: {e}")
        print("This is optional for basic functionality")


def test_data_access():
    """
    Test data access
    """
    print("\nTesting data access...")
    
    # Check if data directory exists
    data_paths = [
        "/content/drive/MyDrive/LaunDetection/data/raw",
        "data/raw",
        "data"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"✓ Found data directory: {path}")
            
            # List files
            try:
                files = os.listdir(path)
                print(f"✓ Found {len(files)} files in {path}")
                if len(files) > 0:
                    print(f"  Sample files: {files[:3]}")
            except Exception as e:
                print(f"⚠️  Could not list files in {path}: {e}")
            break
    else:
        print("⚠️  No data directory found")
        print("Please ensure your data is in /content/drive/MyDrive/LaunDetection/data/raw")


def test_project_structure():
    """
    Test project structure
    """
    print("\nTesting project structure...")
    
    required_dirs = [
        "utils",
        "models", 
        "notebooks",
        "config",
        "colab"
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ Found {directory}/")
        else:
            print(f"✗ Missing {directory}/")
    
    # Check for key files
    key_files = [
        "requirements.txt",
        "config/config.yaml",
        "utils/gpu_utils.py",
        "utils/data_loader_colab.py",
        "notebooks/01_data_exploration.ipynb"
    ]
    
    for file in key_files:
        if os.path.exists(file):
            print(f"✓ Found {file}")
        else:
            print(f"✗ Missing {file}")


def main():
    """
    Main test function
    """
    print("=" * 60)
    print("AML Multi-GNN - Setup Test")
    print("=" * 60)
    
    # Test basic imports
    test_basic_imports()
    
    # Test PyTorch
    test_pytorch()
    
    # Test PyTorch Geometric
    test_pytorch_geometric()
    
    # Test data access
    test_data_access()
    
    # Test project structure
    test_project_structure()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    
    print("\nIf all tests passed, you can proceed with:")
    print("1. Data exploration: %run notebooks/01_data_exploration.ipynb")
    print("2. Or test data loading: python utils/data_loader_colab.py")


if __name__ == "__main__":
    main()

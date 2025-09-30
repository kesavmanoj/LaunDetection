"""
Simplified Colab Setup for AML Multi-GNN Project
Avoids complex PyTorch Geometric installation issues
"""

import os
import sys
import subprocess
from pathlib import Path


def install_basic_requirements():
    """
    Install basic requirements without complex PyTorch Geometric packages
    """
    print("Installing basic requirements...")
    
    # Basic packages that should work without issues
    basic_packages = [
        "pandas",
        "numpy", 
        "matplotlib",
        "seaborn",
        "plotly",
        "scikit-learn",
        "networkx",
        "tqdm",
        "pyyaml",
        "kaggle"
    ]
    
    for package in basic_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")


def install_pytorch_geometric_simple():
    """
    Install PyTorch Geometric with simple method (skip problematic packages)
    """
    print("Installing PyTorch Geometric...")
    
    try:
        # Try simple installation first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])
        print("✓ Installed torch-geometric")
        
        # Skip optional packages that cause build issues
        print("⚠️  Skipping torch-scatter, torch-sparse, torch-cluster (can cause build issues)")
        print("⚠️  These are optional and can be installed later if needed")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install torch-geometric: {e}")
        print("⚠️  PyTorch Geometric installation failed. Some features may not work.")
        return False


def setup_google_drive():
    """
    Setup Google Drive mounting
    """
    print("Setting up Google Drive...")
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully")
        return True
    except ImportError:
        print("✗ Google Colab not detected. This script should be run in Colab.")
        return False
    except Exception as e:
        print(f"⚠️  Google Drive mounting failed: {e}")
        print("This is normal if Drive is already mounted. Continuing...")
        return True


def setup_project_structure():
    """
    Setup project directory structure (use existing Google Drive directories)
    """
    print("Setting up project structure...")
    
    # Check if data exists in Google Drive
    drive_data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    if os.path.exists(drive_data_path):
        print(f"✓ Found existing data directory: {drive_data_path}")
        
        # Create symlink to existing data
        if not os.path.exists("data"):
            os.symlink("/content/drive/MyDrive/LaunDetection/data", "data")
            print("✓ Created symlink to existing data directory")
        
        # Create local results directories only
        local_dirs = [
            "results/experiments",
            "results/models", 
            "results/visualizations"
        ]
        
        for directory in local_dirs:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created {directory}")
            
    else:
        print(f"⚠️  Data directory not found: {drive_data_path}")
        print("Please ensure your data is in /content/drive/MyDrive/LaunDetection/data/raw")
        
        # Create minimal local structure
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/splits", exist_ok=True)
        print("✓ Created minimal local data structure")


def setup_python_path():
    """
    Setup Python path for the project
    """
    print("Setting up Python path...")
    
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"✓ Added {project_root} to Python path")


def test_basic_imports():
    """
    Test if key packages can be imported (basic only)
    """
    print("Testing basic imports...")
    
    test_packages = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns"),
        ("sklearn", "sklearn")
    ]
    
    for package, alias in test_packages:
        try:
            exec(f"import {package} as {alias}")
            print(f"✓ {package} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {package}: {e}")
    
    # Test PyTorch (usually pre-installed in Colab)
    try:
        import torch
        print(f"✓ PyTorch imported successfully (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  GPU not available")
    except ImportError as e:
        print(f"⚠️  PyTorch not available: {e}")
    
    # Skip PyTorch Geometric test to avoid issues
    print("⚠️  PyTorch Geometric test skipped (can be installed later if needed)")


def test_imports():
    """
    Test if key packages can be imported
    """
    print("Testing imports...")
    
    test_packages = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns"),
        ("sklearn", "sklearn")
    ]
    
    for package, alias in test_packages:
        try:
            exec(f"import {package} as {alias}")
            print(f"✓ {package} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {package}: {e}")
    
    # Test PyTorch
    try:
        import torch
        print(f"✓ PyTorch imported successfully (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  GPU not available")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
    
    # Skip PyTorch Geometric test to avoid issues
    print("⚠️  PyTorch Geometric test skipped (can be installed later if needed)")


def main():
    """
    Main setup function - streamlined for existing Google Drive setup
    """
    print("=" * 60)
    print("AML Multi-GNN - Quick Colab Setup")
    print("=" * 60)
    
    # Install basic requirements (skip if already installed)
    print("Checking basic requirements...")
    install_basic_requirements()
    
    # Skip PyTorch Geometric installation to avoid build issues
    print("Skipping PyTorch Geometric installation (can cause build issues)")
    print("⚠️  PyTorch Geometric can be installed later if needed")
    
    # Setup Google Drive (skip if already mounted)
    print("Checking Google Drive...")
    setup_google_drive()
    
    # Setup project structure (use existing directories)
    print("Setting up project structure...")
    setup_project_structure()
    
    # Setup Python path
    setup_python_path()
    
    # Quick test of basic imports only
    print("Testing basic imports...")
    test_basic_imports()
    
    print("\n" + "=" * 60)
    print("Quick setup completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run data exploration: %run notebooks/01_data_exploration.ipynb")
    print("2. Or test data loading: python utils/data_loader_colab.py")
    print("3. Start with Phase 2 implementation")
    print("\nNote: PyTorch Geometric can be installed later if needed:")
    print("!pip install torch-geometric")


if __name__ == "__main__":
    main()

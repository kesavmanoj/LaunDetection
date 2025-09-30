"""
Google Colab Specific Utilities for AML Multi-GNN Project
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess


def is_colab_environment() -> bool:
    """
    Check if running in Google Colab environment
    
    Returns:
        True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def setup_colab_environment():
    """
    Setup Google Colab environment for the project
    """
    if not is_colab_environment():
        print("⚠️  Not running in Google Colab. Some features may not work.")
        return False
    
    print("Setting up Google Colab environment...")
    
    # Mount Google Drive
    mount_google_drive()
    
    # Setup project paths
    setup_project_paths()
    
    # Install additional Colab packages
    install_colab_packages()
    
    return True


def mount_google_drive():
    """
    Mount Google Drive for data persistence
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to mount Google Drive: {e}")
        return False


def setup_project_paths():
    """
    Setup project paths in Colab environment
    """
    # Add project root to Python path
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"✓ Added {project_root} to Python path")
    
    # Use existing data directory in Drive
    drive_data_path = "/content/drive/MyDrive/LaunDetection/data"
    
    # Check if data directory exists
    if os.path.exists(drive_data_path):
        print(f"✓ Found existing data directory: {drive_data_path}")
        
        # Create symlink for easy access
        local_data_path = "data"
        if not os.path.exists(local_data_path):
            os.symlink(drive_data_path, local_data_path)
            print(f"✓ Created symlink: {local_data_path} -> {drive_data_path}")
        
        # Create subdirectories if they don't exist
        subdirs = ["processed", "splits"]
        for subdir in subdirs:
            subdir_path = os.path.join(drive_data_path, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            print(f"✓ Created subdirectory: {subdir_path}")
    else:
        print(f"⚠️  Data directory not found: {drive_data_path}")
        print("Please ensure your data is in /content/drive/MyDrive/LaunDetection/data/raw")


def install_colab_packages():
    """
    Install additional packages specific to Colab
    """
    colab_packages = [
        "google-colab",
        "ipywidgets",
        "tqdm",
        "plotly"
    ]
    
    for package in colab_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True)
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Could not install {package}")


def setup_kaggle_in_colab():
    """
    Setup Kaggle API in Colab environment
    """
    print("Setting up Kaggle API...")
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    print("Please follow these steps to setup Kaggle API:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token' to download kaggle.json")
    print("3. Upload the kaggle.json file to this Colab session")
    print("4. Run: !mv kaggle.json ~/.kaggle/")
    print("5. Run: !chmod 600 ~/.kaggle/kaggle.json")


def check_existing_data():
    """
    Check if data already exists in Google Drive
    """
    drive_data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    if os.path.exists(drive_data_path):
        print(f"✓ Found existing data directory: {drive_data_path}")
        
        # List files in the directory
        try:
            files = os.listdir(drive_data_path)
            print(f"✓ Found {len(files)} files in data directory:")
            for file in files[:10]:  # Show first 10 files
                print(f"  - {file}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")
            return True
        except Exception as e:
            print(f"⚠️  Could not list files: {e}")
            return False
    else:
        print(f"✗ Data directory not found: {drive_data_path}")
        return False


def download_dataset_in_colab(dataset_name: str = "ibm-aml-synthetic-dataset"):
    """
    Download dataset in Colab environment (only if not already present)
    
    Args:
        dataset_name: Kaggle dataset name
    """
    # Check if data already exists
    if check_existing_data():
        print("✓ Data already exists in Google Drive. Skipping download.")
        return True
    
    print(f"Downloading dataset: {dataset_name}")
    
    try:
        # Install kaggle if not already installed
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], 
                      check=True, capture_output=True)
        
        # Create data directory
        os.makedirs("data/raw", exist_ok=True)
        
        # Download dataset
        subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset_name,
            "-p", "data/raw", "--unzip"
        ], check=True)
        
        print("✓ Dataset downloaded successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download dataset: {e}")
        return False


def setup_gpu_in_colab():
    """
    Setup GPU environment in Colab
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ PyTorch version: {torch.__version__}")
            
            # Print GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU memory: {total_memory:.1f} GB")
            
            return True
        else:
            print("⚠️  GPU not available. Please enable GPU in Runtime > Change runtime type")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def create_colab_notebook_template(notebook_name: str, phase: int):
    """
    Create a Colab notebook template for a specific phase
    
    Args:
        notebook_name: Name of the notebook
        phase: Phase number (1-8)
    """
    template = f'''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# AML Multi-GNN - Phase {phase}: {notebook_name}\\n",
    "\\n",
    "This notebook implements Phase {phase} of the AML Multi-GNN project."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Setup environment\\n",
    "import sys\\n",
    "import os\\n",
    "from pathlib import Path\\n",
    "\\n",
    "# Add project root to path\\n",
    "project_root = Path.cwd()\\n",
    "if str(project_root) not in sys.path:\\n",
    "    sys.path.insert(0, str(project_root))\\n",
    "\\n",
    "# Import project utilities\\n",
    "from utils.gpu_utils import setup_gpu_in_colab, print_system_info\\n",
    "from utils.logging_utils import setup_logging\\n",
    "from utils.random_utils import set_random_seed\\n",
    "\\n",
    "# Setup logging\\n",
    "logger = setup_logging(experiment_name=f"phase_{phase}_{notebook_name}")\\n",
    "\\n",
    "# Print system info\\n",
    "print_system_info()"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Phase {phase} implementation\\n",
    "# Add your code here"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "codemirror_mode": {{
    "name": "ipython",
    "version": 3
   }},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}'''
    
    # Save notebook
    notebook_path = f"notebooks/{notebook_name}.ipynb"
    os.makedirs("notebooks", exist_ok=True)
    
    with open(notebook_path, 'w') as f:
        f.write(template)
    
    print(f"✓ Created notebook template: {notebook_path}")


def setup_complete_colab_environment():
    """
    Complete Colab environment setup
    """
    print("=" * 60)
    print("AML Multi-GNN - Complete Colab Setup")
    print("=" * 60)
    
    # Check if in Colab
    if not is_colab_environment():
        print("⚠️  Not running in Google Colab. Some features may not work.")
    
    # Setup environment
    setup_colab_environment()
    
    # Setup GPU
    setup_gpu_in_colab()
    
    # Setup Kaggle
    setup_kaggle_in_colab()
    
    print("\n" + "=" * 60)
    print("Colab environment setup completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Setup Kaggle API (follow instructions above)")
    print("2. Download dataset: download_dataset_in_colab()")
    print("3. Start with Phase 1 implementation")


if __name__ == "__main__":
    setup_complete_colab_environment()

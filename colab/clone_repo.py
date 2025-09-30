"""
Repository Cloning Script for Google Colab
"""

import os
import subprocess
import sys
from pathlib import Path


def clone_repository(repo_url: str = "https://github.com/kesavmanoj/LaunDetection.git"):
    """
    Clone the AML Multi-GNN repository
    
    Args:
        repo_url: GitHub repository URL
    """
    print(f"Cloning repository: {repo_url}")
    
    try:
        # Remove existing directory if it exists
        if os.path.exists("LaunDetection"):
            print("Removing existing LaunDetection directory...")
            subprocess.run(["rm", "-rf", "LaunDetection"], check=True)
        
        # Clone the repository
        subprocess.run(["git", "clone", repo_url], check=True)
        print("✓ Repository cloned successfully")
        
        # Change to project directory
        os.chdir("LaunDetection")
        print("✓ Changed to project directory")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to clone repository: {e}")
        return False


def setup_project_environment():
    """
    Setup project environment after cloning
    """
    print("Setting up project environment...")
    
    # Add project root to Python path
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"✓ Added {project_root} to Python path")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed",
        "data/splits", 
        "results/experiments",
        "results/models",
        "results/visualizations"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Install project requirements
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        print("Installing project requirements...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], 
                          check=True)
            print("✓ Requirements installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install requirements: {e}")
    else:
        print("⚠️  requirements.txt not found")


def download_kaggle_dataset(dataset_name: str = "ibm-aml-synthetic-dataset"):
    """
    Download IBM AML dataset from Kaggle
    
    Args:
        dataset_name: Kaggle dataset name
    """
    print(f"Downloading dataset: {dataset_name}")
    
    try:
        # Check if kaggle is installed
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
        
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
        print("Make sure you have:")
        print("1. Installed kaggle package: pip install kaggle")
        print("2. Uploaded kaggle.json to ~/.kaggle/")
        print("3. Set proper permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False


def setup_colab_notebook():
    """
    Setup Colab notebook environment
    """
    print("Setting up Colab notebook environment...")
    
    # Enable GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  GPU not available. Enable GPU in Runtime > Change runtime type")
    except ImportError:
        print("⚠️  PyTorch not installed")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted")
    except ImportError:
        print("⚠️  Not running in Google Colab")
    except Exception as e:
        print(f"⚠️  Could not mount Google Drive: {e}")


def main():
    """
    Main setup function
    """
    print("=" * 60)
    print("AML Multi-GNN - Repository Setup")
    print("=" * 60)
    
    # Clone repository
    if clone_repository():
        # Setup environment
        setup_project_environment()
        
        # Setup Colab notebook
        setup_colab_notebook()
        
        print("\n" + "=" * 60)
        print("Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Download dataset: python colab/clone_repo.py --download-dataset")
        print("2. Run data exploration: %run notebooks/01_data_exploration.ipynb")
        print("3. Start with Phase 1 implementation")
        
    else:
        print("Setup failed. Please check the repository URL and try again.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup AML Multi-GNN project in Google Colab")
    parser.add_argument("--download-dataset", action="store_true", 
                       help="Download IBM AML dataset from Kaggle")
    parser.add_argument("--repo-url", default="https://github.com/kesavmanoj/LaunDetection.git",
                       help="Repository URL to clone")
    
    args = parser.parse_args()
    
    if args.download_dataset:
        download_kaggle_dataset()
    else:
        main()

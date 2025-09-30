"""
Google Colab Environment Setup for AML Multi-GNN Project
"""

import os
import subprocess
import sys
from pathlib import Path


def install_requirements():
    """
    Install required packages for the project
    """
    print("Installing required packages...")
    
    # Install PyTorch Geometric and related packages with proper syntax
    torch_geometric_packages = [
        ("torch-scatter", "https://data.pyg.org/whl/torch-2.0.0+cu118.html"),
        ("torch-sparse", "https://data.pyg.org/whl/torch-2.0.0+cu118.html"),
        ("torch-cluster", "https://data.pyg.org/whl/torch-2.0.0+cu118.html"),
        ("torch-spline-conv", "https://data.pyg.org/whl/torch-2.0.0+cu118.html")
    ]
    
    for package_name, url in torch_geometric_packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                f"{package_name} -f {url}"
            ])
            print(f"✓ Installed {package_name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package_name}: {e}")
            # Try alternative installation
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"✓ Installed {package_name} (alternative method)")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package_name} with alternative method")
    
    # Install other requirements
    requirements = [
        "torch-geometric",
        "networkx",
        "dgl",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "plotly",
        "seaborn",
        "tqdm",
        "pyyaml",
        "kaggle"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")


def setup_kaggle_api():
    """
    Setup Kaggle API for dataset download
    """
    print("Setting up Kaggle API...")
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    print("Please upload your kaggle.json file to the .kaggle directory")
    print("You can download it from: https://www.kaggle.com/account")
    print("After uploading, run: chmod 600 ~/.kaggle/kaggle.json")


def mount_google_drive():
    """
    Mount Google Drive for data persistence
    """
    print("Mounting Google Drive...")
    
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


def create_project_directories():
    """
    Create project directory structure
    """
    print("Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/splits",
        "results/experiments",
        "results/models",
        "results/visualizations",
        "notebooks",
        "config"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created {directory}")


def setup_environment():
    """
    Complete environment setup for Google Colab
    """
    print("=" * 60)
    print("AML Multi-GNN - Google Colab Environment Setup")
    print("=" * 60)
    
    # Check if running in Colab
    try:
        import google.colab
        print("✓ Running in Google Colab")
    except ImportError:
        print("⚠️  Not running in Google Colab. Some features may not work.")
    
    # Install requirements
    install_requirements()
    
    # Mount Google Drive
    mount_google_drive()
    
    # Create directories
    create_project_directories()
    
    # Setup Kaggle API
    setup_kaggle_api()
    
    print("\n" + "=" * 60)
    print("Environment setup completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Upload your kaggle.json file to ~/.kaggle/")
    print("2. Clone the repository: !git clone https://github.com/kesavmanoj/LaunDetection.git")
    print("3. Change to project directory: %cd LaunDetection")
    print("4. Run the data exploration notebook")


def verify_installation():
    """
    Verify that all packages are installed correctly
    """
    print("Verifying installation...")
    
    packages_to_check = [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("networkx", "NetworkX"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("plotly", "Plotly")
    ]
    
    for package, name in packages_to_check:
        try:
            __import__(package)
            print(f"✓ {name} is available")
        except ImportError:
            print(f"✗ {name} is not available")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  GPU not available")
    except:
        print("✗ Could not check GPU status")


if __name__ == "__main__":
    setup_environment()
    verify_installation()

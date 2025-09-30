"""
Streamlined Colab Setup for Existing Data in Google Drive
"""

import os
import sys
from pathlib import Path
import subprocess


def setup_colab_with_existing_data():
    """
    Complete setup for Colab with existing data in Google Drive
    """
    print("=" * 60)
    print("AML Multi-GNN - Colab Setup with Existing Data")
    print("=" * 60)
    
    # Check if running in Colab
    try:
        import google.colab
        print("✓ Running in Google Colab")
    except ImportError:
        print("⚠️  Not running in Google Colab. Some features may not work.")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully")
    except Exception as e:
        print(f"✗ Failed to mount Google Drive: {e}")
        return False
    
    # Check existing data
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    if os.path.exists(data_path):
        print(f"✓ Found existing data directory: {data_path}")
        
        # List files
        files = os.listdir(data_path)
        print(f"✓ Found {len(files)} files in data directory")
        
        # Create symlink
        if not os.path.exists("data"):
            os.symlink("/content/drive/MyDrive/LaunDetection/data", "data")
            print("✓ Created symlink to data directory")
        
        # Create subdirectories
        subdirs = ["processed", "splits"]
        for subdir in subdirs:
            subdir_path = os.path.join("/content/drive/MyDrive/LaunDetection/data", subdir)
            os.makedirs(subdir_path, exist_ok=True)
            print(f"✓ Created subdirectory: {subdir_path}")
    
    else:
        print(f"✗ Data directory not found: {data_path}")
        print("Please ensure your data is in /content/drive/MyDrive/LaunDetection/data/raw")
        return False
    
    # Install requirements
    print("\nInstalling requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Some packages may not have installed correctly: {e}")
    
    # Setup GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  GPU not available. Enable GPU in Runtime > Change runtime type")
    except ImportError:
        print("⚠️  PyTorch not installed")
    
    # Add project to Python path
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print("✓ Added project root to Python path")
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run data exploration: %run notebooks/01_data_exploration.ipynb")
    print("2. Or run: python utils/data_loader_colab.py")
    print("3. Start with Phase 2 implementation")
    
    return True


def test_data_loading():
    """
    Test data loading with existing data
    """
    print("\nTesting data loading...")
    
    try:
        from utils.data_loader_colab import main as load_data
        unified_df, quality_metrics = load_data()
        
        if unified_df is not None:
            print("✓ Data loading successful")
            print(f"✓ Loaded {len(unified_df)} rows and {len(unified_df.columns)} columns")
            return True
        else:
            print("✗ Data loading failed")
            return False
            
    except Exception as e:
        print(f"✗ Error testing data loading: {e}")
        return False


if __name__ == "__main__":
    # Run setup
    if setup_colab_with_existing_data():
        # Test data loading
        test_data_loading()

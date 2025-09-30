#!/usr/bin/env python3
"""
Colab Setup Script for Clean Training
=====================================

Run this in Google Colab to execute the clean training script.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    packages = [
        "torch",
        "torch-geometric", 
        "networkx",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "tqdm"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except Exception as e:
            print(f"✗ Failed to install {package}: {e}")

def setup_environment():
    """Setup the environment"""
    print("Setting up environment...")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted")
    except Exception as e:
        print(f"✗ Failed to mount Google Drive: {e}")
    
    # Create necessary directories
    os.makedirs("/content/drive/MyDrive/LaunDetection/data/raw", exist_ok=True)
    os.makedirs("/content/drive/MyDrive/LaunDetection/data/processed", exist_ok=True)
    print("✓ Directories created")

def run_training():
    """Run the clean training script"""
    print("Running clean training script...")
    
    try:
        # Import and run the training script
        import sys
        sys.path.append('/content/drive/MyDrive/LaunDetection')
        
        # Run the notebook
        exec(open('/content/drive/MyDrive/LaunDetection/notebooks/06_clean_training.ipynb').read())
        
        print("✓ Training completed successfully!")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        print("Please check the error and try again.")

def main():
    """Main function"""
    print("="*60)
    print("AML Multi-GNN - Clean Training Setup")
    print("="*60)
    
    # Install requirements
    install_requirements()
    
    # Setup environment
    setup_environment()
    
    # Run training
    run_training()
    
    print("\n" + "="*60)
    print("Setup completed!")
    print("="*60)

if __name__ == "__main__":
    main()

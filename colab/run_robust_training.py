#!/usr/bin/env python3
"""
Colab Setup for Robust Enhanced Training
========================================

This script sets up and runs the robust enhanced training in Google Colab.
"""

import os
import sys
import subprocess

def setup_colab_environment():
    """Setup the Colab environment for robust training"""
    print("🚀 Setting up Colab environment for robust training...")
    
    # Install required packages
    packages = [
        "torch",
        "torch-geometric", 
        "networkx",
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "seaborn"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package}")
    
    print("✅ Environment setup complete")

def mount_google_drive():
    """Mount Google Drive"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return False

def run_robust_training():
    """Run the robust training script"""
    print("🎯 Starting robust enhanced training...")
    
    # Change to the project directory
    os.chdir('/content/drive/MyDrive/LaunDetection')
    
    # Run the robust training script
    try:
        exec(open('robust_enhanced_training.py').read())
        print("✅ Robust training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Please check the error and try again")

def main():
    """Main setup and execution function"""
    print("🚀 Robust Enhanced AML Multi-GNN Training - Colab Setup")
    print("=" * 60)
    
    # Setup environment
    setup_colab_environment()
    
    # Mount Google Drive
    if not mount_google_drive():
        print("❌ Cannot proceed without Google Drive access")
        return
    
    # Run robust training
    run_robust_training()

if __name__ == "__main__":
    main()

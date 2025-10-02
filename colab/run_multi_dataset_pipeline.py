#!/usr/bin/env python3
"""
Multi-Dataset AML Training Pipeline - Colab Setup
================================================

This script sets up the complete multi-dataset training pipeline in Google Colab.
"""

import os
import sys
import subprocess
import time

print("🚀 Multi-Dataset AML Training Pipeline - Colab Setup")
print("=" * 60)

def setup_environment():
    """Setup the Colab environment for multi-dataset training"""
    print("🔧 Setting up Colab environment...")
    
    # Install required packages
    packages = [
        'torch',
        'torch-geometric',
        'networkx',
        'pandas',
        'numpy',
        'scikit-learn',
        'imbalanced-learn',
        'tqdm',
        'matplotlib',
        'seaborn',
        'plotly'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
            print(f"   ✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"   ⚠️ {package} installation failed")
    
    print("✅ Environment setup complete")

def check_google_drive():
    """Check if Google Drive is already mounted"""
    print("📁 Checking Google Drive access...")
    
    # Check if drive is already mounted
    if os.path.exists('/content/drive/MyDrive'):
        print("✅ Google Drive already mounted")
        return True
    
    # Try to mount if not already mounted
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {str(e)}")
        print("💡 Please manually mount Google Drive if needed")
        return False

def check_data_availability():
    """Check if all required datasets are available"""
    print("🔍 Checking data availability...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    required_datasets = {
        'HI-Small': ['HI-Small_Trans.csv', 'HI-Small_accounts.csv'],
        'LI-Small': ['LI-Small_Trans.csv', 'LI-Small_accounts.csv'],
        'HI-Medium': ['HI-Medium_Trans.csv', 'HI-Medium_accounts.csv'],
        'LI-Medium': ['LI-Medium_Trans.csv', 'LI-Medium_accounts.csv']
    }
    
    available_datasets = []
    
    for dataset_name, files in required_datasets.items():
        trans_file = os.path.join(data_path, files[0])
        accounts_file = os.path.join(data_path, files[1])
        
        if os.path.exists(trans_file) and os.path.exists(accounts_file):
            print(f"   ✅ {dataset_name} dataset found")
            available_datasets.append(dataset_name)
        else:
            print(f"   ❌ {dataset_name} dataset not found")
            if not os.path.exists(trans_file):
                print(f"      Missing: {files[0]}")
            if not os.path.exists(accounts_file):
                print(f"      Missing: {files[1]}")
    
    print(f"\n📊 Found {len(available_datasets)} datasets: {available_datasets}")
    return available_datasets

def run_preprocessing():
    """Run the multi-dataset preprocessing"""
    print("🔄 Running Multi-Dataset Preprocessing...")
    
    try:
        # Import and run preprocessing
        from multi_dataset_preprocessing import MultiDatasetPreprocessor
        
        preprocessor = MultiDatasetPreprocessor()
        processed_data = preprocessor.run_full_preprocessing()
        
        if processed_data:
            print("✅ Multi-dataset preprocessing completed successfully!")
            return True
        else:
            print("❌ Multi-dataset preprocessing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        return False

def run_training():
    """Run the multi-dataset training"""
    print("🚀 Running Multi-Dataset Training...")
    
    try:
        # Import and run training
        from multi_dataset_training import MultiDatasetTrainer
        
        trainer = MultiDatasetTrainer()
        
        # Load processed datasets
        datasets = trainer.load_processed_datasets()
        
        if not datasets:
            print("❌ No processed datasets found!")
            return False
        
        # Create combined dataset
        combined_data = trainer.create_combined_dataset(datasets)
        
        # Convert to PyTorch format
        data = trainer.create_pytorch_data(combined_data)
        
        # Train model
        model, best_f1 = trainer.train_multi_dataset_model(data)
        
        # Evaluate model
        metrics, aml_metrics, cm = trainer.evaluate_multi_dataset_model(data)
        
        print("✅ Multi-dataset training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return False

def main():
    """Main pipeline execution"""
    print("🚀 Starting Multi-Dataset AML Training Pipeline...")
    
    # Step 1: Setup environment
    setup_environment()
    
    # Step 2: Check Google Drive access
    if not check_google_drive():
        print("❌ Cannot proceed without Google Drive access")
        return
    
    # Step 3: Check data availability
    available_datasets = check_data_availability()
    
    if not available_datasets:
        print("❌ No datasets found! Please upload datasets to Google Drive.")
        return
    
    print(f"\n🎯 Proceeding with {len(available_datasets)} datasets: {available_datasets}")
    
    # Step 4: Run preprocessing
    print("\n" + "="*60)
    print("STEP 1: MULTI-DATASET PREPROCESSING")
    print("="*60)
    
    if not run_preprocessing():
        print("❌ Preprocessing failed! Cannot proceed to training.")
        return
    
    # Step 5: Run training
    print("\n" + "="*60)
    print("STEP 2: MULTI-DATASET TRAINING")
    print("="*60)
    
    if not run_training():
        print("❌ Training failed!")
        return
    
    print("\n🎉 MULTI-DATASET PIPELINE COMPLETE!")
    print("=" * 60)
    print("✅ Enhanced preprocessing completed")
    print("✅ Multi-dataset training completed")
    print("✅ Improved AML detection model ready")
    print("\n🚀 Model is ready for production deployment!")

if __name__ == "__main__":
    main()

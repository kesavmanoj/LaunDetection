#!/usr/bin/env python3
"""
Run Fixed Multi-Dataset Preprocessing
====================================

This script runs the fixed preprocessing with LI-Medium limited to 10M transactions
to prevent RAM crashes in Google Colab.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append('/content/drive/MyDrive/LaunDetection')

from multi_dataset_preprocessing import MultiDatasetPreprocessor

def main():
    """Run the fixed preprocessing pipeline"""
    print("🚀 Running Fixed Multi-Dataset Preprocessing")
    print("=" * 50)
    print("📊 LI-Medium limited to 10M transactions to prevent RAM crashes")
    print("📊 Other datasets will use full data")
    print()
    
    # Initialize preprocessor
    preprocessor = MultiDatasetPreprocessor()
    
    # Run preprocessing
    try:
        preprocessor.run_full_preprocessing()
        print("\n✅ Fixed preprocessing completed successfully!")
        print("📊 LI-Medium limited to 10M transactions")
        print("📊 Other datasets processed in full")
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        print("💡 Try reducing the LI-Medium limit further if still crashing")

if __name__ == "__main__":
    main()

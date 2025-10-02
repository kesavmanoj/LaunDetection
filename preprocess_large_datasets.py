#!/usr/bin/env python3
"""
Preprocess Large Datasets with 5M Transaction Limits
==================================================

This script preprocesses HI-Large and LI-Large datasets with 5M transaction limits
to prevent RAM crashes in Google Colab (11GB RAM).
"""

import sys
import os

# Add the current directory to Python path
sys.path.append('/content/drive/MyDrive/LaunDetection')

from multi_dataset_preprocessing import MultiDatasetPreprocessor

def main():
    """Run preprocessing for large datasets with 5M limits"""
    print("🚀 Preprocessing Large Datasets with 5M Transaction Limits")
    print("=" * 60)
    print("📊 HI-Large and LI-Large limited to 5M transactions each")
    print("📊 Optimized for 11GB Colab RAM to prevent crashes")
    print()
    
    # Initialize preprocessor
    preprocessor = MultiDatasetPreprocessor()
    
    # Run preprocessing
    try:
        preprocessor.run_full_preprocessing()
        print("\n✅ Large dataset preprocessing completed successfully!")
        print("📊 HI-Large and LI-Large processed with 5M transaction limits")
        print("📊 All 6 datasets now available for training")
        print("🚀 Ready for advanced AML detection training!")
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        print("💡 Try reducing limits further if still crashing")

if __name__ == "__main__":
    main()

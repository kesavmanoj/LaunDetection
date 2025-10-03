#!/usr/bin/env python3
"""
Working Preprocessing - Small and Medium Datasets Only
======================================================

This script uses the working preprocessing approach for small and medium datasets.
- HI-Small: Full dataset (5M transactions)
- LI-Small: Full dataset (7M transactions)  
- HI-Medium: Full dataset (32M transactions)
- LI-Medium: Limited to 15M transactions only
"""

import sys
import os

# Add the current directory to Python path
sys.path.append('/content/drive/MyDrive/LaunDetection')

from multi_dataset_preprocessing import MultiDatasetPreprocessor

def main():
    """Run working preprocessing for small and medium datasets"""
    print("🚀 Working Preprocessing - Small and Medium Datasets")
    print("=" * 60)
    print("📊 HI-Small: Full dataset (5M transactions)")
    print("📊 LI-Small: Full dataset (7M transactions)")
    print("📊 HI-Medium: Limited to 15M transactions")
    print("📊 LI-Medium: Limited to 15M transactions")
    print("📊 Large datasets: COMPLETELY SKIPPED")
    print("📊 AML Rate: 5% (reduced from 15% for better data retention)")
    print()
    
    # Initialize preprocessor
    preprocessor = MultiDatasetPreprocessor()
    
    # Run preprocessing
    try:
        preprocessor.run_full_preprocessing()
        print("\n✅ Working preprocessing completed successfully!")
        print("📊 Small and medium datasets processed")
        print("📊 Medium datasets limited to 15M transactions each")
        print("📊 Large datasets skipped")
        print("🚀 Ready for training!")
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        print("💡 This is the working approach that should complete successfully")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run Enhanced Preprocessing for AML Multi-GNN
============================================

This script runs the enhanced preprocessing pipeline for the full dataset.
"""

import sys
import os

# Add the notebooks directory to the path
sys.path.append('notebooks')

# Import and run the enhanced preprocessing
try:
    print("🚀 Starting Enhanced Preprocessing...")
    print("=" * 60)
    
    # Run the enhanced preprocessing notebook as a script
    exec(open('notebooks/08_simple_enhanced_preprocessing.ipynb').read())
    
    print("=" * 60)
    print("✅ Enhanced preprocessing completed successfully!")
    
except Exception as e:
    print(f"❌ Error running enhanced preprocessing: {e}")
    print("Please run the notebook directly:")
    print("%run notebooks/08_simple_enhanced_preprocessing.ipynb")

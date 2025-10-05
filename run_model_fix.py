#!/usr/bin/env python3
"""
Run Model Saving Fix for Google Colab
=====================================

This script fixes model saving issues in Google Colab environment.
"""

import os
import sys

def main():
    """Run the model saving fix"""
    print("🚀 Running Model Saving Fix for Google Colab")
    print("=" * 50)
    
    # Run the fix script
    try:
        exec(open('fix_model_saving.py').read())
        print("\n✅ Model saving fix completed successfully!")
        
        print("\n🎯 Next Steps:")
        print("1. Re-run your training script: !python advanced_aml_detection.py")
        print("2. Models will now be saved to: /content/drive/MyDrive/LaunDetection/models/")
        print("3. Run evaluation: !python balanced_evaluation.py")
        
    except Exception as e:
        print(f"❌ Error running fix: {e}")
        print("💡 Try running the fix manually:")

if __name__ == "__main__":
    main()

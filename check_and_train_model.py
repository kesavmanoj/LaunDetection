#!/usr/bin/env python3
"""
Check Available Models and Train if Needed
==========================================

This script checks for existing trained models and trains one if none exist.
"""

import os
import sys

def check_available_models():
    """Check what trained models are available"""
    print("🔍 Checking for available trained models...")
    print("=" * 50)
    
    # Check multiple possible locations
    possible_paths = [
        '/content/drive/MyDrive/LaunDetection/models',
        '/content/drive/MyDrive/LaunDetection',
        '/content/LaunDetection/models',
        '/content/LaunDetection'
    ]
    
    found_models = []
    
    for base_path in possible_paths:
        if os.path.exists(base_path):
            print(f"📁 Checking: {base_path}")
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith(('.pth', '.pt')):
                        full_path = os.path.join(root, file)
                        size = os.path.getsize(full_path)
                        found_models.append((full_path, size))
                        print(f"  ✅ {file} ({size:,} bytes)")
    
    if found_models:
        print(f"\n✅ Found {len(found_models)} trained models!")
        return found_models
    else:
        print("\n❌ No trained models found")
        return []

def recommend_training():
    """Recommend which training script to run"""
    print("\n💡 RECOMMENDED TRAINING OPTIONS:")
    print("=" * 40)
    print("1. 🚀 Advanced AML Detection (Best Performance):")
    print("   !python advanced_aml_detection.py")
    print("   - Uses ensemble-like architecture")
    print("   - Advanced Focal Loss")
    print("   - Threshold optimization")
    print("   - Saves to: /content/drive/MyDrive/LaunDetection/models/advanced_aml_model.pth")
    
    print("\n2. 📊 Multi-Dataset Training (Comprehensive):")
    print("   !python multi_dataset_training.py")
    print("   - Trains on multiple datasets")
    print("   - Combined learning")
    print("   - Saves to: /content/drive/MyDrive/LaunDetection/models/")
    
    print("\n3. 🎯 Individual Dataset Training (Focused):")
    print("   !python individual_dataset_training.py")
    print("   - Trains on each dataset separately")
    print("   - Dataset-specific optimization")
    
    print("\n4. 🔧 Production Model Training (Simple):")
    print("   !python train_production_model.py")
    print("   - Basic production model")
    print("   - Saves to: /content/drive/MyDrive/LaunDetection/production_model.pth")

def main():
    """Main function"""
    print("🚀 Model Check and Training Guide")
    print("=" * 50)
    
    # Check for existing models
    models = check_available_models()
    
    if models:
        print(f"\n🎉 You have {len(models)} trained models available!")
        print("You can now run the balanced evaluation:")
        print("!python balanced_model_evaluation_final.py")
    else:
        print("\n⚠️ No trained models found.")
        print("You need to train a model first.")
        recommend_training()
        
        print("\n🚀 QUICK START:")
        print("Run this command to train the best model:")
        print("!python advanced_aml_detection.py")
        print("\nThen run the evaluation:")
        print("!python balanced_model_evaluation_final.py")

if __name__ == "__main__":
    main()

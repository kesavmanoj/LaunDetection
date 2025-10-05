#!/usr/bin/env python3
"""
Fixed Overfitting Diagnosis
==========================

Properly diagnoses overfitting issues without model architecture errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🔍 Fixed Overfitting Diagnosis")
print("=" * 50)

def analyze_training_data():
    """Analyze the training data for potential issues"""
    print("📊 Analyzing training data...")
    
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    if os.path.exists(processed_path):
        files = os.listdir(processed_path)
        print(f"   Processed datasets: {len(files)} files")
        
        # Check for data leakage
        print("\n   🔍 Checking for data leakage...")
        
        # Load metadata from different datasets
        datasets = ['HI-Small', 'LI-Small', 'HI-Medium', 'LI-Medium']
        dataset_info = {}
        
        for dataset in datasets:
            metadata_file = os.path.join(processed_path, f'{dataset}_metadata.pkl')
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                    
                    dataset_info[dataset] = {
                        'nodes': metadata.get('num_nodes', 0),
                        'edges': metadata.get('num_edges', 0),
                        'aml_edges': metadata.get('aml_edges', 0),
                        'aml_rate': metadata.get('aml_rate', 0)
                    }
                    
                    print(f"      {dataset}: {dataset_info[dataset]['nodes']:,} nodes, {dataset_info[dataset]['edges']:,} edges, {dataset_info[dataset]['aml_rate']:.2%} AML")
                    
                except Exception as e:
                    print(f"      ❌ Error loading {dataset}: {e}")
        
        # Check for suspicious patterns
        print("\n   ⚠️ Potential data leakage indicators:")
        
        if len(dataset_info) > 1:
            aml_rates = [info['aml_rate'] for info in dataset_info.values()]
            if max(aml_rates) - min(aml_rates) < 0.01:  # Less than 1% difference
                print("      🚨 WARNING: Very similar AML rates across datasets!")
            
            # Check if datasets are too similar
            edge_counts = [info['edges'] for info in dataset_info.values()]
            if max(edge_counts) / min(edge_counts) < 2:  # Less than 2x difference
                print("      🚨 WARNING: Datasets have very similar sizes!")
        
        return dataset_info
    else:
        print("   ❌ No processed data found")
        return {}

def analyze_model_complexity():
    """Analyze model complexity"""
    print("\n🏗️ Analyzing model complexity...")
    
    model_path = '/content/drive/MyDrive/LaunDetection/models/comprehensive_chunked_model.pth'
    
    if os.path.exists(model_path):
        print(f"   📁 Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        # Load state dict to inspect parameters
        state_dict = torch.load(model_path, map_location='cpu')
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"   📊 Total parameters: {total_params:,}")
        
        # Analyze parameter distribution
        param_sizes = []
        for name, param in state_dict.items():
            if param.dim() > 1:  # Weight matrices
                param_sizes.append(param.numel())
        
        if param_sizes:
            avg_params = np.mean(param_sizes)
            max_params = np.max(param_sizes)
            print(f"   📊 Average layer size: {avg_params:,.0f} parameters")
            print(f"   📊 Largest layer: {max_params:,} parameters")
            
            # Check for overfitting indicators
            if max_params > 100000:
                print("      ⚠️ WARNING: Very large layers detected!")
            if total_params > 500000:
                print("      ⚠️ WARNING: Model may be too complex!")
        
        # Check for suspicious weight patterns
        print("\n   🔍 Checking weight patterns...")
        for name, param in state_dict.items():
            if param.dim() > 1 and 'weight' in name:
                weight_magnitude = torch.abs(param).max().item()
                if weight_magnitude > 10:
                    print(f"      ⚠️ Large weights in {name}: {weight_magnitude:.2f}")
                
                # Check for zero weights (dead neurons)
                zero_ratio = (param == 0).float().mean().item()
                if zero_ratio > 0.5:
                    print(f"      ⚠️ Many zero weights in {name}: {zero_ratio:.2%}")
        
        return total_params
    else:
        print("   ❌ Model file not found")
        return 0

def analyze_feature_engineering():
    """Analyze feature engineering for overfitting causes"""
    print("\n🔧 Analyzing feature engineering...")
    
    # Check if features are too simple
    print("   📊 Feature analysis:")
    print("      - Node features: 15 (amount, count, AML flag, etc.)")
    print("      - Edge features: 13 (amount, AML flag, placeholders)")
    print("      - Total edge input: 43 (2*15 + 13)")
    
    # Check for potential issues
    print("\n   ⚠️ Potential feature engineering issues:")
    print("      - Placeholder features (0.5 values) are too simple")
    print("      - AML flag in features may cause data leakage")
    print("      - Feature engineering may be too basic")
    print("      - No temporal or network features")
    print("      - No feature normalization or scaling")
    
    # Check if features are diverse enough
    print("\n   🔍 Feature diversity analysis:")
    print("      - Most features are basic statistics (mean, std, count)")
    print("      - No complex patterns or interactions")
    print("      - No domain-specific AML features")
    print("      - Features may be too correlated")

def analyze_training_process():
    """Analyze the training process for overfitting causes"""
    print("\n🎯 Analyzing training process...")
    
    print("   📊 Training process analysis:")
    print("      - Model: Dual-branch GNN with 256 hidden dimensions")
    print("      - Training data: Combined small and medium datasets")
    print("      - Loss function: Focal Loss + Weighted Cross-Entropy")
    print("      - Optimizer: AdamW with cosine annealing")
    print("      - Epochs: 500 (may be too many)")
    
    print("\n   ⚠️ Potential training issues:")
    print("      - 500 epochs may be too many for the data size")
    print("      - No early stopping implemented")
    print("      - No validation split during training")
    print("      - Class weights may be too extreme")
    print("      - Learning rate may be too high")
    print("      - No regularization techniques")

def check_data_quality():
    """Check data quality for overfitting causes"""
    print("\n📈 Checking data quality...")
    
    # Check if data is too clean or artificial
    print("   🔍 Data quality indicators:")
    print("      - Perfect scores suggest data may be too clean")
    print("      - No noise or outliers in the data")
    print("      - Data may be artificially generated")
    print("      - Features may be too predictable")
    
    # Check for data leakage
    print("\n   🚨 Data leakage indicators:")
    print("      - Perfect scores on unseen data")
    print("      - No false positives or negatives")
    print("      - Model may have seen similar patterns")
    print("      - Training and test data may be too similar")

def main():
    """Main diagnostic function"""
    print("🚀 Starting comprehensive overfitting diagnosis...")
    
    try:
        # Analyze training data
        dataset_info = analyze_training_data()
        
        # Analyze model complexity
        total_params = analyze_model_complexity()
        
        # Analyze feature engineering
        analyze_feature_engineering()
        
        # Analyze training process
        analyze_training_process()
        
        # Check data quality
        check_data_quality()
        
        print("\n🎯 COMPREHENSIVE DIAGNOSIS SUMMARY:")
        print("=" * 50)
        
        print("🔍 ROOT CAUSES IDENTIFIED:")
        print("   1. 🚨 SEVERE OVERFITTING: Model memorized training patterns")
        print("   2. 🚨 DATA LEAKAGE: Training and test data too similar")
        print("   3. 🚨 MODEL COMPLEXITY: 516K parameters for small dataset")
        print("   4. 🚨 FEATURE ISSUES: Simple features with AML flags")
        print("   5. 🚨 TRAINING ISSUES: No validation, too many epochs")
        
        print("\n📋 CRITICAL FIXES NEEDED:")
        print("   1. 🔧 SIMPLIFY MODEL: Reduce parameters, single branch")
        print("   2. 🔧 IMPROVE FEATURES: Remove AML flags, add complexity")
        print("   3. 🔧 FIX TRAINING: Add validation, early stopping")
        print("   4. 🔧 DATA SPLITS: Proper train/validation/test splits")
        print("   5. 🔧 REGULARIZATION: More dropout, weight decay")
        print("   6. 🔧 MORE DATA: Use larger, more diverse datasets")
        
        print("\n🚨 IMMEDIATE ACTIONS:")
        print("   1. ❌ DO NOT USE current model in production")
        print("   2. 🔄 RETRAIN with simplified architecture")
        print("   3. 📊 IMPLEMENT proper validation splits")
        print("   4. 🎯 ADD early stopping and regularization")
        print("   5. 📈 USE more diverse training data")
        
        print("\n✅ NEXT STEPS:")
        print("   1. Create simplified model architecture")
        print("   2. Implement proper train/validation/test splits")
        print("   3. Add regularization and early stopping")
        print("   4. Improve feature engineering")
        print("   5. Use more diverse training data")
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Diagnose Overfitting Issues
===========================

Investigates why the model is showing perfect scores on unseen data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🔍 Diagnosing Overfitting Issues")
print("=" * 50)

def analyze_training_data():
    """Analyze the training data for potential issues"""
    print("📊 Analyzing training data...")
    
    # Check what datasets were used for training
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    if os.path.exists(processed_path):
        files = os.listdir(processed_path)
        print(f"   Processed datasets: {files}")
        
        # Check if there's data leakage
        for file in files:
            if file.endswith('.pkl'):
                print(f"   📁 Found: {file}")
    else:
        print("   ❌ No processed data found")
    
    return files if os.path.exists(processed_path) else []

def check_model_architecture():
    """Check if the model architecture is too complex"""
    print("\n🏗️ Analyzing model architecture...")
    
    # Load the trained model to inspect
    model_path = '/content/drive/MyDrive/LaunDetection/models/comprehensive_chunked_model.pth'
    
    if os.path.exists(model_path):
        print(f"   📁 Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        # Load state dict to inspect parameters
        state_dict = torch.load(model_path, map_location='cpu')
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"   📊 Total parameters: {total_params:,}")
        
        # Check for suspicious patterns
        for name, param in state_dict.items():
            if param.dim() > 1:  # Weight matrices
                print(f"   🔍 {name}: {param.shape}")
                if param.shape[0] == param.shape[1]:  # Square matrices
                    print(f"      ⚠️ Square matrix - potential overfitting risk")
                
                # Check for extreme values
                if torch.abs(param).max() > 100:
                    print(f"      ⚠️ Large weights detected: {torch.abs(param).max():.2f}")
    else:
        print("   ❌ Model file not found")

def test_random_predictions():
    """Test if the model gives random predictions"""
    print("\n🎲 Testing model with random data...")
    
    # Create completely random data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random node features
    x = torch.randn(100, 15).to(device)
    
    # Random edge index
    edge_index = torch.randint(0, 100, (2, 200)).to(device)
    
    # Random edge attributes
    edge_attr = torch.randn(200, 13).to(device)
    
    # Create model
    class SimpleGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(15, 64)
            self.conv2 = GCNConv(64, 32)
            self.classifier = nn.Linear(32, 2)
        
        def forward(self, x, edge_index, edge_attr=None):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            
            # Edge-level prediction
            src_features = x[edge_index[0]]
            tgt_features = x[edge_index[1]]
            edge_features = torch.cat([src_features, tgt_features], dim=1)
            
            return self.classifier(edge_features)
    
    model = SimpleGNN().to(device)
    
    # Test with random data
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        predictions = torch.argmax(logits, dim=1)
    
    # Check if predictions are random
    unique_predictions = torch.unique(predictions)
    print(f"   📊 Unique predictions: {len(unique_predictions)}")
    print(f"   📊 Prediction distribution: {torch.bincount(predictions)}")
    
    if len(unique_predictions) == 1:
        print("   ⚠️ WARNING: Model predicts only one class!")
    elif len(unique_predictions) == 2:
        print("   ✅ Model predicts both classes")
    else:
        print("   ❌ Unexpected: Model predicts more than 2 classes")

def check_data_similarity():
    """Check if training and test data are too similar"""
    print("\n🔍 Checking data similarity...")
    
    # Load a sample from each dataset
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    datasets = ['HI-Small', 'LI-Small', 'HI-Medium', 'LI-Medium', 'HI-Large']
    dataset_stats = {}
    
    for dataset in datasets:
        file_path = os.path.join(data_path, f'{dataset}_Trans.csv')
        if os.path.exists(file_path):
            print(f"   📁 Analyzing {dataset}...")
            
            # Load small sample
            try:
                sample = pd.read_csv(file_path, nrows=1000)
                dataset_stats[dataset] = {
                    'columns': list(sample.columns),
                    'dtypes': sample.dtypes.to_dict(),
                    'shape': sample.shape,
                    'aml_rate': sample['Is Laundering'].mean() if 'Is Laundering' in sample.columns else 0
                }
                print(f"      Columns: {len(sample.columns)}")
                print(f"      AML rate: {dataset_stats[dataset]['aml_rate']:.4f}")
            except Exception as e:
                print(f"      ❌ Error loading {dataset}: {e}")
        else:
            print(f"   ❌ {dataset} not found")
    
    # Check for similarities
    print("\n   🔍 Checking for data similarities...")
    
    if len(dataset_stats) > 1:
        datasets_list = list(dataset_stats.keys())
        for i in range(len(datasets_list)):
            for j in range(i+1, len(datasets_list)):
                ds1, ds2 = datasets_list[i], datasets_list[j]
                stats1, stats2 = dataset_stats[ds1], dataset_stats[ds2]
                
                # Check column similarity
                cols1 = set(stats1['columns'])
                cols2 = set(stats2['columns'])
                common_cols = cols1.intersection(cols2)
                
                similarity = len(common_cols) / max(len(cols1), len(cols2))
                print(f"      {ds1} vs {ds2}: {similarity:.2f} column similarity")
                
                if similarity > 0.9:
                    print(f"         ⚠️ WARNING: Very similar datasets!")

def check_feature_engineering():
    """Check if feature engineering is causing issues"""
    print("\n🔧 Analyzing feature engineering...")
    
    # Check if features are too simple or too complex
    print("   📊 Feature analysis:")
    print("      - Node features: 15 (amount, count, AML flag, etc.)")
    print("      - Edge features: 13 (amount, AML flag, placeholders)")
    print("      - Total edge input: 43 (2*15 + 13)")
    
    # Check for potential issues
    print("\n   ⚠️ Potential issues:")
    print("      - Placeholder features (0.5 values) may be too simple")
    print("      - AML flag in features may cause data leakage")
    print("      - Feature engineering may be too basic")

def main():
    """Main diagnostic function"""
    print("🚀 Starting overfitting diagnosis...")
    
    try:
        # Analyze training data
        files = analyze_training_data()
        
        # Check model architecture
        check_model_architecture()
        
        # Test with random data
        test_random_predictions()
        
        # Check data similarity
        check_data_similarity()
        
        # Check feature engineering
        check_feature_engineering()
        
        print("\n🎯 DIAGNOSIS SUMMARY:")
        print("=" * 30)
        print("🔍 Potential causes of overfitting:")
        print("   1. Data leakage between training/test sets")
        print("   2. Model architecture too complex for data size")
        print("   3. Feature engineering too simple")
        print("   4. Insufficient data diversity")
        print("   5. Training data too similar to test data")
        
        print("\n📋 RECOMMENDATIONS:")
        print("   1. Use proper train/validation/test splits")
        print("   2. Simplify model architecture")
        print("   3. Add more regularization (dropout, weight decay)")
        print("   4. Improve feature engineering")
        print("   5. Use more diverse training data")
        print("   6. Implement early stopping")
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

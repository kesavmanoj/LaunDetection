# ============================================================================
# GOOGLE COLAB CELL - VALIDATE PREPROCESSED DATA (FIXED FOR PYTORCH 2.6)
# Run this to thoroughly validate your data before training
# ============================================================================

# Mount Google Drive
from google.colab import drive
import os

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
else:
    print("Drive already mounted")

# Change to project directory
import sys
sys.path.append('/content/drive/MyDrive/LaunDetection')

# Quick validation without the complex validator
import torch
import numpy as np
from pathlib import Path

def quick_validate_data():
    """Quick but thorough validation of the preprocessed data"""
    
    print("🔍 QUICK DATA VALIDATION")
    print("="*50)
    
    graphs_dir = Path('/content/drive/MyDrive/LaunDetection/data/graphs')
    datasets = ['hi-small', 'li-small']
    
    all_valid = True
    
    for dataset in datasets:
        print(f"\n📊 VALIDATING {dataset.upper()}:")
        print("-" * 30)
        
        splits_file = graphs_dir / f'ibm_aml_{dataset}_fixed_splits.pt'
        
        if not splits_file.exists():
            print(f"❌ File not found: {splits_file}")
            all_valid = False
            continue
        
        try:
            # Load with weights_only=False for PyTorch Geometric data
            data = torch.load(splits_file, map_location='cpu', weights_only=False)
            
            # Basic structure check
            required_keys = ['train', 'val', 'test', 'metadata']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                print(f"❌ Missing keys: {missing_keys}")
                all_valid = False
                continue
            
            print(f"✅ Basic structure: All keys present")
            
            # Edge index validation - THE CRITICAL TEST
            for split_name in ['train', 'val', 'test']:
                split_data = data[split_name]
                edge_index = split_data.edge_index
                num_nodes = split_data.num_nodes
                
                max_idx = edge_index.max().item()
                min_idx = edge_index.min().item()
                
                if min_idx < 0 or max_idx >= num_nodes:
                    print(f"❌ {split_name}: INVALID EDGE INDICES! Range: [{min_idx}, {max_idx}], Nodes: {num_nodes}")
                    all_valid = False
                else:
                    print(f"✅ {split_name}: Valid edge indices [0, {max_idx}] for {num_nodes:,} nodes")
            
            # Data consistency
            train_nodes = data['train'].num_nodes
            val_nodes = data['val'].num_nodes
            test_nodes = data['test'].num_nodes
            
            if train_nodes == val_nodes == test_nodes:
                print(f"✅ Consistent node count: {train_nodes:,}")
            else:
                print(f"❌ Inconsistent nodes: train={train_nodes}, val={val_nodes}, test={test_nodes}")
                all_valid = False
            
            # Feature dimensions
            node_feat_dim = data['train'].x.shape[1]
            edge_feat_dim = data['train'].edge_attr.shape[1]
            print(f"✅ Features: {node_feat_dim} node, {edge_feat_dim} edge")
            
            # Class distribution
            train_labels = data['train'].y.cpu().numpy()
            pos_count = np.sum(train_labels)
            total_count = len(train_labels)
            pos_ratio = pos_count / total_count * 100
            
            print(f"✅ Class distribution: {pos_count:,}/{total_count:,} positive ({pos_ratio:.3f}%)")
            
            # Feature quality check
            node_features = data['train'].x.cpu().numpy()
            edge_features = data['train'].edge_attr.cpu().numpy()
            
            if np.any(np.isnan(node_features)) or np.any(np.isnan(edge_features)):
                print(f"❌ NaN values detected in features")
                all_valid = False
            elif np.any(np.isinf(node_features)) or np.any(np.isinf(edge_features)):
                print(f"❌ Inf values detected in features")
                all_valid = False
            else:
                print(f"✅ No NaN/Inf values in features")
            
            # Edge counts
            train_edges = data['train'].num_edges
            val_edges = data['val'].num_edges
            test_edges = data['test'].num_edges
            total_edges = train_edges + val_edges + test_edges
            
            print(f"✅ Edge counts: {train_edges:,} train, {val_edges:,} val, {test_edges:,} test")
            print(f"   Total: {total_edges:,} edges")
            
        except Exception as e:
            print(f"❌ Error loading {dataset}: {e}")
            all_valid = False
    
    print(f"\n{'='*50}")
    if all_valid:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Data is READY FOR TRAINING")
        print("🚀 You can proceed with colab_simple.py")
    else:
        print("❌ VALIDATION FAILED!")
        print("🔧 Fix the issues above before training")
    print(f"{'='*50}")
    
    return all_valid

# Run the validation
is_ready = quick_validate_data()

if is_ready:
    print("\n🎯 RECOMMENDATION: Proceed with training!")
    print("   Your data has passed all critical validation tests.")
else:
    print("\n⚠️ RECOMMENDATION: Do not train yet!")
    print("   Fix the validation issues first.")

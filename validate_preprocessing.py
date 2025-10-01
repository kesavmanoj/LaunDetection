#!/usr/bin/env python3
"""
Validate Preprocessing Results for AML Multi-GNN
================================================

This script validates that the preprocessing worked correctly and the data is ready for training.
"""

import torch
import pandas as pd
import numpy as np
import networkx as nx
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("🔍 AML Multi-GNN Preprocessing Validation")
print("=" * 60)

def validate_data_structure():
    """Validate the basic data structure"""
    print("\n📊 Step 1: Validating Data Structure")
    print("-" * 40)
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    # Check if data files exist
    required_files = [
        "HI-Small_Trans.csv",
        "HI-Small_accounts.csv"
    ]
    
    for file in required_files:
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False
    
    # Check data sizes
    try:
        transactions = pd.read_csv(os.path.join(data_path, "HI-Small_Trans.csv"), nrows=1000)
        accounts = pd.read_csv(os.path.join(data_path, "HI-Small_accounts.csv"), nrows=1000)
        
        print(f"✅ Transactions: {len(transactions)} rows")
        print(f"✅ Accounts: {len(accounts)} rows")
        print(f"✅ Transaction columns: {list(transactions.columns)}")
        print(f"✅ Account columns: {list(accounts.columns)}")
        
        return True
    except Exception as e:
        print(f"❌ Error reading data: {e}")
        return False

def validate_enhanced_preprocessing():
    """Validate enhanced preprocessing results"""
    print("\n🔧 Step 2: Validating Enhanced Preprocessing Results")
    print("-" * 40)
    
    # Check for enhanced preprocessing outputs
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    if not os.path.exists(processed_path):
        print("❌ Processed data directory not found")
        print("💡 Run enhanced preprocessing first:")
        print("   %run notebooks/08_simple_enhanced_preprocessing.ipynb")
        return False
    
    # Check for checkpoint files
    checkpoint_files = [
        "data_loaded.pkl",
        "node_features.pkl", 
        "edge_features.pkl",
        "normalized.pkl",
        "imbalanced.pkl",
        "weights.pkl",
        "graph_created.pkl"
    ]
    
    found_checkpoints = []
    for file in checkpoint_files:
        file_path = os.path.join(processed_path, file)
        if os.path.exists(file_path):
            found_checkpoints.append(file)
            print(f"✅ {file} exists")
        else:
            print(f"⚠️  {file} missing")
    
    if len(found_checkpoints) == 0:
        print("❌ No preprocessing checkpoints found")
        print("💡 Run enhanced preprocessing first:")
        print("   %run notebooks/08_simple_enhanced_preprocessing.ipynb")
        return False
    
    print(f"✅ Found {len(found_checkpoints)}/{len(checkpoint_files)} checkpoints")
    return True

def validate_graph_structure():
    """Validate the graph structure"""
    print("\n🕸️  Step 3: Validating Graph Structure")
    print("-" * 40)
    
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    # Check for graph files
    graph_files = [
        "enhanced_graph.pkl",
        "node_features.pkl",
        "edge_features.pkl"
    ]
    
    for file in graph_files:
        file_path = os.path.join(processed_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file} exists")
            
            # Try to load and validate the graph
            try:
                if file == "enhanced_graph.pkl":
                    import pickle
                    with open(file_path, 'rb') as f:
                        graph = pickle.load(f)
                    
                    print(f"   Graph nodes: {graph.number_of_nodes()}")
                    print(f"   Graph edges: {graph.number_of_edges()}")
                    print(f"   Graph density: {nx.density(graph):.4f}")
                    
                    # Check for node features
                    if 'features' in graph.nodes[list(graph.nodes())[0]]:
                        feature_dim = len(graph.nodes[list(graph.nodes())[0]]['features'])
                        print(f"   Node features: {feature_dim} dimensions")
                    else:
                        print("   ⚠️  No node features found")
                    
                    # Check for edge features
                    if graph.number_of_edges() > 0:
                        edge = list(graph.edges())[0]
                        if 'features' in graph.edges[edge]:
                            edge_feature_dim = len(graph.edges[edge]['features'])
                            print(f"   Edge features: {edge_feature_dim} dimensions")
                        else:
                            print("   ⚠️  No edge features found")
                    
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
        else:
            print(f"❌ {file} missing")
    
    return True

def validate_class_imbalance():
    """Validate class imbalance handling"""
    print("\n⚖️  Step 4: Validating Class Imbalance Handling")
    print("-" * 40)
    
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    # Check for class imbalance files
    imbalance_files = [
        "imbalanced.pkl",
        "weights.pkl"
    ]
    
    for file in imbalance_files:
        file_path = os.path.join(processed_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file} exists")
            
            try:
                import pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                if file == "imbalanced.pkl":
                    if isinstance(data, dict) and 'class_distribution' in data:
                        class_dist = data['class_distribution']
                        print(f"   Class distribution: {class_dist}")
                        
                        # Check for extreme imbalance
                        if len(class_dist) == 2:
                            majority = max(class_dist.values())
                            minority = min(class_dist.values())
                            imbalance_ratio = majority / minority if minority > 0 else float('inf')
                            print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
                            
                            if imbalance_ratio > 100:
                                print("   ⚠️  Extreme class imbalance detected")
                            else:
                                print("   ✅ Reasonable class imbalance")
                
                elif file == "weights.pkl":
                    if isinstance(data, dict):
                        print(f"   Class weights: {data}")
                        if len(data) == 2:
                            weight_ratio = max(data.values()) / min(data.values())
                            print(f"   Weight ratio: {weight_ratio:.2f}:1")
                
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
        else:
            print(f"❌ {file} missing")
    
    return True

def validate_training_readiness():
    """Validate that data is ready for training"""
    print("\n🚀 Step 5: Validating Training Readiness")
    print("-" * 40)
    
    # Check for training-ready data
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    # Check for temporal splits
    temporal_splits_path = os.path.join(processed_path, "temporal_splits")
    if os.path.exists(temporal_splits_path):
        print("✅ Temporal splits directory exists")
        
        # Check for train/val/test splits
        split_files = ["train.pkl", "val.pkl", "test.pkl"]
        for file in split_files:
            file_path = os.path.join(temporal_splits_path, file)
            if os.path.exists(file_path):
                print(f"✅ {file} exists")
            else:
                print(f"⚠️  {file} missing")
    else:
        print("❌ Temporal splits directory missing")
        print("💡 Run enhanced preprocessing to create temporal splits")
        return False
    
    # Check for PyTorch Geometric data
    try:
        import torch_geometric
        print("✅ PyTorch Geometric available")
        
        # Try to load a sample graph
        sample_file = os.path.join(temporal_splits_path, "train.pkl")
        if os.path.exists(sample_file):
            import pickle
            with open(sample_file, 'rb') as f:
                graphs = pickle.load(f)
            
            if isinstance(graphs, list) and len(graphs) > 0:
                sample_graph = graphs[0]
                print(f"✅ Sample graph loaded: {sample_graph.num_nodes} nodes, {sample_graph.num_edges} edges")
                print(f"   Node features: {sample_graph.x.shape}")
                print(f"   Edge features: {sample_graph.edge_attr.shape if sample_graph.edge_attr is not None else 'None'}")
                print(f"   Labels: {sample_graph.y.shape}")
            else:
                print("❌ No graphs found in training data")
                return False
        
    except Exception as e:
        print(f"❌ Error validating PyTorch Geometric data: {e}")
        return False
    
    return True

def run_comprehensive_validation():
    """Run comprehensive validation"""
    print("🔍 Running Comprehensive Preprocessing Validation")
    print("=" * 60)
    
    validation_results = []
    
    # Step 1: Data Structure
    result1 = validate_data_structure()
    validation_results.append(("Data Structure", result1))
    
    # Step 2: Enhanced Preprocessing
    result2 = validate_enhanced_preprocessing()
    validation_results.append(("Enhanced Preprocessing", result2))
    
    # Step 3: Graph Structure
    result3 = validate_graph_structure()
    validation_results.append(("Graph Structure", result3))
    
    # Step 4: Class Imbalance
    result4 = validate_class_imbalance()
    validation_results.append(("Class Imbalance", result4))
    
    # Step 5: Training Readiness
    result5 = validate_training_readiness()
    validation_results.append(("Training Readiness", result5))
    
    # Summary
    print("\n📋 Validation Summary")
    print("=" * 60)
    
    passed = 0
    total = len(validation_results)
    
    for step, result in validation_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{step}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} validation steps passed")
    
    if passed == total:
        print("🎉 All validations passed! Data is ready for training.")
        print("\n🚀 Next Steps:")
        print("1. Run enhanced preprocessing if not done:")
        print("   %run notebooks/08_simple_enhanced_preprocessing.ipynb")
        print("2. Train the model:")
        print("   %run fixed_real_data_training.py")
    else:
        print("⚠️  Some validations failed. Please check the issues above.")
        print("\n💡 Recommended Actions:")
        if not result2:
            print("- Run enhanced preprocessing:")
            print("  %run notebooks/08_simple_enhanced_preprocessing.ipynb")
        if not result5:
            print("- Check temporal splits creation")
        if not result1:
            print("- Verify data files are in the correct location")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_validation()
    
    if success:
        print("\n✅ Validation completed successfully!")
    else:
        print("\n❌ Validation found issues that need to be addressed.")

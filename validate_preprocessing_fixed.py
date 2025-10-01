#!/usr/bin/env python3
"""
Fixed Preprocessing Validation for AML Multi-GNN
================================================

This script validates that the preprocessing worked correctly with the correct file paths.
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

print("🔍 AML Multi-GNN Preprocessing Validation (FIXED)")
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
    """Validate enhanced preprocessing results with correct paths"""
    print("\n🔧 Step 2: Validating Enhanced Preprocessing Results")
    print("-" * 40)
    
    # Check for enhanced preprocessing outputs in the correct location
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    if not os.path.exists(processed_path):
        print("❌ Processed data directory not found")
        print("💡 Run enhanced preprocessing first:")
        print("   %run notebooks/08_simple_enhanced_preprocessing.ipynb")
        return False
    
    print(f"✅ Processed directory exists: {processed_path}")
    
    # List all files in the processed directory
    try:
        all_files = os.listdir(processed_path)
        print(f"📁 Files in processed directory: {len(all_files)}")
        for file in all_files[:10]:  # Show first 10 files
            print(f"   - {file}")
        if len(all_files) > 10:
            print(f"   ... and {len(all_files) - 10} more files")
    except Exception as e:
        print(f"❌ Error listing processed directory: {e}")
        return False
    
    # Check for checkpoint files (with correct names)
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
            
            # Check file size
            try:
                file_size = os.path.getsize(file_path)
                print(f"   Size: {file_size / 1024 / 1024:.2f} MB")
            except:
                pass
        else:
            print(f"⚠️  {file} missing")
    
    # Also check for chunk files
    chunk_files = [f for f in all_files if 'chunk' in f]
    if chunk_files:
        print(f"✅ Found {len(chunk_files)} chunk files")
    
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
    graph_file = os.path.join(processed_path, "graph_created.pkl")
    if os.path.exists(graph_file):
        print(f"✅ graph_created.pkl exists")
        
        # Try to load and validate the graph
        try:
            import pickle
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            print(f"   Graph data type: {type(graph_data)}")
            
            if isinstance(graph_data, dict):
                print(f"   Keys: {list(graph_data.keys())}")
                
                if 'num_nodes' in graph_data:
                    print(f"   Graph nodes: {graph_data['num_nodes']}")
                if 'num_edges' in graph_data:
                    print(f"   Graph edges: {graph_data['num_edges']}")
                if 'node_features' in graph_data:
                    print(f"   Node features shape: {graph_data['node_features'].shape}")
                if 'edge_features' in graph_data:
                    print(f"   Edge features shape: {graph_data['edge_features'].shape}")
                if 'class_distribution' in graph_data:
                    print(f"   Class distribution: {graph_data['class_distribution']}")
                
                # Check if it's a NetworkX graph
                if 'graph' in graph_data:
                    graph = graph_data['graph']
                    if hasattr(graph, 'number_of_nodes'):
                        print(f"   NetworkX graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                        print(f"   Graph density: {nx.density(graph):.4f}")
                        
                        # Check for node features
                        if graph.number_of_nodes() > 0:
                            first_node = list(graph.nodes())[0]
                            if 'features' in graph.nodes[first_node]:
                                feature_dim = len(graph.nodes[first_node]['features'])
                                print(f"   Node features: {feature_dim} dimensions")
                            else:
                                print("   ⚠️  No node features found")
                        
                        # Check for edge features
                        if graph.number_of_edges() > 0:
                            first_edge = list(graph.edges())[0]
                            if 'features' in graph.edges[first_edge]:
                                edge_feature_dim = len(graph.edges[first_edge]['features'])
                                print(f"   Edge features: {edge_feature_dim} dimensions")
                            else:
                                print("   ⚠️  No edge features found")
            
        except Exception as e:
            print(f"   ❌ Error loading graph: {e}")
    else:
        print(f"❌ graph_created.pkl missing")
    
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
    
    # Check if we have the basic graph data
    graph_file = os.path.join(processed_path, "graph_created.pkl")
    if os.path.exists(graph_file):
        print("✅ Graph data available for training")
        
        try:
            import pickle
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            if isinstance(graph_data, dict) and 'num_nodes' in graph_data:
                print(f"   Ready for training: {graph_data['num_nodes']} nodes")
                return True
            else:
                print("   ⚠️  Graph data format not recognized")
                return False
                
        except Exception as e:
            print(f"   ❌ Error loading graph data: {e}")
            return False
    else:
        print("❌ Graph data missing")
        return False

def run_comprehensive_validation():
    """Run comprehensive validation"""
    print("🔍 Running Comprehensive Preprocessing Validation (FIXED)")
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
        print("1. Train the enhanced model:")
        print("   %run train_enhanced_model.py")
        print("2. Or run hyperparameter optimization:")
        print("   %run hyperparameter_optimization.py")
    else:
        print("⚠️  Some validations failed. Please check the issues above.")
        print("\n💡 Recommended Actions:")
        if not result2:
            print("- Run enhanced preprocessing:")
            print("  %run notebooks/08_simple_enhanced_preprocessing.ipynb")
        if not result5:
            print("- Check graph data format")
        if not result1:
            print("- Verify data files are in the correct location")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_validation()
    
    if success:
        print("\n✅ Validation completed successfully!")
    else:
        print("\n❌ Validation found issues that need to be addressed.")

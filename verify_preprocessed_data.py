#!/usr/bin/env python3
"""
Verify Preprocessed Data - Check what was saved before the crash
This script checks the processed data directory to see what datasets were completed
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    """Verify what preprocessed data was saved"""
    print("🔍 Verifying Preprocessed Data")
    print("=" * 60)
    print("📊 Checking what datasets were saved before the crash")
    print()
    
    # Check processed data directory
    processed_dir = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    if not os.path.exists(processed_dir):
        print("❌ Processed data directory not found!")
        print(f"   Expected: {processed_dir}")
        return
    
    print(f"📁 Checking processed data directory: {processed_dir}")
    print()
    
    # List all files in the directory
    files = os.listdir(processed_dir)
    print(f"📊 Found {len(files)} files in processed directory:")
    print()
    
    # Group files by dataset
    datasets = {}
    for file in files:
        if file.endswith('.pkl'):
            # Extract dataset name from filename
            if '_' in file:
                dataset_name = file.split('_')[0]
                if dataset_name not in datasets:
                    datasets[dataset_name] = []
                datasets[dataset_name].append(file)
    
    # Check each dataset
    for dataset_name, files in datasets.items():
        print(f"🔍 Checking {dataset_name} dataset:")
        print(f"   Files: {', '.join(files)}")
        
        # Check if we have all required files
        required_files = ['_metadata.pkl', '_features.pkl', '_graph.pkl']
        missing_files = []
        
        for req_file in required_files:
            if not any(req_file in f for f in files):
                missing_files.append(req_file)
        
        if missing_files:
            print(f"   ❌ Missing files: {missing_files}")
            print(f"   Status: ⚠️ INCOMPLETE")
        else:
            print(f"   Status: ✅ COMPLETE")
            
            # Try to load and verify one of the files
            try:
                metadata_file = next(f for f in files if '_metadata.pkl' in f)
                metadata_path = os.path.join(processed_dir, metadata_file)
                
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                print(f"   📊 Metadata:")
                print(f"      Dataset: {metadata.get('dataset_name', 'Unknown')}")
                print(f"      Nodes: {metadata.get('num_nodes', 'Unknown'):,}")
                print(f"      Edges: {metadata.get('num_edges', 'Unknown'):,}")
                print(f"      AML Edges: {metadata.get('aml_edges', 'Unknown'):,}")
                print(f"      AML Rate: {metadata.get('aml_rate', 0):.4f}")
                
            except Exception as e:
                print(f"   ❌ Error loading metadata: {str(e)}")
        
        print()
    
    # Summary
    print("📊 Summary:")
    complete_datasets = []
    incomplete_datasets = []
    
    for dataset_name, files in datasets.items():
        required_files = ['_metadata.pkl', '_features.pkl', '_graph.pkl']
        missing_files = []
        
        for req_file in required_files:
            if not any(req_file in f for f in files):
                missing_files.append(req_file)
        
        if missing_files:
            incomplete_datasets.append(dataset_name)
        else:
            complete_datasets.append(dataset_name)
    
    print(f"✅ Complete datasets: {', '.join(complete_datasets) if complete_datasets else 'None'}")
    print(f"⚠️ Incomplete datasets: {', '.join(incomplete_datasets) if incomplete_datasets else 'None'}")
    
    if complete_datasets:
        print(f"\n🎉 {len(complete_datasets)} datasets are ready for training!")
    
    if incomplete_datasets:
        print(f"\n⚠️ {len(incomplete_datasets)} datasets need to be completed:")
        for dataset in incomplete_datasets:
            print(f"   - {dataset}")
    
    print("\n💡 Next steps:")
    if incomplete_datasets:
        print("   1. Run resume_preprocessing.py to complete missing datasets")
    else:
        print("   1. All datasets are complete!")
        print("   2. Ready to start training!")
    
    print("   3. Run multi_dataset_training.py to begin training")

if __name__ == "__main__":
    main()

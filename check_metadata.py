#!/usr/bin/env python3
"""
Check Metadata Content
======================

This script checks what's actually in the metadata files
to understand the data structure better.
"""

import os
import pickle

def check_metadata():
    """Check metadata content for all datasets"""
    print("🔍 Checking Metadata Content...")
    print("=" * 50)
    
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    datasets = ['HI-Small', 'LI-Small', 'HI-Medium', 'LI-Medium', 'HI-Large', 'LI-Large']
    
    for dataset_name in datasets:
        metadata_file = os.path.join(processed_path, f"{dataset_name}_metadata.pkl")
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                
                print(f"\n📊 {dataset_name} Metadata:")
                print(f"   Type: {type(metadata)}")
                
                if isinstance(metadata, dict):
                    print(f"   Keys: {list(metadata.keys())}")
                    for key, value in metadata.items():
                        print(f"   {key}: {value}")
                else:
                    print(f"   Content: {metadata}")
                    
            except Exception as e:
                print(f"   ❌ Error reading {dataset_name}: {str(e)}")
        else:
            print(f"❌ {dataset_name}: Metadata file not found")

if __name__ == "__main__":
    check_metadata()

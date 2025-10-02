#!/usr/bin/env python3
"""
Debug Features Content
======================

This script checks what's actually in the features files
to understand how AML information is stored.
"""

import os
import pickle
import numpy as np

def debug_features():
    """Debug features content for all datasets"""
    print("🔍 Debugging Features Content...")
    print("=" * 50)
    
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    datasets = ['HI-Small', 'LI-Small', 'HI-Medium', 'LI-Medium', 'HI-Large', 'LI-Large']
    
    for dataset_name in datasets:
        features_file = os.path.join(processed_path, f"{dataset_name}_features.pkl")
        
        if os.path.exists(features_file):
            try:
                with open(features_file, 'rb') as f:
                    features = pickle.load(f)
                
                print(f"\n📊 {dataset_name} Features:")
                print(f"   Type: {type(features)}")
                
                if isinstance(features, dict):
                    print(f"   Keys: {list(features.keys())}")
                    for key, value in features.items():
                        if isinstance(value, (list, np.ndarray)):
                            print(f"   {key}: {type(value)} with {len(value)} elements")
                            if len(value) > 0:
                                print(f"      First few values: {value[:5]}")
                        else:
                            print(f"   {key}: {value}")
                elif isinstance(features, (list, np.ndarray)):
                    print(f"   Array with {len(features)} elements")
                    if len(features) > 0:
                        print(f"   First few values: {features[:5]}")
                        if hasattr(features[0], '__len__'):
                            print(f"   Element shape: {features[0].shape if hasattr(features[0], 'shape') else 'N/A'}")
                else:
                    print(f"   Content: {features}")
                    
            except Exception as e:
                print(f"   ❌ Error reading {dataset_name}: {str(e)}")
        else:
            print(f"❌ {dataset_name}: Features file not found")

if __name__ == "__main__":
    debug_features()

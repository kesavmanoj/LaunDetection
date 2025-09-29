# ============================================================================
# GOOGLE COLAB CELL - TRAINING READINESS CHECK
# Quick check to see if data is ready for training
# ============================================================================

# Mount Google Drive
from google.colab import drive
import os

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
else:
    print("Drive already mounted")

import torch
import numpy as np
from pathlib import Path

print("🚀 TRAINING READINESS CHECK")
print("="*40)

graphs_dir = Path('/content/drive/MyDrive/LaunDetection/data/graphs')
datasets = ['hi-small', 'li-small']

ready_datasets = []
issues = []

for dataset in datasets:
    splits_file = graphs_dir / f'ibm_aml_{dataset}_fixed_splits.pt'
    
    if not splits_file.exists():
        issues.append(f"❌ {dataset}: File missing")
        continue
    
    try:
        # Load data
        data = torch.load(splits_file, map_location='cpu', weights_only=False)
        
        # Critical check: Edge indices validity
        train_data = data['train']
        edge_index = train_data.edge_index
        num_nodes = train_data.num_nodes
        
        max_idx = edge_index.max().item()
        min_idx = edge_index.min().item()
        
        if min_idx < 0 or max_idx >= num_nodes:
            issues.append(f"❌ {dataset}: Invalid edge indices [{min_idx}, {max_idx}] for {num_nodes} nodes")
        else:
            ready_datasets.append(dataset)
            print(f"✅ {dataset}: Ready ({train_data.num_edges:,} train edges, {num_nodes:,} nodes)")
            
    except Exception as e:
        issues.append(f"❌ {dataset}: Loading error - {str(e)[:50]}...")

print(f"\n📊 SUMMARY:")
print(f"✅ Ready datasets: {len(ready_datasets)}/2")
print(f"❌ Issues found: {len(issues)}")

if issues:
    print(f"\n🔧 ISSUES TO FIX:")
    for issue in issues:
        print(f"  {issue}")

if len(ready_datasets) >= 1:
    print(f"\n🎯 RECOMMENDATION:")
    print(f"✅ You can train with {ready_datasets} dataset(s)")
    print(f"🚀 Run colab_simple.py to start training")
    
    if len(ready_datasets) == 1:
        print(f"💡 Training will use memory-efficient mode (single dataset)")
else:
    print(f"\n⚠️ RECOMMENDATION:")
    print(f"❌ No datasets are ready for training")
    print(f"🔧 Run colab_preprocess_fixed.py to fix data issues")

print(f"\n{'='*40}")
print(f"Ready to train: {'YES' if len(ready_datasets) >= 1 else 'NO'}")
print(f"{'='*40}")

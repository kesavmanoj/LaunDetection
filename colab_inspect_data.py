# ============================================================================
# GOOGLE COLAB CELL - INSPECT CURRENT DATA FILES
# Run this to see what data files exist and their basic info
# ============================================================================

# Mount Google Drive
from google.colab import drive
import os
from pathlib import Path
import torch

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
else:
    print("Drive already mounted")

print("📁 INSPECTING CURRENT DATA FILES")
print("="*50)

# Check graphs directory
graphs_dir = Path('/content/drive/MyDrive/LaunDetection/data/graphs')

if not graphs_dir.exists():
    print("❌ Graphs directory doesn't exist!")
    print("🔧 Run preprocessing first")
else:
    print(f"📂 Graphs directory: {graphs_dir}")
    
    # List all files
    all_files = list(graphs_dir.glob('*.pt'))
    
    if not all_files:
        print("📭 No .pt files found")
        print("🔧 Run preprocessing to create data files")
    else:
        print(f"\n📋 Found {len(all_files)} data files:")
        
        for file in sorted(all_files):
            size_mb = file.stat().st_size / (1024*1024)
            print(f"  📄 {file.name}: {size_mb:.1f} MB")
            
            # Try to load and inspect
            try:
                if 'splits' in file.name:
                    data = torch.load(file, map_location='cpu', weights_only=False)
                    print(f"    📊 Train: {data['train'].num_edges:,} edges")
                    print(f"    📊 Val: {data['val'].num_edges:,} edges") 
                    print(f"    📊 Test: {data['test'].num_edges:,} edges")
                    print(f"    📊 Nodes: {data['train'].num_nodes:,}")
                    print(f"    📊 Node features: {data['train'].x.shape[1]}")
                    print(f"    📊 Edge features: {data['train'].edge_attr.shape[1]}")
                elif 'complete' in file.name:
                    data = torch.load(file, map_location='cpu')
                    print(f"    📊 Total edges: {data.num_edges:,}")
                    print(f"    📊 Nodes: {data.num_nodes:,}")
                print()
            except Exception as e:
                print(f"    ❌ Error loading: {e}")
                print()

# Check for specific files we expect
expected_files = [
    'ibm_aml_hi-small_fixed_splits.pt',
    'ibm_aml_hi-small_fixed_complete.pt', 
    'ibm_aml_li-small_fixed_splits.pt',
    'ibm_aml_li-small_fixed_complete.pt'
]

print("🎯 EXPECTED FILES CHECK:")
for expected_file in expected_files:
    file_path = graphs_dir / expected_file
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024*1024)
        print(f"  ✅ {expected_file}: {size_mb:.1f} MB")
    else:
        print(f"  ❌ {expected_file}: Missing")

print("\n💡 NEXT STEPS:")
if all((graphs_dir / f).exists() for f in expected_files):
    print("✅ All expected files present - run colab_validate_data.py")
else:
    print("🔧 Missing files - run colab_preprocess_fixed.py first")

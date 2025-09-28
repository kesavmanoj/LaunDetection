# ============================================================================
# GOOGLE COLAB SCRIPT - 10GB RAM OPTIMIZED PREPROCESSING FOR SMALL & MEDIUM DATASETS
# Optimized for 10GB RAM Colab environments - faster processing with larger chunks
# Copy and paste this entire cell into Google Colab
# ============================================================================

# Mount Google Drive and install packages
from google.colab import drive
drive.mount('/content/drive')
!pip install torch torch-geometric scikit-learn tqdm psutil -q

import os
import sys
import gc
import time
import psutil
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# Setup paths for Colab
BASE_DIR = Path('/content/drive/MyDrive/LaunDetection')
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
GRAPHS_DIR = BASE_DIR / 'data' / 'graphs'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories
for directory in [PROCESSED_DATA_DIR, GRAPHS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Add to Python path
sys.path.append(str(BASE_DIR))

print(f"📁 Base directory: {BASE_DIR}")
print(f"📁 Raw data: {RAW_DATA_DIR}")
print(f"📁 Output: {GRAPHS_DIR}")

# Override Config for Colab
class ColabConfig:
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = RAW_DATA_DIR
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
    GRAPHS_DIR = GRAPHS_DIR
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = LOGS_DIR
    
    DATASET_CONFIGS = {
        'HI-Small': {'accounts_file': 'HI-Small_accounts.csv', 'transactions_file': 'HI-Small_Trans.csv'},
        'HI-Medium': {'accounts_file': 'HI-Medium_accounts.csv', 'transactions_file': 'HI-Medium_Trans.csv'},
        'LI-Small': {'accounts_file': 'LI-Small_accounts.csv', 'transactions_file': 'LI-Small_Trans.csv'},
        'LI-Medium': {'accounts_file': 'LI-Medium_accounts.csv', 'transactions_file': 'LI-Medium_Trans.csv'}
    }
    
    # OPTIMIZED FOR 10GB RAM: Larger chunks for speed
    DEFAULT_CHUNK_SIZE = 25000  # Larger chunks for faster processing
    MAX_MEMORY_USAGE_GB = 8     # Use up to 8GB of 10GB available
    GC_FREQUENCY = 20           # Less frequent GC for speed
    
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    NODE_FEATURE_DIM = 10
    EDGE_FEATURE_DIM = 10
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    
    @classmethod
    def create_directories(cls):
        for directory in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.GRAPHS_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def get_dataset_paths(cls, dataset_name):
        if dataset_name not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        config = cls.DATASET_CONFIGS[dataset_name]
        return {
            'accounts': cls.RAW_DATA_DIR / config['accounts_file'],
            'transactions': cls.RAW_DATA_DIR / config['transactions_file']
        }

# Replace config module
import sys
sys.modules['config'] = type(sys)('config')
sys.modules['config'].Config = ColabConfig

# Import the enhanced preprocessor
from preprocessing_enhanced import EnhancedAMLPreprocessor

def check_memory():
    """Check current memory usage"""
    memory = psutil.virtual_memory()
    used_gb = memory.used / 1024**3
    total_gb = memory.total / 1024**3
    percent = memory.percent
    return used_gb, total_gb, percent

def process_small_medium_datasets():
    """Process only small and medium datasets with memory optimization"""
    
    print("🚀 10GB RAM OPTIMIZED PREPROCESSING FOR SMALL & MEDIUM DATASETS")
    print("="*80)
    
    # Target datasets (small and medium only)
    target_datasets = ['HI-Small', 'HI-Medium', 'LI-Small', 'LI-Medium']
    
    # Check available datasets
    available_datasets = []
    dataset_info = {}
    
    for dataset in target_datasets:
        try:
            paths = ColabConfig.get_dataset_paths(dataset)
            if paths['accounts'].exists() and paths['transactions'].exists():
                # Get file sizes
                accounts_size = paths['accounts'].stat().st_size / 1024**2
                transactions_size = paths['transactions'].stat().st_size / 1024**2
                total_size = accounts_size + transactions_size
                
                available_datasets.append(dataset)
                dataset_info[dataset] = {
                    'accounts_size': accounts_size,
                    'transactions_size': transactions_size,
                    'total_size': total_size
                }
                
                print(f"✅ {dataset}: {total_size:.1f} MB total ({accounts_size:.1f} MB accounts + {transactions_size:.1f} MB transactions)")
            else:
                missing = []
                if not paths['accounts'].exists():
                    missing.append("accounts")
                if not paths['transactions'].exists():
                    missing.append("transactions")
                print(f"❌ {dataset}: Missing {', '.join(missing)} file(s)")
        except Exception as e:
            print(f"❌ {dataset}: Error - {e}")
    
    if not available_datasets:
        print("❌ No small/medium datasets found!")
        return
    
    # Sort by size (process smaller first)
    available_datasets.sort(key=lambda x: dataset_info[x]['total_size'])
    
    print(f"\n📊 Processing order: {available_datasets}")
    
    # Check initial memory
    used, total, percent = check_memory()
    print(f"\n💾 Initial memory: {used:.1f}/{total:.1f} GB ({percent:.1f}%)")
    
    # Process each dataset
    results = {}
    total_start_time = time.time()
    
    for i, dataset in enumerate(available_datasets):
        print(f"\n{'='*60}")
        print(f"🔄 PROCESSING DATASET {i+1}/{len(available_datasets)}: {dataset}")
        print(f"{'='*60}")
        
        # Check if already processed
        output_file = GRAPHS_DIR / f'ibm_aml_{dataset.lower()}_enhanced_splits.pt'
        if output_file.exists():
            print(f"⏭️ {dataset} already processed. Skipping...")
            results[dataset] = 'already_processed'
            continue
        
        # Memory check before processing
        used, total, percent = check_memory()
        print(f"💾 Memory before: {used:.1f}/{total:.1f} GB ({percent:.1f}%)")
        
        # OPTIMIZED FOR 10GB RAM: Larger chunk sizes for faster processing
        total_size_mb = dataset_info[dataset]['total_size']
        available_memory_gb = total - used
        
        if total_size_mb < 50:
            chunk_size = 50000  # Much larger for small datasets
        elif total_size_mb < 150:
            chunk_size = 35000  # Larger for small-medium datasets
        elif total_size_mb < 300:
            chunk_size = 25000  # Larger for medium datasets
        elif total_size_mb < 500:
            chunk_size = 15000  # Still larger for larger datasets
        else:
            chunk_size = 10000  # Conservative for very large datasets
        
        # Adjust chunk size based on available memory (more aggressive)
        if available_memory_gb < 4:
            chunk_size = min(chunk_size, 15000)
        elif available_memory_gb < 6:
            chunk_size = min(chunk_size, 25000)
        # No limit if we have 6GB+ available
        
        print(f"📊 Dataset: {total_size_mb:.1f} MB, Available memory: {available_memory_gb:.1f} GB")
        print(f"⚙️ Using chunk size: {chunk_size:,}")
        
        try:
            # Initialize preprocessor with memory-optimized settings
            preprocessor = EnhancedAMLPreprocessor(
                dataset_name=dataset,
                chunk_size=chunk_size,
                log_level='INFO'
            )
            
            # Run preprocessing
            start_time = time.time()
            complete_data, splits_data = preprocessor.run_enhanced_preprocessing()
            processing_time = time.time() - start_time
            
            # Get final statistics
            suspicious_rate = complete_data.y.float().mean().item() * 100
            
            results[dataset] = {
                'status': 'success',
                'nodes': complete_data.num_nodes,
                'edges': complete_data.num_edges,
                'suspicious_rate': suspicious_rate,
                'processing_time': processing_time,
                'node_features': complete_data.x.shape[1],
                'edge_features': complete_data.edge_attr.shape[1]
            }
            
            print(f"✅ {dataset} completed in {processing_time/60:.1f} minutes")
            print(f"📊 {complete_data.num_nodes:,} nodes, {complete_data.num_edges:,} edges")
            print(f"🎯 Suspicious rate: {suspicious_rate:.3f}%")
            print(f"📈 Features: {complete_data.x.shape[1]} node, {complete_data.edge_attr.shape[1]} edge")
            
            # Aggressive cleanup between datasets
            del preprocessor, complete_data, splits_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Memory check after processing
            used, total, percent = check_memory()
            print(f"💾 Memory after: {used:.1f}/{total:.1f} GB ({percent:.1f}%)")
            
        except Exception as e:
            print(f"❌ Failed to process {dataset}: {str(e)}")
            results[dataset] = {'status': 'failed', 'error': str(e)}
            
            # Cleanup on failure
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("🎉 SMALL & MEDIUM DATASET PREPROCESSING COMPLETED!")
    print(f"{'='*80}")
    print(f"⏱️ Total processing time: {total_time/60:.1f} minutes")
    
    print(f"\n📊 Results Summary:")
    successful = 0
    for dataset, result in results.items():
        if isinstance(result, dict) and result.get('status') == 'success':
            successful += 1
            print(f"✅ {dataset}: {result['nodes']:,} nodes, {result['edges']:,} edges, {result['suspicious_rate']:.3f}% suspicious ({result['processing_time']/60:.1f} min)")
        elif result == 'already_processed':
            print(f"⏭️ {dataset}: Already processed")
        else:
            print(f"❌ {dataset}: {result.get('error', 'Unknown error')}")
    
    print(f"\n📁 Processed files saved in: {GRAPHS_DIR}")
    
    # List all processed files
    processed_files = list(GRAPHS_DIR.glob("*_enhanced_splits.pt"))
    print(f"\n📋 Available processed files ({len(processed_files)}):")
    for file in processed_files:
        size_mb = file.stat().st_size / 1024**2
        print(f"  {file.name} ({size_mb:.1f} MB)")
    
    # Final memory check
    used, total, percent = check_memory()
    print(f"\n💾 Final memory usage: {used:.1f}/{total:.1f} GB ({percent:.1f}%)")
    
    print(f"\n🚀 Successfully processed {successful} datasets!")
    print("Ready for GNN training! 🎯")

# Example of how to load processed data
def load_processed_data_example():
    """Example of how to load and use the processed data"""
    print("\n" + "="*60)
    print("📖 EXAMPLE: Loading Processed Data")
    print("="*60)
    
    # Find available processed files
    processed_files = list(GRAPHS_DIR.glob("*_enhanced_splits.pt"))
    
    if not processed_files:
        print("❌ No processed files found. Run preprocessing first!")
        return
    
    # Load the first available file as example
    example_file = processed_files[0]
    print(f"📂 Loading: {example_file.name}")
    
    try:
        splits_data = torch.load(example_file)
        
        # Access different components
        train_data = splits_data['train']
        val_data = splits_data['val']
        test_data = splits_data['test']
        complete_data = splits_data['complete']
        metadata = splits_data['metadata']
        
        print(f"\n📊 Dataset Information:")
        print(f"  Dataset: {metadata['dataset_name']}")
        print(f"  Total nodes: {metadata['num_nodes']:,}")
        print(f"  Total edges: {metadata['num_edges']:,}")
        print(f"  Node features: {metadata['node_feature_dim']}")
        print(f"  Edge features: {metadata['edge_feature_dim']}")
        
        print(f"\n📈 Data Splits:")
        print(f"  Train: {train_data.num_edges:,} edges ({train_data.y.float().mean().item()*100:.3f}% positive)")
        print(f"  Val: {val_data.num_edges:,} edges ({val_data.y.float().mean().item()*100:.3f}% positive)")
        print(f"  Test: {test_data.num_edges:,} edges ({test_data.y.float().mean().item()*100:.3f}% positive)")
        
        print(f"\n✅ Data loaded successfully! Ready for GNN training.")
        
        return splits_data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# Run the processing
if __name__ == "__main__":
    # Process small and medium datasets
    process_small_medium_datasets()
    
    # Show example of loading data
    load_processed_data_example()

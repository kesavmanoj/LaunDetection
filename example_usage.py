"""
Example usage of the Enhanced IBM AML Dataset Preprocessor
This script demonstrates how to use the preprocessing pipeline and load the results.
"""
import torch
from pathlib import Path
from preprocessing_enhanced import EnhancedAMLPreprocessor
from config import Config

def example_preprocessing():
    """Example of running the preprocessing pipeline"""
    print("🚀 Enhanced IBM AML Dataset Preprocessing Example")
    print("="*60)
    
    # Initialize preprocessor for HI-Small dataset
    preprocessor = EnhancedAMLPreprocessor(
        dataset_name='HI-Small',
        chunk_size=25000,  # Smaller chunks for demo
        log_level='INFO'
    )
    
    # Run preprocessing
    print("Starting preprocessing...")
    complete_data, splits_data = preprocessor.run_enhanced_preprocessing()
    
    print("\n✅ Preprocessing completed!")
    return complete_data, splits_data

def example_loading_processed_data():
    """Example of loading already processed data"""
    print("\n📂 Loading Processed Data Example")
    print("="*60)
    
    # Load the splits data
    splits_path = Config.GRAPHS_DIR / 'ibm_aml_hi-small_enhanced_splits.pt'
    
    if splits_path.exists():
        print(f"Loading data from: {splits_path}")
        splits_data = torch.load(splits_path)
        
        # Access different splits
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
        print(f"  Positive rate: {metadata['positive_rate']*100:.3f}%")
        
        print(f"\n📈 Split Information:")
        print(f"  Train: {train_data.num_edges:,} edges")
        print(f"  Validation: {val_data.num_edges:,} edges")
        print(f"  Test: {test_data.num_edges:,} edges")
        
        print(f"\n🔧 Encoders Available:")
        print(f"  Entity classes: {len(metadata['encoders']['entity_classes'])}")
        print(f"  Currency classes: {len(metadata['encoders']['currency_classes'])}")
        print(f"  Payment classes: {len(metadata['encoders']['payment_classes'])}")
        
        return splits_data
    else:
        print(f"❌ Processed data not found at: {splits_path}")
        print("Run preprocessing first!")
        return None

def example_feature_analysis(splits_data):
    """Example of analyzing the processed features"""
    if splits_data is None:
        return
        
    print("\n🔍 Feature Analysis Example")
    print("="*60)
    
    train_data = splits_data['train']
    
    # Node feature analysis
    print("📊 Node Features Analysis:")
    node_features = train_data.x
    print(f"  Shape: {node_features.shape}")
    print(f"  Mean: {node_features.mean(dim=0)[:5]}")  # First 5 features
    print(f"  Std: {node_features.std(dim=0)[:5]}")   # First 5 features
    
    # Edge feature analysis
    print("\n📊 Edge Features Analysis:")
    edge_features = train_data.edge_attr
    print(f"  Shape: {edge_features.shape}")
    print(f"  Mean: {edge_features.mean(dim=0)[:5]}")  # First 5 features
    print(f"  Std: {edge_features.std(dim=0)[:5]}")   # First 5 features
    
    # Label distribution
    print(f"\n🏷️ Label Distribution:")
    labels = train_data.y
    positive_count = labels.sum().item()
    total_count = len(labels)
    print(f"  Positive (laundering): {positive_count:,} ({positive_count/total_count*100:.3f}%)")
    print(f"  Negative (legitimate): {total_count-positive_count:,} ({(total_count-positive_count)/total_count*100:.3f}%)")

def example_graph_statistics(splits_data):
    """Example of computing graph statistics"""
    if splits_data is None:
        return
        
    print("\n📈 Graph Statistics Example")
    print("="*60)
    
    complete_data = splits_data['complete']
    
    # Basic graph properties
    num_nodes = complete_data.num_nodes
    num_edges = complete_data.num_edges
    edge_index = complete_data.edge_index
    
    print(f"📊 Basic Graph Properties:")
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {num_edges:,}")
    print(f"  Density: {num_edges / (num_nodes * (num_nodes - 1)):.6f}")
    
    # Degree statistics
    from torch_geometric.utils import degree
    
    # Out-degree (source nodes)
    out_degrees = degree(edge_index[0], num_nodes=num_nodes)
    # In-degree (destination nodes)  
    in_degrees = degree(edge_index[1], num_nodes=num_nodes)
    
    print(f"\n📊 Degree Statistics:")
    print(f"  Average out-degree: {out_degrees.float().mean().item():.2f}")
    print(f"  Max out-degree: {out_degrees.max().item()}")
    print(f"  Average in-degree: {in_degrees.float().mean().item():.2f}")
    print(f"  Max in-degree: {in_degrees.max().item()}")
    
    # Self-loops
    self_loops = (edge_index[0] == edge_index[1]).sum().item()
    print(f"  Self-loops: {self_loops:,} ({self_loops/num_edges*100:.3f}%)")

def main():
    """Main example execution"""
    print("🎯 IBM AML Dataset Preprocessing - Complete Example")
    print("="*80)
    
    # Create directories
    Config.create_directories()
    
    # Check if processed data already exists
    splits_path = Config.GRAPHS_DIR / 'ibm_aml_hi-small_enhanced_splits.pt'
    
    if splits_path.exists():
        print("✅ Found existing processed data. Loading...")
        splits_data = example_loading_processed_data()
    else:
        print("📝 No processed data found. Running preprocessing...")
        print("Note: Make sure you have the raw CSV files in the data/raw directory!")
        
        try:
            complete_data, splits_data = example_preprocessing()
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("\n📋 To run this example:")
            print("1. Place your IBM AML CSV files in: data/raw/")
            print("2. Files needed: HI-Small_accounts.csv, HI-Small_Trans.csv")
            print("3. Run this script again")
            return
    
    # Run analysis examples
    if splits_data:
        example_feature_analysis(splits_data)
        example_graph_statistics(splits_data)
        
        print(f"\n🎉 Example completed successfully!")
        print(f"📁 Processed files are saved in: {Config.GRAPHS_DIR}")
        print(f"📋 Logs are saved in: {Config.LOGS_DIR}")
        
        print(f"\n🚀 Next steps:")
        print("1. Use the processed data for GNN training")
        print("2. Experiment with different GNN architectures")
        print("3. Evaluate on the test set")

if __name__ == "__main__":
    main()

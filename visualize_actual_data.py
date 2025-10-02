#!/usr/bin/env python3
"""
Visualize Actual Processed Data Format
======================================

This script works with the actual data format found in your Google Drive:
- Separate graph.pkl, features.pkl, and metadata.pkl files
- Multiple datasets (HI-Small, LI-Small, HI-Medium, LI-Medium, HI-Large, LI-Large)
"""

import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def load_dataset_data(processed_path, dataset_name):
    """Load data for a specific dataset"""
    try:
        # Load graph
        graph_path = os.path.join(processed_path, f"{dataset_name}_graph.pkl")
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        
        # Load features
        features_path = os.path.join(processed_path, f"{dataset_name}_features.pkl")
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(processed_path, f"{dataset_name}_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return {
            'graph': graph,
            'features': features,
            'metadata': metadata
        }
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {str(e)}")
        return None

def analyze_dataset(dataset_name, data):
    """Analyze a single dataset"""
    if not data:
        return None
    
    graph = data['graph']
    features = data['features']
    metadata = data['metadata']
    
    # Basic graph info
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    
    # Try to get AML information from metadata or features
    aml_edges = 0
    aml_rate = 0
    
    if isinstance(metadata, dict):
        if 'aml_edges' in metadata:
            aml_edges = metadata['aml_edges']
        elif 'aml_count' in metadata:
            aml_edges = metadata['aml_count']
        elif 'aml_rate' in metadata:
            # Calculate AML edges from rate
            aml_rate = metadata['aml_rate'] * 100  # Convert to percentage
            aml_edges = int(edges * metadata['aml_rate'])
        else:
            print(f"   Available metadata keys: {list(metadata.keys())}")
    
    if aml_edges > 0:
        aml_rate = (aml_edges / edges) * 100
    elif 'aml_rate' in metadata:
        aml_rate = metadata['aml_rate'] * 100
        aml_edges = int(edges * metadata['aml_rate'])
    
    print(f"📊 {dataset_name}:")
    print(f"   Nodes: {nodes:,}")
    print(f"   Edges: {edges:,}")
    print(f"   AML Edges: {aml_edges:,} ({aml_rate:.2f}%)")
    
    return {
        'name': dataset_name,
        'nodes': nodes,
        'edges': edges,
        'aml_edges': aml_edges,
        'aml_rate': aml_rate
    }

def create_comprehensive_visualization(dataset_info):
    """Create comprehensive visualization"""
    if not dataset_info:
        print("❌ No valid datasets to visualize!")
        return
    
    print(f"\n📊 Creating visualization for {len(dataset_info)} datasets...")
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('AML Multi-Dataset Analysis', fontsize=16, fontweight='bold')
    
    names = [d['name'] for d in dataset_info]
    nodes = [d['nodes'] for d in dataset_info]
    edges = [d['edges'] for d in dataset_info]
    aml_rates = [d['aml_rate'] for d in dataset_info]
    aml_edges = [d['aml_edges'] for d in dataset_info]
    
    # Plot 1: Node counts
    bars1 = axes[0, 0].bar(names, nodes, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Number of Nodes per Dataset')
    axes[0, 0].set_ylabel('Nodes')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars1, nodes):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(nodes)*0.01, 
                       f'{value:,}', ha='center', va='bottom')
    
    # Plot 2: Edge counts
    bars2 = axes[0, 1].bar(names, edges, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Number of Edges per Dataset')
    axes[0, 1].set_ylabel('Edges')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars2, edges):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(edges)*0.01, 
                       f'{value:,}', ha='center', va='bottom')
    
    # Plot 3: AML rates
    bars3 = axes[0, 2].bar(names, aml_rates, color='red', alpha=0.7)
    axes[0, 2].set_title('AML Rate per Dataset')
    axes[0, 2].set_ylabel('AML Rate (%)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars3, aml_rates):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(aml_rates)*0.01, 
                       f'{value:.2f}%', ha='center', va='bottom')
    
    # Plot 4: AML edge counts
    bars4 = axes[1, 0].bar(names, aml_edges, color='darkred', alpha=0.7)
    axes[1, 0].set_title('Number of AML Edges')
    axes[1, 0].set_ylabel('AML Edges')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars4, aml_edges):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(aml_edges)*0.01, 
                       f'{value:,}', ha='center', va='bottom')
    
    # Plot 5: Edge-to-Node ratio
    edge_node_ratios = [e/n for e, n in zip(edges, nodes)]
    bars5 = axes[1, 1].bar(names, edge_node_ratios, color='green', alpha=0.7)
    axes[1, 1].set_title('Edge-to-Node Ratio')
    axes[1, 1].set_ylabel('Edges per Node')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars5, edge_node_ratios):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(edge_node_ratios)*0.01, 
                       f'{value:.2f}', ha='center', va='bottom')
    
    # Plot 6: Dataset size comparison (log scale)
    axes[1, 2].bar(names, nodes, alpha=0.5, label='Nodes', color='skyblue')
    axes[1, 2].bar(names, edges, alpha=0.5, label='Edges', color='lightcoral')
    axes[1, 2].set_title('Dataset Size Comparison')
    axes[1, 2].set_ylabel('Count (Log Scale)')
    axes[1, 2].set_yscale('log')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save to Google Drive
    save_path = "/content/drive/MyDrive/LaunDetection/aml_dataset_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    
    plt.show()
    
    # Print comprehensive summary
    print("\n📊 Comprehensive Dataset Summary:")
    print("=" * 50)
    total_nodes = sum(nodes)
    total_edges = sum(edges)
    total_aml = sum(aml_edges)
    overall_aml_rate = (total_aml / total_edges) * 100 if total_edges > 0 else 0
    
    print(f"Total Nodes: {total_nodes:,}")
    print(f"Total Edges: {total_edges:,}")
    print(f"Total AML Edges: {total_aml:,}")
    print(f"Overall AML Rate: {overall_aml_rate:.2f}%")
    print(f"Average Edge-to-Node Ratio: {np.mean(edge_node_ratios):.2f}")
    
    # Dataset breakdown
    print(f"\n📈 Dataset Breakdown:")
    for info in dataset_info:
        print(f"   {info['name']}: {info['nodes']:,} nodes, {info['edges']:,} edges, {info['aml_edges']:,} AML ({info['aml_rate']:.2f}%)")

def main():
    """Main visualization function"""
    print("📊 AML Multi-Dataset Visualization")
    print("=" * 50)
    
    # Set the processed data path
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    if not os.path.exists(processed_path):
        print("❌ Processed data path not found!")
        return
    
    # Available datasets
    available_datasets = ['HI-Small', 'LI-Small', 'HI-Medium', 'LI-Medium', 'HI-Large', 'LI-Large']
    
    print(f"🔍 Analyzing {len(available_datasets)} datasets...")
    
    # Load and analyze each dataset
    dataset_info = []
    for dataset_name in available_datasets:
        print(f"\n📊 Loading {dataset_name}...")
        
        # Check if all required files exist
        graph_file = os.path.join(processed_path, f"{dataset_name}_graph.pkl")
        features_file = os.path.join(processed_path, f"{dataset_name}_features.pkl")
        metadata_file = os.path.join(processed_path, f"{dataset_name}_metadata.pkl")
        
        if all(os.path.exists(f) for f in [graph_file, features_file, metadata_file]):
            data = load_dataset_data(processed_path, dataset_name)
            if data:
                info = analyze_dataset(dataset_name, data)
                if info:
                    dataset_info.append(info)
        else:
            print(f"   ⚠️ Missing files for {dataset_name}")
    
    # Create visualization
    create_comprehensive_visualization(dataset_info)
    
    print("\n🎉 Visualization complete!")
    print("📁 Check your Google Drive for the saved image")

if __name__ == "__main__":
    main()

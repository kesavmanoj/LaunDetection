#!/usr/bin/env python3
"""
Quick AML Data Visualization
============================

Simple script to quickly visualize preprocessed AML datasets
"""

import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def quick_visualize():
    """Quick visualization of processed data"""
    print("📊 Quick AML Data Visualization")
    print("=" * 40)
    
    # Check for processed data
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    # Also check alternative paths
    alt_paths = [
        "/content/drive/MyDrive/LaunDetection/data/processed",
        "/content/drive/MyDrive/LaunDetection/data/processed/",
        "/content/LaunDetection/data/processed",
        "/content/LaunDetection/data/processed/"
    ]
    
    # Find the correct processed path
    found_path = None
    for path in alt_paths:
        if os.path.exists(path):
            found_path = path
            break
    
    if not found_path:
        print("❌ No processed data found!")
        print("💡 Checked paths:")
        for path in alt_paths:
            print(f"   - {path}")
        print("💡 Run preprocessing first: python working_preprocessing.py")
        return
    
    processed_path = found_path
    print(f"✅ Found processed data at: {processed_path}")
    
    # Find processed files
    datasets = []
    for file in os.listdir(processed_path):
        if file.endswith('_processed.pkl'):
            dataset_name = file.replace('_processed.pkl', '')
            datasets.append(dataset_name)
    
    if not datasets:
        print("❌ No processed datasets found!")
        return
    
    print(f"✅ Found {len(datasets)} datasets: {datasets}")
    
    # Load and analyze each dataset
    dataset_info = []
    
    for dataset_name in datasets:
        try:
            file_path = os.path.join(processed_path, f"{dataset_name}_processed.pkl")
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and 'graph' in data:
                graph = data['graph']
                nodes = graph.number_of_nodes()
                edges = graph.number_of_edges()
                
                aml_edges = 0
                if 'edge_labels' in data:
                    aml_edges = sum(data['edge_labels'])
                
                aml_rate = (aml_edges / edges) * 100 if edges > 0 else 0
                
                dataset_info.append({
                    'name': dataset_name,
                    'nodes': nodes,
                    'edges': edges,
                    'aml_edges': aml_edges,
                    'aml_rate': aml_rate
                })
                
                print(f"📊 {dataset_name}: {nodes:,} nodes, {edges:,} edges, {aml_edges:,} AML ({aml_rate:.2f}%)")
                
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {str(e)}")
    
    if not dataset_info:
        print("❌ No valid datasets loaded!")
        return
    
    # Create simple visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('AML Dataset Overview', fontsize=16, fontweight='bold')
    
    names = [d['name'] for d in dataset_info]
    nodes = [d['nodes'] for d in dataset_info]
    edges = [d['edges'] for d in dataset_info]
    aml_rates = [d['aml_rate'] for d in dataset_info]
    aml_edges = [d['aml_edges'] for d in dataset_info]
    
    # Plot 1: Node counts
    axes[0, 0].bar(names, nodes, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Number of Nodes')
    axes[0, 0].set_ylabel('Nodes')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Edge counts
    axes[0, 1].bar(names, edges, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Number of Edges')
    axes[0, 1].set_ylabel('Edges')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: AML rates
    axes[1, 0].bar(names, aml_rates, color='red', alpha=0.7)
    axes[1, 0].set_title('AML Rate (%)')
    axes[1, 0].set_ylabel('AML Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: AML edge counts
    axes[1, 1].bar(names, aml_edges, color='darkred', alpha=0.7)
    axes[1, 1].set_title('Number of AML Edges')
    axes[1, 1].set_ylabel('AML Edges')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/LaunDetection/quick_dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n📊 Dataset Summary:")
    print("=" * 30)
    total_nodes = sum(nodes)
    total_edges = sum(edges)
    total_aml = sum(aml_edges)
    overall_aml_rate = (total_aml / total_edges) * 100 if total_edges > 0 else 0
    
    print(f"Total Nodes: {total_nodes:,}")
    print(f"Total Edges: {total_edges:,}")
    print(f"Total AML Edges: {total_aml:,}")
    print(f"Overall AML Rate: {overall_aml_rate:.2f}%")
    
    print(f"\n✅ Visualization saved to: /content/drive/MyDrive/LaunDetection/quick_dataset_overview.png")

if __name__ == "__main__":
    quick_visualize()

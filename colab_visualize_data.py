#!/usr/bin/env python3
"""
Google Colab AML Data Visualization
===================================

Fixed visualization script for Google Colab environment
"""

import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def find_processed_data():
    """Find processed data in Google Colab"""
    print("🔍 Searching for processed data...")
    
    # Common paths in Google Colab
    search_paths = [
        "/content/drive/MyDrive/LaunDetection/data/processed",
        "/content/drive/MyDrive/LaunDetection/data/processed/",
        "/content/LaunDetection/data/processed",
        "/content/LaunDetection/data/processed/",
        "/content/drive/MyDrive/data/processed",
        "/content/drive/MyDrive/data/processed/",
        "/content/data/processed",
        "/content/data/processed/",
        "/content/processed",
        "/content/processed/"
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            print(f"✅ Found processed data at: {path}")
            return path
    
    print("❌ No processed data found in any location!")
    return None

def list_available_datasets(processed_path):
    """List available processed datasets"""
    print(f"📁 Listing files in {processed_path}...")
    
    try:
        files = os.listdir(processed_path)
        processed_files = [f for f in files if f.endswith('_processed.pkl')]
        
        if processed_files:
            print(f"✅ Found {len(processed_files)} processed datasets:")
            for file in processed_files:
                file_path = os.path.join(processed_path, file)
                size = os.path.getsize(file_path)
                print(f"   📄 {file} ({size:,} bytes)")
            return processed_files
        else:
            print("❌ No processed files found!")
            return []
    except Exception as e:
        print(f"❌ Error listing files: {str(e)}")
        return []

def load_and_analyze_dataset(processed_path, dataset_file):
    """Load and analyze a single dataset"""
    try:
        file_path = os.path.join(processed_path, dataset_file)
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        dataset_name = dataset_file.replace('_processed.pkl', '')
        print(f"\n📊 Analyzing {dataset_name}...")
        
        if isinstance(data, dict) and 'graph' in data:
            graph = data['graph']
            nodes = graph.number_of_nodes()
            edges = graph.number_of_edges()
            
            aml_edges = 0
            if 'edge_labels' in data:
                aml_edges = sum(data['edge_labels'])
            
            aml_rate = (aml_edges / edges) * 100 if edges > 0 else 0
            
            print(f"   📊 Nodes: {nodes:,}")
            print(f"   📊 Edges: {edges:,}")
            print(f"   🚨 AML Edges: {aml_edges:,} ({aml_rate:.2f}%)")
            
            return {
                'name': dataset_name,
                'nodes': nodes,
                'edges': edges,
                'aml_edges': aml_edges,
                'aml_rate': aml_rate
            }
        else:
            print(f"   ❌ Invalid data format")
            return None
            
    except Exception as e:
        print(f"   ❌ Error loading {dataset_file}: {str(e)}")
        return None

def create_visualization(dataset_info):
    """Create visualization from dataset info"""
    if not dataset_info:
        print("❌ No valid datasets to visualize!")
        return
    
    print(f"\n📊 Creating visualization for {len(dataset_info)} datasets...")
    
    # Set up the plot
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
    
    # Add value labels
    for i, v in enumerate(nodes):
        axes[0, 0].text(i, v + max(nodes)*0.01, f'{v:,}', ha='center', va='bottom')
    
    # Plot 2: Edge counts
    axes[0, 1].bar(names, edges, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Number of Edges')
    axes[0, 1].set_ylabel('Edges')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(edges):
        axes[0, 1].text(i, v + max(edges)*0.01, f'{v:,}', ha='center', va='bottom')
    
    # Plot 3: AML rates
    axes[1, 0].bar(names, aml_rates, color='red', alpha=0.7)
    axes[1, 0].set_title('AML Rate (%)')
    axes[1, 0].set_ylabel('AML Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(aml_rates):
        axes[1, 0].text(i, v + max(aml_rates)*0.01, f'{v:.2f}%', ha='center', va='bottom')
    
    # Plot 4: AML edge counts
    axes[1, 1].bar(names, aml_edges, color='darkred', alpha=0.7)
    axes[1, 1].set_title('Number of AML Edges')
    axes[1, 1].set_ylabel('AML Edges')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(aml_edges):
        axes[1, 1].text(i, v + max(aml_edges)*0.01, f'{v:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save to Google Drive
    save_path = "/content/drive/MyDrive/LaunDetection/dataset_overview.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    
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

def main():
    """Main visualization function"""
    print("📊 Google Colab AML Data Visualization")
    print("=" * 50)
    
    # Find processed data
    processed_path = find_processed_data()
    if not processed_path:
        print("💡 Run preprocessing first: python working_preprocessing.py")
        return
    
    # List available datasets
    processed_files = list_available_datasets(processed_path)
    if not processed_files:
        print("💡 No processed datasets found!")
        return
    
    # Load and analyze each dataset
    dataset_info = []
    for dataset_file in processed_files:
        info = load_and_analyze_dataset(processed_path, dataset_file)
        if info:
            dataset_info.append(info)
    
    # Create visualization
    create_visualization(dataset_info)
    
    print("\n🎉 Visualization complete!")
    print("📁 Check your Google Drive for the saved image")

if __name__ == "__main__":
    main()

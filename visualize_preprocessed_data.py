#!/usr/bin/env python3
"""
AML Multi-Dataset Preprocessed Data Visualization
===============================================

This script visualizes the preprocessed AML datasets to understand:
- Dataset distributions and sizes
- AML vs Non-AML ratios
- Graph structure and connectivity
- Feature distributions
- Network topology analysis
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_processed_datasets():
    """Load all processed datasets"""
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    datasets = {}
    available_datasets = ['HI-Small', 'LI-Small', 'HI-Medium', 'LI-Medium']
    
    print("📊 Loading Processed Datasets...")
    print("=" * 50)
    
    for dataset_name in available_datasets:
        try:
            # Load the processed data
            dataset_path = os.path.join(processed_path, f"{dataset_name}_processed.pkl")
            
            if os.path.exists(dataset_path):
                with open(dataset_path, 'rb') as f:
                    data = pickle.load(f)
                
                datasets[dataset_name] = data
                print(f"✅ {dataset_name}: Loaded successfully")
                
                # Print basic info
                if isinstance(data, dict):
                    if 'graph' in data:
                        graph = data['graph']
                        nodes = graph.number_of_nodes()
                        edges = graph.number_of_edges()
                        print(f"   📊 Nodes: {nodes:,}, Edges: {edges:,}")
                        
                        if 'edge_labels' in data:
                            aml_edges = sum(data['edge_labels'])
                            non_aml_edges = len(data['edge_labels']) - aml_edges
                            aml_rate = (aml_edges / len(data['edge_labels'])) * 100
                            print(f"   🚨 AML Edges: {aml_edges:,} ({aml_rate:.2f}%)")
                            print(f"   ✅ Non-AML Edges: {non_aml_edges:,}")
                    else:
                        print(f"   📊 Data keys: {list(data.keys())}")
                else:
                    print(f"   📊 Data type: {type(data)}")
            else:
                print(f"❌ {dataset_name}: File not found")
                
        except Exception as e:
            print(f"❌ {dataset_name}: Error loading - {str(e)}")
    
    return datasets

def create_dataset_overview(datasets):
    """Create overview of all datasets"""
    print("\n📊 Creating Dataset Overview...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('AML Multi-Dataset Overview', fontsize=16, fontweight='bold')
    
    # Collect data for overview
    dataset_names = []
    node_counts = []
    edge_counts = []
    aml_rates = []
    aml_counts = []
    
    for name, data in datasets.items():
        if isinstance(data, dict) and 'graph' in data:
            graph = data['graph']
            nodes = graph.number_of_nodes()
            edges = graph.number_of_edges()
            
            dataset_names.append(name)
            node_counts.append(nodes)
            edge_counts.append(edges)
            
            if 'edge_labels' in data:
                aml_edges = sum(data['edge_labels'])
                total_edges = len(data['edge_labels'])
                aml_rate = (aml_edges / total_edges) * 100 if total_edges > 0 else 0
                aml_rates.append(aml_rate)
                aml_counts.append(aml_edges)
            else:
                aml_rates.append(0)
                aml_counts.append(0)
    
    # Plot 1: Node counts
    axes[0, 0].bar(dataset_names, node_counts, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Number of Nodes per Dataset')
    axes[0, 0].set_ylabel('Number of Nodes')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(node_counts):
        axes[0, 0].text(i, v + max(node_counts)*0.01, f'{v:,}', ha='center', va='bottom')
    
    # Plot 2: Edge counts
    axes[0, 1].bar(dataset_names, edge_counts, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Number of Edges per Dataset')
    axes[0, 1].set_ylabel('Number of Edges')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(edge_counts):
        axes[0, 1].text(i, v + max(edge_counts)*0.01, f'{v:,}', ha='center', va='bottom')
    
    # Plot 3: AML rates
    bars = axes[1, 0].bar(dataset_names, aml_rates, color='red', alpha=0.7)
    axes[1, 0].set_title('AML Rate per Dataset')
    axes[1, 0].set_ylabel('AML Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(aml_rates):
        axes[1, 0].text(i, v + max(aml_rates)*0.01, f'{v:.2f}%', ha='center', va='bottom')
    
    # Plot 4: AML edge counts
    axes[1, 1].bar(dataset_names, aml_counts, color='darkred', alpha=0.7)
    axes[1, 1].set_title('Number of AML Edges per Dataset')
    axes[1, 1].set_ylabel('Number of AML Edges')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(aml_counts):
        axes[1, 1].text(i, v + max(aml_counts)*0.01, f'{v:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/LaunDetection/dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dataset_names, node_counts, edge_counts, aml_rates, aml_counts

def analyze_graph_structure(datasets):
    """Analyze graph structure and connectivity"""
    print("\n🕸️ Analyzing Graph Structure...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Graph Structure Analysis', fontsize=16, fontweight='bold')
    
    dataset_names = []
    density_values = []
    clustering_values = []
    avg_degree_values = []
    component_counts = []
    
    for name, data in datasets.items():
        if isinstance(data, dict) and 'graph' in data:
            graph = data['graph']
            
            # Calculate graph metrics
            density = nx.density(graph)
            clustering = nx.average_clustering(graph)
            avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
            
            # Count connected components
            components = list(nx.connected_components(graph))
            component_count = len(components)
            
            dataset_names.append(name)
            density_values.append(density)
            clustering_values.append(clustering)
            avg_degree_values.append(avg_degree)
            component_counts.append(component_count)
            
            print(f"📊 {name}:")
            print(f"   Density: {density:.4f}")
            print(f"   Clustering: {clustering:.4f}")
            print(f"   Avg Degree: {avg_degree:.2f}")
            print(f"   Components: {component_count}")
    
    # Plot 1: Graph Density
    axes[0, 0].bar(dataset_names, density_values, color='lightblue', alpha=0.7)
    axes[0, 0].set_title('Graph Density per Dataset')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Clustering Coefficient
    axes[0, 1].bar(dataset_names, clustering_values, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Average Clustering Coefficient')
    axes[0, 1].set_ylabel('Clustering Coefficient')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Average Degree
    axes[1, 0].bar(dataset_names, avg_degree_values, color='orange', alpha=0.7)
    axes[1, 0].set_title('Average Node Degree')
    axes[1, 0].set_ylabel('Average Degree')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Connected Components
    axes[1, 1].bar(dataset_names, component_counts, color='purple', alpha=0.7)
    axes[1, 1].set_title('Number of Connected Components')
    axes[1, 1].set_ylabel('Component Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/LaunDetection/graph_structure_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_distributions(datasets):
    """Analyze feature distributions"""
    print("\n📈 Analyzing Feature Distributions...")
    
    for name, data in datasets.items():
        if isinstance(data, dict) and 'node_features' in data and 'edge_features' in data:
            print(f"\n📊 {name} Feature Analysis:")
            
            node_features = data['node_features']
            edge_features = data['edge_features']
            
            print(f"   Node features shape: {node_features.shape}")
            print(f"   Edge features shape: {edge_features.shape}")
            
            # Create feature distribution plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{name} - Feature Distributions', fontsize=16, fontweight='bold')
            
            # Node feature statistics
            node_means = np.mean(node_features, axis=0)
            node_stds = np.std(node_features, axis=0)
            
            axes[0, 0].bar(range(len(node_means)), node_means, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Node Feature Means')
            axes[0, 0].set_xlabel('Feature Index')
            axes[0, 0].set_ylabel('Mean Value')
            
            axes[0, 1].bar(range(len(node_stds)), node_stds, alpha=0.7, color='lightcoral')
            axes[0, 1].set_title('Node Feature Standard Deviations')
            axes[0, 1].set_xlabel('Feature Index')
            axes[0, 1].set_ylabel('Standard Deviation')
            
            # Edge feature statistics
            edge_means = np.mean(edge_features, axis=0)
            edge_stds = np.std(edge_features, axis=0)
            
            axes[1, 0].bar(range(len(edge_means)), edge_means, alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Edge Feature Means')
            axes[1, 0].set_xlabel('Feature Index')
            axes[1, 0].set_ylabel('Mean Value')
            
            axes[1, 1].bar(range(len(edge_stds)), edge_stds, alpha=0.7, color='orange')
            axes[1, 1].set_title('Edge Feature Standard Deviations')
            axes[1, 1].set_xlabel('Feature Index')
            axes[1, 1].set_ylabel('Standard Deviation')
            
            plt.tight_layout()
            plt.savefig(f'/content/drive/MyDrive/LaunDetection/{name}_feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.show()

def create_network_visualization(datasets):
    """Create network visualizations for each dataset"""
    print("\n🕸️ Creating Network Visualizations...")
    
    for name, data in datasets.items():
        if isinstance(data, dict) and 'graph' in data:
            graph = data['graph']
            
            # Create a sample of the graph for visualization (max 1000 nodes)
            if graph.number_of_nodes() > 1000:
                # Get the largest connected component
                components = list(nx.connected_components(graph))
                largest_component = max(components, key=len)
                subgraph = graph.subgraph(largest_component)
                
                # Further sample if still too large
                if subgraph.number_of_nodes() > 1000:
                    nodes_to_keep = list(subgraph.nodes())[:1000]
                    subgraph = subgraph.subgraph(nodes_to_keep)
            else:
                subgraph = graph
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Color nodes by degree
            degrees = dict(subgraph.degree())
            node_colors = [degrees[node] for node in subgraph.nodes()]
            
            # Position nodes using spring layout
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
            
            # Draw the network
            nx.draw(subgraph, pos, 
                   node_color=node_colors,
                   node_size=50,
                   cmap='viridis',
                   alpha=0.7,
                   with_labels=False,
                   edge_color='gray',
                   edge_alpha=0.3)
            
            plt.title(f'{name} - Network Visualization\n(Node color = Degree)', fontsize=14, fontweight='bold')
            plt.colorbar(label='Node Degree')
            plt.axis('off')
            
            plt.savefig(f'/content/drive/MyDrive/LaunDetection/{name}_network_visualization.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ {name}: Network visualization created")

def create_combined_analysis(datasets):
    """Create combined analysis across all datasets"""
    print("\n📊 Creating Combined Analysis...")
    
    # Collect all data
    all_aml_rates = []
    all_edge_counts = []
    all_node_counts = []
    dataset_names = []
    
    for name, data in datasets.items():
        if isinstance(data, dict) and 'graph' in data and 'edge_labels' in data:
            graph = data['graph']
            edge_labels = data['edge_labels']
            
            aml_edges = sum(edge_labels)
            total_edges = len(edge_labels)
            aml_rate = (aml_edges / total_edges) * 100 if total_edges > 0 else 0
            
            dataset_names.append(name)
            all_aml_rates.append(aml_rate)
            all_edge_counts.append(graph.number_of_edges())
            all_node_counts.append(graph.number_of_nodes())
    
    # Create combined visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Combined Multi-Dataset Analysis', fontsize=18, fontweight='bold')
    
    # Plot 1: Dataset sizes comparison
    x = np.arange(len(dataset_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, all_node_counts, width, label='Nodes', alpha=0.8, color='skyblue')
    axes[0, 0].bar(x + width/2, all_edge_counts, width, label='Edges', alpha=0.8, color='lightcoral')
    axes[0, 0].set_title('Dataset Sizes Comparison')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(dataset_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: AML rates
    bars = axes[0, 1].bar(dataset_names, all_aml_rates, color='red', alpha=0.7)
    axes[0, 1].set_title('AML Rates Across Datasets')
    axes[0, 1].set_ylabel('AML Rate (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, rate in zip(bars, all_aml_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       f'{rate:.2f}%', ha='center', va='bottom')
    
    # Plot 3: Edge-to-Node ratio
    edge_node_ratios = [edges/nodes for edges, nodes in zip(all_edge_counts, all_node_counts)]
    axes[1, 0].bar(dataset_names, edge_node_ratios, color='green', alpha=0.7)
    axes[1, 0].set_title('Edge-to-Node Ratio')
    axes[1, 0].set_ylabel('Edges per Node')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Total dataset comparison
    total_edges = sum(all_edge_counts)
    total_nodes = sum(all_node_counts)
    total_aml = sum([rate * edges / 100 for rate, edges in zip(all_aml_rates, all_edge_counts)])
    
    categories = ['Total Nodes', 'Total Edges', 'Total AML Edges']
    values = [total_nodes, total_edges, total_aml]
    colors = ['skyblue', 'lightcoral', 'red']
    
    axes[1, 1].bar(categories, values, color=colors, alpha=0.7)
    axes[1, 1].set_title('Combined Dataset Totals')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + max(values)*0.01, f'{v:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/LaunDetection/combined_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n📊 Combined Dataset Summary:")
    print("=" * 50)
    print(f"Total Nodes: {total_nodes:,}")
    print(f"Total Edges: {total_edges:,}")
    print(f"Total AML Edges: {total_aml:,.0f}")
    print(f"Overall AML Rate: {(total_aml/total_edges)*100:.2f}%")
    print(f"Average Edge-to-Node Ratio: {np.mean(edge_node_ratios):.2f}")

def main():
    """Main visualization function"""
    print("📊 AML Multi-Dataset Preprocessed Data Visualization")
    print("=" * 60)
    
    # Load processed datasets
    datasets = load_processed_datasets()
    
    if not datasets:
        print("❌ No processed datasets found!")
        print("💡 Run preprocessing first: python working_preprocessing.py")
        return
    
    print(f"\n✅ Loaded {len(datasets)} processed datasets")
    
    # Create visualizations
    try:
        # 1. Dataset overview
        dataset_names, node_counts, edge_counts, aml_rates, aml_counts = create_dataset_overview(datasets)
        
        # 2. Graph structure analysis
        analyze_graph_structure(datasets)
        
        # 3. Feature distributions
        analyze_feature_distributions(datasets)
        
        # 4. Network visualizations
        create_network_visualization(datasets)
        
        # 5. Combined analysis
        create_combined_analysis(datasets)
        
        print("\n🎉 All visualizations completed!")
        print("📁 Images saved to: /content/drive/MyDrive/LaunDetection/")
        print("\n📊 Generated Visualizations:")
        print("   • dataset_overview.png - Dataset size comparison")
        print("   • graph_structure_analysis.png - Graph metrics")
        print("   • [Dataset]_feature_distributions.png - Feature analysis")
        print("   • [Dataset]_network_visualization.png - Network plots")
        print("   • combined_analysis.png - Combined analysis")
        
    except Exception as e:
        print(f"❌ Error during visualization: {str(e)}")
        print("💡 Make sure processed data exists and is in correct format")

if __name__ == "__main__":
    main()

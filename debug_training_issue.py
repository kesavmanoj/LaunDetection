#!/usr/bin/env python3
"""
Debug Training Issue for AML Multi-GNN
======================================

This script debugs why the training is getting F1=0.0000.
"""

import torch
import numpy as np
import pickle
import os
from collections import Counter

print("🔍 Debugging Training Issue")
print("=" * 60)

def debug_data_distribution():
    """Debug the data distribution and class imbalance"""
    print("\n📊 Step 1: Analyzing Data Distribution")
    print("-" * 40)
    
    # Load the graph data
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    graph_file = os.path.join(processed_path, "graph_created.pkl")
    
    if not os.path.exists(graph_file):
        print("❌ Graph file not found")
        return
    
    with open(graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    
    print(f"Graph data keys: {list(graph_data.keys())}")
    
    # Check class distribution
    if 'class_distribution' in graph_data:
        class_dist = graph_data['class_distribution']
        print(f"Class distribution: {class_dist}")
        
        if len(class_dist) == 2:
            majority = max(class_dist.values())
            minority = min(class_dist.values())
            imbalance_ratio = majority / minority if minority > 0 else float('inf')
            print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 1000:
                print("⚠️  EXTREME class imbalance detected!")
                print("   This makes training very difficult")
    
    # Check node features
    if 'node_features' in graph_data:
        node_features = graph_data['node_features']
        print(f"Node features type: {type(node_features)}")
        if isinstance(node_features, dict):
            print(f"Number of nodes with features: {len(node_features)}")
            # Check feature values
            sample_features = list(node_features.values())[0]
            print(f"Sample node features: {sample_features}")
            print(f"Feature dimensions: {len(sample_features)}")
    
    # Check edge features
    if 'edge_features' in graph_data:
        edge_features = graph_data['edge_features']
        print(f"Edge features type: {type(edge_features)}")
        if isinstance(edge_features, dict):
            print(f"Number of edges with features: {len(edge_features)}")
            # Check feature values
            sample_features = list(edge_features.values())[0]
            print(f"Sample edge features: {sample_features}")
            print(f"Feature dimensions: {len(sample_features)}")

def debug_networkx_graph():
    """Debug the NetworkX graph structure"""
    print("\n🕸️  Step 2: Analyzing NetworkX Graph")
    print("-" * 40)
    
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    graph_file = os.path.join(processed_path, "graph_created.pkl")
    
    with open(graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    
    if 'graph' in graph_data:
        import networkx as nx
        graph = graph_data['graph']
        
        print(f"NetworkX graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Graph density: {nx.density(graph):.4f}")
        
        # Check if graph is connected
        if nx.is_connected(graph):
            print("✅ Graph is connected")
        else:
            print("⚠️  Graph is not connected")
            print(f"   Number of connected components: {nx.number_connected_components(graph)}")
        
        # Check node features
        if graph.number_of_nodes() > 0:
            first_node = list(graph.nodes())[0]
            if 'features' in graph.nodes[first_node]:
                features = graph.nodes[first_node]['features']
                print(f"Node features for {first_node}: {features}")
                print(f"Feature dimensions: {len(features)}")
            else:
                print("⚠️  No node features found")
        
        # Check edge features
        if graph.number_of_edges() > 0:
            first_edge = list(graph.edges())[0]
            if 'features' in graph.edges[first_edge]:
                features = graph.edges[first_edge]['features']
                print(f"Edge features for {first_edge}: {features}")
                print(f"Feature dimensions: {len(features)}")
            else:
                print("⚠️  No edge features found")
        
        # Check for any AML nodes
        aml_nodes = 0
        for node in graph.nodes():
            if 'features' in graph.nodes[node]:
                features = graph.nodes[node]['features']
                # Check if any feature suggests AML (simplified check)
                if any(f > 0.5 for f in features):
                    aml_nodes += 1
        
        print(f"Potential AML nodes: {aml_nodes}/{graph.number_of_nodes()}")
    
    else:
        print("❌ No NetworkX graph found")

def debug_training_data():
    """Debug the training data creation"""
    print("\n🎯 Step 3: Analyzing Training Data")
    print("-" * 40)
    
    # Simulate the training data creation process
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    graph_file = os.path.join(processed_path, "graph_created.pkl")
    
    with open(graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    
    if 'graph' in graph_data:
        import networkx as nx
        graph = graph_data['graph']
        
        # Check how many training graphs we can create
        training_graphs = []
        
        for i in range(min(20, graph.number_of_nodes())):
            center_node = list(graph.nodes())[i]
            neighbor_nodes = [center_node]
            
            # Add neighbors
            for edge in graph.edges():
                if edge[0] == center_node:
                    neighbor_nodes.append(edge[1])
                elif edge[1] == center_node:
                    neighbor_nodes.append(edge[0])
            
            neighbor_nodes = neighbor_nodes[:5]
            
            if len(neighbor_nodes) > 0:
                # Create subgraph
                subgraph = graph.subgraph(neighbor_nodes)
                if subgraph.number_of_edges() > 0:
                    training_graphs.append(subgraph)
                else:
                    # Create self-loop graph
                    training_graphs.append(graph.subgraph([center_node]))
        
        print(f"Created {len(training_graphs)} training graphs")
        
        # Check graph sizes
        graph_sizes = [g.number_of_nodes() for g in training_graphs]
        print(f"Graph sizes: min={min(graph_sizes)}, max={max(graph_sizes)}, avg={np.mean(graph_sizes):.2f}")
        
        # Check connectivity
        connected_graphs = sum(1 for g in training_graphs if nx.is_connected(g))
        print(f"Connected graphs: {connected_graphs}/{len(training_graphs)}")
        
        # Check for any AML labels
        aml_graphs = 0
        for graph in training_graphs:
            # Simple check for potential AML
            has_aml = False
            for node in graph.nodes():
                if 'features' in graph.nodes[node]:
                    features = graph.nodes[node]['features']
                    if any(f > 0.5 for f in features):
                        has_aml = True
                        break
            if has_aml:
                aml_graphs += 1
        
        print(f"Graphs with potential AML: {aml_graphs}/{len(training_graphs)}")

def suggest_solutions():
    """Suggest solutions based on the analysis"""
    print("\n💡 Suggested Solutions")
    print("-" * 40)
    
    print("Based on the analysis, here are the recommended solutions:")
    print()
    print("1. 🔧 Fix Class Imbalance:")
    print("   - The 9,999:1 ratio is too extreme for training")
    print("   - Use SMOTE or other oversampling techniques")
    print("   - Consider using focal loss instead of cross-entropy")
    print()
    print("2. 📊 Increase Dataset Size:")
    print("   - 78 nodes is too small for meaningful training")
    print("   - Increase sample_size in preprocessing to 50,000+")
    print("   - Use more transactions and accounts")
    print()
    print("3. 🎯 Improve Graph Structure:")
    print("   - The graph has very low density (0.0007)")
    print("   - Add more edges or use different graph construction")
    print("   - Consider using transaction-based edges")
    print()
    print("4. 🚀 Use the Working 4-Node Model:")
    print("   - Your 4-node model achieved F1=1.0000")
    print("   - Scale it up gradually: 4 → 20 → 100 → 1000 nodes")
    print("   - This approach is more reliable")
    print()
    print("5. 🔄 Alternative Training Strategy:")
    print("   - Use node-level classification instead of graph-level")
    print("   - Use edge-level classification for transactions")
    print("   - Implement multi-task learning")

def main():
    """Main debugging function"""
    print("🔍 Debugging Training Issue")
    print("=" * 60)
    
    # Debug data distribution
    debug_data_distribution()
    
    # Debug NetworkX graph
    debug_networkx_graph()
    
    # Debug training data
    debug_training_data()
    
    # Suggest solutions
    suggest_solutions()
    
    print("\n📋 Summary")
    print("-" * 40)
    print("The main issues are:")
    print("1. Extreme class imbalance (9,999:1 ratio)")
    print("2. Small dataset (78 nodes, 2 edges)")
    print("3. Low graph density (0.0007)")
    print("4. No clear AML patterns in the data")
    print()
    print("Recommended next steps:")
    print("1. Use the working 4-node model as a starting point")
    print("2. Gradually scale up the dataset size")
    print("3. Implement better class imbalance handling")
    print("4. Consider alternative graph construction methods")

if __name__ == "__main__":
    main()

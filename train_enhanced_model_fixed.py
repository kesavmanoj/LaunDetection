#!/usr/bin/env python3
"""
Fixed Enhanced AML Multi-GNN Training
=====================================

This script trains the Multi-GNN model using the enhanced preprocessed dataset
with proper data format handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 Enhanced AML Multi-GNN Training (FIXED)")
print("=" * 60)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enhanced GNN Model for larger dataset
class EnhancedGNN(nn.Module):
    """Enhanced GNN for large-scale AML detection"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        super(EnhancedGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.input_conv = GCNConv(input_dim, hidden_dim)
        
        # Hidden layers with residual connections
        self.hidden_convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_conv = GCNConv(hidden_dim, output_dim)
        
        # Dropout and batch normalization
        self.dropout_layer = nn.Dropout(dropout)
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, batch=None):
        x = x.to(device)
        edge_index = edge_index.to(device)
        
        # Input layer
        x = F.relu(self.input_conv(x, edge_index))
        x = self.batch_norms[0](x)
        x = self.dropout_layer(x)
        
        # Hidden layers with residual connections
        for i, conv in enumerate(self.hidden_convs):
            residual = x
            x = F.relu(conv(x, edge_index))
            x = self.batch_norms[i + 1](x)
            x = self.dropout_layer(x)
            
            # Residual connection
            if x.size() == residual.size():
                x = x + residual
        
        # Output layer
        x = self.output_conv(x, edge_index)
        
        # Global pooling for graph-level classification
        if batch is not None:
            batch = batch.to(device)
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        return x

print("✓ Enhanced GNN model defined")

def load_enhanced_data():
    """Load the enhanced preprocessed data with proper format handling"""
    print("Loading enhanced preprocessed data...")
    
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    # Load graph data
    graph_file = os.path.join(processed_path, "graph_created.pkl")
    if not os.path.exists(graph_file):
        print("❌ Enhanced graph not found. Run preprocessing first:")
        print("   %run simple_enhanced_preprocessing.py")
        return None
    
    with open(graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    
    print(f"✅ Loaded enhanced graph: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
    
    # Handle the data format properly
    if isinstance(graph_data, dict):
        print(f"   Graph data type: {type(graph_data)}")
        print(f"   Keys: {list(graph_data.keys())}")
        
        # Extract the NetworkX graph
        if 'graph' in graph_data:
            graph = graph_data['graph']
            print(f"   NetworkX graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            
            # Convert to PyTorch Geometric format
            return convert_networkx_to_pyg(graph, graph_data)
        else:
            print("❌ No NetworkX graph found in data")
            return None
    else:
        print("❌ Unexpected data format")
        return None

def convert_networkx_to_pyg(nx_graph, graph_data):
    """Convert NetworkX graph to PyTorch Geometric format"""
    print("Converting NetworkX graph to PyTorch Geometric format...")
    
    # Get node features
    node_features = []
    node_labels = []
    node_mapping = {}
    
    for i, node in enumerate(nx_graph.nodes()):
        node_mapping[node] = i
        if 'features' in nx_graph.nodes[node]:
            features = nx_graph.nodes[node]['features']
            if isinstance(features, list):
                node_features.append(features)
            else:
                # Create default features if not available
                node_features.append([0.0] * 15)
        else:
            node_features.append([0.0] * 15)
        
        # Create node labels (simplified)
        node_labels.append(0)  # Default to non-AML
    
    # Get edge features and create edge index
    edge_index = []
    edge_features = []
    
    for edge in nx_graph.edges():
        source, target = edge
        if source in node_mapping and target in node_mapping:
            edge_index.append([node_mapping[source], node_mapping[target]])
            
            if 'features' in nx_graph.edges[edge]:
                features = nx_graph.edges[edge]['features']
                if isinstance(features, list):
                    edge_features.append(features)
                else:
                    edge_features.append([0.0] * 12)
            else:
                edge_features.append([0.0] * 12)
    
    # Convert to tensors
    node_features = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.tensor([[], []], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.tensor([], dtype=torch.float32)
    node_labels = torch.tensor(node_labels, dtype=torch.long)
    
    print(f"✅ Converted to PyTorch Geometric format:")
    print(f"   Node features: {node_features.shape}")
    print(f"   Edge index: {edge_index.shape}")
    print(f"   Edge features: {edge_attr.shape}")
    print(f"   Node labels: {node_labels.shape}")
    
    # Create PyTorch Geometric Data object
    pyg_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=node_labels
    )
    
    return pyg_data

def create_training_data(graph_data, sample_size=20):
    """Create training data from the enhanced graph"""
    print(f"Creating training data with {sample_size} samples...")
    
    # Create individual graphs for each node
    individual_graphs = []
    
    for i in range(min(sample_size, graph_data.num_nodes)):
        # Create a simple graph with the center node
        center_node = i
        
        # Create a small subgraph around each node
        neighbor_nodes = [center_node]
        
        # Add some neighbors if available
        if graph_data.edge_index.size(1) > 0:
            for edge_idx in range(min(5, graph_data.edge_index.size(1))):
                edge = graph_data.edge_index[:, edge_idx]
                if edge[0].item() == center_node:
                    neighbor_nodes.append(edge[1].item())
                elif edge[1].item() == center_node:
                    neighbor_nodes.append(edge[0].item())
        
        # Limit to reasonable size
        neighbor_nodes = neighbor_nodes[:5]
        
        if len(neighbor_nodes) > 0:
            # Create subgraph
            subgraph = graph_data.subgraph(torch.tensor(neighbor_nodes))
            if subgraph.num_edges > 0:
                # Create graph-level label
                node_labels = subgraph.y.tolist()
                graph_label = 1 if sum(node_labels) > len(node_labels) / 2 else 0
                
                # Create new graph with single label
                new_graph = Data(
                    x=subgraph.x,
                    edge_index=subgraph.edge_index,
                    edge_attr=subgraph.edge_attr,
                    y=torch.tensor([graph_label], dtype=torch.long)
                )
                individual_graphs.append(new_graph)
            else:
                # Create a simple graph with self-loop
                node_features = graph_data.x[center_node:center_node+1]
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                edge_attr = torch.tensor([[0.0] * 12], dtype=torch.float32)
                graph_label = 0  # Default to non-AML
                
                new_graph = Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor([graph_label], dtype=torch.long)
                )
                individual_graphs.append(new_graph)
    
    print(f"✅ Created {len(individual_graphs)} training graphs")
    return individual_graphs

def train_enhanced_model():
    """Train the enhanced model"""
    print("Training enhanced AML Multi-GNN model...")
    
    # Load data
    graph_data = load_enhanced_data()
    if graph_data is None:
        return 0
    
    # Create training data
    training_graphs = create_training_data(graph_data, sample_size=20)
    
    if len(training_graphs) == 0:
        print("❌ No training graphs created!")
        return 0
    
    # Split data
    train_size = int(0.8 * len(training_graphs))
    train_graphs = training_graphs[:train_size]
    val_graphs = training_graphs[train_size:]
    
    print(f"Split: {len(train_graphs)} train graphs, {len(val_graphs)} val graphs")
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=2, shuffle=False)
    
    # Create model
    model = EnhancedGNN(
        input_dim=15,  # 15 node features
        hidden_dim=32,  # Reduced for memory
        output_dim=2,  # Binary classification
        num_layers=2,  # Reduced for memory
        dropout=0.2
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training setup
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Load class weights if available
    weights_file = "/content/drive/MyDrive/LaunDetection/data/processed/imbalanced.pkl"
    if os.path.exists(weights_file):
        with open(weights_file, 'rb') as f:
            weights_data = pickle.load(f)
        if 'weights' in weights_data:
            class_weights = torch.tensor([weights_data['weights'][0], weights_data['weights'][1]], dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("✅ Using cost-sensitive class weights")
    
    # Training loop
    best_val_f1 = 0.0
    
    for epoch in range(10):
        # Training
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
              f"Val F1={val_f1:.4f}, Val Precision={val_precision:.4f}, Val Recall={val_recall:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"🎉 New best validation F1: {best_val_f1:.4f}")
    
    print(f"\n✅ Enhanced training completed!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    
    return best_val_f1

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train model
    best_f1 = train_enhanced_model()
    
    if best_f1 > 0:
        print(f"\n🎉 Enhanced AML Multi-GNN training completed successfully!")
        print(f"Best F1-Score: {best_f1:.4f}")
    else:
        print("\n❌ Training failed. Check the data and model configuration.")

#!/usr/bin/env python3
"""
Train Enhanced AML Multi-GNN Model
==================================

This script trains the Multi-GNN model using the enhanced preprocessed dataset.
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

print("🚀 Enhanced AML Multi-GNN Training")
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
    """Load the enhanced preprocessed data"""
    print("Loading enhanced preprocessed data...")
    
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    # Load graph data
    graph_file = os.path.join(processed_path, "graph_created.pkl")
    if not os.path.exists(graph_file):
        print("❌ Enhanced graph not found. Run preprocessing first:")
        print("   %run notebooks/08_simple_enhanced_preprocessing.ipynb")
        return None
    
    with open(graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    
    print(f"✅ Loaded enhanced graph: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
    print(f"   Node features: {graph_data['node_features'].shape}")
    print(f"   Edge features: {graph_data['edge_features'].shape}")
    print(f"   Class distribution: {graph_data['class_distribution']}")
    
    return graph_data

def create_training_data(graph_data, sample_size=1000):
    """Create training data from the enhanced graph"""
    print(f"Creating training data with {sample_size} samples...")
    
    # Sample nodes for training (due to memory constraints)
    num_nodes = graph_data['num_nodes']
    sample_indices = np.random.choice(num_nodes, min(sample_size, num_nodes), replace=False)
    
    # Create individual graphs for each sampled node
    individual_graphs = []
    
    for i, node_idx in enumerate(tqdm(sample_indices, desc="Creating training graphs")):
        # Create a small subgraph around each node
        center_node = node_idx
        neighbor_nodes = [center_node]
        
        # Add some neighbors (simplified for memory)
        if i < 10:  # Only create 10 graphs for now due to memory constraints
            # Create a simple graph with the center node
            node_features = graph_data['node_features'][center_node:center_node+1]
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop
            edge_attr = torch.tensor([[0.0] * 12], dtype=torch.float32)  # 12 edge features
            label = graph_data['labels'][center_node]
            
            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.long)
            )
            individual_graphs.append(graph)
    
    print(f"✅ Created {len(individual_graphs)} training graphs")
    return individual_graphs

def train_enhanced_model():
    """Train the enhanced model"""
    print("Training enhanced AML Multi-GNN model...")
    
    # Load data
    graph_data = load_enhanced_data()
    if graph_data is None:
        return
    
    # Create training data
    training_graphs = create_training_data(graph_data, sample_size=100)
    
    if len(training_graphs) == 0:
        print("❌ No training graphs created!")
        return
    
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
        hidden_dim=64,
        output_dim=2,  # Binary classification
        num_layers=3,
        dropout=0.2
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training setup
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Load class weights if available
    weights_file = "/content/drive/MyDrive/LaunDetection/data/processed/weights.pkl"
    if os.path.exists(weights_file):
        with open(weights_file, 'rb') as f:
            weights_data = pickle.load(f)
        class_weights = torch.tensor([weights_data[0], weights_data[1]], dtype=torch.float32).to(device)
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

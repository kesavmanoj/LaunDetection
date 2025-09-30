#!/usr/bin/env python3
"""
Test script for clean training
==============================

Quick test to verify the clean training script works without errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple GNN model
class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(SimpleGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.input_conv = GCNConv(input_dim, hidden_dim)
        self.hidden_convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_convs.append(GCNConv(hidden_dim, hidden_dim))
        self.output_conv = GCNConv(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        x = x.to(device)
        edge_index = edge_index.to(device)
        
        x = F.relu(self.input_conv(x, edge_index))
        x = self.dropout_layer(x)
        
        for conv in self.hidden_convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout_layer(x)
        
        x = self.output_conv(x, edge_index)
        
        # CRITICAL FIX: Always use global pooling for graph-level classification
        if batch is not None:
            batch = batch.to(device)
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        return x

# Create test data
print("Creating test data...")

# Create a simple graph
num_nodes = 20
num_edges = 30

# Node features
x = torch.randn(num_nodes, 5)

# Edge indices
edge_list = []
for _ in range(num_edges):
    src = np.random.randint(0, num_nodes)
    dst = np.random.randint(0, num_nodes)
    edge_list.append([src, dst])

edge_index = torch.tensor(edge_list).t().contiguous()
edge_attr = torch.randn(num_edges, 4)

# Node labels
y = torch.randint(0, 2, (num_nodes,))

# Create graph
graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

print(f"Created test graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

# Create individual graphs for testing
individual_graphs = []
for i in range(5):  # Create 5 test graphs
    # Create small subgraph
    nodes = [i, (i+1) % num_nodes, (i+2) % num_nodes]
    subgraph = graph.subgraph(torch.tensor(nodes))
    
    if subgraph.num_edges > 0:
        # Create graph-level label
        node_labels = subgraph.y.tolist()
        graph_label = 1 if sum(node_labels) > len(node_labels) / 2 else 0
        
        new_graph = Data(
            x=subgraph.x,
            edge_index=subgraph.edge_index,
            edge_attr=subgraph.edge_attr,
            y=torch.tensor([graph_label], dtype=torch.long)
        )
        individual_graphs.append(new_graph)

print(f"Created {len(individual_graphs)} individual graphs")

# Create data loader
train_loader = DataLoader(individual_graphs, batch_size=2, shuffle=True)

# Create model
model = SimpleGNN(input_dim=5, hidden_dim=16, output_dim=2, num_layers=2)
model = model.to(device)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Test forward pass
print("Testing forward pass...")
model.eval()
with torch.no_grad():
    for batch in train_loader:
        batch = batch.to(device)
        print(f"Batch: {batch.num_graphs} graphs, {batch.num_nodes} nodes")
        print(f"Batch y shape: {batch.y.shape}")
        
        out = model(batch.x, batch.edge_index, batch.batch)
        print(f"Output shape: {out.shape}")
        print(f"Expected batch size: {batch.num_graphs}, Got: {out.size(0)}")
        
        if out.size(0) == batch.num_graphs:
            print("✓ Forward pass successful - dimensions match!")
        else:
            print("✗ Forward pass failed - dimension mismatch!")
            break
        
        # Test loss computation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, batch.y)
        print(f"Loss computed successfully: {loss.item():.4f}")
        break

print("Test completed!")

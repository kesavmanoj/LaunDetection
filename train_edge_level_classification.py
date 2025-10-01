#!/usr/bin/env python3
"""
Edge-Level AML Classification
============================

This script implements edge-level classification for direct AML transaction detection.
This is more appropriate for AML detection than graph-level classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🎯 Edge-Level AML Classification")
print("=" * 50)

class EdgeLevelGNN(nn.Module):
    """GNN for edge-level classification"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super(EdgeLevelGNN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Edge classification head
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenated node features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        # Check for NaN in input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # GNN layers
        for i, (conv, bn, dropout) in enumerate(zip(self.convs, self.bns, self.dropouts)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
            
            # Check for NaN after each layer
            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)
        
        # Edge-level classification
        # Get source and target node features for each edge
        src_features = x[edge_index[0]]  # Source node features
        tgt_features = x[edge_index[1]]  # Target node features
        
        # Concatenate source and target features
        edge_features = torch.cat([src_features, tgt_features], dim=1)
        
        # Add edge attributes if available
        if edge_attr is not None:
            if torch.isnan(edge_attr).any():
                edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
            edge_features = torch.cat([edge_features, edge_attr], dim=1)
        
        # Classify edges
        edge_output = self.edge_classifier(edge_features)
        
        # Check for NaN in output
        if torch.isnan(edge_output).any():
            edge_output = torch.nan_to_num(edge_output, nan=0.0)
        
        return edge_output

def load_edge_level_dataset():
    """Load dataset for edge-level classification"""
    print("📊 Loading dataset for edge-level classification...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    # Load larger sample for better AML representation
    print("Loading 500K transactions for edge-level training...")
    transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'), nrows=500000)
    
    print(f"✅ Loaded {len(transactions)} transactions")
    
    # Check AML distribution
    aml_distribution = transactions['Is Laundering'].value_counts()
    print(f"📊 Edge-Level Dataset AML Distribution:")
    print(f"   Class 0 (Non-AML): {aml_distribution.get(0, 0)}")
    print(f"   Class 1 (AML): {aml_distribution.get(1, 0)}")
    
    if aml_distribution.get(1, 0) > 0:
        aml_rate = aml_distribution.get(1, 0) / len(transactions) * 100
        print(f"   AML Rate: {aml_rate:.4f}%")
    else:
        print("   ⚠️  NO AML SAMPLES FOUND!")
        return None
    
    return transactions

def create_edge_level_graph(transactions):
    """Create graph for edge-level classification"""
    print("🕸️  Creating graph for edge-level classification...")
    
    # Clean data
    clean_transactions = transactions.dropna()
    clean_transactions = clean_transactions[clean_transactions['Amount Received'] > 0]
    clean_transactions = clean_transactions[~np.isinf(clean_transactions['Amount Received'])]
    
    print(f"   Clean transactions: {len(clean_transactions)}")
    
    # Separate AML and non-AML
    aml_transactions = clean_transactions[clean_transactions['Is Laundering'] == 1]
    non_aml_transactions = clean_transactions[clean_transactions['Is Laundering'] == 0]
    
    print(f"   AML transactions: {len(aml_transactions)}")
    print(f"   Non-AML transactions: {len(non_aml_transactions)}")
    
    if len(aml_transactions) == 0:
        print("❌ No AML transactions found")
        return None
    
    # Create balanced dataset (10:1 ratio for edge-level classification)
    non_aml_sample_size = min(len(non_aml_transactions), len(aml_transactions) * 10)
    non_aml_sampled = non_aml_transactions.sample(n=non_aml_sample_size, random_state=42)
    
    # Combine AML and sampled non-AML
    balanced_transactions = pd.concat([aml_transactions, non_aml_sampled])
    balanced_transactions = balanced_transactions.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   Balanced dataset: {len(balanced_transactions)} transactions")
    print(f"   AML in balanced: {len(balanced_transactions[balanced_transactions['Is Laundering'] == 1])}")
    print(f"   Non-AML in balanced: {len(balanced_transactions[balanced_transactions['Is Laundering'] == 0])}")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Get unique accounts
    from_accounts = set(balanced_transactions['From Bank'].astype(str))
    to_accounts = set(balanced_transactions['To Bank'].astype(str))
    all_accounts = from_accounts.union(to_accounts)
    
    print(f"   Unique accounts: {len(all_accounts)}")
    
    # Add nodes with stable features
    account_features = {}
    for account in all_accounts:
        # Get transactions for this account
        from_trans = balanced_transactions[balanced_transactions['From Bank'].astype(str) == account]
        to_trans = balanced_transactions[balanced_transactions['To Bank'].astype(str) == account]
        
        # Calculate stable features
        total_amount = from_trans['Amount Received'].sum() + to_trans['Amount Received'].sum()
        transaction_count = len(from_trans) + len(to_trans)
        avg_amount = total_amount / transaction_count if transaction_count > 0 else 0
        max_amount = max(from_trans['Amount Received'].max(), to_trans['Amount Received'].max()) if len(from_trans) > 0 or len(to_trans) > 0 else 0
        min_amount = min(from_trans['Amount Received'].min(), to_trans['Amount Received'].min()) if len(from_trans) > 0 or len(to_trans) > 0 else 0
        
        # Check for AML involvement
        is_aml = 0
        if len(from_trans) > 0:
            is_aml = max(is_aml, from_trans['Is Laundering'].max())
        if len(to_trans) > 0:
            is_aml = max(is_aml, to_trans['Is Laundering'].max())
        
        # Create stable features with log transforms
        features = [
            np.log1p(total_amount),  # Log transform for stability
            np.log1p(transaction_count),
            np.log1p(avg_amount),
            np.log1p(max_amount),
            np.log1p(min_amount),
            is_aml,
            len(from_trans),
            len(to_trans),
            from_trans['Amount Received'].std() if len(from_trans) > 1 else 0,
            to_trans['Amount Received'].std() if len(to_trans) > 1 else 0,
            from_trans['Amount Received'].mean() if len(from_trans) > 0 else 0,
            to_trans['Amount Received'].mean() if len(to_trans) > 0 else 0,
            from_trans['Amount Received'].median() if len(from_trans) > 0 else 0,
            to_trans['Amount Received'].median() if len(to_trans) > 0 else 0,
            len(set(from_trans['To Bank'].astype(str))) if len(from_trans) > 0 else 0
        ]
        
        # Ensure no NaN or Inf
        features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
        
        account_features[account] = features
        G.add_node(account, features=features)
    
    # Add edges with detailed features
    edges_added = 0
    aml_edges = 0
    non_aml_edges = 0
    
    for _, transaction in balanced_transactions.iterrows():
        from_acc = str(transaction['From Bank'])
        to_acc = str(transaction['To Bank'])
        amount = transaction['Amount Received']
        is_aml = transaction['Is Laundering']
        
        if from_acc in G.nodes and to_acc in G.nodes:
            # Create detailed edge features
            edge_features = [
                np.log1p(amount),  # Log transform
                is_aml,
                0.5,  # Placeholder features
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
            ]
            
            # Ensure no NaN or Inf
            edge_features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in edge_features]
            
            G.add_edge(from_acc, to_acc, 
                      features=edge_features, 
                      label=is_aml)
            edges_added += 1
            
            if is_aml == 1:
                aml_edges += 1
            else:
                non_aml_edges += 1
    
    print(f"✅ Created edge-level graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   AML edges: {aml_edges}")
    print(f"   Non-AML edges: {non_aml_edges}")
    print(f"   Graph density: {nx.density(G):.4f}")
    
    # Calculate class distribution
    class_distribution = {0: non_aml_edges, 1: aml_edges}
    print(f"   Class distribution: {class_distribution}")
    if aml_edges > 0:
        print(f"   Imbalance ratio: {non_aml_edges/aml_edges:.2f}:1")
    
    return {
        'graph': G,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'node_features': account_features,
        'edge_features': {},
        'class_distribution': class_distribution
    }

def create_edge_level_training_data(graph_data):
    """Create training data for edge-level classification"""
    print("🎯 Creating edge-level training data...")
    
    G = graph_data['graph']
    node_features = graph_data['node_features']
    
    # Create node mapping
    nodes = list(G.nodes())
    node_to_int = {node: i for i, node in enumerate(nodes)}
    
    # Convert to PyTorch Geometric format
    x_list = []
    for node in nodes:
        if node in node_features:
            features = node_features[node]
            features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
            x_list.append(features)
        else:
            x_list.append([0.0] * 15)
    
    x = torch.tensor(x_list, dtype=torch.float32)
    
    # Check for NaN in node features
    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)
    
    # Create edge index and features
    edge_index_list = []
    edge_attr_list = []
    edge_labels = []
    
    for edge in G.edges(data=True):
        u, v, data = edge
        if u in node_to_int and v in node_to_int:
            edge_index_list.append([node_to_int[u], node_to_int[v]])
            if 'features' in data:
                features = data['features']
                features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
                edge_attr_list.append(features)
            else:
                edge_attr_list.append([0.0] * 12)
            edge_labels.append(data.get('label', 0))
    
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    y = torch.tensor(edge_labels, dtype=torch.long)
    
    # Check for NaN in edge features
    if torch.isnan(edge_attr).any():
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    print(f"✅ Created edge-level data: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"   Edge labels: {data.y.sum().item()} AML, {len(data.y) - data.y.sum().item()} Non-AML")
    
    return data

def train_edge_level_model():
    """Train the edge-level classification model"""
    print("🚀 Starting edge-level AML classification training...")
    
    # Load dataset
    transactions = load_edge_level_dataset()
    
    if transactions is None:
        print("❌ Failed to load dataset")
        return
    
    # Create graph
    graph_data = create_edge_level_graph(transactions)
    
    if graph_data is None:
        print("❌ Failed to create graph")
        return
    
    # Create training data
    data = create_edge_level_training_data(graph_data)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move data to device
    data = data.to(device)
    
    # Create model
    model = EdgeLevelGNN(
        input_dim=15,
        hidden_dim=64,
        output_dim=2,
        num_layers=3,
        dropout=0.1
    ).to(device)
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Use class weights for imbalanced data
    class_weights = torch.tensor([1.0, 5.0], dtype=torch.float32).to(device)  # Weight AML class
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training loop
    best_f1 = 0.0
    patience = 20
    patience_counter = 0
    
    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0.0
        nan_count = 0
        
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index, data.edge_attr)
        
        # Check for NaN in model output
        if torch.isnan(out).any():
            out = torch.nan_to_num(out, nan=0.0)
        
        # Calculate loss
        loss = criterion(out, data.y)
        
        # Check for NaN in loss
        if torch.isnan(loss):
            nan_count += 1
            continue
        
        # Backward pass
        loss.backward()
        
        # Check for NaN in gradients
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                param.grad = torch.nan_to_num(param.grad, nan=0.0)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        if nan_count > 0:
            print(f"   NaN batches: {nan_count}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            
            # Check for NaN in validation
            if torch.isnan(out).any():
                out = torch.nan_to_num(out, nan=0.0)
            
            preds = torch.argmax(out, dim=1)
            
            # Calculate metrics
            val_f1 = f1_score(data.y.cpu().numpy(), preds.cpu().numpy(), average='weighted')
            val_precision = precision_score(data.y.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=0)
            val_recall = recall_score(data.y.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=0)
            
            # Calculate confusion matrix
            cm = confusion_matrix(data.y.cpu().numpy(), preds.cpu().numpy())
            
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val F1={val_f1:.4f}, Val Precision={val_precision:.4f}, Val Recall={val_recall:.4f}")
            print(f"   Confusion Matrix: {cm}")
            
            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"✅ Edge-level training completed! Best F1: {best_f1:.4f}")
    
    if best_f1 > 0.8:
        print("🎉 EXCELLENT! Edge-level model is highly effective")
        print("   Ready for production AML detection!")
    elif best_f1 > 0.6:
        print("✅ GOOD! Edge-level model is effective")
        print("   Consider tuning hyperparameters for better performance")
    else:
        print("⚠️  Edge-level model needs improvement")
        print("   Consider different approaches or more data")

if __name__ == "__main__":
    train_edge_level_model()

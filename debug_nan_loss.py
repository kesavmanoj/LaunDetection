#!/usr/bin/env python3
"""
Debug and Fix NaN Loss Issue
============================

This script identifies and fixes the root causes of NaN loss in training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import subgraph
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🔍 Debug and Fix NaN Loss Issue")
print("=" * 60)

class DebugFocalLoss(nn.Module):
    """Debug Focal Loss with detailed logging"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(DebugFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.debug_count = 0

    def forward(self, inputs, targets):
        self.debug_count += 1
        
        # Check for NaN in inputs
        if torch.isnan(inputs).any():
            print(f"⚠️  NaN detected in inputs at step {self.debug_count}")
            print(f"   Input range: [{inputs.min():.6f}, {inputs.max():.6f}]")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Check for extreme values
        if torch.isinf(inputs).any():
            print(f"⚠️  Inf detected in inputs at step {self.debug_count}")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Use stable cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Check for NaN in CE loss
        if torch.isnan(ce_loss).any():
            print(f"⚠️  NaN in CE loss at step {self.debug_count}")
            print(f"   CE loss range: [{ce_loss.min():.6f}, {ce_loss.max():.6f}]")
            ce_loss = torch.nan_to_num(ce_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Clamp CE loss to prevent numerical issues
        ce_loss = torch.clamp(ce_loss, min=1e-8, max=10.0)
        
        # Calculate pt with stability
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=1e-8, max=1-1e-8)
        
        # Check for NaN in pt
        if torch.isnan(pt).any():
            print(f"⚠️  NaN in pt at step {self.debug_count}")
            print(f"   pt range: [{pt.min():.6f}, {pt.max():.6f}]")
            pt = torch.nan_to_num(pt, nan=0.5, posinf=1.0, neginf=0.0)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        # Check for NaN in focal loss
        if torch.isnan(focal_loss).any():
            print(f"⚠️  NaN in focal loss at step {self.debug_count}")
            print(f"   Focal loss range: [{focal_loss.min():.6f}, {focal_loss.max():.6f}]")
            focal_loss = torch.nan_to_num(focal_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Clamp final loss
        focal_loss = torch.clamp(focal_loss, min=1e-8, max=10.0)
        
        if self.reduction == 'mean':
            result = focal_loss.mean()
        elif self.reduction == 'sum':
            result = focal_loss.sum()
        else:
            result = focal_loss
        
        # Final NaN check
        if torch.isnan(result):
            print(f"❌ Final NaN in loss at step {self.debug_count}")
            result = torch.tensor(0.0, device=result.device, requires_grad=True)
        
        return result

class StableLoss(nn.Module):
    """Stable loss function for extreme imbalance"""
    def __init__(self, reduction='mean'):
        super(StableLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Use standard cross entropy with class weights
        # This is more stable than Focal Loss for extreme imbalance
        return F.cross_entropy(inputs, targets, reduction=self.reduction)

class RobustGNN(nn.Module):
    """Robust GNN with gradient clipping and stability"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(RobustGNN, self).__init__()
        
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
        
        # Output layer with stability
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        # Check for NaN in input
        if torch.isnan(x).any():
            print("⚠️  NaN in input features")
            x = torch.nan_to_num(x, nan=0.0)
        
        # GNN layers with stability
        for i, (conv, bn, dropout) in enumerate(zip(self.convs, self.bns, self.dropouts)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
            
            # Check for NaN after each layer
            if torch.isnan(x).any():
                print(f"⚠️  NaN after layer {i}")
                x = torch.nan_to_num(x, nan=0.0)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Check for NaN before classification
        if torch.isnan(x).any():
            print("⚠️  NaN before classification")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Classification
        output = self.classifier(x)
        
        # Check for NaN in output
        if torch.isnan(output).any():
            print("⚠️  NaN in model output")
            output = torch.nan_to_num(output, nan=0.0)
        
        return output

def debug_data_quality():
    """Debug data quality issues that could cause NaN loss"""
    print("🔍 Debugging data quality...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    # Load a small sample for debugging
    transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'), nrows=10000)
    
    print(f"✅ Loaded {len(transactions)} transactions for debugging")
    
    # Check for NaN values in data
    print("\n📊 Data Quality Check:")
    print(f"   NaN values in Amount Received: {transactions['Amount Received'].isna().sum()}")
    print(f"   NaN values in Is Laundering: {transactions['Is Laundering'].isna().sum()}")
    print(f"   NaN values in From Bank: {transactions['From Bank'].isna().sum()}")
    print(f"   NaN values in To Bank: {transactions['To Bank'].isna().sum()}")
    
    # Check for extreme values
    print(f"\n📈 Value Range Analysis:")
    print(f"   Amount Received range: [{transactions['Amount Received'].min():.2f}, {transactions['Amount Received'].max():.2f}]")
    print(f"   Amount Received mean: {transactions['Amount Received'].mean():.2f}")
    print(f"   Amount Received std: {transactions['Amount Received'].std():.2f}")
    
    # Check for infinite values
    print(f"\n🔍 Infinite Value Check:")
    print(f"   Inf values in Amount Received: {np.isinf(transactions['Amount Received']).sum()}")
    
    # Check class distribution
    aml_dist = transactions['Is Laundering'].value_counts()
    print(f"\n📊 Class Distribution:")
    print(f"   Class 0: {aml_dist.get(0, 0)}")
    print(f"   Class 1: {aml_dist.get(1, 0)}")
    
    return transactions

def create_debug_graph(transactions):
    """Create a debug graph with stability checks"""
    print("🕸️  Creating debug graph with stability checks...")
    
    # Filter out problematic data
    clean_transactions = transactions.dropna()
    clean_transactions = clean_transactions[clean_transactions['Amount Received'] > 0]
    clean_transactions = clean_transactions[~np.isinf(clean_transactions['Amount Received'])]
    
    print(f"   Clean transactions: {len(clean_transactions)}")
    
    # Separate AML and non-AML
    aml_transactions = clean_transactions[clean_transactions['Is Laundering'] == 1]
    non_aml_transactions = clean_transactions[clean_transactions['Is Laundering'] == 0]
    
    print(f"   AML transactions: {len(aml_transactions)}")
    print(f"   Non-AML transactions: {len(non_aml_transactions)}")
    
    # Create balanced sample
    if len(aml_transactions) > 0:
        # Use all AML and sample non-AML
        non_aml_sample = non_aml_transactions.sample(n=min(len(aml_transactions) * 5, len(non_aml_transactions)), random_state=42)
        balanced_transactions = pd.concat([aml_transactions, non_aml_sample])
    else:
        balanced_transactions = non_aml_transactions.sample(n=min(1000, len(non_aml_transactions)), random_state=42)
    
    print(f"   Balanced transactions: {len(balanced_transactions)}")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Get unique accounts
    from_accounts = set(balanced_transactions['From Bank'].astype(str))
    to_accounts = set(balanced_transactions['To Bank'].astype(str))
    all_accounts = from_accounts.union(to_accounts)
    
    print(f"   Unique accounts: {len(all_accounts)}")
    
    # Add nodes with normalized features
    account_features = {}
    for account in all_accounts:
        # Get transactions for this account
        from_trans = balanced_transactions[balanced_transactions['From Bank'].astype(str) == account]
        to_trans = balanced_transactions[balanced_transactions['To Bank'].astype(str) == account]
        
        # Calculate features with stability
        total_amount = from_trans['Amount Received'].sum() + to_trans['Amount Received'].sum()
        transaction_count = len(from_trans) + len(to_trans)
        avg_amount = total_amount / transaction_count if transaction_count > 0 else 0
        
        # Check for AML
        is_aml = 0
        if len(from_trans) > 0:
            is_aml = max(is_aml, from_trans['Is Laundering'].max())
        if len(to_trans) > 0:
            is_aml = max(is_aml, to_trans['Is Laundering'].max())
        
        # Create stable features
        features = [
            np.log1p(total_amount),  # Log transform for stability
            np.log1p(transaction_count),
            np.log1p(avg_amount),
            is_aml,
            len(from_trans),
            len(to_trans),
            0.0,  # Placeholder features
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]
        
        # Ensure no NaN or Inf
        features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
        
        account_features[account] = features
        G.add_node(account, features=features)
    
    # Add edges
    edges_added = 0
    aml_edges = 0
    non_aml_edges = 0
    
    for _, transaction in balanced_transactions.iterrows():
        from_acc = str(transaction['From Bank'])
        to_acc = str(transaction['To Bank'])
        amount = transaction['Amount Received']
        is_aml = transaction['Is Laundering']
        
        if from_acc in G.nodes and to_acc in G.nodes:
            # Create stable edge features
            edge_features = [
                np.log1p(amount),  # Log transform
                is_aml,
                0.5,  # Stable features
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5
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
    
    print(f"✅ Created debug graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   AML edges: {aml_edges}")
    print(f"   Non-AML edges: {non_aml_edges}")
    
    return {
        'graph': G,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'node_features': account_features,
        'edge_features': {},
        'class_distribution': {0: non_aml_edges, 1: aml_edges}
    }

def create_debug_training_data(graph_data, num_samples=1000):
    """Create debug training data with stability checks"""
    print(f"🎯 Creating {num_samples} debug training samples...")
    
    G = graph_data['graph']
    node_features = graph_data['node_features']
    
    # Create node mapping
    nodes = list(G.nodes())
    node_to_int = {node: i for i, node in enumerate(nodes)}
    
    # Convert to PyTorch Geometric format with stability
    x_list = []
    for node in nodes:
        if node in node_features:
            features = node_features[node]
            # Ensure no NaN or Inf
            features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
            x_list.append(features)
        else:
            x_list.append([0.0] * 15)
    
    x = torch.tensor(x_list, dtype=torch.float32)
    
    # Check for NaN in node features
    if torch.isnan(x).any():
        print("⚠️  NaN in node features, fixing...")
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
                # Ensure no NaN or Inf
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
        print("⚠️  NaN in edge features, fixing...")
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # Create training graphs with stability
    training_graphs = []
    aml_graphs = 0
    non_aml_graphs = 0
    
    for _ in range(num_samples):
        if len(nodes) > 5:
            # Sample a subgraph
            sample_nodes = np.random.choice(nodes, size=min(10, len(nodes)), replace=False)
            subgraph = G.subgraph(sample_nodes)
            
            if subgraph.number_of_edges() > 0:
                # Create subgraph data
                subgraph_nodes = list(subgraph.nodes())
                subgraph_node_to_int = {node: i for i, node in enumerate(subgraph_nodes)}
                
                # Create subgraph features
                subgraph_x = []
                for node in subgraph_nodes:
                    if node in node_features:
                        features = node_features[node]
                        features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
                        subgraph_x.append(features)
                    else:
                        subgraph_x.append([0.0] * 15)
                
                subgraph_x = torch.tensor(subgraph_x, dtype=torch.float32)
                
                # Check for NaN
                if torch.isnan(subgraph_x).any():
                    subgraph_x = torch.nan_to_num(subgraph_x, nan=0.0)
                
                # Create subgraph edges
                subgraph_edge_index = []
                subgraph_edge_attr = []
                subgraph_labels = []
                
                for edge in subgraph.edges(data=True):
                    u, v, data = edge
                    if u in subgraph_node_to_int and v in subgraph_node_to_int:
                        subgraph_edge_index.append([subgraph_node_to_int[u], subgraph_node_to_int[v]])
                        if 'features' in data:
                            features = data['features']
                            features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
                            subgraph_edge_attr.append(features)
                        else:
                            subgraph_edge_attr.append([0.0] * 12)
                        subgraph_labels.append(data.get('label', 0))
                
                if len(subgraph_edge_index) > 0:
                    subgraph_edge_index = torch.tensor(subgraph_edge_index, dtype=torch.long).t().contiguous()
                    subgraph_edge_attr = torch.tensor(subgraph_edge_attr, dtype=torch.float32)
                    
                    # Check for NaN
                    if torch.isnan(subgraph_edge_attr).any():
                        subgraph_edge_attr = torch.nan_to_num(subgraph_edge_attr, nan=0.0)
                    
                    subgraph_y = torch.tensor(subgraph_labels, dtype=torch.long)
                    
                    # Create graph-level label
                    graph_label = 1 if sum(subgraph_labels) > len(subgraph_labels) / 2 else 0
                    
                    training_graph = Data(
                        x=subgraph_x,
                        edge_index=subgraph_edge_index,
                        edge_attr=subgraph_edge_attr,
                        y=torch.tensor([graph_label], dtype=torch.long)
                    )
                    training_graphs.append(training_graph)
                    
                    if graph_label == 1:
                        aml_graphs += 1
                    else:
                        non_aml_graphs += 1
    
    print(f"✅ Created {len(training_graphs)} debug training graphs")
    print(f"   AML graphs: {aml_graphs}")
    print(f"   Non-AML graphs: {non_aml_graphs}")
    
    return training_graphs

def train_debug_model():
    """Train debug model with comprehensive NaN checking"""
    print("🚀 Starting debug training with NaN detection...")
    
    # Debug data quality
    transactions = debug_data_quality()
    
    # Create debug graph
    graph_data = create_debug_graph(transactions)
    
    if graph_data is None:
        print("❌ Failed to create debug graph")
        return
    
    print(f"✅ Created debug graph: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
    print(f"   Class distribution: {graph_data['class_distribution']}")
    
    # Create debug training data
    training_graphs = create_debug_training_data(graph_data, num_samples=1000)
    
    if len(training_graphs) == 0:
        print("❌ No training graphs created")
        return
    
    # Split data
    train_size = int(0.8 * len(training_graphs))
    train_graphs = training_graphs[:train_size]
    val_graphs = training_graphs[train_size:]
    
    print(f"   Train: {len(train_graphs)} graphs")
    print(f"   Validation: {len(val_graphs)} graphs")
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)  # Small batch size
    val_loader = DataLoader(val_graphs, batch_size=4, shuffle=False)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = RobustGNN(
        input_dim=15,
        hidden_dim=16,  # Small hidden dim
        output_dim=2,
        num_layers=2,   # Few layers
        dropout=0.1     # Low dropout
    ).to(device)
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Use stable loss instead of Focal Loss
    criterion = StableLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training loop with comprehensive debugging
    best_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(20):
        # Training
        model.train()
        train_loss = 0.0
        nan_count = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            
            # Check for NaN in model output
            if torch.isnan(out).any():
                print(f"⚠️  NaN in model output at epoch {epoch+1}, batch {batch_idx}")
                out = torch.nan_to_num(out, nan=0.0)
            
            # Calculate loss
            loss = criterion(out, batch.y)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"❌ NaN loss at epoch {epoch+1}, batch {batch_idx}")
                nan_count += 1
                continue
            
            # Backward pass
            loss.backward()
            
            # Check for NaN in gradients
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"⚠️  NaN gradient in {name} at epoch {epoch+1}, batch {batch_idx}")
                    param.grad = torch.nan_to_num(param.grad, nan=0.0)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"   NaN batches: {nan_count}/{len(train_loader)}")
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                
                # Check for NaN in validation
                if torch.isnan(out).any():
                    print(f"⚠️  NaN in validation output")
                    out = torch.nan_to_num(out, nan=0.0)
                
                preds = torch.argmax(out, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())
        
        # Calculate metrics
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
        
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val F1={val_f1:.4f}, Val Precision={val_precision:.4f}, Val Recall={val_recall:.4f}")
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"✅ Debug training completed! Best F1: {best_f1:.4f}")
    
    if best_f1 > 0.7:
        print("🎉 SUCCESS! Debug model is learning without NaN loss")
        print("   The fixes work!")
    elif best_f1 > 0.5:
        print("✅ GOOD! Debug model is learning moderately")
        print("   Consider increasing dataset size")
    else:
        print("⚠️  Debug model needs improvement")
        print("   The NaN issue might be more complex")

if __name__ == "__main__":
    train_debug_model()

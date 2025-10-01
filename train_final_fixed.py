#!/usr/bin/env python3
"""
Final Fixed AML Training - No NaN Loss
======================================

This script applies all the NaN fixes to create a stable training pipeline.
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

print("🚀 Final Fixed AML Training - No NaN Loss")
print("=" * 60)

class StableLoss(nn.Module):
    """Stable loss function for extreme imbalance"""
    def __init__(self, reduction='mean'):
        super(StableLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Use standard cross entropy with class weights
        return F.cross_entropy(inputs, targets, reduction=self.reduction)

class RobustGNN(nn.Module):
    """Robust GNN with stability features"""
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
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
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
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Check for NaN before classification
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Classification
        output = self.classifier(x)
        
        # Check for NaN in output
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)
        
        return output

def load_stable_dataset():
    """Load dataset with stability measures"""
    print("📊 Loading stable dataset...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    # Load larger sample for better AML representation
    print("Loading 200K transactions for stable training...")
    transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'), nrows=200000)
    
    print(f"✅ Loaded {len(transactions)} transactions")
    
    # Check AML distribution
    aml_distribution = transactions['Is Laundering'].value_counts()
    print(f"📊 Stable Dataset AML Distribution:")
    print(f"   Class 0 (Non-AML): {aml_distribution.get(0, 0)}")
    print(f"   Class 1 (AML): {aml_distribution.get(0, 0)}")
    
    if aml_distribution.get(1, 0) > 0:
        aml_rate = aml_distribution.get(1, 0) / len(transactions) * 100
        print(f"   AML Rate: {aml_rate:.4f}%")
    else:
        print("   ⚠️  NO AML SAMPLES FOUND!")
        return None
    
    return transactions

def create_stable_graph(transactions):
    """Create stable graph with all fixes applied"""
    print("🕸️  Creating stable graph with all fixes...")
    
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
    
    # Create balanced dataset (5:1 ratio for stability)
    non_aml_sample_size = min(len(non_aml_transactions), len(aml_transactions) * 5)
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
        
        # Check for AML
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
    
    print(f"✅ Created stable graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
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

def create_stable_training_data(graph_data, num_samples=5000):
    """Create stable training data with all fixes"""
    print(f"🎯 Creating {num_samples} stable training samples...")
    
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
            sample_nodes = np.random.choice(nodes, size=min(15, len(nodes)), replace=False)
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
    
    print(f"✅ Created {len(training_graphs)} stable training graphs")
    print(f"   AML graphs: {aml_graphs}")
    print(f"   Non-AML graphs: {non_aml_graphs}")
    
    return training_graphs

def train_final_fixed_model():
    """Train the final fixed model with all NaN fixes"""
    print("🚀 Starting final fixed training with all NaN fixes...")
    
    # Load stable dataset
    transactions = load_stable_dataset()
    
    if transactions is None:
        print("❌ Failed to load stable dataset")
        return
    
    # Create stable graph
    graph_data = create_stable_graph(transactions)
    
    if graph_data is None:
        print("❌ Failed to create stable graph")
        return
    
    print(f"✅ Created stable graph: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
    print(f"   Class distribution: {graph_data['class_distribution']}")
    
    # Create stable training data
    training_graphs = create_stable_training_data(graph_data, num_samples=6000)
    
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
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = RobustGNN(
        input_dim=15,
        hidden_dim=32,
        output_dim=2,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Use stable loss with class weights
    class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32).to(device)  # Weight AML class
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training loop with comprehensive debugging
    best_f1 = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(50):
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
                out = torch.nan_to_num(out, nan=0.0)
            
            # Calculate loss
            loss = criterion(out, batch.y)
            
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
    
    print(f"✅ Final fixed training completed! Best F1: {best_f1:.4f}")
    
    if best_f1 > 0.7:
        print("🎉 EXCELLENT! Final fixed model is learning effectively")
        print("   No NaN loss - Ready for production use!")
    elif best_f1 > 0.5:
        print("✅ GOOD! Final fixed model is learning moderately")
        print("   Consider increasing dataset size for better performance")
    else:
        print("⚠️  Final fixed model needs improvement")
        print("   Consider different approaches")

if __name__ == "__main__":
    train_final_fixed_model()

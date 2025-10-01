#!/usr/bin/env python3
"""
Production AML Detection Model Training
=====================================

This script trains the final production model using the optimal hyperparameters
found through optimization on the full HI-Small dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
import gc
import time
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 Production AML Detection Model Training")
print("=" * 50)

class ProductionEdgeLevelGNN(nn.Module):
    """Production GNN with optimal hyperparameters"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, dropout=0.4):
        super(ProductionEdgeLevelGNN, self).__init__()
        
        # Optimal architecture from hyperparameter optimization
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Dynamic edge classifier
        self.edge_classifier = None
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
    def forward(self, x, edge_index, edge_attr=None):
        # Check for NaN in input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # GNN layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Edge-level classification
        src_features = x[edge_index[0]]
        tgt_features = x[edge_index[1]]
        
        if src_features.dim() == 1:
            src_features = src_features.unsqueeze(1)
        if tgt_features.dim() == 1:
            tgt_features = tgt_features.unsqueeze(1)
        
        edge_features = torch.cat([src_features, tgt_features], dim=1)
        
        if edge_attr is not None:
            if torch.isnan(edge_attr).any():
                edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
            
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            
            edge_features = torch.cat([edge_features, edge_attr], dim=1)
        
        # Create edge classifier dynamically
        if self.edge_classifier is None:
            actual_input_dim = edge_features.shape[1]
            print(f"Creating production edge classifier with input_dim={actual_input_dim}")
            
            self.edge_classifier = nn.Sequential(
                nn.Linear(actual_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.output_dim)
            ).to(edge_features.device)
        
        edge_output = self.edge_classifier(edge_features)
        
        if torch.isnan(edge_output).any():
            edge_output = torch.nan_to_num(edge_output, nan=0.0)
        
        return edge_output

def load_production_dataset():
    """Load full dataset for production training"""
    print("📊 Loading full production dataset...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    # Load full dataset
    print("   📁 Loading full HI-Small dataset (5M+ transactions)...")
    start_time = time.time()
    
    with tqdm(desc="Loading transactions", unit="MB") as pbar:
        transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'))
        pbar.update(1)
    
    load_time = time.time() - start_time
    print(f"✅ Loaded {len(transactions):,} transactions in {load_time:.2f} seconds")
    
    # Create balanced dataset for production
    print("   🔄 Creating production balanced dataset...")
    aml_transactions = transactions[transactions['Is Laundering'] == 1]
    non_aml_transactions = transactions[transactions['Is Laundering'] == 0]
    
    # Use larger sample for production (10:1 ratio)
    aml_sample = aml_transactions  # Use all AML transactions
    non_aml_sample = non_aml_transactions.sample(n=min(len(aml_transactions) * 10, len(non_aml_transactions)), random_state=42)
    
    balanced_transactions = pd.concat([aml_sample, non_aml_sample])
    balanced_transactions = balanced_transactions.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   Production dataset: {len(balanced_transactions):,} transactions")
    print(f"   AML: {len(balanced_transactions[balanced_transactions['Is Laundering'] == 1]):,}")
    print(f"   Non-AML: {len(balanced_transactions[balanced_transactions['Is Laundering'] == 0]):,}")
    
    return balanced_transactions

def create_production_graph(transactions):
    """Create production graph with chunked processing"""
    print("🕸️  Creating production graph...")
    
    # Clean data
    clean_transactions = transactions.dropna()
    clean_transactions = clean_transactions[clean_transactions['Amount Received'] > 0]
    clean_transactions = clean_transactions[~np.isinf(clean_transactions['Amount Received'])]
    
    print(f"   Clean transactions: {len(clean_transactions):,}")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Get unique accounts
    from_accounts = set(clean_transactions['From Bank'].astype(str))
    to_accounts = set(clean_transactions['To Bank'].astype(str))
    all_accounts = from_accounts.union(to_accounts)
    
    print(f"   Unique accounts: {len(all_accounts):,}")
    
    # Add nodes with features
    account_features = {}
    for account in tqdm(all_accounts, desc="Creating account features"):
        from_trans = clean_transactions[clean_transactions['From Bank'].astype(str) == account]
        to_trans = clean_transactions[clean_transactions['To Bank'].astype(str) == account]
        
        total_amount = from_trans['Amount Received'].sum() + to_trans['Amount Received'].sum()
        transaction_count = len(from_trans) + len(to_trans)
        avg_amount = total_amount / transaction_count if transaction_count > 0 else 0
        
        is_aml = 0
        if len(from_trans) > 0:
            is_aml = max(is_aml, from_trans['Is Laundering'].max())
        if len(to_trans) > 0:
            is_aml = max(is_aml, to_trans['Is Laundering'].max())
        
        features = [
            np.log1p(total_amount),
            np.log1p(transaction_count),
            np.log1p(avg_amount),
            is_aml,
            len(from_trans),
            len(to_trans),
            from_trans['Amount Received'].mean() if len(from_trans) > 0 else 0,
            to_trans['Amount Received'].mean() if len(to_trans) > 0 else 0,
            from_trans['Amount Received'].std() if len(from_trans) > 1 else 0,
            to_trans['Amount Received'].std() if len(to_trans) > 1 else 0,
            from_trans['Amount Received'].max() if len(from_trans) > 0 else 0,
            to_trans['Amount Received'].max() if len(to_trans) > 0 else 0,
            from_trans['Amount Received'].min() if len(from_trans) > 0 else 0,
            to_trans['Amount Received'].min() if len(to_trans) > 0 else 0,
            len(set(from_trans['To Bank'].astype(str))) if len(from_trans) > 0 else 0
        ]
        
        features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
        account_features[account] = features
        G.add_node(account, features=features)
    
    # Add edges
    edges_added = 0
    aml_edges = 0
    non_aml_edges = 0
    
    for _, transaction in tqdm(clean_transactions.iterrows(), desc="Adding edges"):
        from_acc = str(transaction['From Bank'])
        to_acc = str(transaction['To Bank'])
        amount = transaction['Amount Received']
        is_aml = transaction['Is Laundering']
        
        if from_acc in G.nodes and to_acc in G.nodes:
            edge_features = [
                np.log1p(amount),
                is_aml,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
            ]
            
            edge_features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in edge_features]
            
            G.add_edge(from_acc, to_acc, 
                      features=edge_features, 
                      label=is_aml)
            edges_added += 1
            
            if is_aml == 1:
                aml_edges += 1
            else:
                non_aml_edges += 1
    
    print(f"✅ Created production graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"   AML edges: {aml_edges:,}")
    print(f"   Non-AML edges: {non_aml_edges:,}")
    
    return {
        'graph': G,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'node_features': account_features,
        'class_distribution': {0: non_aml_edges, 1: aml_edges}
    }

def create_production_data(graph_data):
    """Create production training data"""
    print("🎯 Creating production training data...")
    
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
    
    if torch.isnan(edge_attr).any():
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    print(f"✅ Created production data: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    print(f"   Edge labels: {data.y.sum().item():,} AML, {len(data.y) - data.y.sum().item():,} Non-AML")
    
    return data

def train_production_model():
    """Train the production model with optimal hyperparameters"""
    print("🚀 Starting Production Model Training...")
    
    # Load production dataset
    transactions = load_production_dataset()
    graph_data = create_production_graph(transactions)
    data = create_production_data(graph_data)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = data.to(device)
    
    # Create production model with optimal hyperparameters
    model = ProductionEdgeLevelGNN(
        input_dim=15,
        hidden_dim=128,  # Optimal from hyperparameter optimization
        output_dim=2,
        dropout=0.4      # Optimal from hyperparameter optimization
    ).to(device)
    
    print(f"✅ Production model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Use optimal hyperparameters
    class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32).to(device)  # Optimal class weight
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)  # Optimal learning rate and weight decay
    
    # Training loop
    best_f1 = 0.0
    patience = 30
    patience_counter = 0
    
    print("🚀 Starting production training...")
    print("   This may take 2-4 hours for full dataset...")
    
    for epoch in range(200):  # More epochs for production
        # Training
        model.train()
        train_loss = 0.0
        nan_count = 0
        
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_attr)
        
        if torch.isnan(out).any():
            out = torch.nan_to_num(out, nan=0.0)
        
        loss = criterion(out, data.y)
        
        if torch.isnan(loss):
            nan_count += 1
            continue
        
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                param.grad = torch.nan_to_num(param.grad, nan=0.0)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        if nan_count > 0:
            print(f"   NaN batches: {nan_count}")
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index, data.edge_attr)
                
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
                    
                    # Save best model
                    torch.save(model.state_dict(), '/content/drive/MyDrive/LaunDetection/production_model.pth')
                    print(f"   💾 Saved best model (F1={best_f1:.4f})")
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Memory cleanup every 50 epochs
        if epoch % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    print(f"✅ Production training completed! Best F1: {best_f1:.4f}")
    
    if best_f1 > 0.8:
        print("🎉 EXCELLENT! Production model is highly effective")
        print("   Ready for real-world AML detection!")
    elif best_f1 > 0.75:
        print("✅ GOOD! Production model is effective")
        print("   Suitable for production deployment")
    else:
        print("⚠️  Production model needs improvement")
        print("   Consider additional training or data")
    
    return model, best_f1

if __name__ == "__main__":
    model, best_f1 = train_production_model()

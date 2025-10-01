#!/usr/bin/env python3
"""
Simplified Hyperparameter Optimization for AML Detection
========================================================

This script uses a simplified approach to avoid tensor dimension issues
by using a fixed, proven architecture and optimizing only key hyperparameters.
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
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🔧 Simplified Hyperparameter Optimization for AML Detection")
print("=" * 65)

class SimpleEdgeLevelGNN(nn.Module):
    """Simplified GNN for edge-level classification with fixed architecture"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(SimpleEdgeLevelGNN, self).__init__()
        
        # Fixed architecture - no dynamic creation
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Fixed edge classifier - no dynamic creation
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 12, hidden_dim),  # Fixed input size
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
        
        # Edge-level classification with fixed dimensions
        src_features = x[edge_index[0]]
        tgt_features = x[edge_index[1]]
        
        # Ensure 2D tensors
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
        
        # Fixed edge classifier
        edge_output = self.edge_classifier(edge_features)
        
        if torch.isnan(edge_output).any():
            edge_output = torch.nan_to_num(edge_output, nan=0.0)
        
        return edge_output

def load_optimization_dataset():
    """Load dataset for hyperparameter optimization"""
    print("📊 Loading dataset for simplified optimization...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    # Load full dataset
    print("   📁 Loading full HI-Small dataset...")
    start_time = time.time()
    
    with tqdm(desc="Loading transactions", unit="MB") as pbar:
        transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'))
        pbar.update(1)
    
    load_time = time.time() - start_time
    print(f"✅ Loaded {len(transactions):,} transactions in {load_time:.2f} seconds")
    
    # Create balanced dataset for optimization
    print("   🔄 Creating balanced dataset for optimization...")
    aml_transactions = transactions[transactions['Is Laundering'] == 1]
    non_aml_transactions = transactions[transactions['Is Laundering'] == 0]
    
    # Use smaller sample for faster optimization
    aml_sample = aml_transactions.sample(n=min(1000, len(aml_transactions)), random_state=42)
    non_aml_sample = non_aml_transactions.sample(n=min(5000, len(non_aml_transactions)), random_state=42)
    
    balanced_transactions = pd.concat([aml_sample, non_aml_sample])
    balanced_transactions = balanced_transactions.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   Balanced dataset: {len(balanced_transactions):,} transactions")
    print(f"   AML: {len(balanced_transactions[balanced_transactions['Is Laundering'] == 1]):,}")
    print(f"   Non-AML: {len(balanced_transactions[balanced_transactions['Is Laundering'] == 0]):,}")
    
    return balanced_transactions

def create_optimization_graph(transactions):
    """Create graph for hyperparameter optimization"""
    print("🕸️  Creating optimization graph...")
    
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
    
    print(f"✅ Created optimization graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"   AML edges: {aml_edges:,}")
    print(f"   Non-AML edges: {non_aml_edges:,}")
    
    return {
        'graph': G,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'node_features': account_features,
        'class_distribution': {0: non_aml_edges, 1: aml_edges}
    }

def create_optimization_data(graph_data):
    """Create training data for hyperparameter optimization"""
    print("🎯 Creating optimization training data...")
    
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
    
    print(f"✅ Created optimization data: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    print(f"   Edge labels: {data.y.sum().item():,} AML, {len(data.y) - data.y.sum().item():,} Non-AML")
    
    return data

def train_model_with_config(data, config, device, max_epochs=30):
    """Train model with specific configuration"""
    try:
        model = SimpleEdgeLevelGNN(
            input_dim=15,
            hidden_dim=config['hidden_dim'],
            output_dim=2,
            dropout=config['dropout']
        ).to(device)
        
        # Use class weights
        class_weights = torch.tensor([1.0, config['class_weight']], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
        best_f1 = 0.0
        patience = 10
        patience_counter = 0
        
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index, data.edge_attr)
            
            if torch.isnan(out).any():
                out = torch.nan_to_num(out, nan=0.0)
            
            loss = criterion(out, data.y)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    param.grad = torch.nan_to_num(param.grad, nan=0.0)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index, data.edge_attr)
                    
                    if torch.isnan(out).any():
                        out = torch.nan_to_num(out, nan=0.0)
                    
                    preds = torch.argmax(out, dim=1)
                    val_f1 = f1_score(data.y.cpu().numpy(), preds.cpu().numpy(), average='weighted')
                    
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        break
        
        return best_f1
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        return 0.0

def simple_hyperparameter_optimization():
    """Perform simplified hyperparameter optimization"""
    print("🔧 Starting Simplified Hyperparameter Optimization...")
    
    # Load data
    transactions = load_optimization_dataset()
    graph_data = create_optimization_graph(transactions)
    data = create_optimization_data(graph_data)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = data.to(device)
    
    # Define simplified hyperparameter ranges
    param_ranges = {
        'hidden_dim': [64, 128, 256],
        'dropout': [0.1, 0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'weight_decay': [1e-4, 1e-5, 1e-6],
        'class_weight': [2.0, 3.0, 5.0, 10.0]
    }
    
    n_trials = 30  # Reduced number of trials
    results = []
    best_score = 0.0
    best_config = None
    
    print(f"🔧 Testing {n_trials} simplified hyperparameter combinations...")
    
    for i in tqdm(range(n_trials), desc="Simplified Optimization"):
        # Randomly sample hyperparameters
        config = {}
        for param, values in param_ranges.items():
            config[param] = np.random.choice(values)
        
        try:
            score = train_model_with_config(data, config, device, max_epochs=20)
            results.append((config, score))
            
            if score > best_score:
                best_score = score
                best_config = config
            
            print(f"Trial {i+1}/{n_trials}: F1={score:.4f}")
            
        except Exception as e:
            print(f"Trial {i+1}/{n_trials}: FAILED - {str(e)}")
            results.append((config, 0.0))
            continue
        
        # Memory cleanup
        if i % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Sort results by score
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 Simplified Optimization Results:")
    print(f"   Best F1 Score: {best_score:.4f}")
    print(f"   Best Configuration: {best_config}")
    
    print(f"\n📊 Top 5 Configurations:")
    for i, (config, score) in enumerate(results[:5]):
        print(f"   {i+1}. F1={score:.4f}: {config}")
    
    return best_config, best_score, results

def main():
    """Main optimization function"""
    print("🚀 Starting Simplified Hyperparameter Optimization...")
    
    best_config, best_score, results = simple_hyperparameter_optimization()
    
    print(f"\n🎉 Simplified Optimization Complete!")
    print(f"   Best F1 Score: {best_score:.4f}")
    print(f"   Best Configuration: {best_config}")
    
    # Save results
    results_data = {
        'best_config': best_config,
        'best_score': best_score,
        'all_results': results
    }
    
    with open('/content/drive/MyDrive/LaunDetection/simple_hyperparameter_results.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"✅ Results saved to simple_hyperparameter_results.pkl")
    
    return best_config, best_score

if __name__ == "__main__":
    main()

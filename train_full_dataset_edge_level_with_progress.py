#!/usr/bin/env python3
"""
Full Dataset Edge-Level AML Classification with Progress Bars
============================================================

This script trains the edge-level AML detection model on the full HI-Small dataset
(5M+ transactions) with enhanced preprocessing, memory optimization, and comprehensive progress tracking.
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
import gc
import time
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 Full Dataset Edge-Level AML Classification with Progress Bars")
print("=" * 70)

class EdgeLevelGNN(nn.Module):
    """GNN for edge-level classification - optimized for large datasets"""
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
        
        # Edge classification head - will be created dynamically
        self.edge_classifier = None
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
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
        
        # Create edge classifier dynamically if not exists
        if self.edge_classifier is None:
            input_dim = edge_features.shape[1]
            self.edge_classifier = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, self.output_dim)
            ).to(edge_features.device)
        
        # Classify edges
        edge_output = self.edge_classifier(edge_features)
        
        # Check for NaN in output
        if torch.isnan(edge_output).any():
            edge_output = torch.nan_to_num(edge_output, nan=0.0)
        
        return edge_output

def load_full_dataset():
    """Load the full HI-Small dataset with memory optimization and progress tracking"""
    print("📊 Loading full HI-Small dataset...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    # Progress bar for loading
    print("   📁 Loading full HI-Small dataset (5M+ transactions)...")
    start_time = time.time()
    
    with tqdm(desc="Loading transactions", unit="MB") as pbar:
        transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'))
        pbar.update(1)
    
    load_time = time.time() - start_time
    print(f"✅ Loaded {len(transactions):,} transactions in {load_time:.2f} seconds")
    
    # Progress bar for AML analysis
    print("   📊 Analyzing AML distribution...")
    with tqdm(total=3, desc="Analyzing dataset") as pbar:
        aml_distribution = transactions['Is Laundering'].value_counts()
        pbar.update(1)
        
        print(f"   📊 Full Dataset AML Distribution:")
        print(f"      Class 0 (Non-AML): {aml_distribution.get(0, 0):,}")
        print(f"      Class 1 (AML): {aml_distribution.get(1, 0):,}")
        pbar.update(1)
        
        if aml_distribution.get(1, 0) > 0:
            aml_rate = aml_distribution.get(1, 0) / len(transactions) * 100
            print(f"      AML Rate: {aml_rate:.4f}%")
            print(f"      Total AML transactions: {aml_distribution.get(1, 0):,}")
        else:
            print("      ⚠️  NO AML SAMPLES FOUND!")
            return None
        pbar.update(1)
    
    return transactions

def create_balanced_full_dataset(transactions, target_ratio=5):
    """Create balanced dataset from full dataset with memory optimization and progress tracking"""
    print(f"🔄 Creating balanced dataset with {target_ratio}:1 ratio...")
    
    # Progress bar for data separation
    print("   📊 Separating AML and Non-AML transactions...")
    with tqdm(total=2, desc="Separating data") as pbar:
        aml_transactions = transactions[transactions['Is Laundering'] == 1]
        pbar.update(1)
        non_aml_transactions = transactions[transactions['Is Laundering'] == 0]
        pbar.update(1)
    
    print(f"   Original AML: {len(aml_transactions):,}")
    print(f"   Original Non-AML: {len(non_aml_transactions):,}")
    
    if len(aml_transactions) == 0:
        print("❌ No AML transactions found!")
        return None
    
    # Progress bar for sampling
    print("   🎯 Sampling balanced dataset...")
    non_aml_sample_size = min(len(non_aml_transactions), len(aml_transactions) * target_ratio)
    
    with tqdm(total=non_aml_sample_size, desc="Sampling Non-AML transactions", unit="transactions") as pbar:
        non_aml_sampled = non_aml_transactions.sample(n=non_aml_sample_size, random_state=42)
        pbar.update(non_aml_sample_size)
    
    # Progress bar for combining
    print("   🔄 Combining and shuffling dataset...")
    with tqdm(total=2, desc="Creating balanced dataset") as pbar:
        balanced_transactions = pd.concat([aml_transactions, non_aml_sampled])
        pbar.update(1)
        balanced_transactions = balanced_transactions.sample(frac=1, random_state=42).reset_index(drop=True)
        pbar.update(1)
    
    print(f"   Balanced AML: {len(balanced_transactions[balanced_transactions['Is Laundering'] == 1]):,}")
    print(f"   Balanced Non-AML: {len(balanced_transactions[balanced_transactions['Is Laundering'] == 0]):,}")
    print(f"   Balance ratio: {len(balanced_transactions[balanced_transactions['Is Laundering'] == 0])/len(balanced_transactions[balanced_transactions['Is Laundering'] == 1]):.1f}:1")
    
    # Clear memory
    del aml_transactions, non_aml_transactions, non_aml_sampled
    gc.collect()
    
    return balanced_transactions

def create_full_graph_with_chunking(transactions, chunk_size=100000):
    """Create graph from full dataset with chunked processing and comprehensive progress tracking"""
    print("🕸️  Creating full graph with chunked processing...")
    
    # Progress bar for data cleaning
    print("   🧹 Cleaning and filtering data...")
    with tqdm(total=3, desc="Data cleaning") as pbar:
        clean_transactions = transactions.dropna()
        pbar.update(1)
        clean_transactions = clean_transactions[clean_transactions['Amount Received'] > 0]
        pbar.update(1)
        clean_transactions = clean_transactions[~np.isinf(clean_transactions['Amount Received'])]
        pbar.update(1)
    
    print(f"   Clean transactions: {len(clean_transactions):,}")
    
    # Progress bar for AML separation
    print("   📊 Separating AML and Non-AML transactions...")
    with tqdm(total=2, desc="AML separation") as pbar:
        aml_transactions = clean_transactions[clean_transactions['Is Laundering'] == 1]
        pbar.update(1)
        non_aml_transactions = clean_transactions[clean_transactions['Is Laundering'] == 0]
        pbar.update(1)
    
    print(f"   AML transactions: {len(aml_transactions):,}")
    print(f"   Non-AML transactions: {len(non_aml_transactions):,}")
    
    if len(aml_transactions) == 0:
        print("❌ No AML transactions found")
        return None
    
    # Progress bar for account identification
    print("   🏦 Identifying unique accounts...")
    with tqdm(total=2, desc="Account identification") as pbar:
        from_accounts = set(clean_transactions['From Bank'].astype(str))
        pbar.update(1)
        to_accounts = set(clean_transactions['To Bank'].astype(str))
        all_accounts = from_accounts.union(to_accounts)
        pbar.update(1)
    
    print(f"   Unique accounts: {len(all_accounts):,}")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Progress bar for account feature creation
    print("   🏗️  Creating account features...")
    account_features = {}
    
    for i, account in enumerate(tqdm(all_accounts, desc="Processing accounts", unit="accounts")):
        if i % 10000 == 0:
            gc.collect()  # Memory cleanup every 10k accounts
        
        # Get transactions for this account
        from_trans = clean_transactions[clean_transactions['From Bank'].astype(str) == account]
        to_trans = clean_transactions[clean_transactions['To Bank'].astype(str) == account]
        
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
    
    # Progress bar for edge creation
    print("   🔗 Adding edges with chunked processing...")
    edges_added = 0
    aml_edges = 0
    non_aml_edges = 0
    
    # Calculate total chunks for progress tracking
    total_chunks = (len(clean_transactions) + chunk_size - 1) // chunk_size
    
    with tqdm(total=total_chunks, desc="Processing edges", unit="chunks") as chunk_pbar:
        for i in range(0, len(clean_transactions), chunk_size):
            chunk = clean_transactions.iloc[i:i+chunk_size]
            
            with tqdm(total=len(chunk), desc=f"Chunk {i//chunk_size + 1}/{total_chunks}", leave=False, unit="transactions") as trans_pbar:
                for _, transaction in chunk.iterrows():
                    from_acc = str(transaction['From Bank'])
                    to_acc = str(transaction['To Bank'])
                    amount = transaction['Amount Received']
                    is_aml = transaction['Is Laundering']
                    
                    if from_acc in G.nodes and to_acc in G.nodes:
                        # Create stable edge features
                        edge_features = [
                            np.log1p(amount),  # Log transform
                            is_aml,
                            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
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
                    
                    trans_pbar.update(1)
            
            chunk_pbar.update(1)
            
            # Memory cleanup after each chunk
            if i % (chunk_size * 5) == 0:
                gc.collect()
    
    print(f"✅ Created full graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"   AML edges: {aml_edges:,}")
    print(f"   Non-AML edges: {non_aml_edges:,}")
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

def create_full_training_data(graph_data):
    """Create training data for full dataset with memory optimization and progress tracking"""
    print("🎯 Creating full dataset training data...")
    
    G = graph_data['graph']
    node_features = graph_data['node_features']
    
    # Create node mapping
    nodes = list(G.nodes())
    node_to_int = {node: i for i, node in enumerate(nodes)}
    
    print(f"   Processing {len(nodes):,} nodes...")
    
    # Progress bar for node feature conversion
    print("   🏗️  Converting node features to tensors...")
    x_list = []
    for i, node in enumerate(tqdm(nodes, desc="Converting node features", unit="nodes")):
        if i % 10000 == 0:
            gc.collect()  # Memory cleanup
        
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
    
    # Progress bar for edge feature conversion
    print("   🔗 Converting edge features to tensors...")
    edge_index_list = []
    edge_attr_list = []
    edge_labels = []
    
    for i, edge in enumerate(tqdm(G.edges(data=True), desc="Converting edge features", unit="edges")):
        if i % 50000 == 0:
            gc.collect()  # Memory cleanup every 50k edges
        
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
    
    # Progress bar for tensor creation
    print("   🔧 Creating PyTorch tensors...")
    with tqdm(total=3, desc="Tensor creation") as pbar:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        pbar.update(1)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        pbar.update(1)
        y = torch.tensor(edge_labels, dtype=torch.long)
        pbar.update(1)
    
    # Check for NaN in edge features
    if torch.isnan(edge_attr).any():
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    print(f"✅ Created full dataset data: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    print(f"   Edge labels: {data.y.sum().item():,} AML, {len(data.y) - data.y.sum().item():,} Non-AML")
    
    return data

def train_full_dataset_model():
    """Train the edge-level classification model on full dataset with comprehensive progress tracking"""
    print("🚀 Starting full dataset edge-level AML classification training...")
    
    # Load full dataset
    transactions = load_full_dataset()
    
    if transactions is None:
        print("❌ Failed to load full dataset")
        return
    
    # Create balanced dataset
    balanced_transactions = create_balanced_full_dataset(transactions, target_ratio=5)
    
    if balanced_transactions is None:
        print("❌ Failed to create balanced dataset")
        return
    
    # Create graph with chunked processing
    graph_data = create_full_graph_with_chunking(balanced_transactions, chunk_size=100000)
    
    if graph_data is None:
        print("❌ Failed to create graph")
        return
    
    # Create training data
    data = create_full_training_data(graph_data)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move data to device
    data = data.to(device)
    
    # Create model with larger capacity for full dataset
    print("🏗️  Creating model architecture...")
    with tqdm(total=4, desc="Model creation") as pbar:
        model = EdgeLevelGNN(
            input_dim=15,
            hidden_dim=128,  # Larger hidden dimension for full dataset
            output_dim=2,
            num_layers=4,    # More layers for complex patterns
            dropout=0.2       # Higher dropout for regularization
        ).to(device)
        pbar.update(1)
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        pbar.update(1)
        
        # Use class weights for imbalanced data
        class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32).to(device)  # Weight AML class
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        pbar.update(1)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Lower LR for stability
        pbar.update(1)
    
    # Training loop with memory optimization and progress tracking
    best_f1 = 0.0
    patience = 30
    patience_counter = 0
    
    print("🚀 Starting training on full dataset...")
    print("   This may take 2-4 hours depending on dataset size...")
    
    # Progress bar for training epochs
    with tqdm(total=200, desc="Training epochs", unit="epochs") as epoch_pbar:
        for epoch in range(200):  # More epochs for full dataset
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
            
            # Validation every 10 epochs
            if epoch % 10 == 0:
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
            
            # Memory cleanup every 50 epochs
            if epoch % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
            epoch_pbar.update(1)
    
    print(f"✅ Full dataset training completed! Best F1: {best_f1:.4f}")
    
    if best_f1 > 0.8:
        print("🎉 EXCELLENT! Full dataset model is highly effective")
        print("   Ready for production AML detection on full scale!")
    elif best_f1 > 0.7:
        print("✅ GOOD! Full dataset model is effective")
        print("   Consider tuning hyperparameters for better performance")
    else:
        print("⚠️  Full dataset model needs improvement")
        print("   Consider different approaches or more data")

if __name__ == "__main__":
    train_full_dataset_model()

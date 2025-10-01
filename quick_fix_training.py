#!/usr/bin/env python3
"""
Quick Fix for AML Multi-GNN Training
====================================

This script fixes the data mismatch issue by using the correct column mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported successfully")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# SimpleGNN Model
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

        if batch is not None:
            batch = batch.to(device)
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)

        return x

print("✓ SimpleGNN model defined")

# FIXED data loading function
def load_real_data_fixed(data_path):
    """Load real AML data with proper column mapping"""
    print("Loading real AML data with FIXED column mapping...")

    # Load HI-Small dataset
    transactions_file = os.path.join(data_path, "HI-Small_Trans.csv")
    accounts_file = os.path.join(data_path, "HI-Small_accounts.csv")

    print(f"Loading transactions from {transactions_file}")
    transactions = pd.read_csv(transactions_file, nrows=2000)
    print(f"Loaded {len(transactions)} transactions")

    print(f"Loading accounts from {accounts_file}")
    accounts = pd.read_csv(accounts_file, nrows=1000)
    print(f"Loaded {len(accounts)} accounts")

    return transactions, accounts

# FIXED graph creation function
def create_graph_fixed(transactions, accounts):
    """Create graph with CORRECT column mapping"""
    print("Creating graph with FIXED column mapping...")
    
    print("Transaction columns:", transactions.columns.tolist())
    print("Account columns:", accounts.columns.tolist())
    
    # CORRECT column mapping - use Account and Account.1, NOT From Bank and To Bank
    from_account_col = 'Account'      # String account numbers
    to_account_col = 'Account.1'      # String account numbers
    amount_col = 'Amount Paid'
    timestamp_col = 'Timestamp'
    sar_col = 'Is Laundering'
    
    print(f"✅ Using CORRECT columns: from_account={from_account_col}, to_account={to_account_col}")
    print(f"   amount={amount_col}, time={timestamp_col}, sar={sar_col}")
    
    # Create account features using Account Number as key
    account_features = {}
    
    for _, account in accounts.iterrows():
        account_id = account['Account Number']  # Use Account Number as key
        
        # Default features
        account_features[account_id] = [
            5000.0,  # balance
            0.5,     # risk_score
            1,       # checking
            0,       # savings
            0        # business
        ]
    
    print(f"Created {len(account_features)} account features")
    print(f"Sample account IDs: {list(account_features.keys())[:5]}")
    
    # Process transactions
    edges = []
    edge_features = []
    labels = []
    
    matched_edges = 0
    for i, (_, transaction) in enumerate(transactions.iterrows()):
        if i % 1000 == 0:
            print(f"Processing transaction {i}/{len(transactions)}")
        
        # Use CORRECT columns
        from_account = transaction[from_account_col]  # Account (string)
        to_account = transaction[to_account_col]      # Account.1 (string)
        
        # Get amount
        amount = float(transaction[amount_col]) if pd.notna(transaction[amount_col]) else 1000.0
        
        # Get timestamp
        try:
            timestamp = pd.to_datetime(transaction[timestamp_col])
            hour = timestamp.hour
            day = timestamp.day
            month = timestamp.month
        except:
            hour, day, month = 12, 1, 1
        
        # Get SAR label
        is_sar = int(transaction[sar_col])
        
        # Debug first 5 transactions
        if i < 5:
            print(f"Transaction {i}: from_account={from_account}, to_account={to_account}")
            print(f"  from_account in account_features: {from_account in account_features}")
            print(f"  to_account in account_features: {to_account in account_features}")
        
        # Check if both accounts exist
        if from_account in account_features and to_account in account_features:
            edges.append([from_account, to_account])
            edge_features.append([amount, hour, day, month])
            labels.append(is_sar)
            matched_edges += 1
        else:
            if i < 5:
                print(f"  ✗ Skipped transaction {i}: missing accounts")
    
    print(f"✅ Matched {matched_edges} edges out of {len(transactions)} transactions")
    
    if matched_edges == 0:
        print("❌ ERROR: Still no matches found!")
        print("This means the account numbers in transactions don't match account data.")
        return None
    
    # Create node features
    unique_accounts = list(set([edge[0] for edge in edges] + [edge[1] for edge in edges]))
    node_features = []
    node_labels = []
    
    for account_id in unique_accounts:
        if account_id in account_features:
            node_features.append(account_features[account_id])
            # Node label: 1 if any transaction from this account is SAR
            account_sar = any(labels[i] for i, edge in enumerate(edges) if edge[0] == account_id)
            node_labels.append(1 if account_sar else 0)
        else:
            node_features.append([0, 0, 0, 0, 0])
            node_labels.append(0)
    
    # Convert to tensors
    node_features = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    labels = torch.tensor(node_labels, dtype=torch.long)
    
    print(f"✅ Created graph with {len(unique_accounts)} nodes and {len(edges)} edges")
    
    if len(node_labels) > 0:
        sar_rate = sum(node_labels) / len(node_labels)
        print(f"SAR rate: {sar_rate:.3f}")
    
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)

# Training function
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001):
    """Train model"""
    print(f"Training model for {epochs} epochs...")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
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
    
    return val_f1

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("AML Multi-GNN - Quick Fix Training")
    print("=" * 60)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    transactions, accounts = load_real_data_fixed(data_path)
    
    # Create graph
    graph = create_graph_fixed(transactions, accounts)
    
    if graph is None:
        print("❌ Graph creation failed!")
        exit(1)
    
    print(f"✅ Graph created: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Create individual graphs for training
    individual_graphs = []
    for i in range(min(50, graph.num_nodes)):  # Limit for memory
        center_node = i
        neighbor_nodes = [center_node]
        
        # Add neighbors
        for edge_idx in range(graph.edge_index.size(1)):
            edge = graph.edge_index[:, edge_idx]
            if edge[0].item() == center_node:
                neighbor_nodes.append(edge[1].item())
            elif edge[1].item() == center_node:
                neighbor_nodes.append(edge[0].item())
        
        neighbor_nodes = neighbor_nodes[:5]  # Limit size
        
        if len(neighbor_nodes) > 1:
            subgraph = graph.subgraph(torch.tensor(neighbor_nodes))
            if subgraph.num_edges > 0:
                node_labels = subgraph.y.tolist()
                graph_label = 1 if sum(node_labels) > len(node_labels) / 2 else 0
                
                new_graph = Data(
                    x=subgraph.x,
                    edge_index=subgraph.edge_index,
                    edge_attr=subgraph.edge_attr,
                    y=torch.tensor([graph_label], dtype=torch.long)
                )
                individual_graphs.append(new_graph)
    
    print(f"✅ Created {len(individual_graphs)} individual graphs")
    
    # Split data
    train_size = int(0.8 * len(individual_graphs))
    train_graphs = individual_graphs[:train_size]
    val_graphs = individual_graphs[train_size:]
    
    print(f"Split: {len(train_graphs)} train graphs, {len(val_graphs)} val graphs")
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=4, shuffle=False)
    
    # Create model
    model = SimpleGNN(
        input_dim=5,
        hidden_dim=32,
        output_dim=2,
        num_layers=2,
        dropout=0.1
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    best_f1 = train_model(model, train_loader, val_loader, epochs=5, lr=0.001)
    
    print(f"\n✅ Training completed!")
    print(f"Best validation F1: {best_f1:.4f}")
    
    print("\n" + "=" * 60)
    print("Quick fix training completed successfully!")
    print("=" * 60)

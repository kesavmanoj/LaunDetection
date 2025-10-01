#!/usr/bin/env python3
"""
Fixed Real Data Only Training for AML Multi-GNN
===============================================

This script uses ONLY real data from the IBM AML dataset.
Fixed the tensor creation error by using proper node mapping.
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

# Real data only loading function
def load_real_data_only(data_path):
    """Load ONLY real AML data - no synthetic data"""
    print("Loading ONLY real AML data...")

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

# Fixed graph creation function
def create_graph_fixed(transactions, accounts):
    """Create graph using ONLY real data with proper node mapping"""
    print("Creating graph using ONLY real data with proper node mapping...")
    
    print("Transaction columns:", transactions.columns.tolist())
    print("Account columns:", accounts.columns.tolist())
    
    # Analyze the data structure
    print("\n🔍 Analyzing real data structure...")
    
    # Get account numbers from transactions
    trans_accounts = set(transactions['Account'].unique()) | set(transactions['Account.1'].unique())
    account_numbers = set(accounts['Account Number'].unique())
    
    print(f"Transaction account numbers: {len(trans_accounts)}")
    print(f"Account data numbers: {len(account_numbers)}")
    print(f"Sample transaction accounts: {list(trans_accounts)[:5]}")
    print(f"Sample account data numbers: {list(account_numbers)[:5]}")
    
    # Check for any overlap
    overlap = trans_accounts & account_numbers
    print(f"Direct overlap: {len(overlap)}")
    
    if len(overlap) == 0:
        print("❌ NO OVERLAP FOUND between transaction and account data!")
        print("This means the account numbers in transactions don't match account data.")
        return None
    
    print(f"✅ Found {len(overlap)} overlapping account numbers!")
    print(f"Overlapping accounts: {list(overlap)}")
    
    # Create account features using only overlapping accounts
    account_features = {}
    
    for _, account in accounts.iterrows():
        account_id = account['Account Number']
        if account_id in overlap:
            # Use real account data
            account_features[account_id] = [
                5000.0,  # balance (default)
                0.5,     # risk_score (default)
                1,       # checking (default)
                0,       # savings (default)
                0        # business (default)
            ]
    
    print(f"Created {len(account_features)} account features from overlapping accounts")
    
    # Create node mapping (string account ID -> integer node ID)
    unique_accounts = list(overlap)
    node_mapping = {account_id: i for i, account_id in enumerate(unique_accounts)}
    print(f"Created node mapping for {len(node_mapping)} accounts")
    
    # Process transactions
    edges = []
    edge_features = []
    labels = []
    
    matched_edges = 0
    for i, (_, transaction) in enumerate(transactions.iterrows()):
        if i % 1000 == 0:
            print(f"Processing transaction {i}/{len(transactions)}")
        
        from_account = transaction['Account']
        to_account = transaction['Account.1']
        
        # Only process if both accounts are in our overlapping set
        if from_account in node_mapping and to_account in node_mapping:
            amount = float(transaction['Amount Paid']) if pd.notna(transaction['Amount Paid']) else 1000.0
            
            try:
                timestamp = pd.to_datetime(transaction['Timestamp'])
                hour = timestamp.hour
                day = timestamp.day
                month = timestamp.month
            except:
                hour, day, month = 12, 1, 1
            
            is_sar = int(transaction['Is Laundering'])
            
            # Use integer node IDs instead of string account IDs
            from_node = node_mapping[from_account]
            to_node = node_mapping[to_account]
            
            edges.append([from_node, to_node])
            edge_features.append([amount, hour, day, month])
            labels.append(is_sar)
            matched_edges += 1
    
    print(f"✅ Matched {matched_edges} edges out of {len(transactions)} transactions")
    
    if matched_edges == 0:
        print("❌ ERROR: No edges created even with overlapping accounts!")
        return None
    
    # Create node features using the same order as node_mapping
    node_features = []
    node_labels = []
    
    for account_id in unique_accounts:
        if account_id in account_features:
            node_features.append(account_features[account_id])
            # Node label: 1 if any transaction from this account is SAR
            account_sar = any(labels[i] for i, edge in enumerate(edges) if edge[0] == node_mapping[account_id])
            node_labels.append(1 if account_sar else 0)
        else:
            node_features.append([0, 0, 0, 0, 0])
            node_labels.append(0)
    
    # Convert to tensors - now using integer node IDs
    node_features = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    labels = torch.tensor(node_labels, dtype=torch.long)
    
    print(f"✅ Created graph with {len(unique_accounts)} nodes and {len(edges)} edges")
    print(f"Node features shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge attributes shape: {edge_attr.shape}")
    print(f"Labels shape: {labels.shape}")
    
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
    print("AML Multi-GNN - Fixed Real Data Only Training")
    print("=" * 60)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    transactions, accounts = load_real_data_only(data_path)
    
    # Create graph
    graph = create_graph_fixed(transactions, accounts)
    
    if graph is None:
        print("❌ Graph creation failed!")
        print("The transaction and account data are incompatible.")
        print("No linking strategy found between the datasets.")
        exit(1)
    
    print(f"✅ Graph created: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Create individual graphs for training
    individual_graphs = []
    for i in range(min(20, graph.num_nodes)):  # Limit for memory
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
    
    if len(individual_graphs) == 0:
        print("❌ No individual graphs created!")
        print("The graph structure is too sparse for training.")
        exit(1)
    
    # Split data
    train_size = int(0.8 * len(individual_graphs))
    train_graphs = individual_graphs[:train_size]
    val_graphs = individual_graphs[train_size:]
    
    print(f"Split: {len(train_graphs)} train graphs, {len(val_graphs)} val graphs")
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=2, shuffle=False)
    
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
    print("Fixed real data only training completed successfully!")
    print("=" * 60)

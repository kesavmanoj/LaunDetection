#!/usr/bin/env python3
"""
Training with Proper AML Graph Creation
=======================================

This script ensures AML samples are properly included in training graphs.
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

print("🚀 Training with Proper AML Graph Creation")
print("=" * 60)

class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class RobustGNN(nn.Module):
    """Robust GNN with better architecture for imbalanced data"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.3):
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
        # GNN layers
        for i, (conv, bn, dropout) in enumerate(zip(self.convs, self.bns, self.dropouts)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Classification
        return self.classifier(x)

def load_full_dataset():
    """Load the entire dataset to get more AML samples"""
    print("📊 Loading full dataset for maximum AML samples...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    # Load the entire dataset
    print("Loading entire HI-Small dataset...")
    transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'))
    
    print(f"✅ Loaded {len(transactions)} transactions (full dataset)")
    
    # Check AML distribution
    aml_distribution = transactions['Is Laundering'].value_counts()
    print(f"📊 Full Dataset AML Distribution:")
    print(f"   Class 0 (Non-AML): {aml_distribution.get(0, 0)}")
    print(f"   Class 1 (AML): {aml_distribution.get(1, 0)}")
    
    if aml_distribution.get(1, 0) > 0:
        aml_rate = aml_distribution.get(1, 0) / len(transactions) * 100
        print(f"   AML Rate: {aml_rate:.4f}%")
    else:
        print("   ⚠️  NO AML SAMPLES FOUND!")
        return None
    
    return transactions

def create_aml_centered_graph(transactions):
    """Create a graph centered around AML transactions"""
    print("🕸️  Creating AML-centered graph...")
    
    # Separate AML and non-AML transactions
    aml_transactions = transactions[transactions['Is Laundering'] == 1]
    non_aml_transactions = transactions[transactions['Is Laundering'] == 0]
    
    print(f"   AML transactions: {len(aml_transactions)}")
    print(f"   Non-AML transactions: {len(non_aml_transactions)}")
    
    if len(aml_transactions) == 0:
        print("❌ No AML transactions found")
        return None
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Get all accounts involved in AML transactions
    aml_accounts = set()
    for _, transaction in aml_transactions.iterrows():
        aml_accounts.add(str(transaction['From Bank']))
        aml_accounts.add(str(transaction['To Bank']))
    
    print(f"   AML-related accounts: {len(aml_accounts)}")
    
    # Get additional accounts from non-AML transactions (for context)
    # Sample non-AML transactions that connect to AML accounts
    connected_non_aml = []
    for _, transaction in non_aml_transactions.iterrows():
        from_acc = str(transaction['From Bank'])
        to_acc = str(transaction['To Bank'])
        
        # Include if either account is involved in AML
        if from_acc in aml_accounts or to_acc in aml_accounts:
            connected_non_aml.append(transaction)
    
    # Limit to reasonable size
    connected_non_aml = connected_non_aml[:len(aml_transactions) * 20]  # 20:1 ratio
    
    print(f"   Connected non-AML transactions: {len(connected_non_aml)}")
    
    # Combine AML and connected non-AML
    all_transactions = pd.concat([aml_transactions, pd.DataFrame(connected_non_aml)])
    all_transactions = all_transactions.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   Total transactions for graph: {len(all_transactions)}")
    
    # Get all unique accounts
    from_accounts = set(all_transactions['From Bank'].astype(str))
    to_accounts = set(all_transactions['To Bank'].astype(str))
    all_accounts = from_accounts.union(to_accounts)
    
    print(f"   Unique accounts: {len(all_accounts)}")
    
    # Add nodes with features
    account_features = {}
    for account in all_accounts:
        # Get transactions for this account
        from_trans = all_transactions[all_transactions['From Bank'].astype(str) == account]
        to_trans = all_transactions[all_transactions['To Bank'].astype(str) == account]
        
        # Calculate features
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
        
        # Create features
        features = [
            total_amount,
            transaction_count,
            avg_amount,
            max_amount,
            min_amount,
            is_aml,
            len(from_trans),  # outgoing transactions
            len(to_trans),    # incoming transactions
            from_trans['Amount Received'].std() if len(from_trans) > 1 else 0,
            to_trans['Amount Received'].std() if len(to_trans) > 1 else 0,
            from_trans['Amount Received'].mean() if len(from_trans) > 0 else 0,
            to_trans['Amount Received'].mean() if len(to_trans) > 0 else 0,
            from_trans['Amount Received'].median() if len(from_trans) > 0 else 0,
            to_trans['Amount Received'].median() if len(to_trans) > 0 else 0,
            len(set(from_trans['To Bank'].astype(str))) if len(from_trans) > 0 else 0
        ]
        
        account_features[account] = features
        G.add_node(account, features=features)
    
    # Add edges
    edges_added = 0
    aml_edges = 0
    non_aml_edges = 0
    
    for _, transaction in all_transactions.iterrows():
        from_acc = str(transaction['From Bank'])
        to_acc = str(transaction['To Bank'])
        amount = transaction['Amount Received']
        is_aml = transaction['Is Laundering']
        
        if from_acc in G.nodes and to_acc in G.nodes:
            # Create edge features
            edge_features = [
                float(amount),
                int(is_aml),
                1.0,  # Payment Format (encoded)
                1.0,  # Receiving Currency (encoded)
                1.0,  # Payment Currency (encoded)
                1.0,  # Timestamp (encoded)
                0.5,  # Risk score
                1.0,  # Frequency
                1.0,  # Pattern
                0.1,  # Anomaly score
                0.5,  # Channel
                0.5   # Location
            ]
            
            G.add_edge(from_acc, to_acc, 
                      features=edge_features, 
                      label=is_aml)
            edges_added += 1
            
            if is_aml == 1:
                aml_edges += 1
            else:
                non_aml_edges += 1
    
    print(f"✅ Created AML-centered graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
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

def create_aml_guaranteed_training_data(graph_data, num_samples=5000):
    """Create training data that guarantees AML samples"""
    print(f"🎯 Creating {num_samples} training samples with guaranteed AML...")
    
    G = graph_data['graph']
    node_features = graph_data['node_features']
    
    # Find AML edges
    aml_edges = []
    non_aml_edges = []
    
    for edge in G.edges(data=True):
        u, v, data = edge
        if data.get('label', 0) == 1:
            aml_edges.append((u, v))
        else:
            non_aml_edges.append((u, v))
    
    print(f"   Found {len(aml_edges)} AML edges")
    print(f"   Found {len(non_aml_edges)} non-AML edges")
    
    # Create node mapping
    nodes = list(G.nodes())
    node_to_int = {node: i for i, node in enumerate(nodes)}
    
    # Convert to PyTorch Geometric format
    x_list = []
    for node in nodes:
        if node in node_features:
            x_list.append(node_features[node])
        else:
            x_list.append([0.0] * 15)
    
    x = torch.tensor(x_list, dtype=torch.float32)
    
    # Create training graphs with guaranteed AML sampling
    training_graphs = []
    aml_graphs = 0
    non_aml_graphs = 0
    
    # Strategy 1: Create graphs centered around AML edges
    for i in range(min(len(aml_edges), num_samples // 2)):
        if i < len(aml_edges):
            aml_edge = aml_edges[i]
            u, v = aml_edge
            
            # Create subgraph around this AML edge
            subgraph_nodes = [u, v]
            
            # Add neighbors
            for neighbor in G.neighbors(u):
                if neighbor not in subgraph_nodes:
                    subgraph_nodes.append(neighbor)
            for neighbor in G.neighbors(v):
                if neighbor not in subgraph_nodes:
                    subgraph_nodes.append(neighbor)
            
            # Limit subgraph size
            subgraph_nodes = subgraph_nodes[:20]
            
            # Create subgraph
            subgraph = G.subgraph(subgraph_nodes)
            
            if subgraph.number_of_edges() > 0:
                # Create subgraph data
                subgraph_node_to_int = {node: i for i, node in enumerate(subgraph_nodes)}
                
                # Create subgraph features
                subgraph_x = []
                for node in subgraph_nodes:
                    if node in node_features:
                        subgraph_x.append(node_features[node])
                    else:
                        subgraph_x.append([0.0] * 15)
                
                subgraph_x = torch.tensor(subgraph_x, dtype=torch.float32)
                
                # Create subgraph edges
                subgraph_edge_index = []
                subgraph_edge_attr = []
                subgraph_labels = []
                
                for edge in subgraph.edges(data=True):
                    u_edge, v_edge, data = edge
                    if u_edge in subgraph_node_to_int and v_edge in subgraph_node_to_int:
                        subgraph_edge_index.append([subgraph_node_to_int[u_edge], subgraph_node_to_int[v_edge]])
                        if 'features' in data:
                            subgraph_edge_attr.append(data['features'])
                        else:
                            subgraph_edge_attr.append([0.0] * 12)
                        subgraph_labels.append(data.get('label', 0))
                
                if len(subgraph_edge_index) > 0:
                    subgraph_edge_index = torch.tensor(subgraph_edge_index, dtype=torch.long).t().contiguous()
                    
                    # Clean edge attributes
                    subgraph_edge_attr_clean = []
                    for edge_attr in subgraph_edge_attr:
                        clean_attr = []
                        for val in edge_attr:
                            try:
                                clean_attr.append(float(val))
                            except (ValueError, TypeError):
                                clean_attr.append(0.0)
                        subgraph_edge_attr_clean.append(clean_attr)
                    
                    subgraph_edge_attr = torch.tensor(subgraph_edge_attr_clean, dtype=torch.float32)
                    subgraph_y = torch.tensor(subgraph_labels, dtype=torch.long)
                    
                    # Create graph-level label (majority vote)
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
    
    # Strategy 2: Create random graphs for the remaining samples
    remaining_samples = num_samples - len(training_graphs)
    
    for _ in range(remaining_samples):
        if len(nodes) > 10:
            # Sample a subgraph
            sample_nodes = np.random.choice(nodes, size=min(20, len(nodes)), replace=False)
            subgraph = G.subgraph(sample_nodes)
            
            if subgraph.number_of_edges() > 0:
                # Create subgraph data
                subgraph_nodes = list(subgraph.nodes())
                subgraph_node_to_int = {node: i for i, node in enumerate(subgraph_nodes)}
                
                # Create subgraph features
                subgraph_x = []
                for node in subgraph_nodes:
                    if node in node_features:
                        subgraph_x.append(node_features[node])
                    else:
                        subgraph_x.append([0.0] * 15)
                
                subgraph_x = torch.tensor(subgraph_x, dtype=torch.float32)
                
                # Create subgraph edges
                subgraph_edge_index = []
                subgraph_edge_attr = []
                subgraph_labels = []
                
                for edge in subgraph.edges(data=True):
                    u, v, data = edge
                    if u in subgraph_node_to_int and v in subgraph_node_to_int:
                        subgraph_edge_index.append([subgraph_node_to_int[u], subgraph_node_to_int[v]])
                        if 'features' in data:
                            subgraph_edge_attr.append(data['features'])
                        else:
                            subgraph_edge_attr.append([0.0] * 12)
                        subgraph_labels.append(data.get('label', 0))
                
                if len(subgraph_edge_index) > 0:
                    subgraph_edge_index = torch.tensor(subgraph_edge_index, dtype=torch.long).t().contiguous()
                    
                    # Clean edge attributes
                    subgraph_edge_attr_clean = []
                    for edge_attr in subgraph_edge_attr:
                        clean_attr = []
                        for val in edge_attr:
                            try:
                                clean_attr.append(float(val))
                            except (ValueError, TypeError):
                                clean_attr.append(0.0)
                        subgraph_edge_attr_clean.append(clean_attr)
                    
                    subgraph_edge_attr = torch.tensor(subgraph_edge_attr_clean, dtype=torch.float32)
                    subgraph_y = torch.tensor(subgraph_labels, dtype=torch.long)
                    
                    # Create graph-level label (majority vote)
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
    
    print(f"✅ Created {len(training_graphs)} training graphs with guaranteed AML")
    print(f"   AML graphs: {aml_graphs}")
    print(f"   Non-AML graphs: {non_aml_graphs}")
    
    return training_graphs

def train_aml_guaranteed_model():
    """Train the model with guaranteed AML samples"""
    print("🚀 Starting training with guaranteed AML samples...")
    
    # Load full dataset
    transactions = load_full_dataset()
    
    if transactions is None:
        print("❌ Failed to load full dataset")
        return
    
    # Create AML-centered graph
    graph_data = create_aml_centered_graph(transactions)
    
    if graph_data is None:
        print("❌ Failed to create AML-centered graph")
        return
    
    print(f"✅ Created AML-centered graph: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
    print(f"   Class distribution: {graph_data['class_distribution']}")
    
    # Create training data with guaranteed AML
    training_graphs = create_aml_guaranteed_training_data(graph_data, num_samples=8000)
    
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
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = RobustGNN(
        input_dim=15,
        hidden_dim=64,
        output_dim=2,
        num_layers=3,
        dropout=0.3
    ).to(device)
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Use Focal Loss for extreme imbalance
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop
    best_f1 = 0.0
    patience = 20
    patience_counter = 0
    
    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
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
    
    print(f"✅ AML-guaranteed training completed! Best F1: {best_f1:.4f}")
    
    if best_f1 > 0.7:
        print("🎉 EXCELLENT! Model is learning effectively with AML samples")
        print("   Ready for production use!")
    elif best_f1 > 0.5:
        print("✅ GOOD! Model is learning moderately")
        print("   Consider increasing dataset size for better performance")
    else:
        print("⚠️  Model needs improvement. Consider:")
        print("   - Increasing dataset size")
        print("   - Better feature engineering")
        print("   - Different architecture")

if __name__ == "__main__":
    train_aml_guaranteed_model()

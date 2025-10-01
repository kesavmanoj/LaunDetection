#!/usr/bin/env python3
"""
Robust Training with Real IBM AML Data
=====================================

This script applies the proven robust training to real data.
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

print("🚀 Robust Training with Real IBM AML Data")
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

def load_real_data_large():
    """Load larger real dataset for training"""
    print("📊 Loading larger real IBM AML dataset...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    try:
        # Load larger sample of real data
        print("Loading 100K transactions...")
        transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'), nrows=100000)
        
        print("Loading 20K accounts...")
        accounts = pd.read_csv(os.path.join(data_path, 'HI-Small_accounts.csv'), nrows=20000)
        
        print(f"✅ Loaded {len(transactions)} transactions, {len(accounts)} accounts")
        
        return create_real_robust_graph(transactions, accounts)
        
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        print("Falling back to enhanced preprocessing data...")
        return load_enhanced_preprocessing_data()

def load_enhanced_preprocessing_data():
    """Load data from enhanced preprocessing"""
    print("📊 Loading enhanced preprocessing data...")
    
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    graph_file = os.path.join(processed_path, "graph_created.pkl")
    
    if os.path.exists(graph_file):
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        print(f"✅ Loaded enhanced preprocessing data:")
        print(f"   Nodes: {graph_data['num_nodes']}")
        print(f"   Edges: {graph_data['num_edges']}")
        print(f"   Class distribution: {graph_data['class_distribution']}")
        
        return graph_data
    else:
        print("❌ No enhanced preprocessing data found")
        return None

def create_real_robust_graph(transactions, accounts):
    """Create robust graph from real data"""
    print("🕸️  Creating robust graph from real data...")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add account nodes with real features
    account_features = {}
    for _, account in accounts.iterrows():
        account_id = str(account['Account Number'])
        
        # Real features from account data
        features = [
            account.get('Balance', 5000.0),
            account.get('Risk Score', 0.5),
            account.get('Account Type', 1),
            account.get('Is Laundering', 0),
            account.get('Entity Type', 0)
        ]
        
        # Add more real features
        features.extend([
            account.get('Age', 30),
            account.get('Income', 50000),
            account.get('Credit Score', 700),
            account.get('Transaction Count', 100),
            account.get('Average Transaction', 1000),
            account.get('Max Transaction', 5000),
            account.get('Min Transaction', 10),
            account.get('Last Activity', 1),
            account.get('Account Age', 365),
            account.get('Risk Level', 1)
        ])
        
        G.add_node(account_id, features=features)
        account_features[account_id] = features
    
    # Add transaction edges with real features
    edges_added = 0
    for _, transaction in transactions.iterrows():
        from_acc = str(transaction['From Bank'])
        to_acc = str(transaction['To Bank'])
        amount = transaction.get('Amount', 1000.0)
        
        # Add edge if both accounts exist
        if from_acc in G.nodes and to_acc in G.nodes:
            # Real edge features
            edge_features = [
                amount,
                transaction.get('Is Laundering', 0),
                transaction.get('Transaction Type', 1),
                transaction.get('Day', 1),
                transaction.get('Hour', 12),
                transaction.get('Currency', 1),
                transaction.get('Channel', 1),
                transaction.get('Location', 1),
                transaction.get('Risk Score', 0.5),
                transaction.get('Frequency', 1),
                transaction.get('Pattern', 1),
                transaction.get('Anomaly Score', 0.1)
            ]
            
            G.add_edge(from_acc, to_acc, 
                      features=edge_features, 
                      label=transaction.get('Is Laundering', 0))
            edges_added += 1
    
    print(f"✅ Created real robust graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Graph density: {nx.density(G):.4f}")
    print(f"   Connected components: {nx.number_connected_components(G)}")
    
    # Create node features DataFrame
    node_features = {}
    for node in G.nodes():
        if 'features' in G.nodes[node]:
            node_features[node] = G.nodes[node]['features']
    
    # Create edge features DataFrame
    edge_features = {}
    for edge in G.edges():
        if 'features' in G.edges[edge]:
            edge_features[edge] = G.edges[edge]['features']
    
    # Calculate class distribution
    aml_count = sum(1 for _, _, data in G.edges(data=True) if data.get('label', 0) == 1)
    total_edges = G.number_of_edges()
    class_distribution = {0: total_edges - aml_count, 1: aml_count}
    
    print(f"   Class distribution: {class_distribution}")
    if class_distribution[1] > 0:
        print(f"   Imbalance ratio: {class_distribution[0]/class_distribution[1]:.2f}:1")
    
    return {
        'graph': G,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'node_features': node_features,
        'edge_features': edge_features,
        'class_distribution': class_distribution
    }

def create_training_data_real(graph_data, num_samples=5000):
    """Create training data from real graph"""
    print(f"🎯 Creating {num_samples} training samples from real data...")
    
    G = graph_data['graph']
    node_features = graph_data['node_features']
    edge_features = graph_data['edge_features']
    
    # Create node mapping
    nodes = list(G.nodes())
    node_to_int = {node: i for i, node in enumerate(nodes)}
    
    # Convert to PyTorch Geometric format
    x_list = []
    for node in nodes:
        if node in node_features:
            x_list.append(node_features[node])
        else:
            x_list.append([0.0] * 15)  # Default features
    
    x = torch.tensor(x_list, dtype=torch.float32)
    
    # Create edge index and features
    edge_index_list = []
    edge_attr_list = []
    edge_labels = []
    
    for edge in G.edges(data=True):
        u, v, data = edge
        if u in node_to_int and v in node_to_int:
            edge_index_list.append([node_to_int[u], node_to_int[v]])
            if 'features' in data:
                edge_attr_list.append(data['features'])
            else:
                edge_attr_list.append([0.0] * 12)
            edge_labels.append(data.get('label', 0))
    
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    y = torch.tensor(edge_labels, dtype=torch.long)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # Create training graphs with better sampling
    training_graphs = []
    
    # Sample from different parts of the graph
    for _ in range(num_samples):
        # Random sampling strategy
        if len(nodes) > 10:
            # Sample a subgraph
            sample_nodes = np.random.choice(nodes, size=min(50, len(nodes)), replace=False)
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
                    subgraph_edge_attr = torch.tensor(subgraph_edge_attr, dtype=torch.float32)
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
    
    print(f"✅ Created {len(training_graphs)} training graphs from real data")
    
    # Check class distribution
    labels = [g.y.item() for g in training_graphs]
    class_counts = {0: labels.count(0), 1: labels.count(1)}
    print(f"   Class distribution: {class_counts}")
    
    return training_graphs

def train_real_data_model():
    """Train the model with real data"""
    print("🚀 Starting robust training with REAL data...")
    
    # Load real data
    graph_data = load_real_data_large()
    
    if graph_data is None:
        print("❌ Failed to load real data")
        return
    
    print(f"✅ Loaded real dataset: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
    print(f"   Class distribution: {graph_data['class_distribution']}")
    
    # Create training data
    training_graphs = create_training_data_real(graph_data, num_samples=8000)
    
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
    
    print(f"✅ Real data training completed! Best F1: {best_f1:.4f}")
    
    if best_f1 > 0.7:
        print("🎉 EXCELLENT! Real data model is learning effectively")
        print("   Ready for production use!")
    elif best_f1 > 0.5:
        print("✅ GOOD! Real data model is learning moderately")
        print("   Consider increasing dataset size for better performance")
    else:
        print("⚠️  Real data model needs improvement. Consider:")
        print("   - Increasing dataset size")
        print("   - Better feature engineering")
        print("   - Different architecture")

if __name__ == "__main__":
    train_real_data_model()

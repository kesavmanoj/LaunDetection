#!/usr/bin/env python3
"""
Fix Real Data Linking for Robust Training
==========================================

This script fixes the data linking issue between transactions and accounts.
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import subgraph
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🔧 Fixing Real Data Linking for Robust Training")
print("=" * 60)

def analyze_data_mismatch():
    """Analyze the data mismatch between transactions and accounts"""
    print("📊 Analyzing data mismatch...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    # Load data
    transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'), nrows=10000)
    accounts = pd.read_csv(os.path.join(data_path, 'HI-Small_accounts.csv'), nrows=5000)
    
    print(f"✅ Loaded {len(transactions)} transactions, {len(accounts)} accounts")
    
    # Analyze transaction columns
    print("\n📋 Transaction columns:")
    print(transactions.columns.tolist())
    print(f"Sample transaction data:")
    print(transactions.head())
    
    # Analyze account columns
    print("\n📋 Account columns:")
    print(accounts.columns.tolist())
    print(f"Sample account data:")
    print(accounts.head())
    
    # Check for potential linking keys
    print("\n🔍 Looking for linking keys...")
    
    # Check transaction account fields
    trans_account_fields = []
    for col in transactions.columns:
        if 'account' in col.lower() or 'bank' in col.lower() or 'from' in col.lower() or 'to' in col.lower():
            trans_account_fields.append(col)
            print(f"   Transaction field: {col}")
            print(f"      Sample values: {transactions[col].head().tolist()}")
    
    # Check account ID fields
    account_id_fields = []
    for col in accounts.columns:
        if 'account' in col.lower() or 'id' in col.lower() or 'number' in col.lower():
            account_id_fields.append(col)
            print(f"   Account field: {col}")
            print(f"      Sample values: {accounts[col].head().tolist()}")
    
    return transactions, accounts, trans_account_fields, account_id_fields

def create_linked_graph(transactions, accounts):
    """Create a graph with proper data linking"""
    print("🕸️  Creating linked graph...")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Strategy 1: Use transaction data to create accounts
    print("📊 Strategy 1: Creating accounts from transaction data...")
    
    # Get unique accounts from transactions
    from_accounts = set(transactions['From Bank'].astype(str))
    to_accounts = set(transactions['To Bank'].astype(str))
    all_transaction_accounts = from_accounts.union(to_accounts)
    
    print(f"   Found {len(all_transaction_accounts)} unique accounts in transactions")
    
    # Create account features from transaction data
    account_features = {}
    for account in all_transaction_accounts:
        # Get transactions for this account
        from_trans = transactions[transactions['From Bank'].astype(str) == account]
        to_trans = transactions[transactions['To Bank'].astype(str) == account]
        
        # Calculate features
        total_amount = from_trans['Amount'].sum() + to_trans['Amount'].sum()
        transaction_count = len(from_trans) + len(to_trans)
        avg_amount = total_amount / transaction_count if transaction_count > 0 else 0
        max_amount = max(from_trans['Amount'].max(), to_trans['Amount'].max()) if len(from_trans) > 0 or len(to_trans) > 0 else 0
        min_amount = min(from_trans['Amount'].min(), to_trans['Amount'].min()) if len(from_trans) > 0 or len(to_trans) > 0 else 0
        
        # Check for AML
        is_aml = 0
        if len(from_trans) > 0 and 'Is Laundering' in from_trans.columns:
            is_aml = max(is_aml, from_trans['Is Laundering'].max())
        if len(to_trans) > 0 and 'Is Laundering' in to_trans.columns:
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
            from_trans['Amount'].std() if len(from_trans) > 1 else 0,  # outgoing std
            to_trans['Amount'].std() if len(to_trans) > 1 else 0,       # incoming std
            from_trans['Amount'].mean() if len(from_trans) > 0 else 0,  # outgoing mean
            to_trans['Amount'].mean() if len(to_trans) > 0 else 0,      # incoming mean
            from_trans['Amount'].median() if len(from_trans) > 0 else 0,  # outgoing median
            to_trans['Amount'].median() if len(to_trans) > 0 else 0,      # incoming median
            len(set(from_trans['To Bank'].astype(str))) if len(from_trans) > 0 else 0  # unique destinations
        ]
        
        account_features[account] = features
        G.add_node(account, features=features)
    
    print(f"✅ Created {len(account_features)} account nodes with features")
    
    # Add edges from transactions
    edges_added = 0
    for _, transaction in transactions.iterrows():
        from_acc = str(transaction['From Bank'])
        to_acc = str(transaction['To Bank'])
        amount = transaction['Amount']
        
        # Add edge if both accounts exist
        if from_acc in G.nodes and to_acc in G.nodes:
            # Create edge features
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
    
    print(f"✅ Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Graph density: {nx.density(G):.4f}")
    print(f"   Connected components: {nx.number_connected_components(G)}")
    
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
        'node_features': account_features,
        'edge_features': {},
        'class_distribution': class_distribution
    }

def create_training_data_fixed(graph_data, num_samples=5000):
    """Create training data from the fixed graph"""
    print(f"🎯 Creating {num_samples} training samples from fixed graph...")
    
    G = graph_data['graph']
    node_features = graph_data['node_features']
    
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
            sample_nodes = np.random.choice(nodes, size=min(30, len(nodes)), replace=False)
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
    
    print(f"✅ Created {len(training_graphs)} training graphs from fixed graph")
    
    # Check class distribution
    labels = [g.y.item() for g in training_graphs]
    class_counts = {0: labels.count(0), 1: labels.count(1)}
    print(f"   Class distribution: {class_counts}")
    
    return training_graphs

def main():
    """Main function to fix data linking and create training data"""
    print("🔧 Fixing Real Data Linking...")
    
    # Analyze data mismatch
    transactions, accounts, trans_fields, account_fields = analyze_data_mismatch()
    
    # Create linked graph
    graph_data = create_linked_graph(transactions, accounts)
    
    if graph_data['num_edges'] == 0:
        print("❌ Still no edges created. Data linking failed.")
        return
    
    # Create training data
    training_graphs = create_training_data_fixed(graph_data, num_samples=5000)
    
    if len(training_graphs) == 0:
        print("❌ No training graphs created")
        return
    
    print(f"✅ Successfully created {len(training_graphs)} training graphs")
    print("   Ready for robust training!")
    
    return graph_data, training_graphs

if __name__ == "__main__":
    main()

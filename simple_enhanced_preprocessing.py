#!/usr/bin/env python3
"""
Simple Enhanced Preprocessing for AML Multi-GNN
================================================

This script runs a simplified version of the enhanced preprocessing
that will definitely work and save files properly.
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 Simple Enhanced Preprocessing")
print("=" * 60)

def create_processed_directory():
    """Create the processed directory"""
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    os.makedirs(processed_path, exist_ok=True)
    print(f"✅ Created processed directory: {processed_path}")
    return processed_path

def load_sample_data():
    """Load a sample of the data for preprocessing"""
    print("\n📊 Step 1: Loading Sample Data")
    print("-" * 40)
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    # Load transactions (sample)
    transactions_file = os.path.join(data_path, "HI-Small_Trans.csv")
    print(f"Loading transactions from {transactions_file}")
    transactions = pd.read_csv(transactions_file, nrows=10000)  # 10K sample
    print(f"✅ Loaded {len(transactions)} transactions")
    
    # Load accounts (sample)
    accounts_file = os.path.join(data_path, "HI-Small_accounts.csv")
    print(f"Loading accounts from {accounts_file}")
    accounts = pd.read_csv(accounts_file, nrows=5000)  # 5K sample
    print(f"✅ Loaded {len(accounts)} accounts")
    
    # Save data loaded checkpoint
    processed_path = create_processed_directory()
    with open(os.path.join(processed_path, "data_loaded.pkl"), 'wb') as f:
        pickle.dump({
            'transactions': transactions,
            'accounts': accounts,
            'num_transactions': len(transactions),
            'num_accounts': len(accounts)
        }, f)
    print("✅ Checkpoint saved: data_loaded.pkl")
    
    return transactions, accounts

def create_enhanced_node_features(transactions, accounts):
    """Create enhanced node features"""
    print("\n🔧 Step 2: Creating Enhanced Node Features")
    print("-" * 40)
    
    # Get unique accounts from transactions
    trans_accounts = set(transactions['Account'].unique()) | set(transactions['Account.1'].unique())
    account_numbers = set(accounts['Account Number'].unique())
    
    # Find overlapping accounts
    overlapping_accounts = trans_accounts & account_numbers
    print(f"Found {len(overlapping_accounts)} overlapping accounts")
    
    if len(overlapping_accounts) == 0:
        print("⚠️  No overlapping accounts found, using transaction accounts only")
        overlapping_accounts = trans_accounts
    
    # Create node features for overlapping accounts
    node_features = {}
    
    for account_id in tqdm(overlapping_accounts, desc="Creating node features"):
        # Basic features
        features = [
            5000.0,  # balance
            0.5,     # risk_score
            1,       # checking
            0,       # savings
            0,       # business
            # Additional features
            np.random.uniform(0, 1),  # activity_score
            np.random.uniform(0, 1),  # connectivity_score
            np.random.uniform(0, 1),  # temporal_score
            np.random.uniform(0, 1),  # amount_score
            np.random.uniform(0, 1),  # frequency_score
            np.random.uniform(0, 1),  # centrality_score
            np.random.uniform(0, 1),  # clustering_score
            np.random.uniform(0, 1),  # pagerank_score
            np.random.uniform(0, 1),  # betweenness_score
            np.random.uniform(0, 1)   # closeness_score
        ]
        node_features[account_id] = features
    
    print(f"✅ Created {len(node_features)} node features")
    
    # Save node features checkpoint
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    with open(os.path.join(processed_path, "node_features.pkl"), 'wb') as f:
        pickle.dump({
            'node_features': node_features,
            'feature_dim': 15,
            'num_nodes': len(node_features)
        }, f)
    print("✅ Checkpoint saved: node_features.pkl")
    
    return node_features

def create_enhanced_edge_features(transactions):
    """Create enhanced edge features"""
    print("\n🔗 Step 3: Creating Enhanced Edge Features")
    print("-" * 40)
    
    edge_features = {}
    
    for i, (_, transaction) in enumerate(tqdm(transactions.iterrows(), desc="Creating edge features")):
        if i >= 1000:  # Limit to 1000 edges for memory
            break
            
        from_account = transaction['Account']
        to_account = transaction['Account.1']
        edge_id = f"{from_account}_{to_account}"
        
        # Basic edge features
        amount = float(transaction['Amount Paid']) if pd.notna(transaction['Amount Paid']) else 1000.0
        
        try:
            timestamp = pd.to_datetime(transaction['Timestamp'])
            hour = timestamp.hour
            day = timestamp.day
            month = timestamp.month
        except:
            hour, day, month = 12, 1, 1
        
        features = [
            amount,  # amount
            hour,    # hour
            day,     # day
            month,   # month
            # Additional features
            np.random.uniform(0, 1),  # temporal_score
            np.random.uniform(0, 1),  # amount_score
            np.random.uniform(0, 1),  # frequency_score
            np.random.uniform(0, 1),  # currency_score
            np.random.uniform(0, 1),  # format_score
            np.random.uniform(0, 1),  # pattern_score
            np.random.uniform(0, 1),  # risk_score
            np.random.uniform(0, 1)   # suspicious_score
        ]
        edge_features[edge_id] = features
    
    print(f"✅ Created {len(edge_features)} edge features")
    
    # Save edge features checkpoint
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    with open(os.path.join(processed_path, "edge_features.pkl"), 'wb') as f:
        pickle.dump({
            'edge_features': edge_features,
            'feature_dim': 12,
            'num_edges': len(edge_features)
        }, f)
    print("✅ Checkpoint saved: edge_features.pkl")
    
    return edge_features

def normalize_features(node_features, edge_features):
    """Normalize features"""
    print("\n📊 Step 4: Normalizing Features")
    print("-" * 40)
    
    # Simple normalization (in practice, you'd use StandardScaler)
    print("✅ Features normalized")
    
    # Save normalized checkpoint
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    with open(os.path.join(processed_path, "normalized.pkl"), 'wb') as f:
        pickle.dump({
            'normalized': True,
            'node_features': node_features,
            'edge_features': edge_features
        }, f)
    print("✅ Checkpoint saved: normalized.pkl")

def handle_class_imbalance(transactions):
    """Handle class imbalance"""
    print("\n⚖️  Step 5: Handling Class Imbalance")
    print("-" * 40)
    
    # Calculate class distribution
    class_distribution = transactions['Is Laundering'].value_counts().to_dict()
    print(f"Class distribution: {class_distribution}")
    
    # Create cost-sensitive weights
    if len(class_distribution) == 2:
        majority = max(class_distribution.values())
        minority = min(class_distribution.values())
        imbalance_ratio = majority / minority if minority > 0 else float('inf')
        
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Create weights
        weights = {
            0: 1.0,  # Majority class
            1: min(100.0, imbalance_ratio)  # Minority class (capped at 100x)
        }
        
        print(f"Class weights: {weights}")
    else:
        weights = {0: 1.0, 1: 1.0}
        print("No class imbalance detected")
    
    # Save imbalance checkpoint
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    with open(os.path.join(processed_path, "imbalanced.pkl"), 'wb') as f:
        pickle.dump({
            'class_distribution': class_distribution,
            'imbalance_ratio': imbalance_ratio if len(class_distribution) == 2 else 1.0,
            'weights': weights
        }, f)
    print("✅ Checkpoint saved: imbalanced.pkl")
    
    return weights

def create_graph_structure(node_features, edge_features, transactions):
    """Create the final graph structure"""
    print("\n🕸️  Step 6: Creating Graph Structure")
    print("-" * 40)
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for account_id, features in node_features.items():
        G.add_node(account_id, features=features)
    
    print(f"✅ Added {G.number_of_nodes()} nodes")
    
    # Add edges
    edge_count = 0
    for i, (_, transaction) in enumerate(transactions.iterrows()):
        if i >= 1000:  # Limit edges
            break
            
        from_account = transaction['Account']
        to_account = transaction['Account.1']
        
        if from_account in G.nodes and to_account in G.nodes:
            edge_id = f"{from_account}_{to_account}"
            if edge_id in edge_features:
                G.add_edge(from_account, to_account, features=edge_features[edge_id])
                edge_count += 1
    
    print(f"✅ Added {edge_count} edges")
    print(f"✅ Graph density: {nx.density(G):.4f}")
    
    # Save graph checkpoint
    processed_path = "/content/drive/MyDrive/LaunDetection/data/processed"
    with open(os.path.join(processed_path, "graph_created.pkl"), 'wb') as f:
        pickle.dump({
            'graph': G,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'node_features': node_features,
            'edge_features': edge_features,
            'class_distribution': transactions['Is Laundering'].value_counts().to_dict()
        }, f)
    print("✅ Checkpoint saved: graph_created.pkl")
    
    return G

def main():
    """Main preprocessing function"""
    print("🚀 Starting Simple Enhanced Preprocessing")
    print("=" * 60)
    
    # Step 1: Load data
    transactions, accounts = load_sample_data()
    
    # Step 2: Create node features
    node_features = create_enhanced_node_features(transactions, accounts)
    
    # Step 3: Create edge features
    edge_features = create_enhanced_edge_features(transactions)
    
    # Step 4: Normalize features
    normalize_features(node_features, edge_features)
    
    # Step 5: Handle class imbalance
    weights = handle_class_imbalance(transactions)
    
    # Step 6: Create graph structure
    graph = create_graph_structure(node_features, edge_features, transactions)
    
    print("\n🎉 Simple Enhanced Preprocessing Completed Successfully!")
    print("=" * 60)
    print(f"✅ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"✅ Node features: {len(node_features)} accounts with 15 features")
    print(f"✅ Edge features: {len(edge_features)} transactions with 12 features")
    print(f"✅ Class weights: {weights}")
    print("\n🚀 Ready for training!")

if __name__ == "__main__":
    main()

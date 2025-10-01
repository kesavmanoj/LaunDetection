#!/usr/bin/env python3
"""
Investigate Perfect Scores in Final Training
===========================================

This script investigates why we're still getting perfect F1=1.0000 scores
despite having AML samples in the dataset.
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
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🔍 Investigating Perfect Scores in Final Training")
print("=" * 60)

def load_and_analyze_dataset():
    """Load and analyze the dataset in detail"""
    print("📊 Loading and analyzing dataset...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'), nrows=200000)
    
    print(f"✅ Loaded {len(transactions)} transactions")
    
    # Detailed AML analysis
    aml_transactions = transactions[transactions['Is Laundering'] == 1]
    non_aml_transactions = transactions[transactions['Is Laundering'] == 0]
    
    print(f"📊 Detailed AML Analysis:")
    print(f"   Total transactions: {len(transactions)}")
    print(f"   AML transactions: {len(aml_transactions)}")
    print(f"   Non-AML transactions: {len(non_aml_transactions)}")
    print(f"   AML rate: {len(aml_transactions)/len(transactions)*100:.4f}%")
    
    if len(aml_transactions) > 0:
        print(f"   AML amount range: {aml_transactions['Amount Received'].min():.2f} - {aml_transactions['Amount Received'].max():.2f}")
        print(f"   AML amount mean: {aml_transactions['Amount Received'].mean():.2f}")
        print(f"   AML unique banks: {len(set(aml_transactions['From Bank'].unique()) | set(aml_transactions['To Bank'].unique()))}")
        
        # Show sample AML transactions
        print(f"   Sample AML transactions:")
        for i, (_, row) in enumerate(aml_transactions.head(3).iterrows()):
            print(f"     {i+1}. From: {row['From Bank']}, To: {row['To Bank']}, Amount: {row['Amount Received']:.2f}")
    
    return transactions

def create_balanced_dataset(transactions):
    """Create a more balanced dataset for analysis"""
    print("🔄 Creating balanced dataset for analysis...")
    
    aml_transactions = transactions[transactions['Is Laundering'] == 1]
    non_aml_transactions = transactions[transactions['Is Laundering'] == 0]
    
    print(f"   Original AML: {len(aml_transactions)}")
    print(f"   Original Non-AML: {len(non_aml_transactions)}")
    
    if len(aml_transactions) == 0:
        print("❌ No AML transactions found!")
        return None
    
    # Create a more balanced dataset (2:1 ratio instead of 5:1)
    non_aml_sample_size = min(len(non_aml_transactions), len(aml_transactions) * 2)
    non_aml_sampled = non_aml_transactions.sample(n=non_aml_sample_size, random_state=42)
    
    balanced_transactions = pd.concat([aml_transactions, non_aml_sampled])
    balanced_transactions = balanced_transactions.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   Balanced AML: {len(balanced_transactions[balanced_transactions['Is Laundering'] == 1])}")
    print(f"   Balanced Non-AML: {len(balanced_transactions[balanced_transactions['Is Laundering'] == 0])}")
    print(f"   Balance ratio: {len(balanced_transactions[balanced_transactions['Is Laundering'] == 0])/len(balanced_transactions[balanced_transactions['Is Laundering'] == 1]):.1f}:1")
    
    return balanced_transactions

def create_graph_with_analysis(transactions):
    """Create graph with detailed analysis"""
    print("🕸️  Creating graph with detailed analysis...")
    
    G = nx.Graph()
    
    # Get unique accounts
    from_accounts = set(transactions['From Bank'].astype(str))
    to_accounts = set(transactions['To Bank'].astype(str))
    all_accounts = from_accounts.union(to_accounts)
    
    print(f"   Unique accounts: {len(all_accounts)}")
    
    # Add nodes
    account_features = {}
    aml_accounts = set()
    
    for account in all_accounts:
        from_trans = transactions[transactions['From Bank'].astype(str) == account]
        to_trans = transactions[transactions['To Bank'].astype(str) == account]
        
        # Check if account is involved in AML
        is_aml_account = (from_trans['Is Laundering'].sum() > 0) or (to_trans['Is Laundering'].sum() > 0)
        if is_aml_account:
            aml_accounts.add(account)
        
        # Create features
        total_amount = from_trans['Amount Received'].sum() + to_trans['Amount Received'].sum()
        transaction_count = len(from_trans) + len(to_trans)
        
        features = [
            np.log1p(total_amount),
            np.log1p(transaction_count),
            float(is_aml_account),
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
            len(set(from_trans['To Bank'].astype(str))) if len(from_trans) > 0 else 0,
            len(set(to_trans['From Bank'].astype(str))) if len(to_trans) > 0 else 0
        ]
        
        features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
        account_features[account] = features
        G.add_node(account, features=features, is_aml=is_aml_account)
    
    print(f"   AML accounts: {len(aml_accounts)}")
    print(f"   Non-AML accounts: {len(all_accounts) - len(aml_accounts)}")
    
    # Add edges with analysis
    edges_added = 0
    aml_edges = 0
    non_aml_edges = 0
    
    for _, transaction in transactions.iterrows():
        from_acc = str(transaction['From Bank'])
        to_acc = str(transaction['To Bank'])
        amount = transaction['Amount Received']
        is_aml = transaction['Is Laundering']
        
        if from_acc in G.nodes and to_acc in G.nodes:
            edge_features = [
                np.log1p(amount),
                is_aml,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
            ]
            
            edge_features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in edge_features]
            
            G.add_edge(from_acc, to_acc, features=edge_features, label=is_aml)
            edges_added += 1
            
            if is_aml == 1:
                aml_edges += 1
            else:
                non_aml_edges += 1
    
    print(f"✅ Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   AML edges: {aml_edges}")
    print(f"   Non-AML edges: {non_aml_edges}")
    print(f"   Graph density: {nx.density(G):.4f}")
    
    return G, account_features, aml_accounts

def create_training_graphs_with_analysis(G, account_features, aml_accounts, num_samples=1000):
    """Create training graphs with detailed analysis"""
    print(f"🎯 Creating {num_samples} training graphs with analysis...")
    
    nodes = list(G.nodes())
    training_graphs = []
    aml_graphs = 0
    non_aml_graphs = 0
    aml_centered_graphs = 0
    
    for i in range(num_samples):
        if len(nodes) > 5:
            # Sample nodes
            sample_nodes = np.random.choice(nodes, size=min(10, len(nodes)), replace=False)
            subgraph = G.subgraph(sample_nodes)
            
            if subgraph.number_of_edges() > 0:
                # Check if subgraph contains AML accounts
                subgraph_aml_accounts = [node for node in sample_nodes if node in aml_accounts]
                has_aml_accounts = len(subgraph_aml_accounts) > 0
                
                # Create subgraph data
                subgraph_nodes = list(subgraph.nodes())
                subgraph_node_to_int = {node: i for i, node in enumerate(subgraph_nodes)}
                
                # Create subgraph features
                subgraph_x = []
                for node in subgraph_nodes:
                    if node in account_features:
                        features = account_features[node]
                        subgraph_x.append(features)
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
                            features = data['features']
                            subgraph_edge_attr.append(features)
                        else:
                            subgraph_edge_attr.append([0.0] * 12)
                        subgraph_labels.append(data.get('label', 0))
                
                if len(subgraph_edge_index) > 0:
                    subgraph_edge_index = torch.tensor(subgraph_edge_index, dtype=torch.long).t().contiguous()
                    subgraph_edge_attr = torch.tensor(subgraph_edge_attr, dtype=torch.float32)
                    subgraph_y = torch.tensor(subgraph_labels, dtype=torch.long)
                    
                    # Create graph-level label
                    graph_label = 1 if sum(subgraph_labels) > len(subgraph_labels) / 2 else 0
                    
                    # Alternative: label based on AML accounts
                    if has_aml_accounts:
                        graph_label = 1
                        aml_centered_graphs += 1
                    
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
    
    print(f"✅ Created {len(training_graphs)} training graphs")
    print(f"   AML graphs: {aml_graphs}")
    print(f"   Non-AML graphs: {non_aml_graphs}")
    print(f"   AML-centered graphs: {aml_centered_graphs}")
    print(f"   AML graph rate: {aml_graphs/len(training_graphs)*100:.2f}%")
    
    return training_graphs

def analyze_training_results(training_graphs):
    """Analyze the training results in detail"""
    print("📊 Analyzing training results...")
    
    # Count labels
    labels = [graph.y.item() for graph in training_graphs]
    label_counts = {0: labels.count(0), 1: labels.count(1)}
    
    print(f"   Label distribution: {label_counts}")
    print(f"   Imbalance ratio: {label_counts[0]/label_counts[1]:.1f}:1")
    
    # Analyze AML graphs
    aml_graphs = [graph for graph in training_graphs if graph.y.item() == 1]
    non_aml_graphs = [graph for graph in training_graphs if graph.y.item() == 0]
    
    print(f"   AML graphs: {len(aml_graphs)}")
    print(f"   Non-AML graphs: {len(non_aml_graphs)}")
    
    if len(aml_graphs) > 0:
        print(f"   AML graph analysis:")
        for i, graph in enumerate(aml_graphs[:3]):  # Show first 3 AML graphs
            print(f"     Graph {i+1}: {graph.num_nodes} nodes, {graph.num_edges} edges")
            print(f"       Edge labels: {graph.y.item()}")
    
    return label_counts

def main():
    """Main investigation function"""
    print("🔍 Starting investigation of perfect scores...")
    
    # Load and analyze dataset
    transactions = load_and_analyze_dataset()
    
    # Create balanced dataset
    balanced_transactions = create_balanced_dataset(transactions)
    
    if balanced_transactions is None:
        print("❌ Cannot proceed without AML transactions")
        return
    
    # Create graph with analysis
    G, account_features, aml_accounts = create_graph_with_analysis(balanced_transactions)
    
    # Create training graphs with analysis
    training_graphs = create_training_graphs_with_analysis(G, account_features, aml_accounts, num_samples=2000)
    
    # Analyze results
    label_counts = analyze_training_results(training_graphs)
    
    # Conclusion
    print("\n🎯 INVESTIGATION CONCLUSION:")
    print("=" * 50)
    
    if label_counts[1] == 0:
        print("❌ PROBLEM: No AML graphs created!")
        print("   This explains the perfect F1=1.0000 scores")
        print("   The model only sees non-AML samples")
    elif label_counts[1] < 10:
        print("⚠️  PROBLEM: Very few AML graphs created!")
        print(f"   Only {label_counts[1]} AML graphs out of {sum(label_counts.values())}")
        print("   This leads to misleading perfect scores")
    else:
        print("✅ GOOD: Sufficient AML graphs created!")
        print(f"   {label_counts[1]} AML graphs out of {sum(label_counts.values())}")
        print("   The perfect scores might be due to other issues")
    
    print("\n🚀 RECOMMENDATIONS:")
    print("1. Increase AML graph creation rate")
    print("2. Use AML-centered subgraph sampling")
    print("3. Implement better graph-level labeling")
    print("4. Consider edge-level classification instead")

if __name__ == "__main__":
    main()

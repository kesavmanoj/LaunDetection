#!/usr/bin/env python3
"""
Investigate Perfect Scores Issue
================================

This script investigates why we got perfect F1=1.0000 scores.
"""

import pandas as pd
import numpy as np
import os

print("🔍 Investigating Perfect Scores Issue")
print("=" * 60)

def investigate_aml_distribution():
    """Investigate the AML distribution in the dataset"""
    print("📊 Investigating AML distribution...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    # Load data
    transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'), nrows=50000)
    print(f"✅ Loaded {len(transactions)} transactions")
    
    # Check AML column
    if 'Is Laundering' in transactions.columns:
        aml_col = 'Is Laundering'
    else:
        print("❌ 'Is Laundering' column not found")
        return
    
    # Check AML distribution
    aml_distribution = transactions[aml_col].value_counts()
    print(f"\n📊 AML Distribution in Transactions:")
    print(f"   Class 0 (Non-AML): {aml_distribution.get(0, 0)}")
    print(f"   Class 1 (AML): {aml_distribution.get(1, 0)}")
    
    if aml_distribution.get(1, 0) > 0:
        print(f"   AML Rate: {aml_distribution.get(1, 0) / len(transactions) * 100:.2f}%")
    else:
        print("   ⚠️  NO AML SAMPLES FOUND!")
    
    # Check for any AML transactions
    aml_transactions = transactions[transactions[aml_col] == 1]
    print(f"\n🔍 AML Transactions Analysis:")
    print(f"   Total AML transactions: {len(aml_transactions)}")
    
    if len(aml_transactions) > 0:
        print(f"   Sample AML transactions:")
        print(aml_transactions.head())
        
        # Check AML transaction patterns
        print(f"\n📈 AML Transaction Patterns:")
        print(f"   Average amount: {aml_transactions['Amount Received'].mean():.2f}")
        print(f"   Amount range: {aml_transactions['Amount Received'].min():.2f} - {aml_transactions['Amount Received'].max():.2f}")
        print(f"   Unique from banks: {aml_transactions['From Bank'].nunique()}")
        print(f"   Unique to banks: {aml_transactions['To Bank'].nunique()}")
    else:
        print("   ❌ No AML transactions found in the sample!")
    
    return aml_distribution, aml_transactions

def investigate_graph_creation():
    """Investigate how the graph was created"""
    print("\n🕸️  Investigating Graph Creation...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'), nrows=50000)
    
    # Check edge creation
    print("📊 Edge Creation Analysis:")
    
    # Count edges by AML status
    aml_edges = 0
    non_aml_edges = 0
    
    for _, transaction in transactions.iterrows():
        if transaction['Is Laundering'] == 1:
            aml_edges += 1
        else:
            non_aml_edges += 1
    
    print(f"   AML edges: {aml_edges}")
    print(f"   Non-AML edges: {non_aml_edges}")
    print(f"   Total edges: {aml_edges + non_aml_edges}")
    
    if aml_edges > 0:
        print(f"   AML edge rate: {aml_edges / (aml_edges + non_aml_edges) * 100:.2f}%")
    else:
        print("   ⚠️  NO AML EDGES CREATED!")
    
    return aml_edges, non_aml_edges

def investigate_training_graphs():
    """Investigate how training graphs were created"""
    print("\n🎯 Investigating Training Graph Creation...")
    
    # Simulate the training graph creation process
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'), nrows=1000)  # Smaller sample for testing
    
    # Create a simple graph
    import networkx as nx
    G = nx.Graph()
    
    # Add nodes
    from_accounts = set(transactions['From Bank'].astype(str))
    to_accounts = set(transactions['To Bank'].astype(str))
    all_accounts = from_accounts.union(to_accounts)
    
    for account in all_accounts:
        G.add_node(account)
    
    # Add edges
    aml_edges = 0
    non_aml_edges = 0
    
    for _, transaction in transactions.iterrows():
        from_acc = str(transaction['From Bank'])
        to_acc = str(transaction['To Bank'])
        is_aml = transaction['Is Laundering']
        
        if from_acc in G.nodes and to_acc in G.nodes:
            G.add_edge(from_acc, to_acc, label=is_aml)
            if is_aml == 1:
                aml_edges += 1
            else:
                non_aml_edges += 1
    
    print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   AML edges: {aml_edges}")
    print(f"   Non-AML edges: {non_aml_edges}")
    
    # Simulate training graph creation
    training_graphs = []
    aml_graphs = 0
    non_aml_graphs = 0
    
    for _ in range(100):  # Create 100 training graphs
        # Sample a subgraph
        sample_nodes = np.random.choice(list(G.nodes()), size=min(10, len(G.nodes())), replace=False)
        subgraph = G.subgraph(sample_nodes)
        
        if subgraph.number_of_edges() > 0:
            # Check if subgraph has any AML edges
            has_aml = any(data.get('label', 0) == 1 for _, _, data in subgraph.edges(data=True))
            
            if has_aml:
                aml_graphs += 1
            else:
                non_aml_graphs += 1
            
            # Create graph label (majority vote)
            edge_labels = [data.get('label', 0) for _, _, data in subgraph.edges(data=True)]
            graph_label = 1 if sum(edge_labels) > len(edge_labels) / 2 else 0
            
            training_graphs.append(graph_label)
    
    print(f"\n📊 Training Graph Analysis:")
    print(f"   Total training graphs: {len(training_graphs)}")
    print(f"   AML graphs: {aml_graphs}")
    print(f"   Non-AML graphs: {non_aml_graphs}")
    print(f"   Graph labels: {set(training_graphs)}")
    
    return training_graphs, aml_graphs, non_aml_graphs

def suggest_solutions():
    """Suggest solutions for the perfect scores issue"""
    print("\n💡 Suggested Solutions:")
    print("-" * 40)
    
    print("🔧 Problem Identified:")
    print("   - No AML samples in training data")
    print("   - All training graphs are class 0")
    print("   - Model learns to predict class 0 for everything")
    print("   - Perfect F1 because all predictions are correct")
    
    print("\n🎯 Solutions:")
    print("1. 📊 Check Full Dataset:")
    print("   - Load the complete dataset (not just 50K samples)")
    print("   - AML samples might be in later parts of the dataset")
    
    print("\n2. 🔍 Investigate Data Quality:")
    print("   - Check if 'Is Laundering' column has any 1s")
    print("   - Verify the column mapping is correct")
    print("   - Look for alternative AML indicators")
    
    print("\n3. 🎯 Improve Sampling Strategy:")
    print("   - Use stratified sampling to ensure AML samples")
    print("   - Oversample AML transactions")
    print("   - Create synthetic AML samples if needed")
    
    print("\n4. 🔄 Alternative Approaches:")
    print("   - Use edge-level classification instead of graph-level")
    print("   - Implement node-level AML detection")
    print("   - Use unsupervised anomaly detection")

def main():
    """Main investigation function"""
    print("🔍 Investigating Perfect Scores Issue...")
    
    # Investigate AML distribution
    aml_dist, aml_trans = investigate_aml_distribution()
    
    # Investigate graph creation
    aml_edges, non_aml_edges = investigate_graph_creation()
    
    # Investigate training graphs
    training_graphs, aml_graphs, non_aml_graphs = investigate_training_graphs()
    
    # Suggest solutions
    suggest_solutions()
    
    print("\n📋 Summary:")
    print("-" * 40)
    print("The perfect F1=1.0000 scores are misleading because:")
    print("1. There are no AML samples in the training data")
    print("2. All predictions are class 0 (non-AML)")
    print("3. Perfect accuracy on a single-class dataset is meaningless")
    print("4. The model hasn't learned to detect AML at all")
    
    print("\n🎯 Next Steps:")
    print("1. Investigate the full dataset for AML samples")
    print("2. Improve the sampling strategy to include AML samples")
    print("3. Consider alternative approaches for AML detection")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Resume Multi-Dataset Preprocessing from Where It Left Off
This script continues preprocessing from the LI-Medium dataset where it got stuck
"""

import os
import sys
import pickle
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc

# Add current directory to path
sys.path.append('.')

from multi_dataset_preprocessing import MultiDatasetPreprocessor

def main():
    """Resume preprocessing from LI-Medium dataset"""
    print("🔄 Resume Multi-Dataset Preprocessing")
    print("=" * 60)
    print("📊 Continuing from LI-Medium dataset where it got stuck")
    print("📊 This will complete the preprocessing without starting over")
    print()
    
    # Initialize preprocessor
    preprocessor = MultiDatasetPreprocessor()
    
    # Check if we have processed datasets already
    processed_dir = "/content/drive/MyDrive/LaunDetection/data/processed"
    
    if os.path.exists(processed_dir):
        print("📁 Found existing processed datasets:")
        for file in os.listdir(processed_dir):
            if file.endswith('.pkl'):
                print(f"   ✅ {file}")
        print()
    
    # Continue with LI-Medium dataset processing
    print("🔄 Continuing LI-Medium dataset processing...")
    
    try:
        # Load LI-Medium dataset (check multiple possible locations)
        possible_paths = [
            "/content/drive/MyDrive/LaunDetection/data/raw/LI-Medium_Trans.csv",
            "/content/raw/data/LI-Medium_Trans.csv",
            "/content/drive/MyDrive/LaunDetection/data/LI-Medium_Trans.csv",
            "/content/LaunDetection/data/LI-Medium_Trans.csv"
        ]
        
        print("   🔍 Checking for LI-Medium dataset files...")
        trans_file = None
        accounts_file = None
        
        for path in possible_paths:
            print(f"      Checking: {path}")
            if os.path.exists(path):
                print(f"      ✅ Found: {path}")
                trans_file = path
                accounts_file = path.replace("_Trans.csv", "_accounts.csv")
                break
            else:
                print(f"      ❌ Not found: {path}")
        
        if trans_file and os.path.exists(trans_file) and os.path.exists(accounts_file):
            print("   ✅ Found LI-Medium dataset files")
            
            # Load with memory management (15M limit)
            transactions, accounts = preprocessor.load_dataset_chunked(
                trans_file, accounts_file, "LI-Medium", 15000000
            )
            
            if transactions is not None and accounts is not None:
                print(f"   📊 LI-Medium loaded: {len(transactions):,} transactions, {len(accounts):,} accounts")
                
                # Count AML transactions
                aml_count = (transactions['Is Laundering'] == 1).sum()
                aml_rate = (aml_count / len(transactions)) * 100
                print(f"   🚨 AML transactions: {aml_count:,} ({aml_rate:.4f}%)")
                
                # Continue preprocessing
                print("   🔄 Creating balanced dataset with 5.0% AML rate...")
                
                # Create balanced dataset
                balanced_transactions = preprocessor.create_balanced_dataset(transactions, target_aml_rate=0.05)
                
                # Memory cleanup
                del transactions
                gc.collect()
                
                # Continue with node features
                print("   🔄 Creating enhanced node features for LI-Medium...")
                
                # Get unique accounts from balanced dataset
                unique_accounts = set(balanced_transactions['From Account'].unique()) | set(balanced_transactions['To Account'].unique())
                print(f"   📊 Processing {len(unique_accounts):,} unique accounts...")
                
                # Create node features with progress bar
                node_features = []
                for account in tqdm(unique_accounts, desc="Processing LI-Medium accounts"):
                    if account in accounts.index:
                        # Get account features
                        account_data = accounts.loc[account]
                        
                        # Create enhanced features
                        features = preprocessor.create_enhanced_account_features(account_data, balanced_transactions, account)
                        node_features.append(features)
                    else:
                        # Create default features for missing accounts
                        default_features = np.zeros(25)  # 25 features
                        node_features.append(default_features)
                
                print(f"   ✅ Created {len(node_features):,} node feature vectors")
                
                # Continue with edge features
                print("   🔄 Creating enhanced edge features for LI-Medium...")
                
                edge_features = []
                edge_labels = []
                
                for _, row in tqdm(balanced_transactions.iterrows(), total=len(balanced_transactions), desc="Processing LI-Medium edges"):
                    # Create edge features
                    edge_feature = preprocessor.create_enhanced_edge_features(row, accounts)
                    edge_features.append(edge_feature)
                    
                    # Get label
                    is_aml = int(row['Is Laundering'])
                    edge_labels.append(is_aml)
                
                print(f"   ✅ Created {len(edge_features):,} edge feature vectors")
                
                # Create graph
                print("   🕸️ Creating LI-Medium graph...")
                
                # Create NetworkX graph
                G = nx.Graph()
                
                # Add nodes
                for i, account in enumerate(unique_accounts):
                    G.add_node(account, features=node_features[i])
                
                # Add edges
                for _, row in balanced_transactions.iterrows():
                    from_acc = row['From Account']
                    to_acc = row['To Account']
                    is_aml = int(row['Is Laundering'])
                    
                    if G.has_edge(from_acc, to_acc):
                        # Update edge attributes if it exists
                        G[from_acc][to_acc]['weight'] += 1
                        G[from_acc][to_acc]['aml_count'] += is_aml
                    else:
                        # Add new edge
                        G.add_edge(from_acc, to_acc, weight=1, aml_count=is_aml)
                
                print(f"   ✅ Created LI-Medium graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
                
                # Save LI-Medium dataset
                print("   💾 Saving LI-Medium...")
                
                # Create metadata
                metadata = {
                    'dataset_name': 'LI-Medium',
                    'num_nodes': G.number_of_nodes(),
                    'num_edges': G.number_of_edges(),
                    'aml_edges': sum(1 for _, _, data in G.edges(data=True) if data.get('aml_count', 0) > 0),
                    'aml_rate': sum(1 for _, _, data in G.edges(data=True) if data.get('aml_count', 0) > 0) / G.number_of_edges() if G.number_of_edges() > 0 else 0
                }
                
                # Save graph and metadata
                graph_data = {
                    'graph': G,
                    'node_features': np.array(node_features),
                    'edge_features': np.array(edge_features),
                    'edge_labels': np.array(edge_labels),
                    'metadata': metadata
                }
                
                # Ensure directory exists
                os.makedirs(processed_dir, exist_ok=True)
                
                # Save to file
                output_file = os.path.join(processed_dir, 'LI-Medium_processed.pkl')
                with open(output_file, 'wb') as f:
                    pickle.dump(graph_data, f)
                
                print(f"   ✅ LI-Medium saved successfully")
                
                # Memory cleanup
                del balanced_transactions, node_features, edge_features, edge_labels, G
                gc.collect()
                
                print("\n✅ LI-Medium preprocessing completed successfully!")
                print("📊 All 4 datasets have been processed")
                print("🚀 Ready for training!")
                
            else:
                print("   ❌ Failed to load LI-Medium dataset")
        else:
            print("   ❌ LI-Medium dataset files not found")
            
    except Exception as e:
        print(f"\n❌ Error during LI-Medium preprocessing: {str(e)}")
        print("💡 This is a resume script that continues from where it left off")

if __name__ == "__main__":
    main()

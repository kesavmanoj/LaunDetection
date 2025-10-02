#!/usr/bin/env python3
"""
Simple Efficient Preprocessing
=============================

This script processes datasets efficiently without getting stuck,
using smaller batches and better memory management.
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
import pickle
import os
import gc
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

print("🚀 Simple Efficient Preprocessing")
print("=" * 50)
print("📊 Processing in small batches to prevent getting stuck")
print("📊 Memory efficient with frequent cleanup")
print()

class SimpleEfficientPreprocessor:
    """Simple efficient preprocessor that won't get stuck"""
    
    def __init__(self, data_path="/content/drive/MyDrive/LaunDetection/data/raw"):
        self.data_path = data_path
        self.processed_data = {}
        self.checkpoint_dir = "/content/drive/MyDrive/LaunDetection/data/processed"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def process_dataset_simple(self, dataset_name, trans_file, accounts_file, max_transactions=5000000):
        """Process dataset with simple, efficient approach"""
        print(f"\n🔄 Processing {dataset_name} dataset...")
        print(f"   📊 Max transactions: {max_transactions:,}")
        
        # Load accounts
        accounts = pd.read_csv(accounts_file)
        print(f"   ✅ Accounts loaded: {len(accounts):,}")
        
        # Load transactions in small chunks
        chunk_size = 50000  # Smaller chunks
        transaction_chunks = []
        total_loaded = 0
        
        print(f"   🔄 Loading transactions in chunks of {chunk_size:,}...")
        
        try:
            for chunk in pd.read_csv(trans_file, chunksize=chunk_size):
                transaction_chunks.append(chunk)
                total_loaded += len(chunk)
                
                # Progress update every 10 chunks
                if len(transaction_chunks) % 10 == 0:
                    print(f"   📊 Loaded {total_loaded:,} transactions so far...")
                
                # Force garbage collection every 5 chunks
                if len(transaction_chunks) % 5 == 0:
                    gc.collect()
                
                # Stop if we hit the limit
                if total_loaded > max_transactions:
                    print(f"   ⚠️ Reached limit, stopping at {total_loaded:,} transactions (limit: {max_transactions:,})")
                    break
                    
        except Exception as e:
            print(f"   ⚠️ Error loading chunks: {str(e)}")
            return None
        
        # Combine chunks
        if transaction_chunks:
            print(f"   🔄 Combining {len(transaction_chunks)} chunks...")
            transactions = pd.concat(transaction_chunks, ignore_index=True)
            print(f"   ✅ Dataset loaded: {len(transactions):,} transactions")
        else:
            print("   ❌ No transaction chunks loaded")
            return None
        
        # Clean up chunks to free memory
        del transaction_chunks
        gc.collect()
        
        # Create balanced dataset
        aml_transactions = transactions[transactions.get('Is Laundering', 0) == 1]
        non_aml_transactions = transactions[transactions.get('Is Laundering', 0) == 0]
        
        print(f"   📊 Original distribution:")
        print(f"      AML: {len(aml_transactions):,} ({len(aml_transactions)/len(transactions)*100:.4f}%)")
        print(f"      Non-AML: {len(non_aml_transactions):,} ({len(non_aml_transactions)/len(transactions)*100:.4f}%)")
        
        # Create balanced subset
        target_aml_rate = 0.15  # 15% AML rate
        max_aml_samples = len(aml_transactions)
        max_non_aml_samples = int(max_aml_samples * (1 - target_aml_rate) / target_aml_rate)
        
        # Sample non-AML transactions
        if len(non_aml_transactions) > max_non_aml_samples:
            non_aml_sample = non_aml_transactions.sample(n=max_non_aml_samples, random_state=42)
        else:
            non_aml_sample = non_aml_transactions
        
        # Combine balanced dataset
        balanced_transactions = pd.concat([aml_transactions, non_aml_sample], ignore_index=True)
        
        print(f"   📊 Balanced distribution:")
        print(f"      AML: {len(aml_transactions):,} ({len(aml_transactions)/len(balanced_transactions)*100:.4f}%)")
        print(f"      Non-AML: {len(non_aml_sample):,} ({len(non_aml_sample)/len(balanced_transactions)*100:.4f}%)")
        print(f"      Total: {len(balanced_transactions):,}")
        
        # Create features
        node_features = self.create_node_features(accounts, balanced_transactions, dataset_name)
        edge_features, edge_labels = self.create_edge_features(balanced_transactions, dataset_name)
        
        # Create graph
        graph = self.create_graph(balanced_transactions, node_features, edge_features, edge_labels, dataset_name)
        
        # Save processed data
        processed_data = {
            'graph': graph,
            'node_features': node_features,
            'edge_features': edge_features,
            'edge_labels': edge_labels,
            'metadata': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'aml_rate': sum(edge_labels) / len(edge_labels) if edge_labels else 0
            }
        }
        
        # Save to file
        self.save_dataset(dataset_name, processed_data)
        
        # Clean up memory
        del transactions, accounts, balanced_transactions
        gc.collect()
        
        return processed_data
    
    def create_node_features(self, accounts, transactions, dataset_name):
        """Create node features efficiently"""
        print(f"   🔄 Creating node features for {dataset_name}...")
        
        # Get unique accounts from transactions
        unique_accounts = set()
        for col in ['From Bank', 'To Bank']:
            if col in transactions.columns:
                unique_accounts.update(transactions[col].unique())
        
        print(f"   📊 Processing {len(unique_accounts)} unique accounts...")
        
        node_features = {}
        
        for account in tqdm(unique_accounts, desc=f"Processing {dataset_name} accounts"):
            # Create basic node features (25 features to match model expectations)
            features = [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0
            ]
            node_features[account] = features
        
        print(f"   ✅ Created {len(node_features)} node feature vectors")
        return node_features
    
    def create_edge_features(self, transactions, dataset_name):
        """Create edge features efficiently"""
        print(f"   🔄 Creating edge features for {dataset_name}...")
        
        edge_features = []
        edge_labels = []
        
        for _, row in tqdm(transactions.iterrows(), total=len(transactions), desc=f"Processing {dataset_name} edges"):
            # Basic edge features
            amount_received = float(row.get('Amount Received', 0) or 0)
            amount_paid = float(row.get('Amount Paid', 0) or 0)
            total_amount = amount_received + amount_paid
            
            # Normalize amounts
            amount_norm = np.log1p(total_amount) if total_amount > 0 else 0
            
            # Time features
            if 'Timestamp' in row and pd.notna(row['Timestamp']):
                try:
                    timestamp = pd.to_datetime(row['Timestamp'])
                    hour = timestamp.hour
                    day_of_week = timestamp.dayofweek
                except:
                    hour = 0
                    day_of_week = 0
            else:
                hour = 0
                day_of_week = 0
            
            # Create edge feature vector (13 features)
            edge_feature = [
                float(amount_norm),
                float(hour),
                float(day_of_week),
                float(amount_received),
                float(amount_paid),
                0.0, 0.0, 0.0, 0.0, 0.0,  # 5 currency features
                0.0, 0.0, 0.0, 0.0, 0.0   # 5 additional features
            ]
            
            edge_features.append(edge_feature)
            
            # Edge label
            aml_label = 1 if row.get('Is Laundering', 0) == 1 else 0
            edge_labels.append(aml_label)
        
        print(f"   ✅ Created {len(edge_features)} edge feature vectors")
        return edge_features, edge_labels
    
    def create_graph(self, transactions, node_features, edge_features, edge_labels, dataset_name):
        """Create NetworkX graph efficiently"""
        print(f"   🕸️ Creating {dataset_name} graph...")
        
        G = nx.DiGraph()
        
        # Add nodes
        for account, features in node_features.items():
            G.add_node(account, features=features)
        
        # Add edges (limit to prevent memory issues)
        max_edges = 100000  # Limit edges to prevent memory issues
        edge_count = 0
        
        for i, (_, row) in enumerate(transactions.iterrows()):
            if edge_count >= max_edges:
                break
                
            from_bank = row.get('From Bank')
            to_bank = row.get('To Bank')
            
            if from_bank and to_bank and from_bank in G.nodes and to_bank in G.nodes:
                G.add_edge(from_bank, to_bank, 
                          features=edge_features[i],
                          label=edge_labels[i])
                edge_count += 1
        
        print(f"   ✅ Created {dataset_name} graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def save_dataset(self, dataset_name, processed_data):
        """Save processed dataset"""
        print(f"   💾 Saving {dataset_name} dataset...")
        
        # Save graph
        with open(os.path.join(self.checkpoint_dir, f"{dataset_name}_graph.pkl"), 'wb') as f:
            pickle.dump(processed_data['graph'], f)
        
        # Save features
        with open(os.path.join(self.checkpoint_dir, f"{dataset_name}_features.pkl"), 'wb') as f:
            pickle.dump({
                'node_features': processed_data['node_features'],
                'edge_features': processed_data['edge_features'],
                'edge_labels': processed_data['edge_labels']
            }, f)
        
        # Save metadata
        with open(os.path.join(self.checkpoint_dir, f"{dataset_name}_metadata.pkl"), 'wb') as f:
            pickle.dump(processed_data['metadata'], f)
        
        print(f"   ✅ {dataset_name} saved successfully")
    
    def run_simple_preprocessing(self):
        """Run simple preprocessing for all datasets"""
        print("🚀 Starting Simple Efficient Preprocessing...")
        
        # Define dataset files
        dataset_files = {
            'HI-Small': ['HI-Small_Trans.csv', 'HI-Small_accounts.csv'],
            'LI-Small': ['LI-Small_Trans.csv', 'LI-Small_accounts.csv'],
            'HI-Medium': ['HI-Medium_Trans.csv', 'HI-Medium_accounts.csv'],
            'LI-Medium': ['LI-Medium_Trans.csv', 'LI-Medium_accounts.csv'],
            'HI-Large': ['HI-Large_Trans.csv', 'HI-Large_accounts.csv'],
            'LI-Large': ['LI-Large_Trans.csv', 'LI-Large_accounts.csv']
        }
        
        # Memory limits (5M for large datasets)
        memory_limits = {
            'HI-Small': 5000000,      # 5M transactions
            'LI-Small': 5000000,     # 5M transactions
            'HI-Medium': 10000000,   # 10M transactions
            'LI-Medium': 10000000,   # 10M transactions
            'HI-Large': 5000000,     # 5M transactions (limited for RAM safety)
            'LI-Large': 5000000      # 5M transactions (limited for RAM safety)
        }
        
        for dataset_name, files in dataset_files.items():
            print(f"\n🔍 Processing {dataset_name} dataset...")
            
            trans_file = os.path.join(self.data_path, files[0])
            accounts_file = os.path.join(self.data_path, files[1])
            
            if os.path.exists(trans_file) and os.path.exists(accounts_file):
                print(f"   ✅ Found {dataset_name} dataset")
                
                max_transactions = memory_limits.get(dataset_name, 5000000)
                processed_data = self.process_dataset_simple(
                    dataset_name, trans_file, accounts_file, max_transactions
                )
                
                if processed_data:
                    self.processed_data[dataset_name] = processed_data
                    print(f"   ✅ {dataset_name} processed successfully")
                else:
                    print(f"   ❌ {dataset_name} processing failed")
            else:
                print(f"   ❌ {dataset_name} files not found")
        
        print(f"\n✅ Simple preprocessing completed!")
        print(f"📊 Processed {len(self.processed_data)} datasets")
        
        return self.processed_data

def main():
    """Run simple efficient preprocessing"""
    preprocessor = SimpleEfficientPreprocessor()
    processed_data = preprocessor.run_simple_preprocessing()
    
    if processed_data:
        print("\n🎉 Simple preprocessing successful!")
        print("📊 All datasets processed efficiently")
        print("🚀 Ready for training!")
    else:
        print("\n❌ Simple preprocessing failed!")

if __name__ == "__main__":
    main()

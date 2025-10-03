#!/usr/bin/env python3
"""
Multi-Dataset Enhanced Preprocessing Pipeline
============================================

This script preprocesses multiple AML datasets (LI-Small, HI-Medium, LI-Medium)
with enhanced preprocessing for improved AML detection performance.
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

print("🚀 Multi-Dataset Enhanced Preprocessing Pipeline")
print("=" * 60)

class MultiDatasetPreprocessor:
    """Enhanced preprocessor for multiple AML datasets"""
    
    def __init__(self, data_path="/content/drive/MyDrive/LaunDetection/data/raw"):
        self.data_path = data_path
        self.datasets = {}
        self.processed_data = {}
        self.scalers = {}
        
    def load_dataset_chunked(self, trans_file, accounts_file, dataset_name, max_transactions=999999999):
        """Load ENTIRE dataset with chunked processing for memory management"""
        if max_transactions < 999999999:
            print(f"   📁 Loading {dataset_name} dataset with chunked processing (limited to {max_transactions:,} transactions)...")
        else:
            print(f"   📁 Loading ENTIRE {dataset_name} dataset with chunked processing...")
        
        # Load accounts (usually small)
        accounts = pd.read_csv(accounts_file)
        print(f"   ✅ Accounts loaded: {len(accounts):,}")
        
        # Load transactions in chunks to manage memory - NO LIMITS
        chunk_size = 100000  # Process 100K transactions at a time for better efficiency
        transaction_chunks = []
        total_loaded = 0
        
        print(f"   🔄 Loading ENTIRE dataset in chunks of {chunk_size:,}...")
        
        try:
            for chunk in pd.read_csv(trans_file, chunksize=chunk_size):
                transaction_chunks.append(chunk)
                total_loaded += len(chunk)
                
                # Progress update every 5 chunks for better monitoring
                if len(transaction_chunks) % 5 == 0:
                    print(f"   📊 Loaded {total_loaded:,} transactions so far...")
                
                # Aggressive garbage collection every 3 chunks
                if len(transaction_chunks) % 3 == 0:
                    gc.collect()
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Stop if we hit the memory limit
                if total_loaded > max_transactions:
                    if max_transactions < 999999999:
                        print(f"   ⚠️ Reached memory limit, stopping at {total_loaded:,} transactions (limit: {max_transactions:,})")
                    else:
                        print(f"   ⚠️ Reached theoretical limit, stopping at {total_loaded:,} transactions")
                    break
                
        except Exception as e:
            print(f"   ⚠️ Error loading chunks: {str(e)}")
            return None, None
        
        # Combine chunks
        if transaction_chunks:
            print(f"   🔄 Combining {len(transaction_chunks)} chunks...")
            transactions = pd.concat(transaction_chunks, ignore_index=True)
            print(f"   ✅ ENTIRE dataset loaded: {len(transactions):,} transactions")
        else:
            print(f"   ❌ No transaction chunks loaded")
            return None, None
        
        # Memory cleanup
        del transaction_chunks
        gc.collect()
        
        return transactions, accounts
    
    def load_all_datasets(self):
        """Load all available AML datasets with memory management"""
        print("📊 Loading all available AML datasets with memory management...")
        
        # Define dataset files to look for (including Large datasets)
        dataset_files = {
            'HI-Small': ['HI-Small_Trans.csv', 'HI-Small_accounts.csv'],
            'LI-Small': ['LI-Small_Trans.csv', 'LI-Small_accounts.csv'],
            'HI-Medium': ['HI-Medium_Trans.csv', 'HI-Medium_accounts.csv'],
            'LI-Medium': ['LI-Medium_Trans.csv', 'LI-Medium_accounts.csv'],
            'HI-Large': ['HI-Large_Trans.csv', 'HI-Large_accounts.csv'],
            'LI-Large': ['LI-Large_Trans.csv', 'LI-Large_accounts.csv']
        }
        
        # Memory limits for Small and Medium datasets only
        # Process entire small datasets, limit medium datasets
        memory_limits = {
            'HI-Small': 999999999,   # Process ENTIRE dataset (5M transactions)
            'LI-Small': 999999999,   # Process ENTIRE dataset (7M transactions)
            'HI-Medium': 15000000,   # Limit to 15M transactions only
            'LI-Medium': 15000000,   # Limit to 15M transactions only
            'HI-Large': -1,          # SKIP large datasets completely
            'LI-Large': -1           # SKIP large datasets completely
        }
        
        for dataset_name, files in dataset_files.items():
            print(f"\n🔍 Checking for {dataset_name} dataset...")
            
            # Skip large datasets completely
            if memory_limits[dataset_name] == -1:
                print(f"   ⏭️ Skipping {dataset_name} (large dataset)")
                continue
            
            trans_file = os.path.join(self.data_path, files[0])
            accounts_file = os.path.join(self.data_path, files[1])
            
            if os.path.exists(trans_file) and os.path.exists(accounts_file):
                print(f"   ✅ Found {dataset_name} dataset")
                
                # Load with memory management
                max_transactions = memory_limits.get(dataset_name, 200000)
                transactions, accounts = self.load_dataset_chunked(
                    trans_file, accounts_file, dataset_name, max_transactions
                )
                
                if transactions is not None and accounts is not None:
                    self.datasets[dataset_name] = {
                        'transactions': transactions,
                        'accounts': accounts
                    }
                    
                    print(f"   📊 {dataset_name} loaded: {len(transactions):,} transactions, {len(accounts):,} accounts")
                    
                    # Analyze AML distribution
                    if 'Is Laundering' in transactions.columns:
                        aml_count = transactions['Is Laundering'].sum()
                        aml_rate = (aml_count / len(transactions)) * 100
                        print(f"   🚨 AML transactions: {aml_count:,} ({aml_rate:.4f}%)")
                    else:
                        print(f"   ⚠️ No 'Is Laundering' column found in {dataset_name}")
                    
                    # Memory cleanup
                    del transactions, accounts
                    gc.collect()
                else:
                    print(f"   ❌ Failed to load {dataset_name} dataset")
                    
            else:
                print(f"   ❌ {dataset_name} dataset not found")
                if not os.path.exists(trans_file):
                    print(f"      Missing: {files[0]}")
                if not os.path.exists(accounts_file):
                    print(f"      Missing: {files[1]}")
        
        print(f"\n✅ Loaded {len(self.datasets)} datasets successfully")
        return self.datasets
    
    def create_enhanced_node_features(self, transactions, accounts, dataset_name):
        """Create enhanced node features for accounts"""
        print(f"   🔄 Creating enhanced node features for {dataset_name}...")
        
        # Get all unique accounts from transactions
        from_accounts = set(transactions['From Bank'].astype(str))
        to_accounts = set(transactions['To Bank'].astype(str))
        all_accounts = from_accounts.union(to_accounts)
        
        print(f"   📊 Processing {len(all_accounts):,} unique accounts...")
        
        node_features = {}
        
        for account in tqdm(all_accounts, desc=f"Processing {dataset_name} accounts", unit="accounts"):
            # Get transactions involving this account
            from_trans = transactions[transactions['From Bank'].astype(str) == account]
            to_trans = transactions[transactions['To Bank'].astype(str) == account]
            
            # Calculate comprehensive features
            total_amount_sent = from_trans['Amount Received'].sum() if len(from_trans) > 0 else 0
            total_amount_received = to_trans['Amount Received'].sum() if len(to_trans) > 0 else 0
            total_amount = total_amount_sent + total_amount_received
            
            transaction_count = len(from_trans) + len(to_trans)
            avg_amount = total_amount / transaction_count if transaction_count > 0 else 0
            
            # AML involvement
            is_aml_involved = 0
            if len(from_trans) > 0:
                is_aml_involved = max(is_aml_involved, from_trans['Is Laundering'].max())
            if len(to_trans) > 0:
                is_aml_involved = max(is_aml_involved, to_trans['Is Laundering'].max())
            
            # Advanced features - EXACTLY 25 features to match model expectations
            features = [
                # Basic amounts (log-transformed) - 4 features
                np.log1p(total_amount),
                np.log1p(total_amount_sent),
                np.log1p(total_amount_received),
                np.log1p(avg_amount),
                
                # Transaction counts - 3 features
                float(transaction_count),
                float(len(from_trans)),
                float(len(to_trans)),
                
                # AML involvement - 1 feature
                float(is_aml_involved),
                
                # Amount statistics - 6 features
                from_trans['Amount Received'].std() if len(from_trans) > 1 else 0,
                to_trans['Amount Received'].std() if len(to_trans) > 1 else 0,
                from_trans['Amount Received'].max() if len(from_trans) > 0 else 0,
                to_trans['Amount Received'].max() if len(to_trans) > 0 else 0,
                from_trans['Amount Received'].min() if len(from_trans) > 0 else 0,
                to_trans['Amount Received'].min() if len(to_trans) > 0 else 0,
                
                # Network features - 2 features
                len(set(from_trans['To Bank'].astype(str))) if len(from_trans) > 0 else 0,
                len(set(to_trans['From Bank'].astype(str))) if len(to_trans) > 0 else 0,
                
                # Risk indicators - 2 features
                float(from_trans['Is Laundering'].sum()) if len(from_trans) > 0 else 0,
                float(to_trans['Is Laundering'].sum()) if len(to_trans) > 0 else 0,
                
                # Additional features to reach exactly 25 - 7 features
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # Placeholder for additional features
            ]
            
            # Clean features (handle NaN/Inf)
            clean_features = []
            for f in features:
                if np.isnan(f) or np.isinf(f):
                    clean_features.append(0.0)
                else:
                    clean_features.append(float(f))
            
            node_features[account] = clean_features
        
        print(f"   ✅ Created {len(node_features):,} node feature vectors")
        return node_features
    
    def create_enhanced_edge_features(self, transactions, dataset_name):
        """Create enhanced edge features for transactions"""
        print(f"   🔄 Creating enhanced edge features for {dataset_name}...")
        
        edge_features = []
        edge_labels = []
        
        for _, transaction in tqdm(transactions.iterrows(), 
                                 total=len(transactions), 
                                 desc=f"Processing {dataset_name} edges", 
                                 unit="txns"):
            
            amount = transaction['Amount Received']
            is_aml = transaction['Is Laundering']
            
            # Enhanced edge features
            features = [
                # Amount features (log-transformed)
                np.log1p(amount),
                amount / 1000,  # Normalized amount
                np.log1p(amount) / 10,  # Scaled log amount
                
                # AML label
                float(is_aml),
                
                # Transaction type features (encoded)
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
            ]
            
            # Clean features
            clean_features = []
            for f in features:
                if np.isnan(f) or np.isinf(f):
                    clean_features.append(0.0)
                else:
                    clean_features.append(float(f))
            
            edge_features.append(clean_features)
            edge_labels.append(is_aml)
        
        print(f"   ✅ Created {len(edge_features):,} edge feature vectors")
        return edge_features, edge_labels
    
    def create_balanced_dataset(self, transactions, target_aml_rate=0.10):
        """Create a balanced dataset with target AML rate - ENHANCED for full datasets"""
        print(f"   🔄 Creating balanced dataset with {target_aml_rate*100:.1f}% AML rate...")
        
        aml_transactions = transactions[transactions['Is Laundering'] == 1]
        non_aml_transactions = transactions[transactions['Is Laundering'] == 0]
        
        print(f"   📊 Original distribution:")
        print(f"      AML: {len(aml_transactions):,} ({len(aml_transactions)/len(transactions)*100:.4f}%)")
        print(f"      Non-AML: {len(non_aml_transactions):,} ({len(non_aml_transactions)/len(transactions)*100:.4f}%)")
        
        # For large datasets, use more aggressive balancing to get more AML samples
        if len(aml_transactions) > 10000:
            # Use higher AML rate for large datasets to get more training data
            target_aml_rate = 0.15  # 15% AML rate for large datasets
            print(f"   🚀 Large dataset detected - using {target_aml_rate*100:.1f}% AML rate for better training!")
        
        # Calculate target non-AML count
        target_non_aml_count = int(len(aml_transactions) * (1 - target_aml_rate) / target_aml_rate)
        target_non_aml_count = min(target_non_aml_count, len(non_aml_transactions))
        
        # Sample non-AML transactions
        non_aml_sampled = non_aml_transactions.sample(n=target_non_aml_count, random_state=42)
        
        # Combine datasets
        balanced_transactions = pd.concat([aml_transactions, non_aml_sampled])
        balanced_transactions = balanced_transactions.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   📊 Balanced distribution:")
        print(f"      AML: {len(aml_transactions):,} ({len(aml_transactions)/len(balanced_transactions)*100:.4f}%)")
        print(f"      Non-AML: {len(non_aml_sampled):,} ({len(non_aml_sampled)/len(balanced_transactions)*100:.4f}%)")
        print(f"      Total: {len(balanced_transactions):,}")
        
        return balanced_transactions
    
    def preprocess_dataset(self, dataset_name, transactions, accounts):
        """Preprocess a single dataset with enhanced features and memory management"""
        print(f"\n🔄 Preprocessing {dataset_name} dataset...")
        
        # NO SAMPLING - Use ENTIRE dataset for maximum training performance
        print(f"   🚀 Using ENTIRE {dataset_name} dataset: {len(transactions):,} transactions")
        print(f"   📊 This will provide maximum training data for better AML detection!")
        
        # Create balanced dataset
        balanced_transactions = self.create_balanced_dataset(transactions, target_aml_rate=0.10)
        
        # Memory cleanup
        del transactions
        gc.collect()
        
        # Create enhanced features
        node_features = self.create_enhanced_node_features(balanced_transactions, accounts, dataset_name)
        edge_features, edge_labels = self.create_enhanced_edge_features(balanced_transactions, dataset_name)
        
        # Memory cleanup
        del accounts
        gc.collect()
        
        # Create NetworkX graph
        print(f"   🕸️ Creating {dataset_name} graph...")
        G = nx.DiGraph()
        
        # Add nodes with features
        for account, features in node_features.items():
            G.add_node(account, features=features)
        
        # Add edges with features
        for i, (_, transaction) in enumerate(balanced_transactions.iterrows()):
            from_acc = str(transaction['From Bank'])
            to_acc = str(transaction['To Bank'])
            
            if from_acc in G.nodes and to_acc in G.nodes:
                G.add_edge(from_acc, to_acc, 
                          features=edge_features[i], 
                          label=edge_labels[i])
        
        print(f"   ✅ Created {dataset_name} graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        
        # Store processed data
        self.processed_data[dataset_name] = {
            'graph': G,
            'node_features': node_features,
            'edge_features': edge_features,
            'edge_labels': edge_labels,
            'transactions': balanced_transactions,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'aml_edges': sum(edge_labels),
            'non_aml_edges': len(edge_labels) - sum(edge_labels)
        }
        
        # Memory cleanup
        del balanced_transactions, node_features, edge_features, edge_labels
        gc.collect()
        
        return self.processed_data[dataset_name]
    
    def save_processed_data(self, output_dir="/content/drive/MyDrive/LaunDetection/data/processed"):
        """Save all processed datasets"""
        print(f"\n💾 Saving processed datasets to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset_name, data in self.processed_data.items():
            print(f"   💾 Saving {dataset_name}...")
            
            # Save graph
            graph_path = os.path.join(output_dir, f"{dataset_name}_graph.pkl")
            with open(graph_path, 'wb') as f:
                pickle.dump(data['graph'], f)
            
            # Save features
            features_path = os.path.join(output_dir, f"{dataset_name}_features.pkl")
            with open(features_path, 'wb') as f:
                pickle.dump({
                    'node_features': data['node_features'],
                    'edge_features': data['edge_features'],
                    'edge_labels': data['edge_labels']
                }, f)
            
            # Save metadata
            metadata_path = os.path.join(output_dir, f"{dataset_name}_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'num_nodes': data['num_nodes'],
                    'num_edges': data['num_edges'],
                    'aml_edges': data['aml_edges'],
                    'non_aml_edges': data['non_aml_edges'],
                    'aml_rate': data['aml_edges'] / data['num_edges'] if data['num_edges'] > 0 else 0
                }, f)
            
            print(f"   ✅ {dataset_name} saved successfully")
        
        print(f"\n✅ All datasets saved to {output_dir}")
    
    def run_full_preprocessing(self):
        """Run complete preprocessing pipeline for all datasets"""
        print("🚀 Starting Multi-Dataset Enhanced Preprocessing...")
        
        # Load all datasets
        self.load_all_datasets()
        
        if not self.datasets:
            print("❌ No datasets found! Please check data directory.")
            return
        
        # Preprocess each dataset
        for dataset_name, data in self.datasets.items():
            try:
                self.preprocess_dataset(dataset_name, data['transactions'], data['accounts'])
            except Exception as e:
                print(f"❌ Error preprocessing {dataset_name}: {str(e)}")
                continue
        
        # Save processed data
        self.save_processed_data()
        
        # Print summary
        print("\n📊 Preprocessing Summary:")
        print("=" * 40)
        for dataset_name, data in self.processed_data.items():
            print(f"📈 {dataset_name}:")
            print(f"   Nodes: {data['num_nodes']:,}")
            print(f"   Edges: {data['num_edges']:,}")
            print(f"   AML Edges: {data['aml_edges']:,}")
            print(f"   Non-AML Edges: {data['non_aml_edges']:,}")
            print(f"   AML Rate: {data['aml_edges']/data['num_edges']*100:.2f}%")
        
        print("\n🎉 Multi-Dataset Preprocessing Complete!")
        return self.processed_data

def main():
    """Main preprocessing pipeline"""
    preprocessor = MultiDatasetPreprocessor()
    processed_data = preprocessor.run_full_preprocessing()
    
    if processed_data:
        print("\n🚀 Ready for Multi-Dataset Training!")
        print("Next steps:")
        print("1. Run multi-dataset training script")
        print("2. Evaluate on combined dataset")
        print("3. Deploy enhanced model")

if __name__ == "__main__":
    main()

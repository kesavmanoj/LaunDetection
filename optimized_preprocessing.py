#!/usr/bin/env python3
"""
Optimized Multi-Dataset Preprocessing
====================================

This script uses the full 12GB Colab RAM efficiently for maximum data processing.
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

print("🚀 Optimized Multi-Dataset Preprocessing")
print("=" * 50)
print("📊 Using full 12GB Colab RAM for maximum data processing")
print("📊 Much higher transaction limits for better training")
print()

class OptimizedPreprocessor:
    """Optimized preprocessor using full Colab RAM"""
    
    def __init__(self, data_path="/content/drive/MyDrive/LaunDetection/data/raw"):
        self.data_path = data_path
        self.processed_data = {}
        
    def load_dataset_optimized(self, trans_file, accounts_file, dataset_name, max_transactions=20000000):
        """Load dataset with optimized memory usage"""
        print(f"   📁 Loading {dataset_name} dataset (up to {max_transactions:,} transactions)...")
        
        # Load accounts
        accounts = pd.read_csv(accounts_file)
        print(f"   ✅ Accounts loaded: {len(accounts):,}")
        
        # Load transactions with optimized chunks
        chunk_size = 100000  # Larger chunks for efficiency
        transaction_chunks = []
        total_loaded = 0
        
        print(f"   🔄 Loading dataset in chunks of {chunk_size:,}...")
        
        try:
            for chunk in pd.read_csv(trans_file, chunksize=chunk_size):
                transaction_chunks.append(chunk)
                total_loaded += len(chunk)
                
                # Progress update every 5 chunks
                if len(transaction_chunks) % 5 == 0:
                    print(f"   📊 Loaded {total_loaded:,} transactions so far...")
                
                # Memory management every 10 chunks
                if len(transaction_chunks) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Stop if we hit the limit
                if total_loaded > max_transactions:
                    print(f"   ⚠️ Reached memory limit, stopping at {total_loaded:,} transactions (limit: {max_transactions:,})")
                    break
                    
        except Exception as e:
            print(f"   ⚠️ Error loading chunks: {str(e)}")
            return None, None
        
        # Combine chunks
        if transaction_chunks:
            print(f"   🔄 Combining {len(transaction_chunks)} chunks...")
            transactions = pd.concat(transaction_chunks, ignore_index=True)
            print(f"   ✅ Dataset loaded: {len(transactions):,} transactions")
        else:
            print("   ❌ No transaction chunks loaded")
            return None, None
        
        # Clean up chunks to free memory
        del transaction_chunks
        gc.collect()
        
        return transactions, accounts
    
    def create_enhanced_node_features(self, accounts, transactions, dataset_name):
        """Create enhanced node features with memory management"""
        print(f"   🔄 Creating enhanced node features for {dataset_name}...")
        
        # Get unique accounts from transactions
        unique_accounts = set()
        for col in ['From Bank', 'To Bank']:
            if col in transactions.columns:
                unique_accounts.update(transactions[col].unique())
        
        print(f"   📊 Processing {len(unique_accounts)} unique accounts...")
        
        node_features = {}
        
        for account in tqdm(unique_accounts, desc=f"Processing {dataset_name} accounts"):
            # Account-level features
            account_transactions = transactions[
                (transactions['From Bank'] == account) | 
                (transactions['To Bank'] == account)
            ]
            
            if len(account_transactions) == 0:
                continue
            
            # Basic features
            total_amount = account_transactions['Amount Received'].fillna(0).sum() + \
                          account_transactions['Amount Paid'].fillna(0).sum()
            avg_amount = total_amount / len(account_transactions) if len(account_transactions) > 0 else 0
            max_amount = max(account_transactions['Amount Received'].fillna(0).max(),
                          account_transactions['Amount Paid'].fillna(0).max())
            
            # Transaction frequency
            transaction_count = len(account_transactions)
            
            # Time-based features
            if 'Timestamp' in account_transactions.columns:
                timestamps = pd.to_datetime(account_transactions['Timestamp'], errors='coerce')
                if not timestamps.isna().all():
                    time_span = (timestamps.max() - timestamps.min()).days if len(timestamps.dropna()) > 1 else 0
                    avg_daily_transactions = transaction_count / max(time_span, 1)
                else:
                    time_span = 0
                    avg_daily_transactions = 0
            else:
                time_span = 0
                avg_daily_transactions = 0
            
            # AML involvement
            aml_involvement = 1 if 'Is Laundering' in account_transactions.columns and \
                             account_transactions['Is Laundering'].sum() > 0 else 0
            
            # Create feature vector (25 features to match model expectations)
            features = [
                float(total_amount),
                float(avg_amount),
                float(max_amount),
                float(transaction_count),
                float(time_span),
                float(avg_daily_transactions),
                float(aml_involvement),
                # Add 18 placeholder features to reach 25
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
            
            node_features[account] = features
        
        print(f"   ✅ Created {len(node_features)} node feature vectors")
        return node_features
    
    def create_enhanced_edge_features(self, transactions, dataset_name):
        """Create enhanced edge features"""
        print(f"   🔄 Creating enhanced edge features for {dataset_name}...")
        
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
            
            # Currency features (placeholder)
            currency_features = [0.0, 0.0, 0.0, 0.0, 0.0]  # 5 currency features
            
            # Create edge feature vector (13 features)
            edge_feature = [
                float(amount_norm),
                float(hour),
                float(day_of_week),
                float(amount_received),
                float(amount_paid),
                *currency_features,  # 5 currency features
                0.0, 0.0, 0.0, 0.0, 0.0  # 5 additional features
            ]
            
            edge_features.append(edge_feature)
            
            # Edge label
            aml_label = 1 if row.get('Is Laundering', 0) == 1 else 0
            edge_labels.append(aml_label)
        
        print(f"   ✅ Created {len(edge_features)} edge feature vectors")
        return edge_features, edge_labels
    
    def create_graph(self, transactions, node_features, edge_features, edge_labels, dataset_name):
        """Create NetworkX graph"""
        print(f"   🕸️ Creating {dataset_name} graph...")
        
        G = nx.DiGraph()
        
        # Add nodes
        for account, features in node_features.items():
            G.add_node(account, features=features)
        
        # Add edges
        for i, (_, row) in enumerate(transactions.iterrows()):
            from_bank = row.get('From Bank')
            to_bank = row.get('To Bank')
            
            if from_bank and to_bank and from_bank in G.nodes and to_bank in G.nodes:
                G.add_edge(from_bank, to_bank, 
                          features=edge_features[i],
                          label=edge_labels[i])
        
        print(f"   ✅ Created {dataset_name} graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def process_single_dataset(self, dataset_name, trans_file, accounts_file, max_transactions):
        """Process a single dataset with optimized limits"""
        print(f"\n🔄 Processing {dataset_name} dataset...")
        print(f"   🚀 Using {dataset_name} dataset (up to {max_transactions:,} transactions)")
        
        # Load dataset
        transactions, accounts = self.load_dataset_optimized(trans_file, accounts_file, dataset_name, max_transactions)
        
        if transactions is None or accounts is None:
            print(f"   ❌ Failed to load {dataset_name}")
            return None
        
        # Create balanced dataset
        aml_transactions = transactions[transactions.get('Is Laundering', 0) == 1]
        non_aml_transactions = transactions[transactions.get('Is Laundering', 0) == 0]
        
        print(f"   📊 Original distribution:")
        print(f"      AML: {len(aml_transactions):,} ({len(aml_transactions)/len(transactions)*100:.4f}%)")
        print(f"      Non-AML: {len(non_aml_transactions):,} ({len(non_aml_transactions)/len(transactions)*100:.4f}%)")
        
        # Create balanced subset with higher AML rate for better training
        target_aml_rate = 0.15  # 15% AML rate for better training
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
        node_features = self.create_enhanced_node_features(accounts, balanced_transactions, dataset_name)
        edge_features, edge_labels = self.create_enhanced_edge_features(balanced_transactions, dataset_name)
        
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
        output_dir = "/content/drive/MyDrive/LaunDetection/data/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save graph
        with open(os.path.join(output_dir, f"{dataset_name}_graph.pkl"), 'wb') as f:
            pickle.dump(graph, f)
        
        # Save features
        with open(os.path.join(output_dir, f"{dataset_name}_features.pkl"), 'wb') as f:
            pickle.dump({
                'node_features': node_features,
                'edge_features': edge_features,
                'edge_labels': edge_labels
            }, f)
        
        # Save metadata
        with open(os.path.join(output_dir, f"{dataset_name}_metadata.pkl"), 'wb') as f:
            pickle.dump(processed_data['metadata'], f)
        
        print(f"   💾 Saved {dataset_name} to {output_dir}")
        
        # Clean up memory
        del transactions, accounts, balanced_transactions
        gc.collect()
        
        return processed_data
    
    def run_optimized_preprocessing(self):
        """Run optimized preprocessing on all datasets"""
        print("🚀 Starting Optimized Multi-Dataset Preprocessing...")
        
        # Optimized memory limits to utilize full 12GB Colab RAM
        memory_limits = {
            'HI-Small': 5000000,     # 5M transactions (full dataset)
            'LI-Small': 7000000,     # 7M transactions (full dataset)
            'HI-Medium': 20000000,   # 20M transactions (2/3 of 32M)
            'LI-Medium': 20000000    # 20M transactions (2/3 of 32M)
        }
        
        dataset_files = {
            'HI-Small': ['HI-Small_Trans.csv', 'HI-Small_accounts.csv'],
            'LI-Small': ['LI-Small_Trans.csv', 'LI-Small_accounts.csv'],
            'HI-Medium': ['HI-Medium_Trans.csv', 'HI-Medium_accounts.csv'],
            'LI-Medium': ['LI-Medium_Trans.csv', 'LI-Medium_accounts.csv']
        }
        
        for dataset_name, files in dataset_files.items():
            print(f"\n🔍 Processing {dataset_name} dataset...")
            
            trans_file = os.path.join(self.data_path, files[0])
            accounts_file = os.path.join(self.data_path, files[1])
            
            if os.path.exists(trans_file) and os.path.exists(accounts_file):
                print(f"   ✅ Found {dataset_name} dataset")
                
                max_transactions = memory_limits.get(dataset_name, 10000000)
                processed_data = self.process_single_dataset(
                    dataset_name, trans_file, accounts_file, max_transactions
                )
                
                if processed_data:
                    self.processed_data[dataset_name] = processed_data
                    print(f"   ✅ {dataset_name} processed successfully")
                else:
                    print(f"   ❌ {dataset_name} processing failed")
            else:
                print(f"   ❌ {dataset_name} files not found")
        
        print(f"\n✅ Optimized preprocessing completed!")
        print(f"📊 Processed {len(self.processed_data)} datasets")
        
        return self.processed_data

def main():
    """Run optimized preprocessing"""
    preprocessor = OptimizedPreprocessor()
    processed_data = preprocessor.run_optimized_preprocessing()
    
    if processed_data:
        print("\n🎉 Optimized preprocessing successful!")
        print("📊 All datasets processed with optimized memory limits")
        print("🚀 Ready for training with much more data!")
    else:
        print("\n❌ Optimized preprocessing failed!")

if __name__ == "__main__":
    main()

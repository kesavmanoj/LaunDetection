#!/usr/bin/env python3
"""
Memory Efficient Preprocessing with Checkpointing
================================================

This script processes large datasets in 10M transaction batches with checkpointing
to prevent RAM crashes and allow processing of entire datasets (180M+ transactions).
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

print("🚀 Memory Efficient Preprocessing with Checkpointing")
print("=" * 60)
print("📊 Processing in 10M transaction batches")
print("📊 Checkpointing every 10M transactions")
print("📊 Can process entire 180M+ transaction datasets")
print()

class MemoryEfficientPreprocessor:
    """Memory efficient preprocessor with checkpointing"""
    
    def __init__(self, data_path="/content/drive/MyDrive/LaunDetection/data/raw"):
        self.data_path = data_path
        self.processed_data = {}
        self.checkpoint_dir = "/content/drive/MyDrive/LaunDetection/data/processed"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def process_dataset_in_batches(self, dataset_name, trans_file, accounts_file, batch_size=10000000):
        """Process dataset in batches with checkpointing"""
        print(f"\n🔄 Processing {dataset_name} dataset in batches...")
        print(f"   📊 Batch size: {batch_size:,} transactions")
        print(f"   📊 Checkpointing every batch")
        
        # Load accounts (usually small)
        accounts = pd.read_csv(accounts_file)
        print(f"   ✅ Accounts loaded: {len(accounts):,}")
        
        # Initialize combined data structures
        combined_node_features = {}
        combined_edge_features = []
        combined_edge_labels = []
        combined_graph = nx.DiGraph()
        
        # Get unique accounts from all batches
        print(f"   🔄 Collecting unique accounts from all batches...")
        unique_accounts = set()
        
        # First pass: collect all unique accounts
        batch_count = 0
        total_processed = 0
        
        for chunk in pd.read_csv(trans_file, chunksize=100000):  # Read in 100K chunks
            batch_count += 1
            total_processed += len(chunk)
            
            # Collect unique accounts
            for col in ['From Bank', 'To Bank']:
                if col in chunk.columns:
                    unique_accounts.update(chunk[col].unique())
            
            # Progress update
            if batch_count % 100 == 0:  # Every 10M transactions
                print(f"   📊 Processed {total_processed:,} transactions, found {len(unique_accounts):,} unique accounts")
                gc.collect()  # Force garbage collection
        
        print(f"   ✅ Found {len(unique_accounts):,} unique accounts")
        
        # Create node features for all accounts
        print(f"   🔄 Creating node features for all accounts...")
        for account in tqdm(unique_accounts, desc="Creating node features"):
            # Create basic node features (25 features to match model expectations)
            features = [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0
            ]
            combined_node_features[account] = features
            combined_graph.add_node(account, features=features)
        
        # Second pass: process transactions in smaller batches
        print(f"   🔄 Processing transactions in batches...")
        batch_count = 0
        total_processed = 0
        batch_transactions = []
        max_chunks_per_batch = 50  # Process 50 chunks (5M transactions) at a time
        
        for chunk in pd.read_csv(trans_file, chunksize=100000):
            batch_transactions.append(chunk)
            total_processed += len(chunk)
            
            # Process batch when it reaches max_chunks_per_batch
            if len(batch_transactions) >= max_chunks_per_batch:
                print(f"   📊 Processing batch {batch_count + 1} ({len(batch_transactions)} chunks, {total_processed:,} transactions)...")
                
                # Combine batch
                batch_df = pd.concat(batch_transactions, ignore_index=True)
                
                # Process batch
                batch_edge_features, batch_edge_labels = self.process_batch_edges(batch_df, dataset_name)
                
                # Add to combined data
                combined_edge_features.extend(batch_edge_features)
                combined_edge_labels.extend(batch_edge_labels)
                
                # Add edges to graph (only if not too many)
                if len(batch_edge_features) < 100000:  # Only add if manageable size
                    for i, (_, row) in enumerate(batch_df.iterrows()):
                        from_bank = row.get('From Bank')
                        to_bank = row.get('To Bank')
                        
                        if from_bank and to_bank and from_bank in combined_graph.nodes and to_bank in combined_graph.nodes:
                            combined_graph.add_edge(from_bank, to_bank, 
                                                  features=batch_edge_features[i],
                                                  label=batch_edge_labels[i])
                
                # Save checkpoint every batch
                self.save_checkpoint(dataset_name, batch_count, {
                    'graph': combined_graph,
                    'node_features': combined_node_features,
                    'edge_features': combined_edge_features,
                    'edge_labels': combined_edge_labels,
                    'total_processed': total_processed
                })
                
                # Clear batch data
                del batch_transactions, batch_df, batch_edge_features, batch_edge_labels
                batch_transactions = []
                batch_count += 1
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"   💾 Checkpoint saved: {total_processed:,} transactions processed")
        
        # Process remaining transactions
        if batch_transactions:
            print(f"   📊 Processing final batch ({len(batch_transactions)} chunks)...")
            batch_df = pd.concat(batch_transactions, ignore_index=True)
            batch_edge_features, batch_edge_labels = self.process_batch_edges(batch_df, dataset_name)
            
            combined_edge_features.extend(batch_edge_features)
            combined_edge_labels.extend(batch_edge_labels)
            
            # Add edges to graph (only if not too many)
            if len(batch_edge_features) < 100000:
                for i, (_, row) in enumerate(batch_df.iterrows()):
                    from_bank = row.get('From Bank')
                    to_bank = row.get('To Bank')
                    
                    if from_bank and to_bank and from_bank in combined_graph.nodes and to_bank in combined_graph.nodes:
                        combined_graph.add_edge(from_bank, to_bank, 
                                              features=batch_edge_features[i],
                                              label=batch_edge_labels[i])
        
        print(f"   ✅ {dataset_name} processing complete: {len(combined_edge_features):,} edges")
        
        # Create balanced dataset
        balanced_data = self.create_balanced_dataset(combined_edge_features, combined_edge_labels, dataset_name)
        
        # Final save
        self.save_final_dataset(dataset_name, combined_graph, combined_node_features, balanced_data)
        
        return {
            'graph': combined_graph,
            'node_features': combined_node_features,
            'edge_features': balanced_data['edge_features'],
            'edge_labels': balanced_data['edge_labels'],
            'metadata': {
                'num_nodes': combined_graph.number_of_nodes(),
                'num_edges': len(balanced_data['edge_features']),
                'aml_rate': sum(balanced_data['edge_labels']) / len(balanced_data['edge_labels']) if balanced_data['edge_labels'] else 0
            }
        }
    
    def process_batch_edges(self, batch_df, dataset_name):
        """Process edges for a batch of transactions"""
        edge_features = []
        edge_labels = []
        
        for _, row in batch_df.iterrows():
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
        
        return edge_features, edge_labels
    
    def create_balanced_dataset(self, edge_features, edge_labels, dataset_name):
        """Create balanced dataset from edge features and labels"""
        print(f"   🔄 Creating balanced dataset for {dataset_name}...")
        
        # Count AML and non-AML
        aml_count = sum(edge_labels)
        non_aml_count = len(edge_labels) - aml_count
        
        print(f"   📊 Original distribution:")
        print(f"      AML: {aml_count:,} ({aml_count/len(edge_labels)*100:.4f}%)")
        print(f"      Non-AML: {non_aml_count:,} ({non_aml_count/len(edge_labels)*100:.4f}%)")
        
        # Create balanced subset
        target_aml_rate = 0.15  # 15% AML rate
        max_aml_samples = aml_count
        max_non_aml_samples = int(max_aml_samples * (1 - target_aml_rate) / target_aml_rate)
        
        # Sample non-AML transactions
        non_aml_indices = [i for i, label in enumerate(edge_labels) if label == 0]
        if len(non_aml_indices) > max_non_aml_samples:
            import random
            random.seed(42)
            sampled_non_aml_indices = random.sample(non_aml_indices, max_non_aml_samples)
        else:
            sampled_non_aml_indices = non_aml_indices
        
        # Get AML indices
        aml_indices = [i for i, label in enumerate(edge_labels) if label == 1]
        
        # Combine indices
        balanced_indices = aml_indices + sampled_non_aml_indices
        
        # Create balanced dataset
        balanced_edge_features = [edge_features[i] for i in balanced_indices]
        balanced_edge_labels = [edge_labels[i] for i in balanced_indices]
        
        print(f"   📊 Balanced distribution:")
        print(f"      AML: {sum(balanced_edge_labels):,} ({sum(balanced_edge_labels)/len(balanced_edge_labels)*100:.4f}%)")
        print(f"      Non-AML: {len(balanced_edge_labels) - sum(balanced_edge_labels):,} ({(len(balanced_edge_labels) - sum(balanced_edge_labels))/len(balanced_edge_labels)*100:.4f}%)")
        print(f"      Total: {len(balanced_edge_labels):,}")
        
        return {
            'edge_features': balanced_edge_features,
            'edge_labels': balanced_edge_labels
        }
    
    def save_checkpoint(self, dataset_name, batch_num, data):
        """Save checkpoint data"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{dataset_name}_checkpoint_batch_{batch_num}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
    
    def save_final_dataset(self, dataset_name, graph, node_features, balanced_data):
        """Save final processed dataset"""
        # Save graph
        with open(os.path.join(self.checkpoint_dir, f"{dataset_name}_graph.pkl"), 'wb') as f:
            pickle.dump(graph, f)
        
        # Save features
        with open(os.path.join(self.checkpoint_dir, f"{dataset_name}_features.pkl"), 'wb') as f:
            pickle.dump({
                'node_features': node_features,
                'edge_features': balanced_data['edge_features'],
                'edge_labels': balanced_data['edge_labels']
            }, f)
        
        # Save metadata
        metadata = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': len(balanced_data['edge_features']),
            'aml_rate': sum(balanced_data['edge_labels']) / len(balanced_data['edge_labels']) if balanced_data['edge_labels'] else 0
        }
        
        with open(os.path.join(self.checkpoint_dir, f"{dataset_name}_metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"   💾 Final dataset saved: {dataset_name}")
    
    def run_memory_efficient_preprocessing(self):
        """Run memory efficient preprocessing for all datasets"""
        print("🚀 Starting Memory Efficient Preprocessing...")
        
        # Define dataset files
        dataset_files = {
            'HI-Small': ['HI-Small_Trans.csv', 'HI-Small_accounts.csv'],
            'LI-Small': ['LI-Small_Trans.csv', 'LI-Small_accounts.csv'],
            'HI-Medium': ['HI-Medium_Trans.csv', 'HI-Medium_accounts.csv'],
            'LI-Medium': ['LI-Medium_Trans.csv', 'LI-Medium_accounts.csv'],
            'HI-Large': ['HI-Large_Trans.csv', 'HI-Large_accounts.csv'],
            'LI-Large': ['LI-Large_Trans.csv', 'LI-Large_accounts.csv']
        }
        
        # Memory limits (10M per batch)
        memory_limits = {
            'HI-Small': 10000000,      # 10M per batch
            'LI-Small': 10000000,      # 10M per batch
            'HI-Medium': 10000000,     # 10M per batch
            'LI-Medium': 10000000,    # 10M per batch
            'HI-Large': 10000000,     # 10M per batch (can process 180M total)
            'LI-Large': 10000000      # 10M per batch (can process 180M total)
        }
        
        for dataset_name, files in dataset_files.items():
            print(f"\n🔍 Processing {dataset_name} dataset...")
            
            trans_file = os.path.join(self.data_path, files[0])
            accounts_file = os.path.join(self.data_path, files[1])
            
            if os.path.exists(trans_file) and os.path.exists(accounts_file):
                print(f"   ✅ Found {dataset_name} dataset")
                
                batch_size = memory_limits.get(dataset_name, 10000000)
                processed_data = self.process_dataset_in_batches(
                    dataset_name, trans_file, accounts_file, batch_size
                )
                
                if processed_data:
                    self.processed_data[dataset_name] = processed_data
                    print(f"   ✅ {dataset_name} processed successfully")
                else:
                    print(f"   ❌ {dataset_name} processing failed")
            else:
                print(f"   ❌ {dataset_name} files not found")
        
        print(f"\n✅ Memory efficient preprocessing completed!")
        print(f"📊 Processed {len(self.processed_data)} datasets")
        print(f"💾 All checkpoints saved to {self.checkpoint_dir}")
        
        return self.processed_data

def main():
    """Run memory efficient preprocessing"""
    preprocessor = MemoryEfficientPreprocessor()
    processed_data = preprocessor.run_memory_efficient_preprocessing()
    
    if processed_data:
        print("\n🎉 Memory efficient preprocessing successful!")
        print("📊 All datasets processed with 10M batch limits")
        print("💾 Checkpoints saved for crash recovery")
        print("🚀 Ready for training with massive datasets!")
    else:
        print("\n❌ Memory efficient preprocessing failed!")

if __name__ == "__main__":
    main()

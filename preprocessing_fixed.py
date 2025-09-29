"""
Fixed Preprocessing Script for AML Detection
Addresses the invalid edge indices issue by filtering transactions during preprocessing
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
import time
from tqdm import tqdm
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FixedAMLPreprocessor:
    """
    Fixed preprocessor that handles invalid edge indices during preprocessing
    instead of during training
    """
    
    def __init__(self, dataset_name, base_dir=None):
        self.dataset_name = dataset_name
        self.base_dir = Path(base_dir) if base_dir else Path('/content/drive/MyDrive/LaunDetection')
        
        # Paths
        self.raw_data_dir = self.base_dir / 'data' / 'raw'
        self.processed_data_dir = self.base_dir / 'data' / 'processed'
        self.graphs_dir = self.base_dir / 'data' / 'graphs'
        
        # Create directories
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset paths
        self.dataset_paths = {
            'accounts': self.raw_data_dir / f'{dataset_name}_accounts.csv',
            'transactions': self.raw_data_dir / f'{dataset_name}_Trans.csv'
        }
        
        # Setup logging
        self._setup_logging()
        
        # Processing parameters
        self.chunk_size = 50000  # Smaller chunks for memory efficiency
    
    def _find_column(self, df, possible_names, default_name):
        """Find column by checking multiple possible names"""
        for name in possible_names:
            if name in df.columns:
                return name
        
        # If not found, try case-insensitive search
        for name in possible_names:
            for col in df.columns:
                if name.lower() == col.lower():
                    return col
        
        # Last resort: return default or first available column
        if default_name in df.columns:
            return default_name
        else:
            self.logger.warning(f"Column not found from {possible_names}, using first column: {df.columns[0]}")
            return df.columns[0]
        
    def _setup_logging(self):
        """Setup logging"""
        log_file = self.base_dir / 'logs' / f'preprocessing_fixed_{self.dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        log_file.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_and_validate_accounts(self):
        """Load accounts and create valid account mapping"""
        self.logger.info("Loading and validating accounts...")
        
        # Load accounts
        accounts_df = pd.read_csv(self.dataset_paths['accounts'])
        self.logger.info(f"Loaded {len(accounts_df):,} accounts")
        
        # Debug: Print column names
        self.logger.info(f"Accounts columns: {list(accounts_df.columns)}")
        
        # Handle different possible column names based on the actual data structure
        bank_col = self._find_column(accounts_df, ['Bank ID', 'Bank Name', 'bank_id', 'bank_name', 'Bank'], 'Bank ID')
        account_col = self._find_column(accounts_df, ['Account Number', 'Account', 'account_number', 'account'], 'Account Number')
        
        self.logger.info(f"Using bank column: '{bank_col}', account column: '{account_col}'")
        
        # Create account IDs
        accounts_df['AccountID'] = (accounts_df[bank_col].astype(str) + '_' + 
                                   accounts_df[account_col].astype(str))
        
        # Create account to node mapping
        unique_accounts = accounts_df['AccountID'].unique()
        account_to_node = {acc_id: idx for idx, acc_id in enumerate(unique_accounts)}
        
        self.logger.info(f"Created mapping for {len(account_to_node):,} unique accounts")
        
        # Store for later use
        self.accounts_df = accounts_df
        self.account_to_node = account_to_node
        self.valid_account_ids = set(unique_accounts)
        
        return accounts_df, account_to_node
    
    def analyze_transaction_coverage(self):
        """Analyze what percentage of transactions have valid accounts"""
        self.logger.info("Analyzing transaction coverage...")
        
        total_transactions = 0
        valid_transactions = 0
        missing_src_accounts = set()
        missing_dst_accounts = set()
        
        # Sample first few chunks to get statistics
        chunk_reader = pd.read_csv(self.dataset_paths['transactions'], chunksize=self.chunk_size)
        
        for chunk_idx, df_chunk in enumerate(chunk_reader):
            if chunk_idx >= 10:  # Analyze first 10 chunks for statistics
                break
            
            # Debug: Print column names for first chunk
            if chunk_idx == 0:
                self.logger.info(f"Transaction columns: {list(df_chunk.columns)}")
                
            # Handle different possible column names for transactions
            src_bank_col = self._find_column(df_chunk, ['From Bank', 'from_bank', 'source_bank'], 'From Bank')
            dst_bank_col = self._find_column(df_chunk, ['To Bank', 'to_bank', 'dest_bank'], 'To Bank')
            src_account_col = self._find_column(df_chunk, ['Account', 'account', 'src_account'], 'Account')
            dst_account_col = self._find_column(df_chunk, ['Account.1', 'account.1', 'dest_account'], 'Account.1')
                
            # Create account IDs
            df_chunk['SrcAccountID'] = (df_chunk[src_bank_col].astype(str) + '_' + 
                                       df_chunk[src_account_col].astype(str))
            df_chunk['DstAccountID'] = (df_chunk[dst_bank_col].astype(str) + '_' + 
                                       df_chunk[dst_account_col].astype(str))
            
            # Check validity
            src_valid = df_chunk['SrcAccountID'].isin(self.valid_account_ids)
            dst_valid = df_chunk['DstAccountID'].isin(self.valid_account_ids)
            both_valid = src_valid & dst_valid
            
            total_transactions += len(df_chunk)
            valid_transactions += both_valid.sum()
            
            # Collect missing accounts
            missing_src_accounts.update(df_chunk.loc[~src_valid, 'SrcAccountID'].unique())
            missing_dst_accounts.update(df_chunk.loc[~dst_valid, 'DstAccountID'].unique())
        
        coverage_rate = valid_transactions / total_transactions if total_transactions > 0 else 0
        
        self.logger.info(f"Transaction coverage analysis (first {total_transactions:,} transactions):")
        self.logger.info(f"  Valid transactions: {valid_transactions:,} ({coverage_rate*100:.2f}%)")
        self.logger.info(f"  Invalid transactions: {total_transactions - valid_transactions:,}")
        self.logger.info(f"  Missing source accounts: {len(missing_src_accounts):,}")
        self.logger.info(f"  Missing destination accounts: {len(missing_dst_accounts):,}")
        
        return coverage_rate
    
    def process_transactions_with_filtering(self):
        """Process transactions with proper filtering"""
        self.logger.info("Processing transactions with account filtering...")
        
        # Containers for processed data
        all_edge_features = []
        all_edge_indices = []
        all_labels = []
        all_timestamps = []
        
        # Statistics
        total_processed = 0
        valid_transactions = 0
        filtered_transactions = 0
        
        # Process transactions in chunks
        chunk_reader = pd.read_csv(self.dataset_paths['transactions'], chunksize=self.chunk_size)
        progress_bar = tqdm(desc="Processing transactions", unit="chunks")
        
        for chunk_idx, df_chunk in enumerate(chunk_reader):
            chunk_start_time = time.time()
            
            # Handle different possible column names for transactions
            src_bank_col = self._find_column(df_chunk, ['From Bank', 'from_bank', 'source_bank'], 'From Bank')
            dst_bank_col = self._find_column(df_chunk, ['To Bank', 'to_bank', 'dest_bank'], 'To Bank')
            src_account_col = self._find_column(df_chunk, ['Account', 'account', 'src_account'], 'Account')
            dst_account_col = self._find_column(df_chunk, ['Account.1', 'account.1', 'dest_account'], 'Account.1')
            
            # Create account IDs
            df_chunk['SrcAccountID'] = (df_chunk[src_bank_col].astype(str) + '_' + 
                                       df_chunk[src_account_col].astype(str))
            df_chunk['DstAccountID'] = (df_chunk[dst_bank_col].astype(str) + '_' + 
                                       df_chunk[dst_account_col].astype(str))
            
            # CRITICAL FIX: Filter transactions with invalid accounts BEFORE processing
            src_valid = df_chunk['SrcAccountID'].isin(self.valid_account_ids)
            dst_valid = df_chunk['DstAccountID'].isin(self.valid_account_ids)
            valid_mask = src_valid & dst_valid
            
            # Statistics
            total_processed += len(df_chunk)
            valid_count = valid_mask.sum()
            valid_transactions += valid_count
            filtered_transactions += len(df_chunk) - valid_count
            
            if valid_count == 0:
                progress_bar.update(1)
                continue
            
            # Filter to only valid transactions
            df_valid = df_chunk[valid_mask].copy()
            
            # Map to node indices (now guaranteed to be valid)
            df_valid['SrcID'] = df_valid['SrcAccountID'].map(self.account_to_node)
            df_valid['DstID'] = df_valid['DstAccountID'].map(self.account_to_node)
            
            # Verify mapping worked (should always be true now)
            mapping_valid = df_valid['SrcID'].notna() & df_valid['DstID'].notna()
            if not mapping_valid.all():
                self.logger.warning(f"Chunk {chunk_idx}: Some mappings failed despite filtering!")
                df_valid = df_valid[mapping_valid]
            
            if len(df_valid) == 0:
                progress_bar.update(1)
                continue
            
            # Create edge features
            edge_features = self._extract_edge_features(df_valid)
            
            # Create edge indices (now guaranteed to be valid)
            edge_indices = np.column_stack([
                df_valid['SrcID'].values.astype(int),
                df_valid['DstID'].values.astype(int)
            ])
            
            # Labels and timestamps
            labels = df_valid['Is Laundering'].values
            timestamps = pd.to_datetime(df_valid['Timestamp'], errors='coerce')
            
            # Store processed data
            all_edge_features.append(edge_features)
            all_edge_indices.append(edge_indices)
            all_labels.append(labels)
            all_timestamps.append(timestamps.values)
            
            # Progress update
            chunk_time = time.time() - chunk_start_time
            progress_bar.set_postfix({
                'valid': f'{valid_count:,}',
                'filtered': f'{len(df_chunk) - valid_count:,}',
                'time': f'{chunk_time:.1f}s'
            })
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Combine all processed data
        self.logger.info("Combining processed data...")
        final_edge_features = np.vstack(all_edge_features)
        final_edge_indices = np.vstack(all_edge_indices)
        final_labels = np.concatenate(all_labels)
        final_timestamps = np.concatenate(all_timestamps)
        
        # Final statistics
        self.logger.info(f"Transaction processing completed:")
        self.logger.info(f"  Total transactions processed: {total_processed:,}")
        self.logger.info(f"  Valid transactions kept: {valid_transactions:,} ({valid_transactions/total_processed*100:.2f}%)")
        self.logger.info(f"  Invalid transactions filtered: {filtered_transactions:,} ({filtered_transactions/total_processed*100:.2f}%)")
        self.logger.info(f"  Final edge features shape: {final_edge_features.shape}")
        self.logger.info(f"  Final edge indices shape: {final_edge_indices.shape}")
        
        # Verify edge indices are valid
        max_node_id = len(self.account_to_node) - 1
        max_edge_idx = final_edge_indices.max()
        min_edge_idx = final_edge_indices.min()
        
        self.logger.info(f"Edge index validation:")
        self.logger.info(f"  Max node ID available: {max_node_id}")
        self.logger.info(f"  Edge indices range: [{min_edge_idx}, {max_edge_idx}]")
        
        if max_edge_idx > max_node_id or min_edge_idx < 0:
            raise ValueError(f"Invalid edge indices detected! This should not happen with fixed preprocessing.")
        else:
            self.logger.info(f"  ✅ All edge indices are valid!")
        
        return final_edge_features, final_edge_indices, final_labels, final_timestamps
    
    def _extract_edge_features(self, df_chunk):
        """Extract edge features from transaction chunk"""
        # Handle missing values
        df_chunk['Amount Received'] = pd.to_numeric(df_chunk['Amount Received'], errors='coerce').fillna(0)
        df_chunk['Amount Paid'] = pd.to_numeric(df_chunk['Amount Paid'], errors='coerce').fillna(0)
        
        # Create features
        num_transactions = len(df_chunk)
        edge_features = np.zeros((num_transactions, 10))
        
        # Basic amount features
        edge_features[:, 0] = np.log1p(df_chunk['Amount Received'].values)
        edge_features[:, 1] = np.log1p(df_chunk['Amount Paid'].values)
        
        # Time features
        timestamps = pd.to_datetime(df_chunk['Timestamp'], errors='coerce')
        edge_features[:, 2] = timestamps.dt.hour.fillna(12).values / 24.0
        edge_features[:, 3] = timestamps.dt.dayofweek.fillna(3).values / 6.0
        
        # Currency and payment features
        df_chunk['Receiving Currency'] = df_chunk['Receiving Currency'].fillna('Unknown')
        df_chunk['Payment Currency'] = df_chunk['Payment Currency'].fillna('Unknown')
        df_chunk['Payment Format'] = df_chunk['Payment Format'].fillna('Unknown')
        
        # Binary features
        edge_features[:, 4] = (df_chunk['Receiving Currency'] != df_chunk['Payment Currency']).astype(float)
        edge_features[:, 5] = (df_chunk['Payment Format'] == 'Bitcoin').astype(float)
        edge_features[:, 6] = (df_chunk['Amount Received'] > df_chunk['Amount Received'].quantile(0.95)).astype(float)
        edge_features[:, 7] = (df_chunk['Amount Received'] % 1000 == 0).astype(float)
        
        # Exchange rate
        exchange_rate = df_chunk['Amount Received'] / df_chunk['Amount Paid'].clip(lower=0.01)
        edge_features[:, 8] = np.clip(exchange_rate.fillna(1.0).values, 0, 10)
        
        # Self-transaction
        edge_features[:, 9] = (df_chunk['SrcID'] == df_chunk['DstID']).astype(float)
        
        return edge_features
    
    def create_node_features(self):
        """Create node features from accounts"""
        self.logger.info("Creating node features...")
        
        # Sort accounts by the mapping order
        sorted_accounts = sorted(self.account_to_node.items(), key=lambda x: x[1])
        account_ids_ordered = [acc_id for acc_id, _ in sorted_accounts]
        
        # Create mapping from AccountID back to accounts dataframe
        accounts_with_id = self.accounts_df.set_index('AccountID')
        
        # Initialize node features
        num_nodes = len(self.account_to_node)
        node_features = np.zeros((num_nodes, 10))
        
        for node_idx, account_id in enumerate(account_ids_ordered):
            if account_id in accounts_with_id.index:
                account_row = accounts_with_id.loc[account_id]
                
                # Basic account features
                node_features[node_idx, 0] = 1.0  # Account exists
                
                # Use Bank ID for hashing (more stable than Bank Name)
                bank_id = account_row.get('Bank ID', account_row.get('Bank Name', ''))
                node_features[node_idx, 1] = hash(str(bank_id)) % 1000 / 1000.0
                
                # Entity features
                entity_id = account_row.get('Entity ID', '')
                node_features[node_idx, 2] = hash(str(entity_id)) % 1000 / 1000.0
                
                # Entity type (Corporation vs Sole Proprietorship)
                entity_name = str(account_row.get('Entity Name', ''))
                if 'corporation' in entity_name.lower():
                    node_features[node_idx, 3] = 1.0
                elif 'sole proprietorship' in entity_name.lower():
                    node_features[node_idx, 3] = 0.5
                else:
                    node_features[node_idx, 3] = 0.0
                
                # Fill remaining features with defaults
                for i in range(4, 10):
                    node_features[node_idx, i] = 0.1 * i  # Simple default features
        
        self.logger.info(f"Created node features: {node_features.shape}")
        return node_features
    
    def create_temporal_splits(self, timestamps, labels):
        """Create temporal train/val/test splits"""
        self.logger.info("Creating temporal splits...")
        
        # Convert timestamps to datetime
        valid_timestamps = pd.to_datetime(timestamps, errors='coerce')
        valid_mask = valid_timestamps.notna()
        
        if valid_mask.sum() == 0:
            # Fallback to random splits if no valid timestamps
            self.logger.warning("No valid timestamps found, using random splits")
            indices = np.arange(len(labels))
            np.random.shuffle(indices)
            
            train_size = int(0.7 * len(indices))
            val_size = int(0.15 * len(indices))
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
        else:
            # Temporal splits
            sorted_indices = np.argsort(valid_timestamps[valid_mask])
            valid_indices = np.where(valid_mask)[0][sorted_indices]
            
            train_size = int(0.7 * len(valid_indices))
            val_size = int(0.15 * len(valid_indices))
            
            train_indices = valid_indices[:train_size]
            val_indices = valid_indices[train_size:train_size + val_size]
            test_indices = valid_indices[train_size + val_size:]
        
        self.logger.info(f"Created splits: {len(train_indices):,} train, {len(val_indices):,} val, {len(test_indices):,} test")
        return train_indices, val_indices, test_indices
    
    def save_processed_data(self, edge_features, edge_indices, labels, node_features, timestamps):
        """Save the processed data"""
        self.logger.info("Saving processed data...")
        
        # Convert to tensors
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_indices.T, dtype=torch.long)  # Transpose for PyG format
        y = torch.tensor(labels, dtype=torch.long)
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Create splits
        train_indices, val_indices, test_indices = self.create_temporal_splits(timestamps, labels)
        
        # Create data objects
        complete_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        splits_data = {
            'train': Data(
                x=x,
                edge_index=edge_index[:, train_indices],
                edge_attr=edge_attr[train_indices],
                y=y[train_indices]
            ),
            'val': Data(
                x=x,
                edge_index=edge_index[:, val_indices],
                edge_attr=edge_attr[val_indices],
                y=y[val_indices]
            ),
            'test': Data(
                x=x,
                edge_index=edge_index[:, test_indices],
                edge_attr=edge_attr[test_indices],
                y=y[test_indices]
            ),
            'metadata': {
                'num_nodes': len(node_features),
                'num_edges': len(labels),
                'node_feature_dim': node_features.shape[1],
                'edge_feature_dim': edge_features.shape[1],
                'num_classes': 2,
                'dataset_name': self.dataset_name
            }
        }
        
        # Save files
        output_file = self.graphs_dir / f'ibm_aml_{self.dataset_name.lower()}_fixed_splits.pt'
        torch.save(splits_data, output_file)
        
        self.logger.info(f"Saved processed data to: {output_file}")
        self.logger.info(f"File size: {output_file.stat().st_size / 1024**2:.1f} MB")
        
        return complete_data, splits_data
    
    def process_dataset(self):
        """Main processing function"""
        self.logger.info(f"Starting fixed preprocessing for {self.dataset_name}")
        start_time = time.time()
        
        try:
            # Step 1: Load and validate accounts
            accounts_df, account_to_node = self.load_and_validate_accounts()
            
            # Step 2: Analyze transaction coverage
            coverage_rate = self.analyze_transaction_coverage()
            
            # Step 3: Process transactions with filtering
            edge_features, edge_indices, labels, timestamps = self.process_transactions_with_filtering()
            
            # Step 4: Create node features
            node_features = self.create_node_features()
            
            # Step 5: Save processed data
            complete_data, splits_data = self.save_processed_data(edge_features, edge_indices, labels, node_features, timestamps)
            
            # Final summary
            total_time = time.time() - start_time
            self.logger.info(f"Fixed preprocessing completed in {total_time/60:.1f} minutes")
            self.logger.info(f"✅ No invalid edge indices - all transactions filtered properly!")
            
            return complete_data, splits_data
            
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            raise

def process_datasets(datasets=['HI-Small', 'LI-Small']):
    """Process multiple datasets with fixed preprocessing"""
    print("🔧 FIXED PREPROCESSING - NO MORE INVALID EDGE INDICES")
    print("="*60)
    
    results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*40}")
        print(f"PROCESSING {dataset_name}")
        print(f"{'='*40}")
        
        try:
            processor = FixedAMLPreprocessor(dataset_name)
            complete_data, splits_data = processor.process_dataset()
            results[dataset_name] = {'complete': complete_data, 'splits': splits_data}
            print(f"✅ {dataset_name} processed successfully!")
            
        except Exception as e:
            print(f"❌ Failed to process {dataset_name}: {e}")
            results[dataset_name] = None
    
    print(f"\n🎉 Fixed preprocessing completed!")
    print(f"📁 Processed files saved in: data/graphs/")
    print(f"🔍 Files will have '_fixed_splits.pt' suffix")
    
    return results

if __name__ == "__main__":
    # Process both datasets
    results = process_datasets(['HI-Small', 'LI-Small'])

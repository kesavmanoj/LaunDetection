"""
Enhanced IBM AML Dataset Preprocessing with Advanced Features
Includes comprehensive logging, memory management, and temporal features.
"""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from tqdm import tqdm
import os
import gc
import time
import logging
import psutil
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import Config

class EnhancedAMLPreprocessor:
    """Enhanced preprocessor for large-scale AML datasets with advanced features"""
    
    def __init__(self, dataset_name='HI-Small', chunk_size=None, log_level='INFO'):
        """
        Initialize the enhanced preprocessor
        
        Args:
            dataset_name: Name of dataset to process (HI-Small, HI-Medium, etc.)
            chunk_size: Size of chunks for processing (auto-determined if None)
            log_level: Logging level
        """
        self.dataset_name = dataset_name
        self.chunk_size = chunk_size or Config.DEFAULT_CHUNK_SIZE
        
        # Setup directories
        Config.create_directories()
        self.dataset_paths = Config.get_dataset_paths(dataset_name)
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Initialize encoders
        self.entity_le = LabelEncoder()
        self.currency_le = LabelEncoder()
        self.payment_le = LabelEncoder()
        self.scaler = RobustScaler()  # More robust to outliers
        
        # Data containers
        self.account_stats = defaultdict(lambda: {
            'outgoing_count': 0, 'incoming_count': 0,
            'outgoing_amounts': [], 'incoming_amounts': [],
            'outgoing_suspicious': 0, 'incoming_suspicious': 0,
            'outgoing_hours': [], 'incoming_hours': [],
            'payment_formats': set(), 'currencies': set(),
            'cross_currency': 0, 'round_amounts': 0, 
            'self_transactions': 0, 'timestamps': []
        })
        
        # Memory tracking
        self.process = psutil.Process()
        self.max_memory_mb = 0
        
        self.logger.info(f"Initialized Enhanced AML Preprocessor for {dataset_name}")
        self.logger.info(f"Chunk size: {self.chunk_size:,}")
        
    def _setup_logging(self, log_level):
        """Setup comprehensive logging"""
        log_dir = Config.LOGS_DIR
        log_file = log_dir / f"preprocessing_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=Config.LOG_FORMAT,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")
        
    def _monitor_memory(self):
        """Monitor and log memory usage"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.max_memory_mb = max(self.max_memory_mb, memory_mb)
        
        if memory_mb > Config.MAX_MEMORY_USAGE_GB * 1024:
            self.logger.warning(f"High memory usage: {memory_mb:.1f} MB")
            gc.collect()
            
        return memory_mb
        
    def load_accounts_efficiently(self):
        """Load and process accounts data with enhanced error handling"""
        self.logger.info("Loading accounts data...")
        accounts_path = self.dataset_paths['accounts']
        
        if not accounts_path.exists():
            raise FileNotFoundError(f"Accounts file not found: {accounts_path}")
            
        try:
            # Try loading full file first
            df_accounts = pd.read_csv(accounts_path)
            self.logger.info(f"Loaded {len(df_accounts):,} account records")
            
        except (MemoryError, pd.errors.EmptyDataError) as e:
            self.logger.warning(f"Loading full file failed: {e}. Using chunked loading...")
            chunks = []
            
            for chunk in tqdm(pd.read_csv(accounts_path, chunksize=self.chunk_size), 
                            desc="Loading account chunks"):
                chunks.append(chunk)
                
            df_accounts = pd.concat(chunks, ignore_index=True)
            self.logger.info(f"Loaded {len(df_accounts):,} account records in chunks")
            
        # Validate required columns
        required_cols = ['Bank ID', 'Account Number', 'Entity Name']
        missing_cols = [col for col in required_cols if col not in df_accounts.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in accounts: {missing_cols}")
            
        # Process accounts
        self.logger.info("Processing account data...")
        df_accounts['FullAccountID'] = (df_accounts['Bank ID'].astype(str) + '_' + 
                                       df_accounts['Account Number'].astype(str))
        
        # Handle missing entity names
        df_accounts['Entity Name'] = df_accounts['Entity Name'].fillna('Unknown')
        
        # Fit entity encoder
        unique_entities = df_accounts['Entity Name'].unique()
        self.entity_le.fit(unique_entities)
        df_accounts['EntityTypeEncoded'] = self.entity_le.transform(df_accounts['Entity Name'])
        
        self.logger.info(f"Entity types: {len(self.entity_le.classes_):,}")
        self.logger.info(f"Sample types: {list(self.entity_le.classes_[:5])}")
        
        memory_mb = self._monitor_memory()
        self.logger.info(f"Memory usage after accounts loading: {memory_mb:.1f} MB")
        
        return df_accounts
    
    def analyze_transactions_structure(self):
        """Analyze transaction file structure and get metadata"""
        self.logger.info("Analyzing transaction file structure...")
        transactions_path = self.dataset_paths['transactions']
        
        if not transactions_path.exists():
            raise FileNotFoundError(f"Transactions file not found: {transactions_path}")
            
        # Sample first chunk to understand structure
        sample_chunk = pd.read_csv(transactions_path, nrows=1000)
        
        # Validate required columns
        required_cols = ['From Bank', 'Account', 'To Bank', 'Account.1', 
                        'Timestamp', 'Amount Received', 'Amount Paid', 
                        'Receiving Currency', 'Payment Currency', 
                        'Payment Format', 'Is Laundering']
        
        missing_cols = [col for col in required_cols if col not in sample_chunk.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in transactions: {missing_cols}")
            
        # Get file metadata
        file_size = transactions_path.stat().st_size
        estimated_rows = file_size // (sample_chunk.memory_usage(deep=True).sum() // len(sample_chunk))
        
        self.logger.info(f"Transaction file size: {file_size / 1024**2:.1f} MB")
        self.logger.info(f"Estimated rows: {estimated_rows:,}")
        self.logger.info(f"Sample columns: {list(sample_chunk.columns)}")
        
        return sample_chunk, estimated_rows
        
    def process_transactions_first_pass(self):
        """First pass: collect metadata and fit encoders"""
        self.logger.info("First pass: analyzing transaction data structure...")
        
        sample_chunk, estimated_rows = self.analyze_transactions_structure()
        transactions_path = self.dataset_paths['transactions']
        
        # Initialize containers
        all_currencies = set()
        all_payment_formats = set()
        unique_accounts = set()
        total_processed = 0
        suspicious_count = 0
        timestamps = []
        
        # Process in chunks
        chunk_reader = pd.read_csv(transactions_path, chunksize=self.chunk_size)
        progress_bar = tqdm(desc="First pass - analyzing data", unit="rows")
        
        for chunk_idx, df_chunk in enumerate(chunk_reader):
            chunk_start_time = time.time()
            
            # Create account IDs
            df_chunk['SrcAccountID'] = (df_chunk['From Bank'].astype(str) + '_' + 
                                      df_chunk['Account'].astype(str))
            df_chunk['DstAccountID'] = (df_chunk['To Bank'].astype(str) + '_' + 
                                      df_chunk['Account.1'].astype(str))
            
            # Collect unique values
            unique_accounts.update(df_chunk['SrcAccountID'].unique())
            unique_accounts.update(df_chunk['DstAccountID'].unique())
            
            # Handle missing currencies and payment formats
            df_chunk['Receiving Currency'] = df_chunk['Receiving Currency'].fillna('Unknown')
            df_chunk['Payment Currency'] = df_chunk['Payment Currency'].fillna('Unknown')
            df_chunk['Payment Format'] = df_chunk['Payment Format'].fillna('Unknown')
            
            all_currencies.update(df_chunk['Receiving Currency'].unique())
            all_currencies.update(df_chunk['Payment Currency'].unique())
            all_payment_formats.update(df_chunk['Payment Format'].unique())
            
            # Collect timestamps for temporal analysis
            df_chunk['Timestamp'] = pd.to_datetime(df_chunk['Timestamp'], errors='coerce')
            valid_timestamps = df_chunk['Timestamp'].dropna()
            if len(valid_timestamps) > 0:
                timestamps.extend([valid_timestamps.min(), valid_timestamps.max()])
            
            # Count suspicious transactions
            suspicious_count += df_chunk['Is Laundering'].sum()
            total_processed += len(df_chunk)
            
            # Update progress
            chunk_time = time.time() - chunk_start_time
            memory_mb = self._monitor_memory()
            
            progress_bar.set_postfix({
                'Chunk': f'{chunk_idx+1}',
                'Processed': f'{total_processed:,}',
                'Suspicious': f'{suspicious_count:,}',
                'Memory': f'{memory_mb:.0f}MB',
                'Time/chunk': f'{chunk_time:.1f}s'
            })
            progress_bar.update(len(df_chunk))
            
            # Memory management
            if chunk_idx % Config.GC_FREQUENCY == 0:
                gc.collect()
                
        progress_bar.close()
        
        # Temporal analysis
        if timestamps:
            min_timestamp = min(timestamps)
            max_timestamp = max(timestamps)
            time_span = max_timestamp - min_timestamp
            
            self.logger.info(f"Temporal span: {min_timestamp} to {max_timestamp}")
            self.logger.info(f"Time range: {time_span.days} days")
        
        self.logger.info(f"First pass completed:")
        self.logger.info(f"  Total transactions: {total_processed:,}")
        self.logger.info(f"  Suspicious transactions: {suspicious_count:,} ({suspicious_count/total_processed*100:.3f}%)")
        self.logger.info(f"  Unique accounts: {len(unique_accounts):,}")
        self.logger.info(f"  Currencies: {len(all_currencies)}")
        self.logger.info(f"  Payment formats: {len(all_payment_formats)}")
        
        # Fit encoders
        self.logger.info("Fitting encoders...")
        self.currency_le.fit(list(all_currencies))
        self.payment_le.fit(list(all_payment_formats))
        
        # Create account mapping
        self.logger.info("Creating account mappings...")
        unique_accounts_list = sorted(list(unique_accounts))
        account_id_map = {acc: idx for idx, acc in enumerate(unique_accounts_list)}
        
        return account_id_map, total_processed, suspicious_count, min_timestamp, max_timestamp
    
    def extract_enhanced_features(self, account_id_map, total_transactions):
        """Second pass: extract comprehensive features with temporal information"""
        self.logger.info("Second pass: extracting enhanced features...")
        transactions_path = self.dataset_paths['transactions']
        
        # Containers for final data
        all_edge_features = []
        all_edge_indices = []
        all_labels = []
        all_timestamps = []
        
        # Progress tracking
        chunk_reader = pd.read_csv(transactions_path, chunksize=self.chunk_size)
        progress_bar = tqdm(total=total_transactions, desc="Extracting features", unit="transactions")
        
        processed_count = 0
        
        for chunk_idx, df_chunk in enumerate(chunk_reader):
            chunk_start_time = time.time()
            
            # Create account IDs and mappings
            df_chunk['SrcAccountID'] = (df_chunk['From Bank'].astype(str) + '_' + 
                                      df_chunk['Account'].astype(str))
            df_chunk['DstAccountID'] = (df_chunk['To Bank'].astype(str) + '_' + 
                                      df_chunk['Account.1'].astype(str))
            df_chunk['SrcID'] = df_chunk['SrcAccountID'].map(account_id_map)
            df_chunk['DstID'] = df_chunk['DstAccountID'].map(account_id_map)
            
            # Handle missing data
            df_chunk['Receiving Currency'] = df_chunk['Receiving Currency'].fillna('Unknown')
            df_chunk['Payment Currency'] = df_chunk['Payment Currency'].fillna('Unknown')
            df_chunk['Payment Format'] = df_chunk['Payment Format'].fillna('Unknown')
            
            # Enhanced temporal features
            df_chunk['Timestamp'] = pd.to_datetime(df_chunk['Timestamp'], errors='coerce')
            df_chunk['Hour'] = df_chunk['Timestamp'].dt.hour.astype(float)
            df_chunk['DayOfWeek'] = df_chunk['Timestamp'].dt.dayofweek.astype(float)
            df_chunk['Month'] = df_chunk['Timestamp'].dt.month.astype(float)
            df_chunk['Day'] = df_chunk['Timestamp'].dt.day.astype(float)
            df_chunk['Quarter'] = df_chunk['Timestamp'].dt.quarter.astype(float)
            df_chunk['IsWeekend'] = (df_chunk['Timestamp'].dt.dayofweek >= 5).astype(float)
            
            # Business hours feature (9 AM to 5 PM)
            df_chunk['BusinessHours'] = ((df_chunk['Hour'] >= 9) & (df_chunk['Hour'] <= 17)).astype(float)
            
            # Currency and payment encoding
            df_chunk['RecCurrencyEnc'] = self.currency_le.transform(df_chunk['Receiving Currency'])
            df_chunk['PayCurrencyEnc'] = self.currency_le.transform(df_chunk['Payment Currency'])
            df_chunk['PaymentFormatEnc'] = self.payment_le.transform(df_chunk['Payment Format'])
            
            # Enhanced financial features
            df_chunk['AmountDifference'] = np.abs(df_chunk['Amount Received'] - df_chunk['Amount Paid'])
            df_chunk['ExchangeRate'] = df_chunk['Amount Received'] / df_chunk['Amount Paid'].clip(lower=0.01)
            df_chunk['RoundAmount'] = (df_chunk['Amount Received'] % 1000 == 0).astype(float)
            df_chunk['CrossCurrency'] = (df_chunk['Receiving Currency'] != df_chunk['Payment Currency']).astype(float)
            df_chunk['SelfTransaction'] = (df_chunk['SrcID'] == df_chunk['DstID']).astype(float)
            
            # Additional amount-based features
            df_chunk['LargeAmount'] = (df_chunk['Amount Received'] > df_chunk['Amount Received'].quantile(0.95)).astype(float)
            df_chunk['SmallAmount'] = (df_chunk['Amount Received'] < df_chunk['Amount Received'].quantile(0.05)).astype(float)
            
            # Extract comprehensive edge features (16 features)
            edge_features = np.column_stack([
                np.log10(df_chunk['Amount Received'].clip(lower=1)),      # 0
                np.log10(df_chunk['Amount Paid'].clip(lower=1)),          # 1
                df_chunk['Hour'],                                          # 2
                df_chunk['DayOfWeek'],                                     # 3
                df_chunk['Month'],                                         # 4
                df_chunk['Quarter'],                                       # 5
                df_chunk['IsWeekend'],                                     # 6
                df_chunk['BusinessHours'],                                 # 7
                df_chunk['RecCurrencyEnc'],                               # 8
                df_chunk['PayCurrencyEnc'],                               # 9
                df_chunk['PaymentFormatEnc'],                             # 10
                np.log10(df_chunk['AmountDifference'].clip(lower=0.01)),  # 11
                df_chunk['ExchangeRate'].clip(0, 10),                     # 12
                df_chunk['RoundAmount'],                                   # 13
                df_chunk['CrossCurrency'],                                 # 14
                df_chunk['SelfTransaction']                                # 15
            ])
            
            # Edge indices (filter out invalid mappings)
            valid_mask = df_chunk['SrcID'].notna() & df_chunk['DstID'].notna()
            edge_indices = np.column_stack([
                df_chunk.loc[valid_mask, 'SrcID'].astype(int),
                df_chunk.loc[valid_mask, 'DstID'].astype(int)
            ])
            
            # Labels and timestamps
            labels = df_chunk.loc[valid_mask, 'Is Laundering'].values
            timestamps = df_chunk.loc[valid_mask, 'Timestamp'].values
            
            # Filter features for valid edges only
            edge_features = edge_features[valid_mask]
            
            # Store chunk data
            all_edge_features.append(edge_features)
            all_edge_indices.append(edge_indices)
            all_labels.append(labels)
            all_timestamps.append(timestamps)
            
            # Update account statistics for enhanced node features
            self._update_account_statistics(df_chunk, valid_mask)
            
            processed_count += len(df_chunk)
            
            # Update progress
            chunk_time = time.time() - chunk_start_time
            memory_mb = self._monitor_memory()
            
            progress_bar.set_postfix({
                'Chunk': f'{chunk_idx+1}',
                'Valid': f'{valid_mask.sum()}/{len(df_chunk)}',
                'Memory': f'{memory_mb:.0f}MB',
                'Time': f'{chunk_time:.1f}s'
            })
            progress_bar.update(len(df_chunk))
            
            # Memory management
            if chunk_idx % Config.GC_FREQUENCY == 0:
                gc.collect()
                
        progress_bar.close()
        
        # Combine all chunks
        self.logger.info("Combining feature chunks...")
        final_edge_features = np.vstack(all_edge_features)
        final_edge_indices = np.vstack(all_edge_indices)
        final_labels = np.concatenate(all_labels)
        final_timestamps = np.concatenate(all_timestamps)
        
        self.logger.info(f"Feature extraction completed:")
        self.logger.info(f"  Edge features shape: {final_edge_features.shape}")
        self.logger.info(f"  Edge indices shape: {final_edge_indices.shape}")
        self.logger.info(f"  Labels shape: {final_labels.shape}")
        self.logger.info(f"  Valid edges: {len(final_labels):,}")
        
        return final_edge_features, final_edge_indices, final_labels, final_timestamps
    
    def _update_account_statistics(self, df_chunk, valid_mask):
        """Update account statistics for node feature creation"""
        valid_chunk = df_chunk[valid_mask]
        
        for _, row in valid_chunk.iterrows():
            src_id = int(row['SrcID'])
            dst_id = int(row['DstID'])
            
            # Update source account stats
            self.account_stats[src_id]['outgoing_count'] += 1
            self.account_stats[src_id]['outgoing_amounts'].append(row['Amount Paid'])
            self.account_stats[src_id]['outgoing_suspicious'] += row['Is Laundering']
            self.account_stats[src_id]['outgoing_hours'].append(row['Hour'])
            self.account_stats[src_id]['payment_formats'].add(row['PaymentFormatEnc'])
            self.account_stats[src_id]['currencies'].add(row['PayCurrencyEnc'])
            self.account_stats[src_id]['cross_currency'] += row['CrossCurrency']
            self.account_stats[src_id]['round_amounts'] += row['RoundAmount']
            self.account_stats[src_id]['self_transactions'] += row['SelfTransaction']
            if pd.notna(row['Timestamp']):
                self.account_stats[src_id]['timestamps'].append(row['Timestamp'])
            
            # Update destination account stats
            self.account_stats[dst_id]['incoming_count'] += 1
            self.account_stats[dst_id]['incoming_amounts'].append(row['Amount Received'])
            self.account_stats[dst_id]['incoming_suspicious'] += row['Is Laundering']
            self.account_stats[dst_id]['incoming_hours'].append(row['Hour'])
            self.account_stats[dst_id]['currencies'].add(row['RecCurrencyEnc'])
            if pd.notna(row['Timestamp']):
                self.account_stats[dst_id]['timestamps'].append(row['Timestamp'])
    
    def create_enhanced_node_features(self, num_accounts, df_accounts):
        """Create comprehensive node features with 25 dimensions"""
        self.logger.info("Creating enhanced node features...")
        
        node_features = np.zeros((num_accounts, Config.NODE_FEATURE_DIM))
        
        # Create account lookup for entity types
        account_to_entity = {}
        for _, row in df_accounts.iterrows():
            key = f"{row['Bank ID']}_{row['Account Number']}"
            account_to_entity[key] = row['EntityTypeEncoded']
        
        progress_bar = tqdm(range(num_accounts), desc="Creating node features")
        
        for node_id in progress_bar:
            stats = self.account_stats[node_id]
            
            # Basic entity information (features 0-2)
            node_features[node_id, 0] = node_id % len(self.entity_le.classes_)  # Entity type proxy
            node_features[node_id, 1] = node_id % 100  # Bank ID proxy
            
            # Transaction counts (features 2-4)
            total_txns = stats['outgoing_count'] + stats['incoming_count']
            node_features[node_id, 2] = total_txns
            node_features[node_id, 3] = stats['outgoing_count']
            node_features[node_id, 4] = stats['incoming_count']
            
            # Amount statistics (features 5-12)
            if stats['outgoing_amounts']:
                out_amounts = np.array(stats['outgoing_amounts'])
                node_features[node_id, 5] = np.log10(np.mean(out_amounts))
                node_features[node_id, 6] = np.log10(np.std(out_amounts) + 1)
                node_features[node_id, 7] = np.log10(np.max(out_amounts))
                node_features[node_id, 8] = np.log10(np.min(out_amounts))
            
            if stats['incoming_amounts']:
                in_amounts = np.array(stats['incoming_amounts'])
                node_features[node_id, 9] = np.log10(np.mean(in_amounts))
                node_features[node_id, 10] = np.log10(np.std(in_amounts) + 1)
                node_features[node_id, 11] = np.log10(np.max(in_amounts))
                node_features[node_id, 12] = np.log10(np.min(in_amounts))
            
            # Suspicious activity rates (features 13-15)
            if total_txns > 0:
                node_features[node_id, 13] = (stats['outgoing_suspicious'] + stats['incoming_suspicious']) / total_txns
                node_features[node_id, 14] = stats['outgoing_suspicious'] / max(stats['outgoing_count'], 1)
                node_features[node_id, 15] = stats['incoming_suspicious'] / max(stats['incoming_count'], 1)
            
            # Temporal patterns (features 16-18)
            all_hours = stats['outgoing_hours'] + stats['incoming_hours']
            if all_hours:
                node_features[node_id, 16] = np.mean(all_hours)  # Average transaction hour
                node_features[node_id, 17] = np.std(all_hours)   # Hour variability
                
            # Activity diversity (features 18-21)
            node_features[node_id, 18] = len(stats['payment_formats'])
            node_features[node_id, 19] = len(stats['currencies'])
            node_features[node_id, 20] = stats['cross_currency'] / max(total_txns, 1)
            node_features[node_id, 21] = stats['round_amounts'] / max(total_txns, 1)
            
            # Network behavior (features 22-24)
            node_features[node_id, 22] = stats['self_transactions']
            node_features[node_id, 23] = abs(stats['outgoing_count'] - stats['incoming_count'])  # Imbalance
            node_features[node_id, 24] = max(stats['outgoing_count'], stats['incoming_count']) / max(total_txns, 1)  # Dominance
            
            if node_id % 10000 == 0:
                progress_bar.set_postfix({'Processed': f'{node_id:,}'})
                
        progress_bar.close()
        
        # Handle NaN values and normalize
        node_features = np.nan_to_num(node_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self.logger.info("Normalizing node features...")
        node_features = self.scaler.fit_transform(node_features)
        
        self.logger.info(f"Node features created: {node_features.shape}")
        return node_features
    
    def create_temporal_splits(self, timestamps, labels, edge_indices):
        """Create chronological train/validation/test splits"""
        self.logger.info("Creating temporal data splits...")
        
        # Convert timestamps to pandas datetime for sorting
        timestamp_df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'index': range(len(timestamps))
        }).sort_values('timestamp')
        
        # Calculate split points based on time
        total_edges = len(timestamps)
        train_size = int(Config.TRAIN_RATIO * total_edges)
        val_size = int(Config.VAL_RATIO * total_edges)
        
        # Get indices for each split (chronologically ordered)
        sorted_indices = timestamp_df['index'].values
        train_indices = sorted_indices[:train_size]
        val_indices = sorted_indices[train_size:train_size + val_size]
        test_indices = sorted_indices[train_size + val_size:]
        
        self.logger.info(f"Temporal splits created:")
        self.logger.info(f"  Train: {len(train_indices):,} edges ({len(train_indices)/total_edges*100:.1f}%)")
        self.logger.info(f"  Val: {len(val_indices):,} edges ({len(val_indices)/total_edges*100:.1f}%)")
        self.logger.info(f"  Test: {len(test_indices):,} edges ({len(test_indices)/total_edges*100:.1f}%)")
        
        # Log temporal ranges
        train_times = timestamp_df.iloc[:train_size]['timestamp']
        val_times = timestamp_df.iloc[train_size:train_size + val_size]['timestamp']
        test_times = timestamp_df.iloc[train_size + val_size:]['timestamp']
        
        self.logger.info(f"Temporal ranges:")
        self.logger.info(f"  Train: {train_times.min()} to {train_times.max()}")
        self.logger.info(f"  Val: {val_times.min()} to {val_times.max()}")
        self.logger.info(f"  Test: {test_times.min()} to {test_times.max()}")
        
        return train_indices, val_indices, test_indices
    
    def save_enhanced_data(self, edge_features, edge_indices, labels, node_features, timestamps):
        """Save processed data with enhanced splits and metadata"""
        self.logger.info("Saving enhanced processed data...")
        
        # Convert to tensors
        self.logger.info("Converting to PyTorch tensors...")
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_indices.T, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Create temporal splits
        train_indices, val_indices, test_indices = self.create_temporal_splits(timestamps, labels, edge_indices)
        
        # Create complete data object
        complete_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )
        
        self.logger.info(f"Complete dataset statistics:")
        self.logger.info(f"  Nodes: {complete_data.num_nodes:,}")
        self.logger.info(f"  Edges: {complete_data.num_edges:,}")
        self.logger.info(f"  Node features: {complete_data.x.shape}")
        self.logger.info(f"  Edge features: {complete_data.edge_attr.shape}")
        self.logger.info(f"  Positive labels: {complete_data.y.sum().item():,} ({complete_data.y.float().mean().item()*100:.3f}%)")
        
        # Create split datasets
        splits_data = {
            'train': Data(
                x=complete_data.x,
                edge_index=complete_data.edge_index[:, train_indices],
                edge_attr=complete_data.edge_attr[train_indices],
                y=complete_data.y[train_indices]
            ),
            'val': Data(
                x=complete_data.x,
                edge_index=complete_data.edge_index[:, val_indices],
                edge_attr=complete_data.edge_attr[val_indices],
                y=complete_data.y[val_indices]
            ),
            'test': Data(
                x=complete_data.x,
                edge_index=complete_data.edge_index[:, test_indices],
                edge_attr=complete_data.edge_attr[test_indices],
                y=complete_data.y[test_indices]
            ),
            'complete': complete_data,
            'metadata': {
                'dataset_name': self.dataset_name,
                'num_nodes': complete_data.num_nodes,
                'num_edges': complete_data.num_edges,
                'node_feature_dim': complete_data.x.shape[1],
                'edge_feature_dim': complete_data.edge_attr.shape[1],
                'positive_rate': complete_data.y.float().mean().item(),
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'encoders': {
                    'entity_classes': self.entity_le.classes_.tolist(),
                    'currency_classes': self.currency_le.classes_.tolist(),
                    'payment_classes': self.payment_le.classes_.tolist()
                }
            }
        }
        
        # Save files
        complete_path = Config.GRAPHS_DIR / f'ibm_aml_{self.dataset_name.lower()}_enhanced_complete.pt'
        splits_path = Config.GRAPHS_DIR / f'ibm_aml_{self.dataset_name.lower()}_enhanced_splits.pt'
        
        self.logger.info("Saving complete dataset...")
        torch.save(complete_data, complete_path)
        
        self.logger.info("Saving splits and metadata...")
        torch.save(splits_data, splits_path)
        
        self.logger.info(f"Files saved:")
        self.logger.info(f"  Complete: {complete_path}")
        self.logger.info(f"  Splits: {splits_path}")
        
        return complete_data, splits_data
    
    def run_enhanced_preprocessing(self):
        """Run the complete enhanced preprocessing pipeline"""
        start_time = time.time()
        
        self.logger.info("="*80)
        self.logger.info("🚀 STARTING ENHANCED IBM AML PREPROCESSING PIPELINE")
        self.logger.info("="*80)
        
        try:
            # Step 1: Load accounts
            self.logger.info("Step 1: Loading account data...")
            df_accounts = self.load_accounts_efficiently()
            
            # Step 2: First pass - analyze transactions
            self.logger.info("Step 2: Analyzing transaction structure...")
            account_id_map, total_transactions, suspicious_count, min_time, max_time = self.process_transactions_first_pass()
            
            # Step 3: Second pass - extract features
            self.logger.info("Step 3: Extracting enhanced features...")
            edge_features, edge_indices, labels, timestamps = self.extract_enhanced_features(account_id_map, total_transactions)
            
            # Step 4: Create node features
            self.logger.info("Step 4: Creating enhanced node features...")
            num_accounts = len(account_id_map)
            node_features = self.create_enhanced_node_features(num_accounts, df_accounts)
            
            # Step 5: Save processed data
            self.logger.info("Step 5: Saving enhanced processed data...")
            complete_data, splits_data = self.save_enhanced_data(edge_features, edge_indices, labels, node_features, timestamps)
            
            # Final summary
            total_time = time.time() - start_time
            
            self.logger.info("="*80)
            self.logger.info("🎉 ENHANCED PREPROCESSING COMPLETED SUCCESSFULLY!")
            self.logger.info("="*80)
            
            self.logger.info(f"⏱️ Total processing time: {total_time/60:.1f} minutes")
            self.logger.info(f"💾 Peak memory usage: {self.max_memory_mb:.1f} MB")
            
            self.logger.info(f"📊 Final enhanced dataset statistics:")
            self.logger.info(f"  Dataset: {self.dataset_name}")
            self.logger.info(f"  Accounts (nodes): {complete_data.num_nodes:,}")
            self.logger.info(f"  Transactions (edges): {complete_data.num_edges:,}")
            self.logger.info(f"  Suspicious rate: {complete_data.y.float().mean().item()*100:.3f}%")
            self.logger.info(f"  Node features: {complete_data.x.shape[1]} dimensions")
            self.logger.info(f"  Edge features: {complete_data.edge_attr.shape[1]} dimensions")
            self.logger.info(f"  Temporal range: {min_time} to {max_time}")
            
            # Split statistics
            train_data = splits_data['train']
            val_data = splits_data['val']
            test_data = splits_data['test']
            
            self.logger.info(f"📈 Data splits:")
            self.logger.info(f"  Train: {train_data.num_edges:,} edges, {train_data.y.float().mean().item()*100:.3f}% positive")
            self.logger.info(f"  Val: {val_data.num_edges:,} edges, {val_data.y.float().mean().item()*100:.3f}% positive")
            self.logger.info(f"  Test: {test_data.num_edges:,} edges, {test_data.y.float().mean().item()*100:.3f}% positive")
            
            return complete_data, splits_data
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            raise


def main():
    """Main execution function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced IBM AML Dataset Preprocessing')
    parser.add_argument('--dataset', default='HI-Small', 
                       choices=['HI-Small', 'HI-Medium', 'HI-Large', 'LI-Small', 'LI-Medium', 'LI-Large'],
                       help='Dataset to process')
    parser.add_argument('--chunk_size', type=int, default=None,
                       help='Chunk size for processing (auto-determined if not specified)')
    parser.add_argument('--log_level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Initialize and run preprocessing
    preprocessor = EnhancedAMLPreprocessor(
        dataset_name=args.dataset,
        chunk_size=args.chunk_size,
        log_level=args.log_level
    )
    
    # Run complete preprocessing
    complete_data, splits_data = preprocessor.run_enhanced_preprocessing()
    
    print(f"\n🚀 Enhanced preprocessing completed for {args.dataset}!")
    print(f"📁 Files saved in: {Config.GRAPHS_DIR}")
    print(f"📋 Logs saved in: {Config.LOGS_DIR}")
    print("\n✅ Ready for GNN training!")


if __name__ == "__main__":
    main()

"""
Data Loading and Processing Utilities for AML Multi-GNN Project
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import logging


class AMLDataset(Dataset):
    """
    Dataset class for AML transaction data
    """
    
    def __init__(self, data: Dict[str, Any], node_features: torch.Tensor, 
                 edge_features: torch.Tensor, labels: torch.Tensor):
        """
        Initialize AML dataset
        
        Args:
            data: Graph data dictionary
            node_features: Node feature tensor
            edge_features: Edge feature tensor
            labels: Label tensor
        """
        self.data = data
        self.node_features = node_features
        self.edge_features = edge_features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'node_features': self.node_features,
            'edge_features': self.edge_features,
            'labels': self.labels[idx],
            'data': self.data
        }


def load_ibm_aml_dataset(data_path: str) -> Dict[str, Any]:
    """
    Load IBM AML dataset from CSV files
    
    Args:
        data_path: Path to dataset directory
    
    Returns:
        Dictionary containing loaded data
    """
    logger = logging.getLogger("AML_MultiGNN")
    
    try:
        # Load transaction data
        transactions_path = os.path.join(data_path, "transactions.csv")
        if os.path.exists(transactions_path):
            transactions = pd.read_csv(transactions_path)
            logger.info(f"Loaded {len(transactions)} transactions")
        else:
            raise FileNotFoundError(f"Transactions file not found: {transactions_path}")
        
        # Load account data
        accounts_path = os.path.join(data_path, "accounts.csv")
        if os.path.exists(accounts_path):
            accounts = pd.read_csv(accounts_path)
            logger.info(f"Loaded {len(accounts)} accounts")
        else:
            # Create accounts from transaction data
            accounts = create_accounts_from_transactions(transactions)
            logger.info(f"Created {len(accounts)} accounts from transactions")
        
        # Load labels if available
        labels_path = os.path.join(data_path, "labels.csv")
        if os.path.exists(labels_path):
            labels = pd.read_csv(labels_path)
            logger.info(f"Loaded {len(labels)} labels")
        else:
            labels = None
            logger.warning("No labels file found")
        
        return {
            'transactions': transactions,
            'accounts': accounts,
            'labels': labels
        }
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def create_accounts_from_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Create account information from transaction data
    
    Args:
        transactions: Transaction DataFrame
    
    Returns:
        Account DataFrame
    """
    # Get unique accounts from both source and destination
    source_accounts = transactions[['source_account', 'source_bank']].rename(
        columns={'source_account': 'account', 'source_bank': 'bank'}
    )
    dest_accounts = transactions[['destination_account', 'destination_bank']].rename(
        columns={'destination_account': 'account', 'destination_bank': 'bank'}
    )
    
    # Combine and get unique accounts
    all_accounts = pd.concat([source_accounts, dest_accounts]).drop_duplicates()
    
    # Add additional account features
    all_accounts['account_type'] = 'unknown'  # Default type
    all_accounts['balance'] = 0.0  # Default balance
    
    return all_accounts


def preprocess_transaction_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess transaction data for graph construction
    
    Args:
        data: Raw data dictionary
    
    Returns:
        Preprocessed data dictionary
    """
    logger = logging.getLogger("AML_MultiGNN")
    
    transactions = data['transactions'].copy()
    accounts = data['accounts'].copy()
    
    # Handle missing values
    transactions = transactions.fillna({
        'amount': 0.0,
        'currency': 'USD',
        'payment_type': 'unknown'
    })
    
    # Convert timestamps
    if 'timestamp' in transactions.columns:
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        transactions['hour'] = transactions['timestamp'].dt.hour
        transactions['day_of_week'] = transactions['timestamp'].dt.dayofweek
    
    # Create account mappings
    account_to_id = {account: idx for idx, account in enumerate(accounts['account'].unique())}
    
    # Map account names to IDs
    transactions['source_id'] = transactions['source_account'].map(account_to_id)
    transactions['dest_id'] = transactions['destination_account'].map(account_to_id)
    
    # Remove transactions with missing account mappings
    transactions = transactions.dropna(subset=['source_id', 'dest_id'])
    
    logger.info(f"Preprocessed {len(transactions)} transactions")
    logger.info(f"Unique accounts: {len(account_to_id)}")
    
    return {
        'transactions': transactions,
        'accounts': accounts,
        'account_to_id': account_to_id,
        'labels': data.get('labels', None)
    }


def create_graph_features(data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create node and edge features for graph construction
    
    Args:
        data: Preprocessed data dictionary
    
    Returns:
        Tuple of (node_features, edge_features, labels)
    """
    logger = logging.getLogger("AML_MultiGNN")
    
    transactions = data['transactions']
    accounts = data['accounts']
    account_to_id = data['account_to_id']
    
    # Create node features
    node_features = []
    for account in accounts['account']:
        if account in account_to_id:
            # Basic account features
            features = [
                1.0,  # Account exists
                0.0,  # Account type (encoded)
                0.0,  # Bank ID (encoded)
                0.0   # Balance (normalized)
            ]
            node_features.append(features)
    
    node_features = torch.tensor(node_features, dtype=torch.float32)
    
    # Create edge features
    edge_features = []
    for _, transaction in transactions.iterrows():
        # Basic transaction features
        features = [
            float(transaction['amount']),
            float(transaction.get('hour', 12)),  # Hour of day
            float(transaction.get('day_of_week', 0)),  # Day of week
            1.0 if transaction['currency'] == 'USD' else 0.0,  # Currency
            1.0 if transaction['payment_type'] == 'transfer' else 0.0  # Payment type
        ]
        edge_features.append(features)
    
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    
    # Create labels (if available)
    if data.get('labels') is not None:
        labels = torch.tensor(data['labels']['is_illicit'].values, dtype=torch.long)
    else:
        # Create dummy labels (all legitimate for now)
        labels = torch.zeros(len(transactions), dtype=torch.long)
    
    logger.info(f"Created node features: {node_features.shape}")
    logger.info(f"Created edge features: {edge_features.shape}")
    logger.info(f"Created labels: {labels.shape}")
    
    return node_features, edge_features, labels


def create_data_splits(data: Dict[str, Any], train_ratio: float = 0.6, 
                      val_ratio: float = 0.2, test_ratio: float = 0.2) -> Dict[str, Any]:
    """
    Create temporal train/validation/test splits
    
    Args:
        data: Preprocessed data dictionary
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    
    Returns:
        Dictionary containing data splits
    """
    logger = logging.getLogger("AML_MultiGNN")
    
    transactions = data['transactions']
    
    # Sort by timestamp for temporal split
    if 'timestamp' in transactions.columns:
        transactions = transactions.sort_values('timestamp')
    
    n_total = len(transactions)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = transactions.iloc[:n_train]
    val_data = transactions.iloc[n_train:n_train + n_val]
    test_data = transactions.iloc[n_train + n_val:]
    
    logger.info(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }


def create_dataloader(dataset: AMLDataset, batch_size: int = 32, 
                     shuffle: bool = True, **kwargs) -> DataLoader:
    """
    Create DataLoader for AML dataset
    
    Args:
        dataset: AML dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        **kwargs: Additional DataLoader arguments
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )


def save_processed_data(data: Dict[str, Any], save_path: str):
    """
    Save processed data to disk
    
    Args:
        data: Processed data dictionary
        save_path: Path to save data
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save transactions
    data['transactions'].to_csv(os.path.join(save_path, 'transactions.csv'), index=False)
    
    # Save accounts
    data['accounts'].to_csv(os.path.join(save_path, 'accounts.csv'), index=False)
    
    # Save account mapping
    if 'account_to_id' in data:
        pd.DataFrame(list(data['account_to_id'].items()), 
                    columns=['account', 'id']).to_csv(
            os.path.join(save_path, 'account_mapping.csv'), index=False)
    
    print(f"Processed data saved to: {save_path}")


def load_processed_data(load_path: str) -> Dict[str, Any]:
    """
    Load processed data from disk
    
    Args:
        load_path: Path to load data from
    
    Returns:
        Loaded data dictionary
    """
    data = {}
    
    # Load transactions
    transactions_path = os.path.join(load_path, 'transactions.csv')
    if os.path.exists(transactions_path):
        data['transactions'] = pd.read_csv(transactions_path)
    
    # Load accounts
    accounts_path = os.path.join(load_path, 'accounts.csv')
    if os.path.exists(accounts_path):
        data['accounts'] = pd.read_csv(accounts_path)
    
    # Load account mapping
    mapping_path = os.path.join(load_path, 'account_mapping.csv')
    if os.path.exists(mapping_path):
        mapping_df = pd.read_csv(mapping_path)
        data['account_to_id'] = dict(zip(mapping_df['account'], mapping_df['id']))
    
    return data


if __name__ == "__main__":
    # Test data utilities
    print("Testing data utilities...")
    
    # Create sample data for testing
    sample_transactions = pd.DataFrame({
        'source_account': ['A1', 'A2', 'A1'],
        'destination_account': ['A2', 'A3', 'A3'],
        'amount': [100.0, 200.0, 150.0],
        'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'currency': ['USD', 'USD', 'USD'],
        'payment_type': ['transfer', 'transfer', 'transfer']
    })
    
    sample_data = {
        'transactions': sample_transactions,
        'accounts': create_accounts_from_transactions(sample_transactions)
    }
    
    # Test preprocessing
    processed_data = preprocess_transaction_data(sample_data)
    print("Preprocessing completed successfully")
    
    # Test feature creation
    node_features, edge_features, labels = create_graph_features(processed_data)
    print(f"Node features shape: {node_features.shape}")
    print(f"Edge features shape: {edge_features.shape}")
    print(f"Labels shape: {labels.shape}")

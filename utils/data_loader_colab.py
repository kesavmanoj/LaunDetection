"""
Specialized Data Loading Utilities for Google Colab with Existing Data
"""

import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging


def load_existing_data(data_path: str = "/content/drive/MyDrive/LaunDetection/data/raw", 
                      use_small_dataset: bool = True) -> Dict[str, Any]:
    """
    Load existing IBM AML dataset from Google Drive
    
    Args:
        data_path: Path to existing data directory
        use_small_dataset: If True, only load HI-Small dataset (recommended for exploration)
    
    Returns:
        Dictionary containing loaded data
    """
    logger = logging.getLogger("AML_MultiGNN")
    
    print(f"Loading data from: {data_path}")
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # List all files in the directory
    files = os.listdir(data_path)
    print(f"Found {len(files)} files in data directory:")
    for file in files[:10]:  # Show first 10 files
        print(f"  - {file}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
    
    data = {}
    
    # Look for common file patterns
    csv_files = [f for f in files if f.endswith('.csv')]
    parquet_files = [f for f in files if f.endswith('.parquet')]
    
    print(f"Found {len(csv_files)} CSV files and {len(parquet_files)} Parquet files")
    
    # Filter files based on dataset size preference
    if use_small_dataset:
        print("Using HI-Small dataset for faster loading...")
        target_files = [f for f in csv_files if 'HI-Small' in f]
        print(f"Loading {len(target_files)} HI-Small files:")
    else:
        print("Loading all CSV files (this may take longer)...")
        target_files = csv_files
    
    # Load CSV files with progress tracking
    for i, csv_file in enumerate(target_files):
        file_path = os.path.join(data_path, csv_file)
        file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
        
        print(f"Loading {csv_file} ({file_size:.1f} MB)... [{i+1}/{len(target_files)}]")
        
        try:
            # For large files, read in chunks to avoid memory issues
            if file_size > 100:  # If file is larger than 100MB
                print(f"  Large file detected, reading in chunks...")
                df = pd.read_csv(file_path, chunksize=10000)
                # For now, just read the first chunk to get structure
                df = next(df)
                print(f"  ✓ Loaded sample of {csv_file}: {df.shape} (first 10k rows)")
            else:
                df = pd.read_csv(file_path)
                print(f"  ✓ Loaded {csv_file}: {df.shape}")
            
            data[csv_file.replace('.csv', '')] = df
            
        except Exception as e:
            print(f"  ✗ Failed to load {csv_file}: {e}")
            continue
    
    # Load Parquet files
    for parquet_file in parquet_files:
        file_path = os.path.join(data_path, parquet_file)
        try:
            df = pd.read_parquet(file_path)
            data[parquet_file.replace('.parquet', '')] = df
            print(f"✓ Loaded {parquet_file}: {df.shape}")
        except Exception as e:
            print(f"✗ Failed to load {parquet_file}: {e}")
    
    print(f"✓ Successfully loaded {len(data)} data files")
    return data


def identify_data_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify the structure of the loaded data
    
    Args:
        data: Dictionary containing loaded dataframes
    
    Returns:
        Dictionary with identified data structure
    """
    logger = logging.getLogger("AML_MultiGNN")
    
    structure = {
        'transactions': None,
        'accounts': None,
        'labels': None,
        'other_files': []
    }
    
    for name, df in data.items():
        print(f"\nAnalyzing {name} ({df.shape}):")
        print(f"Columns: {list(df.columns)}")
        
        # Try to identify transaction data
        if any(col in df.columns for col in ['source', 'destination', 'amount', 'timestamp']):
            structure['transactions'] = name
            print(f"  → Identified as transactions data")
        
        # Try to identify account data
        elif any(col in df.columns for col in ['account', 'bank', 'type', 'balance']):
            structure['accounts'] = name
            print(f"  → Identified as accounts data")
        
        # Try to identify labels
        elif any(col in df.columns for col in ['label', 'illicit', 'fraud', 'is_illicit']):
            structure['labels'] = name
            print(f"  → Identified as labels data")
        
        else:
            structure['other_files'].append(name)
            print(f"  → Unidentified data")
    
    return structure


def create_unified_dataframe(data: Dict[str, Any], structure: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a unified dataframe from the identified data structure
    
    Args:
        data: Dictionary containing loaded dataframes
        structure: Identified data structure
    
    Returns:
        Unified dataframe
    """
    logger = logging.getLogger("AML_MultiGNN")
    
    # Start with transactions data
    if structure['transactions']:
        transactions_df = data[structure['transactions']].copy()
        print(f"Using {structure['transactions']} as base transactions data")
    else:
        raise ValueError("No transactions data found")
    
    # Add account information if available
    if structure['accounts']:
        accounts_df = data[structure['accounts']]
        print(f"Adding account information from {structure['accounts']}")
        
        # Merge account information with transactions
        # This is a simplified merge - you may need to adjust based on your data structure
        if 'source_account' in transactions_df.columns and 'account' in accounts_df.columns:
            transactions_df = transactions_df.merge(
                accounts_df, 
                left_on='source_account', 
                right_on='account', 
                how='left',
                suffixes=('', '_source')
            )
        
        if 'destination_account' in transactions_df.columns and 'account' in accounts_df.columns:
            transactions_df = transactions_df.merge(
                accounts_df, 
                left_on='destination_account', 
                right_on='account', 
                how='left',
                suffixes=('', '_dest')
            )
    
    # Add labels if available
    if structure['labels']:
        labels_df = data[structure['labels']]
        print(f"Adding labels from {structure['labels']}")
        
        # Merge labels with transactions
        # This is a simplified merge - you may need to adjust based on your data structure
        if 'transaction_id' in transactions_df.columns and 'transaction_id' in labels_df.columns:
            transactions_df = transactions_df.merge(labels_df, on='transaction_id', how='left')
        elif 'id' in transactions_df.columns and 'id' in labels_df.columns:
            transactions_df = transactions_df.merge(labels_df, on='id', how='left')
    
    print(f"Unified dataframe shape: {transactions_df.shape}")
    return transactions_df


def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze data quality and provide insights
    
    Args:
        df: Unified dataframe
    
    Returns:
        Dictionary with data quality metrics
    """
    logger = logging.getLogger("AML_MultiGNN")
    
    quality_metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    
    # Calculate missing value percentages
    quality_metrics['missing_percentages'] = {
        col: (df[col].isnull().sum() / len(df)) * 100 
        for col in df.columns
    }
    
    # Identify potential issues
    issues = []
    if quality_metrics['missing_percentages']:
        high_missing = {k: v for k, v in quality_metrics['missing_percentages'].items() if v > 50}
        if high_missing:
            issues.append(f"High missing values: {high_missing}")
    
    quality_metrics['issues'] = issues
    
    return quality_metrics


def print_data_summary(df: pd.DataFrame, quality_metrics: Dict[str, Any]):
    """
    Print comprehensive data summary
    
    Args:
        df: Unified dataframe
        quality_metrics: Data quality metrics
    """
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {quality_metrics['memory_usage']:.2f} MB")
    print(f"Columns: {list(df.columns)}")
    
    print("\nData Types:")
    for col, dtype in quality_metrics['data_types'].items():
        print(f"  {col}: {dtype}")
    
    print("\nMissing Values:")
    for col, missing in quality_metrics['missing_values'].items():
        if missing > 0:
            percentage = quality_metrics['missing_percentages'][col]
            print(f"  {col}: {missing} ({percentage:.1f}%)")
    
    if quality_metrics['issues']:
        print("\nPotential Issues:")
        for issue in quality_metrics['issues']:
            print(f"  ⚠️  {issue}")
    
    print("=" * 60)


def main():
    """
    Main function to load and analyze existing data
    """
    print("Loading existing IBM AML dataset from Google Drive...")
    
    try:
        # Load data
        data = load_existing_data()
        
        # Identify structure
        structure = identify_data_structure(data)
        
        # Create unified dataframe
        unified_df = create_unified_dataframe(data, structure)
        
        # Analyze data quality
        quality_metrics = analyze_data_quality(unified_df)
        
        # Print summary
        print_data_summary(unified_df, quality_metrics)
        
        return unified_df, quality_metrics
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


if __name__ == "__main__":
    main()

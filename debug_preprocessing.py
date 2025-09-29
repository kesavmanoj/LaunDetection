"""
Debug script to analyze preprocessing issues and fix invalid edge indices
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path

def analyze_preprocessing_issue(dataset_name='HI-Small'):
    """Analyze why we have invalid edge indices"""
    
    base_dir = Path('/content/drive/MyDrive/LaunDetection')
    raw_data_dir = base_dir / 'data' / 'raw'
    
    # Load raw data
    accounts_file = raw_data_dir / f'{dataset_name}_accounts.csv'
    transactions_file = raw_data_dir / f'{dataset_name}_Trans.csv'
    
    print(f"🔍 Analyzing {dataset_name} preprocessing issues...")
    
    # Load accounts
    accounts_df = pd.read_csv(accounts_file)
    print(f"📊 Accounts loaded: {len(accounts_df):,} accounts")
    
    # Create account ID mapping
    accounts_df['AccountID'] = (accounts_df['Bank'].astype(str) + '_' + 
                               accounts_df['Account'].astype(str))
    
    account_ids = set(accounts_df['AccountID'].unique())
    print(f"📊 Unique account IDs: {len(account_ids):,}")
    
    # Sample transactions to check
    print(f"📊 Sampling transactions to check for missing accounts...")
    
    chunk_size = 50000
    missing_src_accounts = set()
    missing_dst_accounts = set()
    total_transactions = 0
    valid_transactions = 0
    
    chunk_reader = pd.read_csv(transactions_file, chunksize=chunk_size)
    
    for chunk_idx, df_chunk in enumerate(chunk_reader):
        if chunk_idx >= 5:  # Just check first 5 chunks
            break
            
        # Create transaction account IDs
        df_chunk['SrcAccountID'] = (df_chunk['From Bank'].astype(str) + '_' + 
                                   df_chunk['Account'].astype(str))
        df_chunk['DstAccountID'] = (df_chunk['To Bank'].astype(str) + '_' + 
                                   df_chunk['Account.1'].astype(str))
        
        # Check which accounts are missing
        src_missing = ~df_chunk['SrcAccountID'].isin(account_ids)
        dst_missing = ~df_chunk['DstAccountID'].isin(account_ids)
        
        missing_src_accounts.update(df_chunk.loc[src_missing, 'SrcAccountID'].unique())
        missing_dst_accounts.update(df_chunk.loc[dst_missing, 'DstAccountID'].unique())
        
        total_transactions += len(df_chunk)
        valid_transactions += len(df_chunk[~src_missing & ~dst_missing])
        
        print(f"  Chunk {chunk_idx}: {len(df_chunk):,} transactions, "
              f"{src_missing.sum():,} missing src, {dst_missing.sum():,} missing dst")
    
    print(f"\n📊 Analysis Results:")
    print(f"  Total transactions sampled: {total_transactions:,}")
    print(f"  Valid transactions: {valid_transactions:,}")
    print(f"  Invalid transactions: {total_transactions - valid_transactions:,}")
    print(f"  Missing source accounts: {len(missing_src_accounts):,}")
    print(f"  Missing destination accounts: {len(missing_dst_accounts):,}")
    
    # Show some examples
    if missing_src_accounts:
        print(f"\n🔍 Sample missing source accounts:")
        for i, acc in enumerate(list(missing_src_accounts)[:5]):
            print(f"    {acc}")
    
    if missing_dst_accounts:
        print(f"\n🔍 Sample missing destination accounts:")
        for i, acc in enumerate(list(missing_dst_accounts)[:5]):
            print(f"    {acc}")
    
    return {
        'total_accounts': len(account_ids),
        'missing_src': len(missing_src_accounts),
        'missing_dst': len(missing_dst_accounts),
        'valid_rate': valid_transactions / total_transactions
    }

def fix_preprocessing_issue():
    """Create a fixed preprocessing approach"""
    
    print(f"\n🔧 SOLUTION: Modified preprocessing approach")
    print(f"="*50)
    
    solution = """
    The issue is that transactions reference accounts that don't exist in the accounts file.
    
    CURRENT PROBLEM:
    1. Accounts file has ~500K accounts
    2. Transactions reference accounts not in the accounts file
    3. This creates NaN values in SrcID/DstID mapping
    4. NaN becomes invalid node indices (like 705906 when we only have 515088 nodes)
    
    SOLUTION OPTIONS:
    
    Option 1: FILTER APPROACH (Recommended)
    - Only keep transactions between accounts that exist in accounts file
    - This is what we're currently doing with edge validation
    - Pros: Clean data, no invalid indices
    - Cons: Lose some transactions
    
    Option 2: EXPAND APPROACH
    - Create dummy accounts for missing account IDs
    - Add them to the accounts dataframe with default features
    - Pros: Keep all transactions
    - Cons: Artificial accounts with no real features
    
    Option 3: HYBRID APPROACH
    - Filter out transactions with missing accounts
    - But do this during preprocessing, not during training
    - This prevents the invalid indices issue entirely
    """
    
    print(solution)
    
    return "filter"  # Recommended approach

if __name__ == "__main__":
    # Analyze the issue
    results = analyze_preprocessing_issue('HI-Small')
    
    # Suggest solution
    solution = fix_preprocessing_issue()
    
    print(f"\n✅ Recommended: Use filtering approach during preprocessing")
    print(f"   This will prevent invalid edge indices entirely")

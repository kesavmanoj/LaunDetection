"""
Debug script to check the actual column names in the raw data files
"""

import pandas as pd
from pathlib import Path

def check_column_names():
    """Check the actual column names in the raw data files"""
    
    base_dir = Path('/content/drive/MyDrive/LaunDetection')
    raw_data_dir = base_dir / 'data' / 'raw'
    
    datasets = ['HI-Small', 'LI-Small']
    
    for dataset in datasets:
        print(f"\n{'='*40}")
        print(f"CHECKING {dataset} COLUMNS")
        print(f"{'='*40}")
        
        # Check accounts file
        accounts_file = raw_data_dir / f'{dataset}_accounts.csv'
        if accounts_file.exists():
            print(f"📊 {dataset} Accounts file:")
            accounts_df = pd.read_csv(accounts_file, nrows=5)  # Just read first 5 rows
            print(f"  Columns: {list(accounts_df.columns)}")
            print(f"  Shape: {accounts_df.shape}")
            print(f"  Sample data:")
            print(accounts_df.head(2))
        else:
            print(f"❌ Accounts file not found: {accounts_file}")
        
        # Check transactions file
        transactions_file = raw_data_dir / f'{dataset}_Trans.csv'
        if transactions_file.exists():
            print(f"\n📊 {dataset} Transactions file:")
            trans_df = pd.read_csv(transactions_file, nrows=5)  # Just read first 5 rows
            print(f"  Columns: {list(trans_df.columns)}")
            print(f"  Shape: {trans_df.shape}")
            print(f"  Sample data:")
            print(trans_df.head(2))
        else:
            print(f"❌ Transactions file not found: {transactions_file}")

if __name__ == "__main__":
    check_column_names()

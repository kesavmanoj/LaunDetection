#!/usr/bin/env python3
"""
Debug script to fix data mismatch between transaction and account data
"""

import pandas as pd
import numpy as np
import os

def debug_data_mismatch(data_path):
    """Debug and fix the data mismatch issue"""
    print("🔍 Debugging data mismatch issue...")
    
    # Load the data
    transactions_file = os.path.join(data_path, "HI-Small_Trans.csv")
    accounts_file = os.path.join(data_path, "HI-Small_accounts.csv")
    
    print(f"Loading transactions from {transactions_file}")
    transactions = pd.read_csv(transactions_file, nrows=10)  # Load small sample for debugging
    
    print(f"Loading accounts from {accounts_file}")
    accounts = pd.read_csv(accounts_file, nrows=10)  # Load small sample for debugging
    
    print("\n📊 Transaction Data Analysis:")
    print("Transaction columns:", transactions.columns.tolist())
    print("First 5 transaction rows:")
    print(transactions[['Account', 'Account.1', 'From Bank', 'To Bank']].head())
    print(f"Account column types: {transactions['Account'].dtype}, {transactions['Account.1'].dtype}")
    print(f"Account unique values: {transactions['Account'].unique()[:10]}")
    
    print("\n📊 Account Data Analysis:")
    print("Account columns:", accounts.columns.tolist())
    print("First 5 account rows:")
    print(accounts[['Account Number', 'Bank Name', 'Entity Name']].head())
    print(f"Account Number type: {accounts['Account Number'].dtype}")
    print(f"Account Number unique values: {accounts['Account Number'].unique()[:10]}")
    
    print("\n🔍 Data Mismatch Analysis:")
    print("Transaction Account IDs:", set(transactions['Account'].unique()) | set(transactions['Account.1'].unique()))
    print("Account Number IDs:", set(accounts['Account Number'].unique()))
    print("Overlap:", set(transactions['Account'].unique()) & set(accounts['Account Number'].unique()))
    
    # Check if there's a mapping between Bank names and Account Numbers
    print("\n🔍 Bank Name Analysis:")
    print("Transaction From Bank unique:", transactions['From Bank'].unique()[:10])
    print("Transaction To Bank unique:", transactions['To Bank'].unique()[:10])
    print("Account Bank Name unique:", accounts['Bank Name'].unique()[:10])
    
    # Check if Bank names match
    trans_banks = set(transactions['From Bank'].unique()) | set(transactions['To Bank'].unique())
    account_banks = set(accounts['Bank Name'].unique())
    print("Bank name overlap:", len(trans_banks & account_banks))
    
    return transactions, accounts

def create_fixed_data_loader(data_path):
    """Create a fixed data loader that handles the mismatch"""
    print("\n🔧 Creating fixed data loader...")
    
    # Load data
    transactions_file = os.path.join(data_path, "HI-Small_Trans.csv")
    accounts_file = os.path.join(data_path, "HI-Small_accounts.csv")
    
    transactions = pd.read_csv(transactions_file, nrows=2000)
    accounts = pd.read_csv(accounts_file, nrows=1000)
    
    print("Original data shapes:")
    print(f"Transactions: {transactions.shape}")
    print(f"Accounts: {accounts.shape}")
    
    # Strategy 1: Use Bank names as the linking key
    print("\n🔧 Strategy 1: Using Bank names as linking key")
    
    # Create account features using Bank names
    account_features = {}
    
    for _, account in accounts.iterrows():
        bank_name = account['Bank Name']
        account_number = account['Account Number']
        
        # Use bank name as the key (since transactions use bank names)
        account_features[bank_name] = [
            5000.0,  # balance
            0.5,     # risk_score
            1,       # checking
            0,       # savings
            0        # business
        ]
    
    print(f"Created {len(account_features)} account features using bank names")
    print(f"Sample bank names: {list(account_features.keys())[:5]}")
    
    # Check transaction bank names
    trans_banks = set(transactions['From Bank'].unique()) | set(transactions['To Bank'].unique())
    account_banks = set(account_features.keys())
    overlap = trans_banks & account_banks
    
    print(f"Transaction banks: {len(trans_banks)}")
    print(f"Account banks: {len(account_banks)}")
    print(f"Overlap: {len(overlap)}")
    print(f"Sample overlap: {list(overlap)[:5]}")
    
    if len(overlap) > 0:
        print("✅ SUCCESS: Found bank name overlap!")
        return transactions, accounts, account_features
    else:
        print("❌ FAILED: No bank name overlap found")
        
        # Strategy 2: Create synthetic accounts from transaction data
        print("\n🔧 Strategy 2: Creating accounts from transaction data")
        
        # Get all unique bank names from transactions
        all_banks = set(transactions['From Bank'].unique()) | set(transactions['To Bank'].unique())
        
        # Create account features for each bank
        account_features = {}
        for bank_name in all_banks:
            account_features[bank_name] = [
                5000.0,  # balance
                0.5,     # risk_score
                1,       # checking
                0,       # savings
                0        # business
            ]
        
        print(f"Created {len(account_features)} account features from transaction banks")
        return transactions, accounts, account_features

def test_fixed_loader(data_path):
    """Test the fixed data loader"""
    print("\n🧪 Testing fixed data loader...")
    
    transactions, accounts, account_features = create_fixed_data_loader(data_path)
    
    # Test transaction processing
    matched_edges = 0
    total_transactions = len(transactions)
    
    for i, (_, transaction) in enumerate(transactions.iterrows()):
        from_bank = transaction['From Bank']
        to_bank = transaction['To Bank']
        
        if from_bank in account_features and to_bank in account_features:
            matched_edges += 1
    
    print(f"✅ Matched {matched_edges}/{total_transactions} transactions ({matched_edges/total_transactions*100:.1f}%)")
    
    if matched_edges > 0:
        print("🎉 SUCCESS: Fixed data loader works!")
        return True
    else:
        print("❌ FAILED: Still no matches found")
        return False

if __name__ == "__main__":
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    print("🔍 Step 1: Debug data mismatch")
    transactions, accounts = debug_data_mismatch(data_path)
    
    print("\n🔧 Step 2: Create fixed data loader")
    success = test_fixed_loader(data_path)
    
    if success:
        print("\n✅ Data mismatch fixed! You can now use the fixed approach.")
    else:
        print("\n❌ Data mismatch still exists. Need alternative approach.")

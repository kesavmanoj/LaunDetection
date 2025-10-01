#!/usr/bin/env python3
"""
Test script to verify the data mismatch fix
"""

import pandas as pd
import os

def test_data_mismatch_fix():
    """Test the data mismatch fix"""
    print("🔍 Testing data mismatch fix...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    # Load small samples for testing
    transactions_file = os.path.join(data_path, "HI-Small_Trans.csv")
    accounts_file = os.path.join(data_path, "HI-Small_accounts.csv")
    
    print(f"Loading transactions from {transactions_file}")
    transactions = pd.read_csv(transactions_file, nrows=10)
    
    print(f"Loading accounts from {accounts_file}")
    accounts = pd.read_csv(accounts_file, nrows=10)
    
    print("\n📊 Transaction Data Analysis:")
    print("Transaction columns:", transactions.columns.tolist())
    print("First 5 transaction rows:")
    print(transactions[['From Bank', 'To Bank', 'Account', 'Account.1']].head())
    
    print("\n📊 Account Data Analysis:")
    print("Account columns:", accounts.columns.tolist())
    print("First 5 account rows:")
    print(accounts[['Bank Name', 'Account Number', 'Entity Name']].head())
    
    print("\n🔍 Data Mismatch Analysis:")
    print("Transaction From Bank unique:", transactions['From Bank'].unique()[:10])
    print("Transaction To Bank unique:", transactions['To Bank'].unique()[:10])
    print("Account Bank Name unique:", accounts['Bank Name'].unique()[:10])
    
    # Check if Bank names match
    trans_banks = set(transactions['From Bank'].unique()) | set(transactions['To Bank'].unique())
    account_banks = set(accounts['Bank Name'].unique())
    overlap = trans_banks & account_banks
    
    print(f"\n✅ Bank name overlap: {len(overlap)}")
    print(f"Sample overlap: {list(overlap)[:5]}")
    
    if len(overlap) > 0:
        print("🎉 SUCCESS: Found bank name overlap!")
        print("The data mismatch fix should work!")
        return True
    else:
        print("❌ FAILED: No bank name overlap found")
        print("Need to investigate further...")
        return False

if __name__ == "__main__":
    success = test_data_mismatch_fix()
    
    if success:
        print("\n✅ Data mismatch fix verified! You can now run the fixed training script.")
        print("Run: python notebooks/06_clean_training_fixed.py")
    else:
        print("\n❌ Data mismatch still exists. Need alternative approach.")

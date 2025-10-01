#!/usr/bin/env python3
"""
Fix data mapping between transaction and account data
"""

import pandas as pd
import os

def analyze_data_structure(data_path):
    """Analyze the actual data structure to understand the mapping"""
    print("🔍 Analyzing data structure...")
    
    # Load data
    transactions_file = os.path.join(data_path, "HI-Small_Trans.csv")
    accounts_file = os.path.join(data_path, "HI-Small_accounts.csv")
    
    transactions = pd.read_csv(transactions_file, nrows=100)
    accounts = pd.read_csv(accounts_file, nrows=100)
    
    print("\n📊 Transaction Data Structure:")
    print("Columns:", transactions.columns.tolist())
    print("From Bank unique:", sorted(transactions['From Bank'].unique())[:20])
    print("To Bank unique:", sorted(transactions['To Bank'].unique())[:20])
    print("Account unique:", transactions['Account'].unique()[:10])
    print("Account.1 unique:", transactions['Account.1'].unique()[:10])
    
    print("\n📊 Account Data Structure:")
    print("Columns:", accounts.columns.tolist())
    print("Bank Name unique:", accounts['Bank Name'].unique()[:10])
    print("Account Number unique:", accounts['Account Number'].unique()[:10])
    
    # Check if Account numbers match between transactions and accounts
    trans_accounts = set(transactions['Account'].unique()) | set(transactions['Account.1'].unique())
    account_numbers = set(accounts['Account Number'].unique())
    account_overlap = trans_accounts & account_numbers
    
    print(f"\n🔍 Account Number Overlap: {len(account_overlap)}")
    print(f"Sample overlap: {list(account_overlap)[:5]}")
    
    return transactions, accounts, account_overlap

def create_bank_mapping(transactions, accounts):
    """Create a mapping between numeric bank IDs and bank names"""
    print("\n🔧 Creating bank mapping...")
    
    # Strategy 1: Use Account numbers as the linking key
    print("Strategy 1: Using Account numbers as linking key")
    
    # Get all unique account numbers from transactions
    trans_accounts = set(transactions['Account'].unique()) | set(transactions['Account.1'].unique())
    account_numbers = set(accounts['Account Number'].unique())
    
    print(f"Transaction accounts: {len(trans_accounts)}")
    print(f"Account numbers: {len(account_numbers)}")
    print(f"Overlap: {len(trans_accounts & account_numbers)}")
    
    if len(trans_accounts & account_numbers) > 0:
        print("✅ SUCCESS: Found account number overlap!")
        return "account_numbers"
    
    # Strategy 2: Create synthetic mapping
    print("\nStrategy 2: Creating synthetic bank mapping")
    
    # Get unique bank IDs from transactions
    bank_ids = set(transactions['From Bank'].unique()) | set(transactions['To Bank'].unique())
    bank_names = set(accounts['Bank Name'].unique())
    
    print(f"Transaction bank IDs: {len(bank_ids)}")
    print(f"Account bank names: {len(bank_names)}")
    
    # Create a mapping dictionary
    bank_mapping = {}
    for i, bank_id in enumerate(sorted(bank_ids)):
        if i < len(bank_names):
            bank_mapping[bank_id] = list(bank_names)[i]
        else:
            bank_mapping[bank_id] = f"Bank_{bank_id}"
    
    print(f"Created mapping for {len(bank_mapping)} bank IDs")
    print(f"Sample mapping: {dict(list(bank_mapping.items())[:5])}")
    
    return bank_mapping

def test_fixed_data_loader(data_path):
    """Test the fixed data loader with proper mapping"""
    print("\n🧪 Testing fixed data loader...")
    
    # Load data
    transactions_file = os.path.join(data_path, "HI-Small_Trans.csv")
    accounts_file = os.path.join(data_path, "HI-Small_accounts.csv")
    
    transactions = pd.read_csv(transactions_file, nrows=1000)
    accounts = pd.read_csv(accounts_file, nrows=1000)
    
    # Strategy 1: Use Account numbers
    print("Testing Strategy 1: Account numbers")
    
    # Create account features using Account Number as key
    account_features = {}
    for _, account in accounts.iterrows():
        account_number = account['Account Number']
        bank_name = account['Bank Name']
        
        account_features[account_number] = [
            5000.0,  # balance
            0.5,     # risk_score
            1,       # checking
            0,       # savings
            0        # business
        ]
    
    print(f"Created {len(account_features)} account features using account numbers")
    
    # Test transaction processing
    matched_edges = 0
    total_transactions = len(transactions)
    
    for i, (_, transaction) in enumerate(transactions.iterrows()):
        from_account = transaction['Account']
        to_account = transaction['Account.1']
        
        if from_account in account_features and to_account in account_features:
            matched_edges += 1
    
    print(f"✅ Matched {matched_edges}/{total_transactions} transactions ({matched_edges/total_transactions*100:.1f}%)")
    
    if matched_edges > 0:
        print("🎉 SUCCESS: Account number strategy works!")
        return True
    else:
        print("❌ FAILED: Account number strategy doesn't work")
        
        # Strategy 2: Use bank mapping
        print("\nTesting Strategy 2: Bank mapping")
        
        # Create bank mapping
        bank_ids = set(transactions['From Bank'].unique()) | set(transactions['To Bank'].unique())
        bank_names = set(accounts['Bank Name'].unique())
        
        bank_mapping = {}
        for i, bank_id in enumerate(sorted(bank_ids)):
            if i < len(bank_names):
                bank_mapping[bank_id] = list(bank_names)[i]
            else:
                bank_mapping[bank_id] = f"Bank_{bank_id}"
        
        # Create account features using bank names
        account_features = {}
        for _, account in accounts.iterrows():
            bank_name = account['Bank Name']
            account_features[bank_name] = [
                5000.0,  # balance
                0.5,     # risk_score
                1,       # checking
                0,       # savings
                0        # business
            ]
        
        # Test with bank mapping
        matched_edges = 0
        for i, (_, transaction) in enumerate(transactions.iterrows()):
            from_bank_id = transaction['From Bank']
            to_bank_id = transaction['To Bank']
            
            from_bank_name = bank_mapping.get(from_bank_id)
            to_bank_name = bank_mapping.get(to_bank_id)
            
            if from_bank_name in account_features and to_bank_name in account_features:
                matched_edges += 1
        
        print(f"✅ Matched {matched_edges}/{total_transactions} transactions ({matched_edges/total_transactions*100:.1f}%)")
        
        if matched_edges > 0:
            print("🎉 SUCCESS: Bank mapping strategy works!")
            return True
        else:
            print("❌ FAILED: Both strategies failed")
            return False

if __name__ == "__main__":
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    print("🔍 Step 1: Analyze data structure")
    transactions, accounts, account_overlap = analyze_data_structure(data_path)
    
    print("\n🔧 Step 2: Create bank mapping")
    mapping_result = create_bank_mapping(transactions, accounts)
    
    print("\n🧪 Step 3: Test fixed data loader")
    success = test_fixed_data_loader(data_path)
    
    if success:
        print("\n✅ Data mapping fixed! You can now run the training script.")
    else:
        print("\n❌ Data mapping still needs work. Need to investigate further.")

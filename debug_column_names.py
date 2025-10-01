#!/usr/bin/env python3
"""
Debug Column Names in Real Dataset
===================================

This script checks the actual column names in the real dataset.
"""

import pandas as pd
import os

print("🔍 Debugging Column Names in Real Dataset")
print("=" * 60)

def debug_column_names():
    """Debug the actual column names in the dataset"""
    print("📊 Checking column names in real dataset...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    try:
        # Load a small sample to check columns
        transactions = pd.read_csv(os.path.join(data_path, 'HI-Small_Trans.csv'), nrows=5)
        accounts = pd.read_csv(os.path.join(data_path, 'HI-Small_accounts.csv'), nrows=5)
        
        print("✅ Successfully loaded data samples")
        
        # Check transaction columns
        print("\n📋 Transaction columns:")
        print(f"   Total columns: {len(transactions.columns)}")
        for i, col in enumerate(transactions.columns):
            print(f"   {i+1:2d}. {col}")
        
        print(f"\n📋 Sample transaction data:")
        print(transactions.head())
        
        # Check account columns
        print("\n📋 Account columns:")
        print(f"   Total columns: {len(accounts.columns)}")
        for i, col in enumerate(accounts.columns):
            print(f"   {i+1:2d}. {col}")
        
        print(f"\n📋 Sample account data:")
        print(accounts.head())
        
        # Look for amount-related columns
        print("\n🔍 Looking for amount-related columns in transactions:")
        amount_cols = []
        for col in transactions.columns:
            if 'amount' in col.lower() or 'value' in col.lower() or 'money' in col.lower():
                amount_cols.append(col)
                print(f"   Found: {col}")
        
        if not amount_cols:
            print("   No amount-related columns found")
            print("   Available columns:", list(transactions.columns))
        
        # Look for bank-related columns
        print("\n🔍 Looking for bank-related columns in transactions:")
        bank_cols = []
        for col in transactions.columns:
            if 'bank' in col.lower() or 'from' in col.lower() or 'to' in col.lower():
                bank_cols.append(col)
                print(f"   Found: {col}")
        
        if not bank_cols:
            print("   No bank-related columns found")
            print("   Available columns:", list(transactions.columns))
        
        # Look for AML-related columns
        print("\n🔍 Looking for AML-related columns in transactions:")
        aml_cols = []
        for col in transactions.columns:
            if 'launder' in col.lower() or 'aml' in col.lower() or 'suspicious' in col.lower():
                aml_cols.append(col)
                print(f"   Found: {col}")
        
        if not aml_cols:
            print("   No AML-related columns found")
            print("   Available columns:", list(transactions.columns))
        
        return transactions, accounts, amount_cols, bank_cols, aml_cols
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, [], [], []

def suggest_column_mapping(transactions, accounts, amount_cols, bank_cols, aml_cols):
    """Suggest column mapping based on available columns"""
    print("\n💡 Suggested Column Mapping:")
    print("-" * 40)
    
    if transactions is not None:
        print("📋 Transaction columns available:")
        for col in transactions.columns:
            print(f"   - {col}")
        
        # Suggest mappings
        print("\n🎯 Suggested mappings:")
        
        # Amount mapping
        if amount_cols:
            print(f"   Amount: {amount_cols[0]}")
        else:
            print("   Amount: Not found - need to identify amount column")
        
        # Bank mapping
        if bank_cols:
            print(f"   From Bank: {bank_cols[0] if len(bank_cols) > 0 else 'Not found'}")
            print(f"   To Bank: {bank_cols[1] if len(bank_cols) > 1 else bank_cols[0] if len(bank_cols) > 0 else 'Not found'}")
        else:
            print("   From Bank: Not found - need to identify source column")
            print("   To Bank: Not found - need to identify destination column")
        
        # AML mapping
        if aml_cols:
            print(f"   Is Laundering: {aml_cols[0]}")
        else:
            print("   Is Laundering: Not found - need to identify AML column")
        
        # Other important columns
        print("\n🔍 Other important columns to identify:")
        print("   - Transaction Type")
        print("   - Day/Hour (temporal)")
        print("   - Currency")
        print("   - Channel")
        print("   - Location")
        print("   - Risk Score")

def main():
    """Main debugging function"""
    print("🔍 Debugging Column Names...")
    
    # Debug column names
    transactions, accounts, amount_cols, bank_cols, aml_cols = debug_column_names()
    
    if transactions is not None:
        # Suggest column mapping
        suggest_column_mapping(transactions, accounts, amount_cols, bank_cols, aml_cols)
        
        print("\n📋 Next Steps:")
        print("1. Identify the correct column names")
        print("2. Update the training script with correct column names")
        print("3. Test the updated script")
    else:
        print("❌ Could not load data for debugging")

if __name__ == "__main__":
    main()

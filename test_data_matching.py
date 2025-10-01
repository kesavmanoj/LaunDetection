#!/usr/bin/env python3
"""
Test data matching for IBM AML data
===================================

Test the data matching logic to ensure edges are created properly.
"""

import pandas as pd
import numpy as np

# Simulate the real IBM AML data structure
print("Testing data matching for IBM AML data...")

# Simulate transaction data
transactions_data = {
    'Timestamp': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-01 12:00:00'],
    'From Bank': ['Bank A', 'Bank B', 'Bank A'],
    'Account': ['ACC001', 'ACC002', 'ACC001'],
    'To Bank': ['Bank C', 'Bank D', 'Bank B'],
    'Account.1': ['ACC003', 'ACC004', 'ACC002'],
    'Amount Received': [1000, 2000, 1500],
    'Receiving Currency': ['USD', 'EUR', 'USD'],
    'Amount Paid': [1000, 2000, 1500],
    'Payment Currency': ['USD', 'EUR', 'USD'],
    'Payment Format': ['Wire', 'ACH', 'Wire'],
    'Is Laundering': [0, 1, 0]
}

transactions = pd.DataFrame(transactions_data)
print("Transaction data:")
print(transactions.head())

# Simulate account data
accounts_data = {
    'Bank Name': ['Bank A', 'Bank B', 'Bank C', 'Bank D'],
    'Bank ID': ['B001', 'B002', 'B003', 'B004'],
    'Account Number': ['Bank A', 'Bank B', 'Bank C', 'Bank D'],  # Use bank names as account numbers
    'Entity ID': ['E001', 'E002', 'E003', 'E004'],
    'Entity Name': ['Entity A', 'Entity B', 'Entity C', 'Entity D']
}

accounts = pd.DataFrame(accounts_data)
print("\nAccount data:")
print(accounts.head())

# Test column mapping
print("\nTesting column mapping...")

# Transaction columns
from_col = 'From Bank'
to_col = 'To Bank'
amount_col = 'Amount Paid'
timestamp_col = 'Timestamp'
sar_col = 'Is Laundering'

print(f"Using columns: from={from_col}, to={to_col}, amount={amount_col}, time={timestamp_col}, sar={sar_col}")

# Account columns
account_id_col = 'Account Number'
print(f"Account ID column: {account_id_col}")

# Create account features
account_features = {}
for _, account in accounts.iterrows():
    account_id = account[account_id_col]
    account_features[account_id] = [5000.0, 0.5, 1, 0, 0]  # Default features

print(f"Account features keys: {list(account_features.keys())}")

# Test edge creation
edges = []
edge_features = []
labels = []

print("\nTesting edge creation...")
for i, (_, transaction) in enumerate(transactions.iterrows()):
    from_acc = transaction[from_col]
    to_acc = transaction[to_col]
    amount = float(transaction[amount_col])
    is_sar = int(transaction[sar_col])
    
    print(f"Transaction {i}: from_acc={from_acc}, to_acc={to_acc}")
    print(f"  from_acc in account_features: {from_acc in account_features}")
    print(f"  to_acc in account_features: {to_acc in account_features}")
    
    if from_acc in account_features and to_acc in account_features:
        edges.append([from_acc, to_acc])
        edge_features.append([amount, 12, 1, 1])
        labels.append(is_sar)
        print(f"  ✓ Edge created")
    else:
        print(f"  ✗ Edge not created - account mismatch")

print(f"\nCreated {len(edges)} edges out of {len(transactions)} transactions")
print(f"Edges: {edges}")
print(f"Labels: {labels}")

print("\nData matching test completed!")

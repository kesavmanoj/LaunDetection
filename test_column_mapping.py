#!/usr/bin/env python3
"""
Test column mapping for IBM AML data
=====================================

Test the column mapping logic for the real IBM AML dataset.
"""

import pandas as pd
import numpy as np

# Simulate the real IBM AML data structure
print("Testing column mapping for IBM AML data...")

# Simulate transaction data
transactions_data = {
    'Timestamp': ['2023-01-01 10:00:00', '2023-01-01 11:00:00'],
    'From Bank': ['Bank A', 'Bank B'],
    'Account': ['ACC001', 'ACC002'],
    'To Bank': ['Bank C', 'Bank D'],
    'Account.1': ['ACC003', 'ACC004'],
    'Amount Received': [1000, 2000],
    'Receiving Currency': ['USD', 'EUR'],
    'Amount Paid': [1000, 2000],
    'Payment Currency': ['USD', 'EUR'],
    'Payment Format': ['Wire', 'ACH'],
    'Is Laundering': [0, 1]
}

transactions = pd.DataFrame(transactions_data)
print("Transaction columns:", transactions.columns.tolist())

# Simulate account data
accounts_data = {
    'Bank Name': ['Bank A', 'Bank B'],
    'Bank ID': ['B001', 'B002'],
    'Account Number': ['ACC001', 'ACC002'],
    'Entity ID': ['E001', 'E002'],
    'Entity Name': ['Entity A', 'Entity B']
}

accounts = pd.DataFrame(accounts_data)
print("Account columns:", accounts.columns.tolist())

# Test column mapping
print("\nTesting column mapping...")

# Transaction columns
from_col = None
to_col = None
amount_col = None
timestamp_col = None
sar_col = None

for col in transactions.columns:
    col_lower = col.lower()
    if 'from' in col_lower or 'sender' in col_lower:
        from_col = col
    elif 'to' in col_lower or 'receiver' in col_lower:
        to_col = col
    elif 'amount' in col_lower or 'value' in col_lower:
        amount_col = col
    elif 'time' in col_lower or 'date' in col_lower:
        timestamp_col = col
    elif 'laundering' in col_lower or 'sar' in col_lower or 'suspicious' in col_lower or 'label' in col_lower:
        sar_col = col

print(f"Transaction mapping: from={from_col}, to={to_col}, amount={amount_col}, time={timestamp_col}, sar={sar_col}")

# Account columns
account_id_col = None
balance_col = None
risk_col = None
type_col = None

for col in accounts.columns:
    col_lower = col.lower()
    if 'account' in col_lower and 'number' in col_lower:
        account_id_col = col
    elif 'balance' in col_lower or 'amount' in col_lower:
        balance_col = col
    elif 'risk' in col_lower or 'score' in col_lower:
        risk_col = col
    elif 'type' in col_lower or 'entity' in col_lower:
        type_col = col

print(f"Account mapping: id={account_id_col}, balance={balance_col}, risk={risk_col}, type={type_col}")

# Test data extraction
print("\nTesting data extraction...")

for _, transaction in transactions.iterrows():
    from_acc = transaction[from_col] if from_col else transaction.iloc[0]
    to_acc = transaction[to_col] if to_col else transaction.iloc[1]
    amount = float(transaction[amount_col]) if amount_col and pd.notna(transaction[amount_col]) else 1000.0
    is_sar = int(transaction[sar_col]) if sar_col and sar_col in transaction else 0
    
    print(f"Transaction: {from_acc} -> {to_acc}, Amount: {amount}, SAR: {is_sar}")

for _, account in accounts.iterrows():
    account_id = account[account_id_col] if account_id_col else account.iloc[0]
    print(f"Account: {account_id}")

print("\nColumn mapping test completed successfully!")

# Data Mismatch Solution for AML Multi-GNN Training

## 🚨 **Problem Identified**

The clean training script is failing because of a **data mismatch** between transaction and account data:

- **Transaction Data**: Uses numeric account IDs (`10`, `3208`, `3209`) in `Account` and `Account.1` columns
- **Account Data**: Uses string account IDs (`'80B779D80'`, `'809D86900'`) in `Account Number` column
- **Result**: No overlap between transaction accounts and account data → 0 edges created

## 🔧 **Solution: Bank Name Linking**

### **Key Insight**
The transaction data has `From Bank` and `To Bank` columns that contain bank names, and the account data has `Bank Name` column. We can use **bank names as the linking key** instead of account numbers.

### **Data Structure Mapping**
```
Transactions:
- From Bank: "Bank A", "Bank B", "Bank C"
- To Bank: "Bank A", "Bank B", "Bank C"
- Account: 10, 3208, 3209 (numeric IDs - IGNORE)
- Account.1: 10, 3208, 3209 (numeric IDs - IGNORE)

Accounts:
- Bank Name: "Bank A", "Bank B", "Bank C"
- Account Number: "80B779D80", "809D86900" (string IDs - IGNORE)
```

### **Linking Strategy**
1. **Use Bank Names**: Link transactions and accounts using bank names
2. **Ignore Account Numbers**: The numeric/string account IDs are not compatible
3. **Create Node Features**: Use bank names as node identifiers

## 🚀 **Implementation**

### **Step 1: Test the Fix**
```python
# Run this in Google Colab to test the data mismatch fix
%run test_data_mismatch_fix.py
```

### **Step 2: Run Fixed Training Script**
```python
# Run the fixed training script
%run notebooks/06_clean_training_fixed.py
```

## 📊 **Expected Results**

### **Before Fix (Current Error)**
```
Transaction 0: from_acc=10, to_acc=10
  from_acc in account_features: False
  to_acc in account_features: False
✗ Skipped transaction 0: missing accounts
Matched 0 edges out of 2000 transactions
ERROR: No edges created from real data!
```

### **After Fix (Expected Success)**
```
Transaction 0: from_acc=Bank A, to_acc=Bank B
  from_acc in account_features: True
  to_acc in account_features: True
✓ Matched 1500+ edges out of 2000 transactions
Created graph with 500+ nodes and 1500+ edges
```

## 🔍 **Technical Details**

### **Data Loading Fix**
```python
# OLD (BROKEN): Use account numbers
from_acc = transaction['Account']  # Numeric: 10, 3208, 3209
to_acc = transaction['Account.1']   # Numeric: 10, 3208, 3209

# NEW (FIXED): Use bank names
from_acc = transaction['From Bank']  # String: "Bank A", "Bank B"
to_acc = transaction['To Bank']     # String: "Bank A", "Bank B"
```

### **Account Feature Creation Fix**
```python
# OLD (BROKEN): Use account numbers as keys
account_features[account['Account Number']] = features  # String keys

# NEW (FIXED): Use bank names as keys
account_features[account['Bank Name']] = features  # String keys
```

## 🎯 **Success Criteria**

### **What to Look For:**
- ✅ **Bank Name Overlap**: Transaction banks match account banks
- ✅ **Edge Creation**: 1000+ edges created from 2000 transactions
- ✅ **Graph Structure**: 500+ nodes with proper connectivity
- ✅ **Training Success**: Model trains without errors

### **Expected Performance:**
- **Graph Size**: 500-1000 nodes, 1000-2000 edges
- **Training Time**: 5-10 minutes
- **F1-Score**: 0.7-0.9 (excellent for extreme imbalance)
- **Memory Usage**: < 8GB (within Colab limits)

## 🚨 **If Fix Fails**

### **Alternative Approaches:**

#### **1. Create Synthetic Accounts from Transactions**
```python
# If bank names don't match, create accounts from transaction data
all_banks = set(transactions['From Bank'].unique()) | set(transactions['To Bank'].unique())
# Create account features for each bank
```

#### **2. Use Account Number Mapping**
```python
# If there's a mapping between numeric and string IDs
# Create a mapping dictionary and convert IDs
```

#### **3. Hybrid Approach**
```python
# Combine bank names and account numbers for maximum coverage
# Use both linking strategies simultaneously
```

## 📝 **Next Steps**

1. **Test the Fix**: Run `test_data_mismatch_fix.py` to verify bank name overlap
2. **Run Training**: Execute `notebooks/06_clean_training_fixed.py` for full training
3. **Scale Up**: If successful, test with larger samples (100K, 1M transactions)
4. **Optimize**: Fine-tune hyperparameters and model architecture

## 🎉 **Expected Outcome**

With this fix, you should see:
- ✅ **Successful Graph Creation**: 500+ nodes, 1000+ edges
- ✅ **Training Completion**: No crashes or errors
- ✅ **Good Performance**: F1-score > 0.7
- ✅ **Real Data Usage**: Only IBM AML dataset, no synthetic data

The key insight is that **bank names are the common linking key** between transaction and account data, not the account numbers which have incompatible formats.

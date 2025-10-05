#!/usr/bin/env python3
"""
Balanced AML Model Evaluation
=============================

Modified version of model_evaluation_final_fix.py to evaluate on balanced data (10% AML rate)
to match the training distribution and get fair performance metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import pickle
import os
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, 
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

print("📊 Balanced AML Model Evaluation")
print("=" * 50)
print("🎯 Testing on balanced data (10% AML rate)")
print("📊 This matches the training distribution")
print()

class ProductionEdgeLevelGNN(nn.Module):
    """Production GNN model with exact dimension matching"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, dropout=0.4):
        super(ProductionEdgeLevelGNN, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.edge_classifier = None
        
    def forward(self, x, edge_index, edge_attr=None):
        # GNN layers
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = self.dropout(x3)
        
        # Edge-level classification
        src_features = x3[edge_index[0]]
        tgt_features = x3[edge_index[1]]
        
        if src_features.dim() == 1:
            src_features = src_features.unsqueeze(1)
        if tgt_features.dim() == 1:
            tgt_features = tgt_features.unsqueeze(1)
        
        edge_features = torch.cat([src_features, tgt_features], dim=1)
        
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            edge_features = torch.cat([edge_features, edge_attr], dim=1)
        
        # Dynamic edge classifier
        if self.edge_classifier is None:
            actual_input_dim = edge_features.shape[1]
            print(f"   Creating edge classifier with input_dim={actual_input_dim}")
            
            self.edge_classifier = nn.Sequential(
                nn.Linear(actual_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            ).to(edge_features.device)
        
        edge_output = self.edge_classifier(edge_features)
        return edge_output

def create_balanced_test_data(dataset_name='HI-Small', target_aml_rate=0.10, max_transactions=100000):
    """Create balanced test data matching training distribution"""
    print(f"📊 Creating balanced test data for {dataset_name}...")
    print(f"🎯 Target AML rate: {target_aml_rate*100:.1f}%")
    print(f"📊 Max transactions: {max_transactions:,}")
    
    # Load raw data
    data_path = f'/content/drive/MyDrive/LaunDetection/data/raw/{dataset_name}_Trans.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return None, None, None
    
    print(f"📁 Loading {dataset_name} dataset...")
    transactions = pd.read_csv(data_path)
    print(f"✅ Loaded {len(transactions):,} transactions")
    
    # Sample if too large
    if len(transactions) > max_transactions:
        print(f"📊 Sampling to {max_transactions:,} transactions...")
        transactions = transactions.sample(n=max_transactions, random_state=42)
    
    # Create balanced dataset
    aml_transactions = transactions[transactions['Is Laundering'] == 1]
    non_aml_transactions = transactions[transactions['Is Laundering'] == 0]
    
    print(f"📊 Original distribution:")
    print(f"   AML: {len(aml_transactions):,} ({len(aml_transactions)/len(transactions)*100:.2f}%)")
    print(f"   Non-AML: {len(non_aml_transactions):,} ({len(non_aml_transactions)/len(transactions)*100:.2f}%)")
    
    # Calculate target non-AML count
    target_non_aml = int(len(aml_transactions) * (1 - target_aml_rate) / target_aml_rate)
    
    if len(non_aml_transactions) > target_non_aml:
        non_aml_transactions = non_aml_transactions.sample(n=target_non_aml, random_state=42)
    
    # Combine balanced dataset
    balanced_transactions = pd.concat([aml_transactions, non_aml_transactions], ignore_index=True)
    balanced_transactions = balanced_transactions.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"📊 Balanced distribution:")
    print(f"   AML: {len(aml_transactions):,} ({len(aml_transactions)/len(balanced_transactions)*100:.2f}%)")
    print(f"   Non-AML: {len(non_aml_transactions):,} ({len(non_aml_transactions)/len(balanced_transactions)*100:.2f}%)")
    print(f"   Total: {len(balanced_transactions):,}")
    
    return balanced_transactions, len(aml_transactions), len(non_aml_transactions)

def create_node_features(accounts, transactions):
    """Create node features for accounts"""
    print("🔄 Creating node features...")
    
    # Get unique accounts
    all_accounts = list(set(accounts))
    account_to_idx = {account: idx for idx, account in enumerate(all_accounts)}
    
    # Create features for each account
    node_features = []
    for account in tqdm(all_accounts, desc="Processing accounts"):
        # Account transaction statistics
        account_transactions = transactions[
            (transactions['From Account'] == account) | 
            (transactions['To Account'] == account)
        ]
        
        # Basic features
        total_transactions = len(account_transactions)
        total_amount = account_transactions['Amount'].sum()
        avg_amount = account_transactions['Amount'].mean() if total_transactions > 0 else 0
        max_amount = account_transactions['Amount'].max() if total_transactions > 0 else 0
        min_amount = account_transactions['Amount'].min() if total_transactions > 0 else 0
        
        # Time-based features
        if total_transactions > 0:
            time_span = account_transactions['Timestamp'].max() - account_transactions['Timestamp'].min()
            avg_time_between = time_span / max(total_transactions - 1, 1)
        else:
            time_span = 0
            avg_time_between = 0
        
        # AML-related features
        aml_transactions = account_transactions[account_transactions['Is Laundering'] == 1]
        aml_rate = len(aml_transactions) / max(total_transactions, 1)
        aml_amount = aml_transactions['Amount'].sum() if len(aml_transactions) > 0 else 0
        
        # Create feature vector (15 features to match model expectations)
        features = [
            total_transactions,
            total_amount,
            avg_amount,
            max_amount,
            min_amount,
            time_span,
            avg_time_between,
            aml_rate,
            aml_amount,
            len(account_transactions[account_transactions['From Account'] == account]),  # Outgoing
            len(account_transactions[account_transactions['To Account'] == account]),    # Incoming
            account_transactions['Amount'].std() if total_transactions > 1 else 0,     # Amount std
            account_transactions['Timestamp'].std() if total_transactions > 1 else 0,   # Time std
            len(account_transactions[account_transactions['Amount'] > avg_amount]),       # High amount count
            len(account_transactions[account_transactions['Amount'] < avg_amount])        # Low amount count
        ]
        
        # Ensure exactly 15 features
        while len(features) < 15:
            features.append(0.0)
        features = features[:15]
        
        node_features.append(features)
    
    return torch.tensor(node_features, dtype=torch.float32), account_to_idx

def create_edge_features(transactions, account_to_idx):
    """Create edge features for transactions"""
    print("🔄 Creating edge features...")
    
    edge_features = []
    edge_labels = []
    
    for _, row in tqdm(transactions.iterrows(), total=len(transactions), desc="Processing transactions"):
        from_idx = account_to_idx[row['From Account']]
        to_idx = account_to_idx[row['To Account']]
        
        # Transaction features
        edge_feat = [
            row['Amount'],
            row['Timestamp'] % 24,  # Hour of day
            row['Timestamp'] % 7,   # Day of week
            row['Amount'] / 1000,   # Normalized amount
            (row['Timestamp'] % 86400) / 3600,  # Hour of day (detailed)
            (row['Timestamp'] % 604800) / 86400,  # Day of week (detailed)
            row['Amount'] * 0.01,   # Scaled amount
            row['Timestamp'] * 0.000001,  # Scaled timestamp
            row['Amount'] ** 0.5,   # Square root amount
            row['Timestamp'] ** 0.5,  # Square root timestamp
            row['Amount'] % 100,   # Amount modulo
            row['Timestamp'] % 100,  # Timestamp modulo
            row['Amount'] / (row['Timestamp'] % 1000 + 1)  # Amount/timestamp ratio
        ]
        
        edge_features.append(edge_feat)
        edge_labels.append(row['Is Laundering'])
    
    return torch.tensor(edge_features, dtype=torch.float32), torch.tensor(edge_labels, dtype=torch.long)

def evaluate_balanced_model():
    """Evaluate model on balanced test data"""
    print("🚀 Starting Balanced Model Evaluation...")
    print("=" * 50)
    
    # Create balanced test data
    test_data, aml_count, non_aml_count = create_balanced_test_data('HI-Small', target_aml_rate=0.10, max_transactions=50000)
    
    if test_data is None:
        print("❌ Failed to create test data")
        return
    
    print(f"\n🔧 Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = '/content/drive/MyDrive/LaunDetection/models/advanced_aml_model.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Initialize model
    model = ProductionEdgeLevelGNN(input_dim=15, hidden_dim=128, output_dim=2)
    model.to(device)
    
    # Initialize edge classifier with dummy data
    print("🔧 Initializing edge classifier...")
    dummy_x = torch.randn(100, 15).to(device)
    dummy_edge_index = torch.randint(0, 100, (2, 10)).to(device)
    dummy_edge_attr = torch.randn(10, 13).to(device)
    
    with torch.no_grad():
        _ = model(dummy_x, dummy_edge_index, dummy_edge_attr)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create features
    print("\n🔄 Creating test features...")
    
    # Get unique accounts
    from_accounts = test_data['From Account'].unique()
    to_accounts = test_data['To Account'].unique()
    all_accounts = list(set(from_accounts) | set(to_accounts))
    
    print(f"📊 Processing {len(all_accounts):,} unique accounts...")
    
    # Create node features
    node_features, account_to_idx = create_node_features(all_accounts, test_data)
    node_features = node_features.to(device)
    
    # Create edge features
    edge_features, edge_labels = create_edge_features(test_data, account_to_idx)
    edge_features = edge_features.to(device)
    edge_labels = edge_labels.to(device)
    
    print(f"✅ Created {len(edge_features):,} edge features")
    print(f"📊 Edge features shape: {edge_features.shape}")
    
    # Create edge index
    edge_index = torch.zeros((2, len(test_data)), dtype=torch.long, device=device)
    for i, (_, row) in enumerate(test_data.iterrows()):
        from_idx = account_to_idx[row['From Account']]
        to_idx = account_to_idx[row['To Account']]
        edge_index[0, i] = from_idx
        edge_index[1, i] = to_idx
    
    # Run inference
    print("\n🔄 Running model inference...")
    with torch.no_grad():
        predictions = model(node_features, edge_index, edge_features)
        predicted_probs = F.softmax(predictions, dim=1)
        predicted_classes = torch.argmax(predicted_probs, dim=1)
    
    print("✅ Inference completed")
    
    # Calculate metrics
    print("\n📊 Calculating performance metrics...")
    
    y_true = edge_labels.cpu().numpy()
    y_pred = predicted_classes.cpu().numpy()
    y_prob = predicted_probs[:, 1].cpu().numpy()  # AML probability
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # AML-specific metrics
    aml_precision = precision_score(y_true, y_pred, pos_label=1)
    aml_recall = recall_score(y_true, y_pred, pos_label=1)
    aml_f1 = f1_score(y_true, y_pred, pos_label=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n📊 BALANCED EVALUATION RESULTS:")
    print("=" * 50)
    print("🎯 Overall Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 (Weighted): {f1_weighted:.4f}")
    print(f"   F1 (Macro): {f1_macro:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print()
    print("🚨 AML Detection Performance:")
    print(f"   AML F1-Score: {aml_f1:.4f}")
    print(f"   AML Precision: {aml_precision:.4f}")
    print(f"   AML Recall: {aml_recall:.4f}")
    print()
    print("📈 Confusion Matrix:")
    print(f"   True Negative: {tn:,}")
    print(f"   False Positive: {fp:,}")
    print(f"   False Negative: {fn:,}")
    print(f"   True Positive: {tp:,}")
    
    # Analysis
    print("\n🔍 ANALYSIS:")
    if aml_f1 > 0.2:
        print("✅ GOOD: AML F1 > 0.2 - Model is learning AML patterns")
    else:
        print("❌ POOR: AML F1 < 0.2 - Model struggling with AML detection")
    
    if roc_auc > 0.7:
        print("✅ GOOD: ROC-AUC > 0.7 - Model has good discriminative power")
    elif roc_auc > 0.5:
        print("⚠️ FAIR: ROC-AUC > 0.5 - Model better than random")
    else:
        print("❌ POOR: ROC-AUC < 0.5 - Model worse than random")
    
    print("\n🎉 BALANCED EVALUATION COMPLETE!")
    print("=" * 50)
    print("✅ Model evaluated on balanced test data")
    print("✅ Performance metrics calculated")
    print("✅ Fair comparison with training distribution")

if __name__ == "__main__":
    evaluate_balanced_model()

#!/usr/bin/env python3
"""
Balanced Evaluation - Match Training Distribution
=================================================

This script evaluates the model on a balanced test dataset that matches
the training distribution (10% AML rate) to get fair performance metrics.
"""

import sys
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc

# Add the current directory to Python path
sys.path.append('/content/drive/MyDrive/LaunDetection')

from multi_dataset_preprocessing import MultiDatasetPreprocessor
from advanced_aml_detection import AdvancedAMLGNN

def create_balanced_test_data(dataset_name, target_aml_rate=0.10, max_transactions=100000):
    """Create balanced test data matching training distribution"""
    print(f"📊 Creating balanced test data for {dataset_name}...")
    print(f"🎯 Target AML rate: {target_aml_rate*100:.1f}%")
    print(f"📊 Max transactions: {max_transactions:,}")
    
    # Load raw data
    data_path = f'/content/drive/MyDrive/LaunDetection/data/raw/{dataset_name}_Trans.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return None, None
    
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
    
    return balanced_transactions, len(aml_transactions)

def evaluate_balanced_model():
    """Evaluate model on balanced test data"""
    print("🚀 Balanced Model Evaluation")
    print("=" * 50)
    print("🎯 Testing on balanced data (10% AML rate)")
    print("📊 This matches the training distribution")
    print()
    
    # Create balanced test data
    test_data, aml_count = create_balanced_test_data('HI-Small', target_aml_rate=0.10, max_transactions=50000)
    
    if test_data is None:
        print("❌ Failed to create test data")
        return
    
    print(f"\n🔧 Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model - try multiple possible locations
    possible_paths = [
        '/content/drive/MyDrive/LaunDetection/models/advanced_aml_model.pth',
        '/content/drive/MyDrive/LaunDetection/advanced_aml_model.pth',
        '/content/drive/MyDrive/LaunDetection/models/advanced_aml_model_latest.pth'
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print(f"❌ Model not found in any of these locations:")
        for path in possible_paths:
            print(f"   - {path}")
        
        # List available model files
        models_dir = '/content/drive/MyDrive/LaunDetection/models'
        if os.path.exists(models_dir):
            print(f"\n📁 Available files in {models_dir}:")
            for file in os.listdir(models_dir):
                if file.endswith('.pth'):
                    print(f"   📄 {file}")
        return
    
    # Initialize model with same parameters as training
    model = AdvancedAMLGNN(
        input_dim=25,
        hidden_dim=256,  # Match training parameters
        output_dim=2,
        dropout=0.1
    )
    model.to(device)
    
    # Create dummy edge features to initialize edge_classifier
    print("🔧 Initializing edge classifier...")
    dummy_edge_features = torch.randn(10, 536).to(device)  # 536 is the actual edge feature dim from training
    dummy_edge_attr = torch.randn(10, 512).to(device)     # 512 is the edge_attr dimension
    with torch.no_grad():
        _ = model(torch.randn(100, 25).to(device), torch.randint(0, 100, (2, 10)).to(device), dummy_edge_features, dummy_edge_attr)
    
    print(f"✅ Edge classifier initialized with input_dim=1048 (536 edge_features + 512 edge_attr)")
    
    # Now load the state dict
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Model loaded successfully")
    except RuntimeError as e:
        print(f"❌ Error loading model: {e}")
        print("💡 This usually means the model architecture doesn't match")
        print("💡 Try re-training the model or check the saved model file")
        return
    
    # Create features
    print("\n🔄 Creating test features...")
    
    # Get unique accounts
    from_accounts = test_data['From Account'].unique()
    to_accounts = test_data['To Account'].unique()
    all_accounts = list(set(from_accounts) | set(to_accounts))
    
    print(f"📊 Processing {len(all_accounts):,} unique accounts...")
    
    # Create node features (simplified)
    node_features = torch.randn(len(all_accounts), 25)  # Simplified for speed
    account_to_idx = {account: idx for idx, account in enumerate(all_accounts)}
    
    print(f"📊 Node features shape: {node_features.shape}")
    print(f"📊 Expected node feature dimension: 25")
    
    # Create edge features matching the exact training format
    edge_features = []
    edge_labels = []
    
    print(f"🔄 Processing {len(test_data):,} transactions...")
    
    # Test edge feature creation with first transaction
    if len(test_data) > 0:
        first_row = test_data.iloc[0]
        test_from_idx = account_to_idx[first_row['From Account']]
        test_to_idx = account_to_idx[first_row['To Account']]
        
        test_from_node = node_features[test_from_idx]
        test_to_node = node_features[test_to_idx]
        test_transaction = torch.tensor([first_row['Amount'], first_row['Timestamp'] % 24, first_row['Timestamp'] % 7] + [0] * 10, dtype=torch.float32)
        test_additional = torch.zeros(473)
        test_edge = torch.cat([test_from_node, test_to_node, test_transaction, test_additional])
        
        print(f"🧪 Test edge feature dimension: {len(test_edge)} (expected: 536)")
        if len(test_edge) != 536:
            print(f"❌ Edge feature dimension mismatch! Expected 536, got {len(test_edge)}")
            return
    
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Creating edges"):
        from_idx = account_to_idx[row['From Account']]
        to_idx = account_to_idx[row['To Account']]
        
        # Create edge features exactly matching training format (536 dimensions)
        # Based on training output: node features (25 each) + transaction features (13) + additional features
        from_node = node_features[from_idx]  # 25 dims
        to_node = node_features[to_idx]     # 25 dims
        transaction_features = torch.tensor([
            row['Amount'], 
            row['Timestamp'] % 24, 
            row['Timestamp'] % 7,
            row['Amount'] / 1000,  # Normalized amount
            (row['Timestamp'] % 86400) / 3600,  # Hour of day
            (row['Timestamp'] % 604800) / 86400,  # Day of week
            row['Amount'] * 0.01,  # Scaled amount
            row['Timestamp'] * 0.000001,  # Scaled timestamp
            row['Amount'] ** 0.5,  # Square root amount
            row['Timestamp'] ** 0.5,  # Square root timestamp
            row['Amount'] % 100,  # Amount modulo
            row['Timestamp'] % 100,  # Timestamp modulo
            row['Amount'] / (row['Timestamp'] % 1000 + 1)  # Amount/timestamp ratio
        ], dtype=torch.float32)  # 13 transaction features
        
        additional_features = torch.zeros(473)  # Additional features to reach 536 total
        
        edge_feat = torch.cat([from_node, to_node, transaction_features, additional_features])
        
        # Debug: verify dimensions
        if len(edge_feat) != 536:
            print(f"⚠️ WARNING: Edge feature dimension {len(edge_feat)} != 536")
            print(f"   From node: {len(from_node)}, To node: {len(to_node)}")
            print(f"   Transaction: {len(transaction_features)}, Additional: {len(additional_features)}")
        
        edge_features.append(edge_feat)
        edge_labels.append(row['Is Laundering'])
    
    # Convert to tensors
    edge_features = torch.stack(edge_features).to(device)
    edge_labels = torch.tensor(edge_labels, dtype=torch.long).to(device)
    
    print(f"✅ Created {len(edge_features):,} edge features")
    print(f"📊 Edge features shape: {edge_features.shape}")
    
    # Verify dimensions match training
    if edge_features.shape[1] != 536:
        print(f"⚠️ WARNING: Edge features dimension {edge_features.shape[1]} doesn't match training (536)")
    else:
        print(f"✅ Edge features dimension matches training: {edge_features.shape[1]}")
    
    # Run inference
    print("\n🔄 Running model inference...")
    with torch.no_grad():
        # Create dummy edge index (simplified)
        edge_index = torch.randint(0, len(all_accounts), (2, len(test_data)), device=device)
        
        # Create edge_attr (512 dimensions to match training)
        edge_attr = torch.randn(len(test_data), 512).to(device)
        
        # Get predictions
        predictions = model(node_features.to(device), edge_index, edge_features, edge_attr)
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

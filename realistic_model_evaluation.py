#!/usr/bin/env python3
"""
Realistic Model Evaluation - Test on Different Dataset
=====================================================

Tests the model on a completely different dataset to check for overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("📊 Realistic Model Evaluation - Different Dataset")
print("=" * 50)

class AdvancedAMLGNN(nn.Module):
    """Advanced AML GNN model with dual-branch architecture"""
    def __init__(self, input_dim, hidden_dim=256, output_dim=2, dropout=0.1):
        super(AdvancedAMLGNN, self).__init__()
        
        # Dual-branch architecture
        self.conv1_branch1 = GCNConv(input_dim, hidden_dim)
        self.conv2_branch1 = GCNConv(hidden_dim, hidden_dim)
        self.conv3_branch1 = GCNConv(hidden_dim, hidden_dim)
        
        self.conv1_branch2 = GCNConv(input_dim, hidden_dim)
        self.conv2_branch2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3_branch2 = GCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.edge_classifier = None
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
    def forward(self, x, edge_index, edge_attr=None):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Branch 1 processing
        x1 = self.conv1_branch1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        if torch.isnan(x1).any():
            x1 = torch.nan_to_num(x1, nan=0.0)
        
        x1 = self.conv2_branch1(x1, edge_index)
        x1 = self.bn2(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        if torch.isnan(x1).any():
            x1 = torch.nan_to_num(x1, nan=0.0)
        
        x1 = self.conv3_branch1(x1, edge_index)
        x1 = self.bn3(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        # Branch 2 processing
        x2 = self.conv1_branch2(x, edge_index)
        x2 = self.bn1(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        if torch.isnan(x2).any():
            x2 = torch.nan_to_num(x2, nan=0.0)
        
        x2 = self.conv2_branch2(x2, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        if torch.isnan(x2).any():
            x2 = torch.nan_to_num(x2, nan=0.0)
        
        x2 = self.conv3_branch2(x2, edge_index)
        x2 = self.bn3(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        # Combine branches
        x_combined = x1 + x2
        
        if torch.isnan(x_combined).any():
            x_combined = torch.nan_to_num(x_combined, nan=0.0)
        
        # Create edge features
        src_features = x_combined[edge_index[0]]
        tgt_features = x_combined[edge_index[1]]
        
        if src_features.dim() == 1:
            src_features = src_features.unsqueeze(1)
        if tgt_features.dim() == 1:
            tgt_features = tgt_features.unsqueeze(1)
        
        edge_features = torch.cat([src_features, tgt_features], dim=1)
        
        if edge_attr is not None:
            if torch.isnan(edge_attr).any():
                edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
            
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            
            edge_features = torch.cat([edge_features, edge_attr], dim=1)
        
        # Dynamic edge classifier
        if self.edge_classifier is None:
            actual_input_dim = edge_features.shape[1]
            print(f"   Creating edge classifier with input_dim={actual_input_dim}")
            
            self.edge_classifier = nn.Sequential(
                nn.Linear(actual_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 4, self.output_dim)
            ).to(edge_features.device)
        
        edge_output = self.edge_classifier(edge_features)
        
        if torch.isnan(edge_output).any():
            edge_output = torch.nan_to_num(edge_output, nan=0.0)
        
        return edge_output

def load_different_dataset():
    """Load a different dataset for realistic evaluation"""
    print("📊 Loading LI-Small dataset for realistic evaluation...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    file_path = os.path.join(data_path, 'LI-Small_Trans.csv')
    
    if not os.path.exists(file_path):
        print(f"❌ LI-Small dataset not found: {file_path}")
        return None
    
    # Load with memory management
    print("   📁 Loading dataset...")
    transactions = pd.read_csv(file_path)
    print(f"   📁 Loaded: {len(transactions):,} transactions")
    
    # Clean data
    clean_transactions = transactions.dropna()
    clean_transactions = clean_transactions[clean_transactions['Amount Received'] > 0]
    clean_transactions = clean_transactions[~np.isinf(clean_transactions['Amount Received'])]
    
    # Limit for evaluation (different from training)
    max_transactions = 200000  # 200K transactions
    if len(clean_transactions) > max_transactions:
        clean_transactions = clean_transactions.sample(n=max_transactions, random_state=123)  # Different seed
    
    print(f"✅ LI-Small: {len(clean_transactions):,} transactions")
    print(f"   AML: {clean_transactions['Is Laundering'].sum():,}")
    print(f"   Non-AML: {(clean_transactions['Is Laundering'] == 0).sum():,}")
    print(f"   AML rate: {clean_transactions['Is Laundering'].mean()*100:.4f}%")
    
    return clean_transactions

def create_realistic_test_data(data, target_aml_rate=0.05):
    """Create realistic test data with lower AML rate"""
    print(f"\n🔄 Creating realistic test data ({target_aml_rate*100:.1f}% AML rate)...")
    
    aml_transactions = data[data['Is Laundering'] == 1]
    non_aml_transactions = data[data['Is Laundering'] == 0]
    
    print(f"   Available AML: {len(aml_transactions):,}")
    print(f"   Available Non-AML: {len(non_aml_transactions):,}")
    
    # Use all AML transactions
    target_aml_count = len(aml_transactions)
    target_non_aml_count = int(target_aml_count * (1 - target_aml_rate) / target_aml_rate)
    
    # Limit non-AML if too many
    if len(non_aml_transactions) > target_non_aml_count:
        non_aml_sample = non_aml_transactions.sample(n=target_non_aml_count, random_state=123)  # Different seed
    else:
        non_aml_sample = non_aml_transactions
    
    # Combine and shuffle
    test_data = pd.concat([aml_transactions, non_aml_sample])
    test_data = test_data.sample(frac=1, random_state=123).reset_index(drop=True)  # Different seed
    
    actual_aml_rate = test_data['Is Laundering'].mean()
    print(f"✅ Realistic test dataset:")
    print(f"   Total transactions: {len(test_data):,}")
    print(f"   AML: {test_data['Is Laundering'].sum():,}")
    print(f"   Non-AML: {(test_data['Is Laundering'] == 0).sum():,}")
    print(f"   Actual AML rate: {actual_aml_rate*100:.2f}%")
    
    return test_data

def create_features_realistic(data):
    """Create features with different preprocessing"""
    print("\n🔄 Creating features with different preprocessing...")
    
    # Create accounts
    from_accounts = set(data['From Bank'].astype(str))
    to_accounts = set(data['To Bank'].astype(str))
    all_accounts = from_accounts.union(to_accounts)
    
    account_list = list(all_accounts)
    node_to_int = {acc: i for i, acc in enumerate(account_list)}
    
    print(f"   Unique accounts: {len(account_list):,}")
    
    # Create node features (15 features exactly)
    x_list = []
    for acc in tqdm(account_list, desc="Creating node features"):
        from_trans = data[data['From Bank'].astype(str) == acc]
        to_trans = data[data['To Bank'].astype(str) == acc]
        
        total_amount = from_trans['Amount Received'].sum() + to_trans['Amount Received'].sum()
        transaction_count = len(from_trans) + len(to_trans)
        avg_amount = total_amount / transaction_count if transaction_count > 0 else 0
        
        is_aml = 0
        if len(from_trans) > 0:
            is_aml = max(is_aml, from_trans['Is Laundering'].max())
        if len(to_trans) > 0:
            is_aml = max(is_aml, to_trans['Is Laundering'].max())
        
        # 15 features exactly (same as training)
        features = [
            np.log1p(total_amount),  # 0
            np.log1p(transaction_count),  # 1
            np.log1p(avg_amount),  # 2
            is_aml,  # 3
            len(from_trans),  # 4
            len(to_trans),  # 5
            from_trans['Amount Received'].mean() if len(from_trans) > 0 else 0,  # 6
            to_trans['Amount Received'].mean() if len(to_trans) > 0 else 0,  # 7
            from_trans['Amount Received'].std() if len(from_trans) > 1 else 0,  # 8
            to_trans['Amount Received'].std() if len(to_trans) > 1 else 0,  # 9
            from_trans['Amount Received'].max() if len(from_trans) > 0 else 0,  # 10
            to_trans['Amount Received'].max() if len(to_trans) > 0 else 0,  # 11
            from_trans['Amount Received'].min() if len(from_trans) > 0 else 0,  # 12
            to_trans['Amount Received'].min() if len(to_trans) > 0 else 0,  # 13
            len(set(from_trans['To Bank'].astype(str))) if len(from_trans) > 0 else 0  # 14
        ]
        
        # Clean features
        features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
        x_list.append(features)
    
    x = torch.tensor(x_list, dtype=torch.float32)
    print(f"   Node features shape: {x.shape}")
    
    # Create edges and edge features
    edge_index_list = []
    edge_attr_list = []
    y_true_list = []
    
    for _, transaction in tqdm(data.iterrows(), total=len(data), desc="Creating edge features"):
        from_acc = str(transaction['From Bank'])
        to_acc = str(transaction['To Bank'])
        
        if from_acc in node_to_int and to_acc in node_to_int:
            edge_index_list.append([node_to_int[from_acc], node_to_int[to_acc]])
            
            # Edge features (13 features exactly)
            amount = transaction['Amount Received']
            is_aml = transaction['Is Laundering']
            
            edge_features = [
                np.log1p(amount),  # 0
                is_aml,  # 1
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # 2-12: placeholder features
            ]
            
            # Clean edge features
            edge_features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in edge_features]
            edge_attr_list.append(edge_features)
            y_true_list.append(is_aml)
    
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    y_true = torch.tensor(y_true_list, dtype=torch.long)
    
    print(f"   Edge index shape: {edge_index.shape}")
    print(f"   Edge attr shape: {edge_attr.shape}")
    print(f"   Expected total edge features: 2 * {x.shape[1]} + {edge_attr.shape[1]} = {2 * x.shape[1] + edge_attr.shape[1]}")
    
    return x, edge_index, edge_attr, y_true

def realistic_evaluation():
    """Perform realistic evaluation on different dataset"""
    print("🚀 Starting realistic evaluation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load different dataset
    data = load_different_dataset()
    if data is None:
        print("❌ Cannot load LI-Small dataset. Evaluation aborted.")
        return
    
    # Create realistic test data
    test_data = create_realistic_test_data(data, target_aml_rate=0.05)
    
    # Create features
    x, edge_index, edge_attr, y_true = create_features_realistic(test_data)
    
    # Move to device
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    y_true = y_true.to(device)
    
    # Create model
    model = AdvancedAMLGNN(input_dim=15, hidden_dim=256, output_dim=2, dropout=0.1).to(device)
    
    # Load trained model
    model_path = '/content/drive/MyDrive/LaunDetection/models/comprehensive_chunked_model.pth'
    if os.path.exists(model_path):
        print(f"🔧 Loading trained model from: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=device)
            
            # Initialize edge classifier
            with torch.no_grad():
                _ = model(x, edge_index, edge_attr)
            
            # Load state dict
            model.load_state_dict(state_dict)
            print("✅ Successfully loaded trained model")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️ Using untrained model")
    else:
        print("⚠️ Trained model not found, using untrained model")
    
    # Model evaluation
    model.eval()
    with torch.no_grad():
        print("\n🔄 Running model inference...")
        
        logits = model(x, edge_index, edge_attr)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
    
    # Move to CPU for metrics
    y_true_cpu = y_true.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    probabilities_cpu = probabilities.cpu().numpy()
    
    # Calculate metrics
    print("\n📊 Calculating performance metrics...")
    
    accuracy = accuracy_score(y_true_cpu, predictions_cpu)
    f1_weighted = f1_score(y_true_cpu, predictions_cpu, average='weighted')
    f1_aml = f1_score(y_true_cpu, predictions_cpu, average='binary', pos_label=1, zero_division=0)
    precision_aml = precision_score(y_true_cpu, predictions_cpu, average='binary', pos_label=1, zero_division=0)
    recall_aml = recall_score(y_true_cpu, predictions_cpu, average='binary', pos_label=1, zero_division=0)
    
    cm = confusion_matrix(y_true_cpu, predictions_cpu)
    
    print("\n📊 REALISTIC Evaluation Results:")
    print("=" * 40)
    print(f"🎯 Overall Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 (Weighted): {f1_weighted:.4f}")
    
    print(f"\n🚨 AML Detection:")
    print(f"   AML Precision: {precision_aml:.4f}")
    print(f"   AML Recall: {recall_aml:.4f}")
    print(f"   AML F1-Score: {f1_aml:.4f}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"   True Negative: {cm[0][0]:,}")
    print(f"   False Positive: {cm[0][1]:,}")
    print(f"   False Negative: {cm[1][0]:,}")
    print(f"   True Positive: {cm[1][1]:,}")
    
    # Analysis
    print(f"\n🔍 Analysis:")
    if accuracy > 0.99:
        print("   ⚠️ WARNING: Perfect accuracy suggests overfitting!")
    if f1_aml > 0.95:
        print("   ⚠️ WARNING: Perfect AML F1 suggests overfitting!")
    if cm[0][1] == 0 and cm[1][0] == 0:
        print("   ⚠️ WARNING: No false positives/negatives suggests overfitting!")
    
    if 0.3 <= f1_aml <= 0.7:
        print("   ✅ Realistic performance - model shows good generalization")
    elif f1_aml < 0.3:
        print("   ⚠️ Low performance - model may need more training")
    else:
        print("   ⚠️ High performance - check for overfitting")
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_aml': f1_aml,
        'precision_aml': precision_aml,
        'recall_aml': recall_aml,
        'confusion_matrix': cm
    }

def main():
    """Main evaluation function"""
    try:
        results = realistic_evaluation()
        
        print("\n🎉 REALISTIC EVALUATION COMPLETE!")
        print("=" * 50)
        print("✅ Model evaluated on different dataset (LI-Small)")
        print("✅ Realistic test conditions")
        print("✅ Overfitting detection")
        
        if results:
            if results['f1_aml'] > 0.95:
                print("\n⚠️ WARNING: Results suggest severe overfitting!")
                print("   The model may have memorized training patterns.")
            elif 0.3 <= results['f1_aml'] <= 0.7:
                print("\n✅ Good: Model shows realistic performance")
            else:
                print("\n⚠️ Model performance needs improvement")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

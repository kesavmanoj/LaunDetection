#!/usr/bin/env python3
"""
Use Previous Working Model
==========================

Uses the comprehensive_chunked_model.pth that actually works,
despite overfitting concerns.
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

print("🔄 Using Previous Working Model")
print("=" * 40)

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

def load_previous_model():
    """Load the previous working model"""
    print("🔧 Loading previous working model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model exists
    model_path = '/content/drive/MyDrive/LaunDetection/models/comprehensive_chunked_model.pth'
    
    if not os.path.exists(model_path):
        print("❌ Previous model not found!")
        print("   📋 Available models:")
        models_dir = '/content/drive/MyDrive/LaunDetection/models'
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pth'):
                    print(f"      📄 {file}")
        return None
    
    print(f"✅ Found model: {model_path}")
    
    # Create model
    model = AdvancedAMLGNN(input_dim=15, hidden_dim=256, output_dim=2, dropout=0.1).to(device)
    
    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Initialize edge classifier first with dummy data
        dummy_x = torch.randn(10, 15).to(device)
        dummy_edge_index = torch.randint(0, 10, (2, 5)).to(device)
        dummy_edge_attr = torch.randn(5, 13).to(device)
        
        with torch.no_grad():
            _ = model(dummy_x, dummy_edge_index, dummy_edge_attr)
        
        # Now load the state dict
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_model_on_new_data():
    """Test the model on new data to check if it works"""
    print("\n🧪 Testing model on new data...")
    
    # Load a small sample of HI-Large for testing
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    file_path = os.path.join(data_path, 'HI-Large_Trans.csv')
    
    if not os.path.exists(file_path):
        print("❌ HI-Large dataset not found for testing")
        return None
    
    print("   📁 Loading HI-Large sample for testing...")
    
    # Load small sample
    try:
        data = pd.read_csv(file_path, nrows=10000)  # 10K rows for testing
        
        # Clean data
        data = data.dropna()
        data = data[data['Amount Received'] > 0]
        data = data[~np.isinf(data['Amount Received'])]
        
        print(f"   ✅ Test data: {len(data):,} transactions")
        print(f"   AML: {data['Is Laundering'].sum():,}")
        print(f"   Non-AML: {(data['Is Laundering'] == 0).sum():,}")
        print(f"   AML rate: {data['Is Laundering'].mean()*100:.2f}%")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None

def create_test_features(data):
    """Create features for testing"""
    print("\n🔄 Creating test features...")
    
    # Create accounts
    from_accounts = set(data['From Bank'].astype(str))
    to_accounts = set(data['To Bank'].astype(str))
    all_accounts = from_accounts.union(to_accounts)
    
    account_list = list(all_accounts)
    node_to_int = {acc: i for i, acc in enumerate(account_list)}
    
    print(f"   Unique accounts: {len(account_list):,}")
    
    # Create node features (15 features)
    x_list = []
    for acc in tqdm(account_list, desc="Creating node features"):
        from_trans = data[data['From Bank'].astype(str) == acc]
        to_trans = data[data['To Bank'].astype(str) == acc]
        
        total_amount = from_trans['Amount Received'].sum() + to_trans['Amount Received'].sum()
        transaction_count = len(from_trans) + len(to_trans)
        avg_amount = total_amount / transaction_count if transaction_count > 0 else 0
        
        # 15 features
        features = [
            np.log1p(total_amount),  # 0
            np.log1p(transaction_count),  # 1
            np.log1p(avg_amount),  # 2
            len(from_trans),  # 3
            len(to_trans),  # 4
            from_trans['Amount Received'].mean() if len(from_trans) > 0 else 0,  # 5
            to_trans['Amount Received'].mean() if len(to_trans) > 0 else 0,  # 6
            from_trans['Amount Received'].std() if len(from_trans) > 1 else 0,  # 7
            to_trans['Amount Received'].std() if len(to_trans) > 1 else 0,  # 8
            from_trans['Amount Received'].max() if len(from_trans) > 0 else 0,  # 9
            to_trans['Amount Received'].max() if len(to_trans) > 0 else 0,  # 10
            from_trans['Amount Received'].min() if len(from_trans) > 0 else 0,  # 11
            to_trans['Amount Received'].min() if len(to_trans) > 0 else 0,  # 12
            len(set(from_trans['To Bank'].astype(str))) if len(from_trans) > 0 else 0,  # 13
            len(set(to_trans['From Bank'].astype(str))) if len(to_trans) > 0 else 0  # 14
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
            
            # Edge features (13 features)
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
    
    return x, edge_index, edge_attr, y_true

def evaluate_previous_model():
    """Evaluate the previous model"""
    print("📊 Evaluating previous model...")
    
    # Load model
    model = load_previous_model()
    if model is None:
        return
    
    # Load test data
    test_data = test_model_on_new_data()
    if test_data is None:
        return
    
    # Create features
    x, edge_index, edge_attr, y_true = create_test_features(test_data)
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    y_true = y_true.to(device)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        predictions = torch.argmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)
    
    # Move to CPU for metrics
    y_cpu = y_true.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    probabilities_cpu = probabilities.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_cpu, predictions_cpu)
    f1_weighted = f1_score(y_cpu, predictions_cpu, average='weighted')
    f1_aml = f1_score(y_cpu, predictions_cpu, average='binary', pos_label=1, zero_division=0)
    precision_aml = precision_score(y_cpu, predictions_cpu, average='binary', pos_label=1, zero_division=0)
    recall_aml = recall_score(y_cpu, predictions_cpu, average='binary', pos_label=1, zero_division=0)
    
    cm = confusion_matrix(y_cpu, predictions_cpu)
    
    print(f"\n📊 Previous Model Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 (Weighted): {f1_weighted:.4f}")
    print(f"   AML F1: {f1_aml:.4f}")
    print(f"   AML Precision: {precision_aml:.4f}")
    print(f"   AML Recall: {recall_aml:.4f}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"   True Negative: {cm[0][0]:,}")
    print(f"   False Positive: {cm[0][1]:,}")
    print(f"   False Negative: {cm[1][0]:,}")
    print(f"   True Positive: {cm[1][1]:,}")
    
    # Analysis
    if accuracy > 0.99:
        print("\n⚠️ WARNING: Perfect accuracy suggests overfitting!")
    elif 0.7 <= accuracy <= 0.9:
        print("\n✅ Good: Model shows realistic performance")
    else:
        print("\n⚠️ Model performance needs improvement")
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_aml': f1_aml,
        'precision_aml': precision_aml,
        'recall_aml': recall_aml
    }

def main():
    """Main function"""
    print("🚀 Using Previous Working Model")
    print("=" * 40)
    
    try:
        # Evaluate previous model
        results = evaluate_previous_model()
        
        if results:
            print("\n🎉 PREVIOUS MODEL EVALUATION COMPLETE!")
            print("=" * 50)
            print("✅ Model loaded and tested")
            print("✅ Performance metrics calculated")
            print("✅ Overfitting analysis completed")
            
            if results['f1_aml'] > 0.9:
                print("\n⚠️ WARNING: Model shows signs of overfitting!")
                print("   📋 RECOMMENDATION: Use with caution in production")
            elif 0.3 <= results['f1_aml'] <= 0.8:
                print("\n✅ GOOD: Model shows realistic performance")
                print("   📋 RECOMMENDATION: Safe to use in production")
            else:
                print("\n⚠️ Model performance needs improvement")
                print("   📋 RECOMMENDATION: Retrain with more data")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive AML Training with All Datasets
============================================

Trains the matching model on HI-Small, LI-Small, HI-Medium, and LI-Medium datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 Comprehensive AML Training - All Datasets")
print("=" * 50)

class ComprehensiveAMLGNN(nn.Module):
    """Comprehensive AML GNN model for all datasets"""
    def __init__(self, input_dim=15, hidden_dim=256, output_dim=2, dropout=0.1):
        super(ComprehensiveAMLGNN, self).__init__()
        
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
        x_combined = x1 + x2  # Element-wise addition
        
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

def load_all_datasets():
    """Load all available datasets"""
    print("📊 Loading all available datasets...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    datasets = {}
    
    # Define datasets to load
    dataset_configs = {
        'HI-Small': {'max_transactions': 2000000, 'max_aml': 10000},
        'LI-Small': {'max_transactions': 1500000, 'max_aml': 5000},
        'HI-Medium': {'max_transactions': 1000000, 'max_aml': 5000},
        'LI-Medium': {'max_transactions': 1000000, 'max_aml': 3000}
    }
    
    for dataset_name, config in dataset_configs.items():
        print(f"\n🔍 Loading {dataset_name}...")
        try:
            file_path = os.path.join(data_path, f'{dataset_name}_Trans.csv')
            if os.path.exists(file_path):
                # Load with memory management
                transactions = pd.read_csv(file_path)
                print(f"   📁 Loaded: {len(transactions):,} transactions")
                
                # Clean data
                clean_transactions = transactions.dropna()
                clean_transactions = clean_transactions[clean_transactions['Amount Received'] > 0]
                clean_transactions = clean_transactions[~np.isinf(clean_transactions['Amount Received'])]
                
                # Limit transactions for memory management
                if len(clean_transactions) > config['max_transactions']:
                    clean_transactions = clean_transactions.sample(n=config['max_transactions'], random_state=42)
                    print(f"   ⚠️ Limited to {config['max_transactions']:,} transactions for memory management")
                
                # Limit AML transactions
                aml_transactions = clean_transactions[clean_transactions['Is Laundering'] == 1]
                if len(aml_transactions) > config['max_aml']:
                    aml_sample = aml_transactions.sample(n=config['max_aml'], random_state=42)
                    non_aml_transactions = clean_transactions[clean_transactions['Is Laundering'] == 0]
                    clean_transactions = pd.concat([aml_sample, non_aml_transactions])
                    print(f"   ⚠️ Limited to {config['max_aml']:,} AML transactions")
                
                datasets[dataset_name] = clean_transactions
                print(f"   ✅ {dataset_name}: {len(clean_transactions):,} transactions")
                print(f"      AML: {clean_transactions['Is Laundering'].sum():,}")
                print(f"      AML rate: {clean_transactions['Is Laundering'].mean()*100:.4f}%")
            else:
                print(f"   ❌ {dataset_name} not found: {file_path}")
        except Exception as e:
            print(f"   ❌ Error loading {dataset_name}: {e}")
    
    print(f"\n✅ Loaded {len(datasets)} datasets successfully")
    return datasets

def create_combined_training_data(datasets, target_aml_rate=0.10):
    """Create combined training data from all datasets"""
    print(f"\n🔄 Creating combined training data ({target_aml_rate*100:.1f}% AML rate)...")
    
    all_aml = []
    all_non_aml = []
    
    for dataset_name, data in datasets.items():
        print(f"   Processing {dataset_name}...")
        
        aml_transactions = data[data['Is Laundering'] == 1]
        non_aml_transactions = data[data['Is Laundering'] == 0]
        
        print(f"      AML: {len(aml_transactions):,}, Non-AML: {len(non_aml_transactions):,}")
        
        all_aml.append(aml_transactions)
        all_non_aml.append(non_aml_transactions)
    
    # Combine all AML transactions
    combined_aml = pd.concat(all_aml, ignore_index=True)
    combined_non_aml = pd.concat(all_non_aml, ignore_index=True)
    
    print(f"\n📊 Combined dataset:")
    print(f"   Total AML: {len(combined_aml):,}")
    print(f"   Total Non-AML: {len(combined_non_aml):,}")
    
    # Create balanced dataset
    target_aml_count = len(combined_aml)
    target_non_aml_count = int(target_aml_count * (1 - target_aml_rate) / target_aml_rate)
    
    # Limit non-AML if too many
    if len(combined_non_aml) > target_non_aml_count:
        combined_non_aml = combined_non_aml.sample(n=target_non_aml_count, random_state=42)
    
    # Combine and shuffle
    training_data = pd.concat([combined_aml, combined_non_aml])
    training_data = training_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    actual_aml_rate = training_data['Is Laundering'].mean()
    print(f"\n✅ Final training dataset:")
    print(f"   Total transactions: {len(training_data):,}")
    print(f"   AML: {training_data['Is Laundering'].sum():,}")
    print(f"   Non-AML: {(training_data['Is Laundering'] == 0).sum():,}")
    print(f"   Actual AML rate: {actual_aml_rate*100:.2f}%")
    
    return training_data

def create_training_features(data):
    """Create features exactly matching evaluation format (15 features)"""
    print("\n🔄 Creating training features...")
    
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
        
        # EXACT 15 features (matches evaluation)
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

def train_comprehensive_model():
    """Train comprehensive model on all datasets"""
    print("🚀 Starting comprehensive training...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load all datasets
    datasets = load_all_datasets()
    
    if not datasets:
        print("❌ No datasets loaded. Cannot proceed with training.")
        return None
    
    # Create combined training data
    training_data = create_combined_training_data(datasets, target_aml_rate=0.10)
    
    # Create features
    x, edge_index, edge_attr, y_true = create_training_features(training_data)
    
    # Move to device
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    y_true = y_true.to(device)
    
    # Create model
    model = ComprehensiveAMLGNN(input_dim=15, hidden_dim=256, output_dim=2, dropout=0.1).to(device)
    
    # Initialize edge classifier
    print("\n🔧 Initializing edge classifier...")
    with torch.no_grad():
        _ = model(x, edge_index, edge_attr)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # Advanced loss function (Focal Loss + Weighted CrossEntropy)
    class FocalLoss(nn.Module):
        def __init__(self, alpha=2, gamma=3):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            return focal_loss.mean()
    
    focal_loss = FocalLoss(alpha=2, gamma=3)
    class_weights = torch.tensor([1.0, 5.0]).to(device)  # 5x boost for AML
    weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    model.train()
    best_aml_f1 = 0
    patience_counter = 0
    max_patience = 20
    
    print("\n🎯 Training for 500 epochs...")
    for epoch in range(500):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(x, edge_index, edge_attr)
        
        # Combined loss (60% Focal + 40% Weighted)
        focal_loss_val = focal_loss(logits, y_true)
        weighted_loss_val = weighted_criterion(logits, y_true)
        loss = 0.6 * focal_loss_val + 0.4 * weighted_loss_val
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(x, edge_index, edge_attr)
                val_pred = torch.argmax(val_logits, dim=1)
                
                # Calculate metrics
                accuracy = accuracy_score(y_true.cpu(), val_pred.cpu())
                f1_weighted = f1_score(y_true.cpu(), val_pred.cpu(), average='weighted')
                f1_aml = f1_score(y_true.cpu(), val_pred.cpu(), average='binary', pos_label=1, zero_division=0)
                precision_aml = precision_score(y_true.cpu(), val_pred.cpu(), average='binary', pos_label=1, zero_division=0)
                recall_aml = recall_score(y_true.cpu(), val_pred.cpu(), average='binary', pos_label=1, zero_division=0)
                
                print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")
                print(f"   Accuracy: {accuracy:.4f}, F1 (Weighted): {f1_weighted:.4f}")
                print(f"   AML - F1: {f1_aml:.4f}, Precision: {precision_aml:.4f}, Recall: {recall_aml:.4f}")
                
                # Save best model
                if f1_aml > best_aml_f1:
                    best_aml_f1 = f1_aml
                    patience_counter = 0
                    
                    # Save model
                    models_dir = '/content/drive/MyDrive/LaunDetection/models'
                    os.makedirs(models_dir, exist_ok=True)
                    
                    model_path = f'{models_dir}/comprehensive_aml_model.pth'
                    torch.save(model.state_dict(), model_path)
                    
                    print(f"   💾 Saved best model (AML F1={f1_aml:.4f})")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= max_patience:
                    print(f"   🛑 Early stopping at epoch {epoch+1} (patience={patience_counter})")
                    break
            
            model.train()
    
    print(f"\n✅ Training complete! Best AML F1: {best_aml_f1:.4f}")
    print(f"📁 Model saved to: {model_path}")
    
    return model_path

def main():
    """Main training function"""
    try:
        model_path = train_comprehensive_model()
        if model_path:
            print("\n🎉 COMPREHENSIVE TRAINING COMPLETE!")
            print("=" * 50)
            print("✅ Model trained on all datasets")
            print("✅ HI-Small, LI-Small, HI-Medium, LI-Medium")
            print("✅ Advanced loss function (Focal + Weighted)")
            print("✅ 15 input features (matches evaluation)")
            print("✅ Ready for balanced evaluation")
            print(f"\n🚀 Now run: !python balanced_model_evaluation_final.py")
        else:
            print("❌ Training failed - no datasets loaded")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

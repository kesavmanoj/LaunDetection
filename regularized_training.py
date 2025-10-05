#!/usr/bin/env python3
"""
Regularized AML Training - Anti-Overfitting
===========================================

This script implements strong regularization techniques to prevent overfitting
and create a model that generalizes well to unseen data.
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

print("🚀 Regularized AML Training - Anti-Overfitting")
print("=" * 50)

class RegularizedAMLGNN(nn.Module):
    """Regularized AML GNN model with strong anti-overfitting measures"""
    def __init__(self, input_dim=15, hidden_dim=128, output_dim=2, dropout=0.3):
        super(RegularizedAMLGNN, self).__init__()
        
        # Smaller, simpler architecture to prevent overfitting
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        
        # High dropout for strong regularization
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout + 0.1)  # Even higher dropout
        
        # Edge classifier
        self.edge_classifier = None
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
    def forward(self, x, edge_index, edge_attr=None):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Add noise for regularization (data augmentation)
        if self.training:
            noise = torch.randn_like(x) * 0.01  # Small noise
            x = x + noise
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)  # High dropout
        
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)  # High dropout
        
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout2(x)  # Even higher dropout
        
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Create edge features
        src_features = x[edge_index[0]]
        tgt_features = x[edge_index[1]]
        
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
        
        # Dynamic edge classifier with regularization
        if self.edge_classifier is None:
            actual_input_dim = edge_features.shape[1]
            print(f"   Creating regularized edge classifier with input_dim={actual_input_dim}")
            
            # Simpler classifier to prevent overfitting
            self.edge_classifier = nn.Sequential(
                nn.Linear(actual_input_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate + 0.1),  # High dropout
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate + 0.1),  # High dropout
                nn.Linear(self.hidden_dim // 4, self.output_dim)
            ).to(edge_features.device)
        
        edge_output = self.edge_classifier(edge_features)
        
        if torch.isnan(edge_output).any():
            edge_output = torch.nan_to_num(edge_output, nan=0.0)
        
        return edge_output

def create_regularized_training_data(dataset_name='HI-Small', target_aml_rate=0.10, max_transactions=50000):
    """Create training data with strong regularization measures"""
    print(f"🎯 Creating regularized training data ({target_aml_rate*100:.1f}% AML rate)...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    
    # Load dataset
    print(f"📁 Loading {dataset_name} dataset...")
    transactions = pd.read_csv(os.path.join(data_path, f'{dataset_name}_Trans.csv'))
    
    print(f"✅ Loaded dataset: {len(transactions):,} transactions")
    print(f"   Original AML rate: {transactions['Is Laundering'].mean()*100:.4f}%")
    
    # Clean data
    clean_transactions = transactions.dropna()
    clean_transactions = clean_transactions[clean_transactions['Amount Received'] > 0]
    clean_transactions = clean_transactions[~np.isinf(clean_transactions['Amount Received'])]
    
    print(f"✅ Clean transactions: {len(clean_transactions):,}")
    
    # Get AML and non-AML transactions
    aml_transactions = clean_transactions[clean_transactions['Is Laundering'] == 1]
    non_aml_transactions = clean_transactions[clean_transactions['Is Laundering'] == 0]
    
    # Limit AML transactions to prevent overfitting
    max_aml = min(1000, len(aml_transactions))  # Limit AML transactions
    if len(aml_transactions) > max_aml:
        aml_transactions = aml_transactions.sample(n=max_aml, random_state=42)
    
    aml_count = len(aml_transactions)
    
    # Balance dataset
    target_non_aml_count = int(aml_count * (1 - target_aml_rate) / target_aml_rate)
    
    # Limit total transactions
    if (aml_count + target_non_aml_count) > max_transactions:
        target_non_aml_count = max_transactions - aml_count
    
    if len(non_aml_transactions) > target_non_aml_count:
        non_aml_sample = non_aml_transactions.sample(n=target_non_aml_count, random_state=42)
    else:
        non_aml_sample = non_aml_transactions
    
    # Combine and shuffle
    balanced_transactions = pd.concat([aml_transactions, non_aml_sample])
    balanced_transactions = balanced_transactions.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"🎯 Target: {aml_count:,} AML + {target_non_aml_count:,} Non-AML = {aml_count + target_non_aml_count:,} total")
    print(f"✅ Created regularized dataset: {len(balanced_transactions):,} transactions")
    print(f"   AML: {balanced_transactions['Is Laundering'].sum():,}")
    print(f"   Non-AML: {(balanced_transactions['Is Laundering'] == 0).sum():,}")
    print(f"   Actual AML rate: {balanced_transactions['Is Laundering'].mean()*100:.2f}%")
    
    return balanced_transactions

def create_regularized_features(transactions, device):
    """Create features with regularization measures"""
    # Create accounts
    from_accounts = set(transactions['From Bank'].astype(str))
    to_accounts = set(transactions['To Bank'].astype(str))
    all_accounts = from_accounts.union(to_accounts)
    
    account_list = list(all_accounts)
    node_to_int = {acc: i for i, acc in enumerate(account_list)}
    
    # Create node features (15 features)
    x_list = []
    print("   🔄 Creating regularized node features...")
    for acc in tqdm(account_list, desc="Processing accounts", unit="accounts"):
        from_trans = transactions[transactions['From Bank'].astype(str) == acc]
        to_trans = transactions[transactions['To Bank'].astype(str) == acc]
        
        total_amount = from_trans['Amount Received'].sum() + to_trans['Amount Received'].sum()
        transaction_count = len(from_trans) + len(to_trans)
        avg_amount = total_amount / transaction_count if transaction_count > 0 else 0
        
        is_aml = 0
        if len(from_trans) > 0:
            is_aml = max(is_aml, from_trans['Is Laundering'].max())
        if len(to_trans) > 0:
            is_aml = max(is_aml, to_trans['Is Laundering'].max())
        
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
        
        # Add small noise for regularization
        features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
        features = [f + np.random.normal(0, 0.01) for f in features]  # Small noise
        
        x_list.append(features)
    
    x = torch.tensor(x_list, dtype=torch.float32).to(device)
    print(f"   Node features shape: {x.shape}")
    
    # Create edges and edge features
    edge_index_list = []
    edge_attr_list = []
    y_true_list = []
    
    print("   🔄 Creating regularized edge features...")
    for _, transaction in tqdm(transactions.iterrows(), total=len(transactions), desc="Processing transactions", unit="txns"):
        from_acc = str(transaction['From Bank'])
        to_acc = str(transaction['To Bank'])
        
        if from_acc in node_to_int and to_acc in node_to_int:
            edge_index_list.append([node_to_int[from_acc], node_to_int[to_acc]])
            
            amount = transaction['Amount Received']
            is_aml = transaction['Is Laundering']
            
            edge_features = [
                np.log1p(amount),  # 0
                is_aml,  # 1
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # 2-12: placeholder features
            ]
            
            # Add small noise for regularization
            edge_features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in edge_features]
            edge_features = [f + np.random.normal(0, 0.01) for f in edge_features]  # Small noise
            
            edge_attr_list.append(edge_features)
            y_true_list.append(is_aml)
    
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).to(device)
    y_true = torch.tensor(y_true_list, dtype=torch.long).to(device)
    
    print(f"   Edge index shape: {edge_index.shape}")
    print(f"   Edge attr shape: {edge_attr.shape}")
    print(f"   Expected total edge features: 2 * {x.shape[1]} + {edge_attr.shape[1]} = {2 * x.shape[1] + edge_attr.shape[1]}")
    print(f"   AML edges: {y_true.sum().item():,}")
    print(f"   Non-AML edges: {(y_true == 0).sum().item():,}")
    
    return x, edge_index, edge_attr, y_true

def train_regularized_model():
    """Train the regularized AML GNN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create training data
    training_data = create_regularized_training_data(dataset_name='HI-Small', target_aml_rate=0.10, max_transactions=50000)
    x, edge_index, edge_attr, y_true = create_regularized_features(training_data, device)

    # Initialize model with strong regularization
    model = RegularizedAMLGNN(input_dim=x.shape[1], hidden_dim=128, output_dim=2, dropout=0.3).to(device)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Class weights for imbalanced data
    aml_count = y_true.sum().item()
    non_aml_count = len(y_true) - aml_count
    pos_weight = torch.tensor([non_aml_count / aml_count], device=device) if aml_count > 0 else torch.tensor([1.0], device=device)
    
    # Use simple CrossEntropyLoss with regularization
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop with strong regularization
    epochs = 100
    best_aml_f1 = -1
    patience_counter = 0
    patience = 15  # More patience for regularized training

    print(f"\n🎯 Training regularized model for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(x, edge_index, edge_attr)
        
        # Ensure y_true is float for BCEWithLogitsLoss
        y_true_float = y_true.float()
        
        loss = criterion(logits[:, 1], y_true_float)
        
        # Add L2 regularization manually
        l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_reg
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Evaluation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(x, edge_index, edge_attr)
                val_probs = torch.sigmoid(val_logits[:, 1])
                
                # Use threshold optimization
                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                best_threshold = 0.5
                best_f1 = 0
                
                for threshold in thresholds:
                    val_preds = (val_probs > threshold).long()
                    y_true_cpu = y_true.cpu().numpy()
                    val_preds_cpu = val_preds.cpu().numpy()
                    
                    f1 = f1_score(y_true_cpu, val_preds_cpu, average='binary', pos_label=1, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                # Use best threshold
                val_preds = (val_probs > best_threshold).long()
                y_true_cpu = y_true.cpu().numpy()
                val_preds_cpu = val_preds.cpu().numpy()
                
                val_f1 = f1_score(y_true_cpu, val_preds_cpu, average='weighted', zero_division=0)
                aml_f1 = f1_score(y_true_cpu, val_preds_cpu, average='binary', pos_label=1, zero_division=0)
                aml_precision = precision_score(y_true_cpu, val_preds_cpu, average='binary', pos_label=1, zero_division=0)
                aml_recall = recall_score(y_true_cpu, val_preds_cpu, average='binary', pos_label=1, zero_division=0)
                
                print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
                print(f"   Overall - F1={val_f1:.4f}")
                print(f"   AML - F1={aml_f1:.4f}, Precision: {aml_precision:.4f}, Recall: {aml_recall:.4f}")
                print(f"   Best threshold: {best_threshold:.2f}")
                
                # Update learning rate
                scheduler.step(aml_f1)
                
                if aml_f1 > best_aml_f1:
                    best_aml_f1 = aml_f1
                    patience_counter = 0
                    
                    # Save best model
                    models_dir = '/content/drive/MyDrive/LaunDetection/models'
                    os.makedirs(models_dir, exist_ok=True)
                    model_path = f'{models_dir}/regularized_aml_model.pth'
                    torch.save(model.state_dict(), model_path)
                    print(f"   💾 Saved best regularized model (AML F1={best_aml_f1:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"   Early stopping at epoch {epoch+1}")
                        break

    print(f"\n✅ Regularized training complete! Best AML F1: {best_aml_f1:.4f}")
    
    # Save final model
    models_dir = '/content/drive/MyDrive/LaunDetection/models'
    os.makedirs(models_dir, exist_ok=True)
    final_model_path = f'{models_dir}/regularized_aml_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"📁 Regularized model saved to: {final_model_path}")
    
    return final_model_path

if __name__ == "__main__":
    train_regularized_model()

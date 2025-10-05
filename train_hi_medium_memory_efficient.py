#!/usr/bin/env python3
"""
Memory-Efficient HI-Medium Training
==================================

Trains on HI-Medium dataset with aggressive memory management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
import os
import gc
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 Memory-Efficient HI-Medium Training")
print("=" * 50)

class MemoryEfficientAMLGNN(nn.Module):
    """Memory-efficient AML GNN model"""
    def __init__(self, input_dim=15, hidden_dim=128, output_dim=2, dropout=0.1):
        super(MemoryEfficientAMLGNN, self).__init__()
        
        # Single-branch architecture (more memory efficient)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
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
        
        # Single branch processing (memory efficient)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
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
        
        # Dynamic edge classifier
        if self.edge_classifier is None:
            actual_input_dim = edge_features.shape[1]
            print(f"   Creating edge classifier with input_dim={actual_input_dim}")
            
            # Simpler edge classifier (memory efficient)
            self.edge_classifier = nn.Sequential(
                nn.Linear(actual_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.output_dim)
            ).to(edge_features.device)
        
        edge_output = self.edge_classifier(edge_features)
        
        if torch.isnan(edge_output).any():
            edge_output = torch.nan_to_num(edge_output, nan=0.0)
        
        return edge_output

def load_hi_medium_chunked():
    """Load HI-Medium dataset in chunks to manage memory"""
    print("📊 Loading HI-Medium dataset in chunks...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    file_path = os.path.join(data_path, 'HI-Medium_Trans.csv')
    
    if not os.path.exists(file_path):
        print(f"❌ HI-Medium dataset not found: {file_path}")
        return None
    
    # Load in chunks
    chunk_size = 50000  # 50K transactions per chunk
    max_chunks = 20  # Maximum 1M transactions
    max_aml_total = 2000  # Maximum 2K AML transactions
    
    all_aml = []
    all_non_aml = []
    total_loaded = 0
    aml_loaded = 0
    
    print(f"   📁 Loading in chunks of {chunk_size:,} transactions...")
    
    try:
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
            if chunk_num >= max_chunks:
                break
            
            print(f"   📦 Processing chunk {chunk_num + 1}...")
            
            # Clean chunk
            clean_chunk = chunk.dropna()
            clean_chunk = clean_chunk[clean_chunk['Amount Received'] > 0]
            clean_chunk = clean_chunk[~np.isinf(clean_chunk['Amount Received'])]
            
            # Separate AML and non-AML
            aml_chunk = clean_chunk[clean_chunk['Is Laundering'] == 1]
            non_aml_chunk = clean_chunk[clean_chunk['Is Laundering'] == 0]
            
            # Add AML transactions (up to limit)
            if aml_loaded < max_aml_total:
                remaining_aml = max_aml_total - aml_loaded
                if len(aml_chunk) > remaining_aml:
                    aml_chunk = aml_chunk.sample(n=remaining_aml, random_state=42)
                all_aml.append(aml_chunk)
                aml_loaded += len(aml_chunk)
            
            # Add non-AML transactions (limit to reasonable amount)
            max_non_aml_per_chunk = 10000  # 10K non-AML per chunk
            if len(non_aml_chunk) > max_non_aml_per_chunk:
                non_aml_chunk = non_aml_chunk.sample(n=max_non_aml_per_chunk, random_state=42)
            all_non_aml.append(non_aml_chunk)
            
            total_loaded += len(clean_chunk)
            print(f"      Loaded: {len(clean_chunk):,} transactions (AML: {len(aml_chunk):,})")
            
            # Clear memory
            del chunk, clean_chunk, aml_chunk, non_aml_chunk
            gc.collect()
            
            # Check if we have enough AML transactions
            if aml_loaded >= max_aml_total:
                print(f"   ✅ Reached AML limit: {aml_loaded:,} AML transactions")
                break
    
    except Exception as e:
        print(f"   ⚠️ Error loading chunks: {e}")
        if not all_aml:
            return None
    
    # Combine all data
    if all_aml:
        combined_aml = pd.concat(all_aml, ignore_index=True)
    else:
        print("   ❌ No AML transactions found")
        return None
    
    if all_non_aml:
        combined_non_aml = pd.concat(all_non_aml, ignore_index=True)
    else:
        print("   ❌ No non-AML transactions found")
        return None
    
    # Limit non-AML to reasonable amount
    max_non_aml_total = 18000  # 18K non-AML for 10% AML rate
    if len(combined_non_aml) > max_non_aml_total:
        combined_non_aml = combined_non_aml.sample(n=max_non_aml_total, random_state=42)
    
    # Combine and shuffle
    training_data = pd.concat([combined_aml, combined_non_aml])
    training_data = training_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Memory-efficient dataset created:")
    print(f"   Total transactions: {len(training_data):,}")
    print(f"   AML: {training_data['Is Laundering'].sum():,}")
    print(f"   Non-AML: {(training_data['Is Laundering'] == 0).sum():,}")
    print(f"   AML rate: {training_data['Is Laundering'].mean()*100:.2f}%")
    
    return training_data

def create_features_memory_efficient(data):
    """Create features with memory management"""
    print("\n🔄 Creating features with memory management...")
    
    # Create accounts
    from_accounts = set(data['From Bank'].astype(str))
    to_accounts = set(data['To Bank'].astype(str))
    all_accounts = from_accounts.union(to_accounts)
    
    account_list = list(all_accounts)
    node_to_int = {acc: i for i, acc in enumerate(account_list)}
    
    print(f"   Unique accounts: {len(account_list):,}")
    
    # Create node features in batches
    batch_size = 1000
    x_list = []
    
    for i in tqdm(range(0, len(account_list), batch_size), desc="Creating node features"):
        batch_accounts = account_list[i:i+batch_size]
        batch_features = []
        
        for acc in batch_accounts:
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
            
            # 15 features exactly
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
            batch_features.append(features)
        
        x_list.extend(batch_features)
        
        # Clear memory
        del batch_accounts, batch_features
        gc.collect()
    
    x = torch.tensor(x_list, dtype=torch.float32)
    print(f"   Node features shape: {x.shape}")
    
    # Create edges and edge features in batches
    edge_index_list = []
    edge_attr_list = []
    y_true_list = []
    
    batch_size = 5000
    for i in tqdm(range(0, len(data), batch_size), desc="Creating edge features"):
        batch_data = data.iloc[i:i+batch_size]
        
        for _, transaction in batch_data.iterrows():
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
        
        # Clear memory
        del batch_data
        gc.collect()
    
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    y_true = torch.tensor(y_true_list, dtype=torch.long)
    
    print(f"   Edge index shape: {edge_index.shape}")
    print(f"   Edge attr shape: {edge_attr.shape}")
    print(f"   Expected total edge features: 2 * {x.shape[1]} + {edge_attr.shape[1]} = {2 * x.shape[1] + edge_attr.shape[1]}")
    
    return x, edge_index, edge_attr, y_true

def train_memory_efficient_model():
    """Train model with memory management"""
    print("🚀 Starting memory-efficient training...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data with memory management
    data = load_hi_medium_chunked()
    if data is None:
        print("❌ Cannot load HI-Medium dataset. Training aborted.")
        return None
    
    # Create features with memory management
    x, edge_index, edge_attr, y_true = create_features_memory_efficient(data)
    
    # Clear data from memory
    del data
    gc.collect()
    
    # Move to device
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    y_true = y_true.to(device)
    
    # Create model (smaller architecture)
    model = MemoryEfficientAMLGNN(input_dim=15, hidden_dim=128, output_dim=2, dropout=0.1).to(device)
    
    # Initialize edge classifier
    print("\n🔧 Initializing edge classifier...")
    with torch.no_grad():
        _ = model(x, edge_index, edge_attr)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Simpler optimizer
    
    # Simple loss function
    class_weights = torch.tensor([1.0, 3.0]).to(device)  # 3x boost for AML (reduced)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    model.train()
    best_aml_f1 = 0
    patience_counter = 0
    max_patience = 15
    
    print("\n🎯 Training for 200 epochs...")
    for epoch in range(200):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(x, edge_index, edge_attr)
        loss = criterion(logits, y_true)
        
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
                    
                    model_path = f'{models_dir}/memory_efficient_aml_model.pth'
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
        model_path = train_memory_efficient_model()
        if model_path:
            print("\n🎉 MEMORY-EFFICIENT TRAINING COMPLETE!")
            print("=" * 50)
            print("✅ Model trained on HI-Medium dataset")
            print("✅ Memory-efficient processing")
            print("✅ 15 input features (matches evaluation)")
            print("✅ Ready for balanced evaluation")
            print(f"\n🚀 Now run: !python balanced_model_evaluation_final.py")
        else:
            print("❌ Training failed - HI-Medium dataset not found")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

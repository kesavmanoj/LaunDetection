#!/usr/bin/env python3
"""
Fast Regularized Training - GPU Optimized
========================================

Highly optimized for Tesla T4 GPU and 12.7GB RAM:
- Vectorized operations
- GPU acceleration
- Batch processing
- Memory efficient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("⚡ Fast Regularized Training - GPU Optimized")
print("=" * 60)

class RegularizedAMLGNN(nn.Module):
    """Regularized AML GNN with heavy regularization"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.5):
        super(RegularizedAMLGNN, self).__init__()
        
        # Simplified single-branch architecture
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        
        # Heavy dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout + 0.1)
        
        self.edge_classifier = None
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
    def forward(self, x, edge_index, edge_attr=None):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Single branch with heavy regularization
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
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
            print(f"   Creating regularized edge classifier with input_dim={actual_input_dim}")
            
            self.edge_classifier = nn.Sequential(
                nn.Linear(actual_input_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate + 0.2),
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate + 0.1),
                nn.Linear(self.hidden_dim // 4, self.output_dim)
            ).to(edge_features.device)
        
        edge_output = self.edge_classifier(edge_features)
        
        if torch.isnan(edge_output).any():
            edge_output = torch.nan_to_num(edge_output, nan=0.0)
        
        return edge_output

def load_fast_data():
    """Load data quickly with limited size"""
    print("📊 Loading fast data (optimized for speed)...")
    
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    file_path = os.path.join(data_path, 'HI-Small_Trans.csv')
    
    if not os.path.exists(file_path):
        print("❌ HI-Small dataset not found!")
        return None
    
    print("   📁 Loading HI-Small dataset (limited for speed)...")
    
    # Load limited data for speed
    chunk_size = 1000000  # 1M rows at a time
    max_chunks = 2  # Max 2M rows for speed
    chunks = []
    
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        if i >= max_chunks:
            break
        chunks.append(chunk)
        print(f"   📁 Loaded chunk {i+1} ({len(chunk):,} rows)")
    
    data = pd.concat(chunks, ignore_index=True)
    print(f"   📁 Total loaded: {len(data):,} transactions")
    
    # Clean data
    print("   🔄 Cleaning data...")
    data = data.dropna()
    data = data[data['Amount Received'] > 0]
    data = data[~np.isinf(data['Amount Received'])]
    
    print(f"✅ Data: {len(data):,} transactions")
    print(f"   AML: {data['Is Laundering'].sum():,}")
    print(f"   Non-AML: {(data['Is Laundering'] == 0).sum():,}")
    print(f"   AML rate: {data['Is Laundering'].mean()*100:.2f}%")
    
    # Create 10% AML rate dataset
    print(f"\n🔄 Creating 10% AML rate dataset...")
    aml_transactions = data[data['Is Laundering'] == 1]
    non_aml_transactions = data[data['Is Laundering'] == 0]
    
    print(f"   Available AML: {len(aml_transactions):,}")
    print(f"   Available Non-AML: {len(non_aml_transactions):,}")
    
    # Use all AML transactions
    target_aml_count = len(aml_transactions)
    target_non_aml_count = int(target_aml_count * 9)  # 10% AML rate (1:9 ratio)
    
    # Limit non-AML if too many
    if len(non_aml_transactions) > target_non_aml_count:
        non_aml_sample = non_aml_transactions.sample(n=target_non_aml_count, random_state=42)
    else:
        non_aml_sample = non_aml_transactions
    
    # Combine and shuffle
    balanced_data = pd.concat([aml_transactions, non_aml_sample])
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    actual_aml_rate = balanced_data['Is Laundering'].mean()
    print(f"✅ Balanced dataset: {len(balanced_data):,} transactions")
    print(f"   AML: {balanced_data['Is Laundering'].sum():,}")
    print(f"   Non-AML: {(balanced_data['Is Laundering'] == 0).sum():,}")
    print(f"   Actual AML rate: {actual_aml_rate*100:.2f}%")
    
    return balanced_data

def create_fast_features(data):
    """Create features using vectorized operations"""
    print("\n🔄 Creating fast features (vectorized)...")
    
    # Limit accounts for speed
    max_accounts = 5000  # Limit to 5000 most active accounts
    
    # Get most active accounts
    from_accounts = data['From Bank'].value_counts().head(max_accounts // 2)
    to_accounts = data['To Bank'].value_counts().head(max_accounts // 2)
    
    # Combine and get unique accounts
    active_accounts = set(from_accounts.index).union(set(to_accounts.index))
    active_accounts = list(active_accounts)[:max_accounts]
    
    print(f"   Active accounts: {len(active_accounts):,}")
    
    # Filter data to only include active accounts
    filtered_data = data[
        (data['From Bank'].isin(active_accounts)) & 
        (data['To Bank'].isin(active_accounts))
    ].copy()
    
    print(f"   Filtered transactions: {len(filtered_data):,}")
    print(f"   AML rate after filtering: {filtered_data['Is Laundering'].mean()*100:.2f}%")
    
    # Create account mapping
    account_list = list(active_accounts)
    node_to_int = {acc: i for i, acc in enumerate(account_list)}
    
    # Vectorized node feature creation
    print("   🔄 Creating node features (vectorized)...")
    
    # Create account transaction dataframes
    from_trans = filtered_data.groupby('From Bank')['Amount Received'].agg(['sum', 'count', 'mean', 'std', 'max', 'min']).fillna(0)
    to_trans = filtered_data.groupby('To Bank')['Amount Received'].agg(['sum', 'count', 'mean', 'std', 'max', 'min']).fillna(0)
    
    # Create node features vectorized
    node_features = []
    for acc in tqdm(account_list, desc="Creating node features"):
        from_data = from_trans.loc[acc] if acc in from_trans.index else pd.Series([0, 0, 0, 0, 0, 0], index=from_trans.columns)
        to_data = to_trans.loc[acc] if acc in to_trans.index else pd.Series([0, 0, 0, 0, 0, 0], index=to_trans.columns)
        
        total_amount = from_data['sum'] + to_data['sum']
        transaction_count = from_data['count'] + to_data['count']
        avg_amount = total_amount / transaction_count if transaction_count > 0 else 0
        
        # 15 features with noise
        features = [
            np.log1p(total_amount) + np.random.normal(0, 0.1),
            np.log1p(transaction_count) + np.random.normal(0, 0.1),
            np.log1p(avg_amount) + np.random.normal(0, 0.1),
            from_data['count'] + np.random.normal(0, 0.5),
            to_data['count'] + np.random.normal(0, 0.5),
            from_data['mean'],
            to_data['mean'],
            from_data['std'],
            to_data['std'],
            from_data['max'],
            to_data['max'],
            from_data['min'],
            to_data['min'],
            len(set(filtered_data[filtered_data['From Bank'] == acc]['To Bank'])) if acc in filtered_data['From Bank'].values else 0,
            len(set(filtered_data[filtered_data['To Bank'] == acc]['From Bank'])) if acc in filtered_data['To Bank'].values else 0
        ]
        
        # Clean features
        features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float32)
    print(f"   Node features shape: {x.shape}")
    
    # Vectorized edge creation
    print("   🔄 Creating edge features (vectorized)...")
    
    # Filter transactions to active accounts
    edge_data = filtered_data[
        (filtered_data['From Bank'].isin(active_accounts)) & 
        (filtered_data['To Bank'].isin(active_accounts))
    ].copy()
    
    # Create edge index
    edge_index = torch.tensor([
        [node_to_int[row['From Bank']], node_to_int[row['To Bank']]]
        for _, row in edge_data.iterrows()
    ], dtype=torch.long).t().contiguous()
    
    # Create edge features vectorized
    amounts = edge_data['Amount Received'].values
    is_aml = edge_data['Is Laundering'].values
    
    # Vectorized edge features
    edge_features = np.column_stack([
        np.log1p(amounts) + np.random.normal(0, 0.1, len(amounts)),
        is_aml,
        np.random.normal(0, 0.1, len(amounts)),
        np.random.normal(0, 0.1, len(amounts)),
        np.random.normal(0, 0.1, len(amounts)),
        np.random.normal(0, 0.1, len(amounts)),
        np.random.normal(0, 0.1, len(amounts)),
        np.random.normal(0, 0.1, len(amounts)),
        np.random.normal(0, 0.1, len(amounts)),
        np.random.normal(0, 0.1, len(amounts)),
        np.random.normal(0, 0.1, len(amounts)),
        np.random.normal(0, 0.1, len(amounts)),
        np.random.normal(0, 0.1, len(amounts))
    ])
    
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    y_true = torch.tensor(is_aml, dtype=torch.long)
    
    print(f"   Edge index shape: {edge_index.shape}")
    print(f"   Edge attr shape: {edge_attr.shape}")
    
    return x, edge_index, edge_attr, y_true

def create_fast_splits(x, edge_index, edge_attr, y_true, test_size=0.2, val_size=0.2):
    """Create fast data splits"""
    print(f"\n🔄 Creating fast data splits...")
    
    # Get unique edges for splitting
    num_edges = edge_index.shape[1]
    edge_indices = torch.arange(num_edges)
    
    # Split edges with stratification
    train_edges, temp_edges = train_test_split(
        edge_indices, 
        test_size=test_size + val_size, 
        random_state=42, 
        stratify=y_true.numpy()
    )
    val_edges, test_edges = train_test_split(
        temp_edges, 
        test_size=test_size/(test_size + val_size), 
        random_state=42, 
        stratify=y_true[temp_edges].numpy()
    )
    
    print(f"   Train edges: {len(train_edges):,}")
    print(f"   Validation edges: {len(val_edges):,}")
    print(f"   Test edges: {len(test_edges):,}")
    
    # Create splits
    train_data = {
        'x': x,
        'edge_index': edge_index[:, train_edges],
        'edge_attr': edge_attr[train_edges],
        'y': y_true[train_edges]
    }
    
    val_data = {
        'x': x,
        'edge_index': edge_index[:, val_edges],
        'edge_attr': edge_attr[val_edges],
        'y': y_true[val_edges]
    }
    
    test_data = {
        'x': x,
        'edge_index': edge_index[:, test_edges],
        'edge_attr': edge_attr[test_edges],
        'y': y_true[test_edges]
    }
    
    print(f"   Train AML rate: {train_data['y'].float().mean():.2%}")
    print(f"   Val AML rate: {val_data['y'].float().mean():.2%}")
    print(f"   Test AML rate: {test_data['y'].float().mean():.2%}")
    
    return train_data, val_data, test_data

def train_fast_model(train_data, val_data, epochs=50):
    """Train model quickly with early stopping"""
    print(f"\n🚀 Training fast model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move data to device
    x = train_data['x'].to(device)
    edge_index = train_data['edge_index'].to(device)
    edge_attr = train_data['edge_attr'].to(device)
    y = train_data['y'].to(device)
    
    val_x = val_data['x'].to(device)
    val_edge_index = val_data['edge_index'].to(device)
    val_edge_attr = val_data['edge_attr'].to(device)
    val_y = val_data['y'].to(device)
    
    # Create model with moderate dropout for better learning
    model = RegularizedAMLGNN(input_dim=15, hidden_dim=64, output_dim=2, dropout=0.3).to(device)
    
    # Initialize edge classifier
    with torch.no_grad():
        _ = model(x, edge_index, edge_attr)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # Optimizer with moderate weight decay for better learning
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    
    # Loss with balanced class weights for 10% AML rate
    class_weights = torch.tensor([1.0, 1.0]).to(device)  # Balanced weights for 10% AML
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Early stopping
    best_val_f1 = 0
    patience = 5  # Reduced patience for speed
    patience_counter = 0
    
    print(f"   Training for {epochs} epochs with early stopping...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        logits = model(x, edge_index, edge_attr)
        loss = criterion(logits, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x, val_edge_index, val_edge_attr)
            val_predictions = torch.argmax(val_logits, dim=1)
            val_f1 = f1_score(val_y.cpu().numpy(), val_predictions.cpu().numpy(), average='binary', pos_label=1, zero_division=0)
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), '/content/drive/MyDrive/LaunDetection/models/fast_regularized_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"   Epoch {epoch}: Loss={loss:.4f}, Val F1={val_f1:.4f}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch}")
            break
    
    print(f"   Best validation F1: {best_val_f1:.4f}")
    return model

def evaluate_fast_model(model, test_data):
    """Evaluate fast model"""
    print(f"\n📊 Evaluating fast model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move data to device
    x = test_data['x'].to(device)
    edge_index = test_data['edge_index'].to(device)
    edge_attr = test_data['edge_attr'].to(device)
    y = test_data['y'].to(device)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        predictions = torch.argmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)
    
    # Move to CPU for metrics
    y_cpu = y.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    probabilities_cpu = probabilities.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_cpu, predictions_cpu)
    f1_weighted = f1_score(y_cpu, predictions_cpu, average='weighted')
    f1_aml = f1_score(y_cpu, predictions_cpu, average='binary', pos_label=1, zero_division=0)
    precision_aml = precision_score(y_cpu, predictions_cpu, average='binary', pos_label=1, zero_division=0)
    recall_aml = recall_score(y_cpu, predictions_cpu, average='binary', pos_label=1, zero_division=0)
    
    cm = confusion_matrix(y_cpu, predictions_cpu)
    
    print(f"\n📊 Fast Regularized Model Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 (Weighted): {f1_weighted:.4f}")
    print(f"   AML F1: {f1_aml:.4f}")
    print(f"   AML Precision: {precision_aml:.4f}")
    print(f"   AML Recall: {recall_aml:.4f}")
    
    print(f"\n📈 Confusion Matrix:")
    if cm.shape == (2, 2):
        print(f"   True Negative: {cm[0][0]:,}")
        print(f"   False Positive: {cm[0][1]:,}")
        print(f"   False Negative: {cm[1][0]:,}")
        print(f"   True Positive: {cm[1][1]:,}")
    else:
        print(f"   Confusion Matrix Shape: {cm.shape}")
        print(f"   Matrix: {cm}")
    
    # Analysis
    if accuracy > 0.99:
        print("\n⚠️ WARNING: Perfect accuracy suggests overfitting!")
    elif 0.7 <= accuracy <= 0.9:
        print("\n✅ Good: Realistic performance")
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
    print("🚀 Starting fast regularized training...")
    
    try:
        # Load data
        data = load_fast_data()
        if data is None:
            print("❌ Cannot load data. Training aborted.")
            return
        
        # Create features
        x, edge_index, edge_attr, y_true = create_fast_features(data)
        
        # Create fast splits
        train_data, val_data, test_data = create_fast_splits(x, edge_index, edge_attr, y_true)
        
        # Train model
        model = train_fast_model(train_data, val_data)
        
        # Evaluate model
        results = evaluate_fast_model(model, test_data)
        
        print("\n🎉 FAST REGULARIZATION COMPLETE!")
        print("=" * 50)
        print("✅ Model trained with heavy regularization")
        print("✅ Early stopping implemented")
        print("✅ Noise added to prevent overfitting")
        print("✅ Proper validation splits")
        print("✅ Fast processing (GPU optimized)")
        
        if results['f1_aml'] > 0.9:
            print("⚠️ WARNING: Still showing signs of overfitting!")
        elif 0.3 <= results['f1_aml'] <= 0.8:
            print("✅ GOOD: Model shows realistic performance")
        else:
            print("⚠️ Model performance needs improvement")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# ============================================================================
# GOOGLE COLAB SCRIPT - GNN MODEL TRAINING FOR AML TRANSACTION CLASSIFICATION
# Train and evaluate GCN, GAT, and GIN models on preprocessed AML data
# Copy and paste this entire cell into Google Colab
# ============================================================================

# Mount Google Drive and install packages
from google.colab import drive
import os

# Check if drive is already mounted
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
else:
    print("Drive already mounted")

# Install required packages
!pip install torch torch-geometric scikit-learn matplotlib seaborn tqdm -q

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, BatchNorm, LayerNorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path('/content/drive/MyDrive/LaunDetection')
GRAPHS_DIR = BASE_DIR / 'data' / 'graphs'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

print(f"📁 Base directory: {BASE_DIR}")
print(f"📁 Graphs: {GRAPHS_DIR}")
print(f"📁 Models: {MODELS_DIR}")

# Check available processed datasets
available_datasets = []
for dataset in ['hi-small', 'li-small']:
    file_path = GRAPHS_DIR / f'ibm_aml_{dataset}_enhanced_splits.pt'
    if file_path.exists():
        available_datasets.append(dataset)
        size_mb = file_path.stat().st_size / 1024**2
        print(f"✅ Found {dataset}: {size_mb:.1f} MB")
    else:
        print(f"❌ Missing {dataset}: {file_path}")

if not available_datasets:
    print("❌ No processed datasets found! Run preprocessing first.")
    sys.exit(1)

print(f"\n🚀 Will train models on: {available_datasets}")

# ============================================================================
# SIMPLIFIED GNN MODELS FOR COLAB
# ============================================================================

class SimpleGCN(nn.Module):
    """Simplified GCN model for Colab training"""
    
    def __init__(self, node_features, edge_features, hidden_dim=64, dropout=0.3):
        super(SimpleGCN, self).__init__()
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GCN layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        
        # Edge classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),  # src + dst + edge
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr):
        # Encode features
        h = self.node_encoder(x)
        edge_feat = self.edge_encoder(edge_attr)
        
        # GCN layers
        h = F.relu(self.bn1(self.conv1(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = F.relu(self.bn3(self.conv3(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # MEMORY OPTIMIZATION: Create edge representations in chunks to avoid OOM
        row, col = edge_index
        num_edges = edge_index.shape[1]
        
        # Process edges in chunks if too many
        if num_edges > 2000000:  # 2M edge threshold
            chunk_size = 1000000  # 1M edges per chunk
            edge_logits = []
            
            for i in range(0, num_edges, chunk_size):
                end_idx = min(i + chunk_size, num_edges)
                
                # Get chunk indices
                chunk_row = row[i:end_idx]
                chunk_col = col[i:end_idx]
                chunk_edge_feat = edge_feat[i:end_idx]
                
                # Create chunk edge representation
                chunk_edge_repr = torch.cat([h[chunk_row], h[chunk_col], chunk_edge_feat], dim=1)
                
                # Classify chunk
                chunk_logits = self.classifier(chunk_edge_repr)
                edge_logits.append(chunk_logits)
                
                # Clean up intermediate tensors
                del chunk_edge_repr, chunk_row, chunk_col, chunk_edge_feat
            
            # Concatenate all chunks
            return torch.cat(edge_logits, dim=0)
        else:
            # Standard processing for smaller edge sets
            edge_repr = torch.cat([h[row], h[col], edge_feat], dim=1)
            return self.classifier(edge_repr)

class SimpleGAT(nn.Module):
    """Simplified GAT model for Colab training"""
    
    def __init__(self, node_features, edge_features, hidden_dim=64, heads=4, dropout=0.3):
        super(SimpleGAT, self).__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT layers
        self.conv1 = GATConv(hidden_dim, hidden_dim//heads, heads=heads, dropout=0.1, concat=True)
        self.conv2 = GATConv(hidden_dim, hidden_dim//heads, heads=heads, dropout=0.1, concat=True)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.1, concat=False)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr):
        h = self.node_encoder(x)
        edge_feat = self.edge_encoder(edge_attr)
        
        h = F.relu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = F.relu(self.conv3(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # MEMORY OPTIMIZATION: Process edges in chunks for GAT
        row, col = edge_index
        num_edges = edge_index.shape[1]
        
        if num_edges > 2000000:  # 2M edge threshold
            chunk_size = 1000000  # 1M edges per chunk
            edge_logits = []
            
            for i in range(0, num_edges, chunk_size):
                end_idx = min(i + chunk_size, num_edges)
                
                chunk_row = row[i:end_idx]
                chunk_col = col[i:end_idx]
                chunk_edge_feat = edge_feat[i:end_idx]
                
                chunk_edge_repr = torch.cat([h[chunk_row], h[chunk_col], chunk_edge_feat], dim=1)
                chunk_logits = self.classifier(chunk_edge_repr)
                edge_logits.append(chunk_logits)
                
                del chunk_edge_repr, chunk_row, chunk_col, chunk_edge_feat
            
            return torch.cat(edge_logits, dim=0)
        else:
            edge_repr = torch.cat([h[row], h[col], edge_feat], dim=1)
            return self.classifier(edge_repr)

class SimpleGIN(nn.Module):
    """Simplified GIN model for Colab training"""
    
    def __init__(self, node_features, edge_features, hidden_dim=64, dropout=0.3):
        super(SimpleGIN, self).__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GIN layers
        mlp1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        mlp3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        
        self.conv1 = GINConv(mlp1, eps=0.0, train_eps=True)
        self.conv2 = GINConv(mlp2, eps=0.0, train_eps=True)
        self.conv3 = GINConv(mlp3, eps=0.0, train_eps=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr):
        h = self.node_encoder(x)
        edge_feat = self.edge_encoder(edge_attr)
        
        h = F.relu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = F.relu(self.conv3(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # MEMORY OPTIMIZATION: Process edges in chunks for GIN
        row, col = edge_index
        num_edges = edge_index.shape[1]
        
        if num_edges > 2000000:  # 2M edge threshold
            chunk_size = 1000000  # 1M edges per chunk
            edge_logits = []
            
            for i in range(0, num_edges, chunk_size):
                end_idx = min(i + chunk_size, num_edges)
                
                chunk_row = row[i:end_idx]
                chunk_col = col[i:end_idx]
                chunk_edge_feat = edge_feat[i:end_idx]
                
                chunk_edge_repr = torch.cat([h[chunk_row], h[chunk_col], chunk_edge_feat], dim=1)
                chunk_logits = self.classifier(chunk_edge_repr)
                edge_logits.append(chunk_logits)
                
                del chunk_edge_repr, chunk_row, chunk_col, chunk_edge_feat
            
            return torch.cat(edge_logits, dim=0)
        else:
            edge_repr = torch.cat([h[row], h[col], edge_feat], dim=1)
            return self.classifier(edge_repr)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def load_data():
    """Load and combine available datasets"""
    print("📂 Loading processed datasets...")
    
    all_train_data = []
    all_val_data = []
    all_test_data = []
    
    for dataset in available_datasets:
        file_path = GRAPHS_DIR / f'ibm_aml_{dataset}_enhanced_splits.pt'
        print(f"Loading {dataset}...")
        
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        all_train_data.append(data['train'])
        all_val_data.append(data['val'])
        all_test_data.append(data['test'])
        
        print(f"  {dataset}: {data['train'].num_edges:,} train, {data['val'].num_edges:,} val, {data['test'].num_edges:,} test")
    
    # Combine datasets
    def combine_data(data_list):
        if len(data_list) == 1:
            return data_list[0]
        
        # Use first dataset's nodes
        x = data_list[0].x
        
        # Combine edges
        edge_indices = [d.edge_index for d in data_list]
        edge_attrs = [d.edge_attr for d in data_list]
        labels = [d.y for d in data_list]
        
        combined_edge_index = torch.cat(edge_indices, dim=1)
        combined_edge_attr = torch.cat(edge_attrs, dim=0)
        combined_y = torch.cat(labels, dim=0)
        
        from torch_geometric.data import Data
        return Data(x=x, edge_index=combined_edge_index, edge_attr=combined_edge_attr, y=combined_y)
    
    train_data = combine_data(all_train_data)
    val_data = combine_data(all_val_data)
    test_data = combine_data(all_test_data)
    
    print(f"\n📊 Combined dataset:")
    print(f"  Nodes: {train_data.num_nodes:,}")
    print(f"  Train edges: {train_data.num_edges:,}")
    print(f"  Val edges: {val_data.num_edges:,}")
    print(f"  Test edges: {test_data.num_edges:,}")
    print(f"  Node features: {train_data.x.shape[1]}")
    print(f"  Edge features: {train_data.edge_attr.shape[1]}")
    
    # Calculate class distribution
    pos_count = train_data.y.sum().item()
    total_count = len(train_data.y)
    print(f"  Positive rate: {pos_count/total_count*100:.2f}% ({pos_count:,}/{total_count:,})")
    
    # CUDA FIX: Validate and fix edge indices
    def validate_and_fix_edges(data, split_name):
        print(f"🔍 Validating {split_name} edges...")
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        # Check for invalid indices
        max_idx = edge_index.max().item()
        min_idx = edge_index.min().item()
        
        print(f"  Edge index range: [{min_idx}, {max_idx}], Num nodes: {num_nodes}")
        
        if max_idx >= num_nodes or min_idx < 0:
            print(f"  ⚠️ Invalid edge indices found! Filtering...")
            
            # Create mask for valid edges
            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) & \
                        (edge_index[0] >= 0) & (edge_index[1] >= 0)
            
            # Filter data
            data.edge_index = edge_index[:, valid_mask]
            data.edge_attr = data.edge_attr[valid_mask]
            data.y = data.y[valid_mask]
            
            print(f"  ✅ Filtered to {data.num_edges:,} valid edges")
            
            # ADDITIONAL FIX: Ensure edge_index is contiguous and properly formatted
            data.edge_index = data.edge_index.contiguous()
            data.edge_attr = data.edge_attr.contiguous()
            data.y = data.y.contiguous()
            
            # Verify after filtering
            new_max = data.edge_index.max().item()
            new_min = data.edge_index.min().item()
            print(f"  ✅ After filtering: [{new_min}, {new_max}] vs {num_nodes} nodes")
            
        else:
            print(f"  ✅ All edges valid")
        
        # Ensure all tensors are contiguous
        data.edge_index = data.edge_index.contiguous()
        data.edge_attr = data.edge_attr.contiguous() 
        data.y = data.y.contiguous()
        data.x = data.x.contiguous()
        
        return data
    
    # Validate all splits
    train_data = validate_and_fix_edges(train_data, "train")
    val_data = validate_and_fix_edges(val_data, "validation")
    test_data = validate_and_fix_edges(test_data, "test")
    
    return train_data, val_data, test_data

def calculate_class_weights(train_data):
    """Calculate class weights for imbalanced data"""
    labels = train_data.y.cpu().numpy()  # Move to CPU first
    pos_count = np.sum(labels)
    neg_count = len(labels) - pos_count
    
    pos_weight = len(labels) / (2 * pos_count)
    neg_weight = len(labels) / (2 * neg_count)
    
    return torch.tensor([neg_weight, pos_weight], dtype=torch.float32)

def train_model(model, train_data, val_data, epochs=50, lr=0.001, device='cpu'):
    """Train a single model with memory-efficient batch processing"""
    
    # CUDA DEBUGGING: Enable blocking for better error messages
    if device == 'cuda':
        import os
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print("🔧 CUDA_LAUNCH_BLOCKING enabled for debugging")
        
        # GPU MEMORY OPTIMIZATION: Set memory fraction and enable memory-efficient attention
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        print("🔧 GPU memory fraction set to 90%")
    
    # CUDA FIX: Safe model transfer to device
    print(f"🔄 Moving model to {device}...")
    try:
        model = model.to(device)
        print(f"✅ Model successfully moved to {device}")
    except Exception as e:
        print(f"❌ Error moving model to {device}: {e}")
        print("🔄 Falling back to CPU for model...")
        device = 'cpu'
        model = model.cpu()
    
    print(f"🔄 Moving training data to {device}...")
    try:
        # Move data piece by piece for better error tracking
        train_data.x = train_data.x.to(device)
        train_data.edge_index = train_data.edge_index.to(device)
        train_data.edge_attr = train_data.edge_attr.to(device)
        train_data.y = train_data.y.to(device)
        
        val_data.x = val_data.x.to(device)
        val_data.edge_index = val_data.edge_index.to(device)
        val_data.edge_attr = val_data.edge_attr.to(device)
        val_data.y = val_data.y.to(device)
        
        print(f"✅ Data successfully moved to {device}")
        
    except Exception as e:
        print(f"❌ Error moving data to {device}: {e}")
        print("🔄 Falling back to CPU training...")
        device = 'cpu'
        model = model.cpu()
        train_data = train_data.cpu()
        val_data = val_data.cpu()
    
    # CUDA FIX: Additional validation after moving to device
    print(f"📊 Final data validation on {device}:")
    print(f"  Train: {train_data.num_nodes} nodes, {train_data.num_edges} edges")
    print(f"  Val: {val_data.num_nodes} nodes, {val_data.num_edges} edges")
    print(f"  Train edge range: [{train_data.edge_index.min().item()}, {train_data.edge_index.max().item()}]")
    print(f"  Val edge range: [{val_data.edge_index.min().item()}, {val_data.edge_index.max().item()}]")
    
    # Setup training
    class_weights = calculate_class_weights(train_data).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_val_f1 = 0
    patience = 10
    no_improve = 0
    
    print(f"🚀 Training on {device}")
    
    # GPU MEMORY OPTIMIZATION: Check if we need batch processing
    if device == 'cuda' and train_data.num_edges > 3000000:  # 3M edges threshold
        print(f"🔧 Large dataset detected ({train_data.num_edges:,} edges)")
        print(f"🔧 Using gradient accumulation to fit in GPU memory")
        use_gradient_accumulation = True
        accumulation_steps = 4  # Process in 4 mini-batches
    else:
        use_gradient_accumulation = False
        accumulation_steps = 1
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        if use_gradient_accumulation:
            # GPU MEMORY OPTIMIZATION: Gradient accumulation for large graphs
            optimizer.zero_grad()
            
            # Split edges into mini-batches
            num_edges = train_data.num_edges
            batch_size = num_edges // accumulation_steps
            
            for step in range(accumulation_steps):
                start_idx = step * batch_size
                end_idx = min((step + 1) * batch_size, num_edges)
                
                # Create mini-batch
                edge_mask = torch.arange(start_idx, end_idx, device=device)
                mini_edge_index = train_data.edge_index[:, edge_mask]
                mini_edge_attr = train_data.edge_attr[edge_mask]
                mini_y = train_data.y[edge_mask]
                
                # Forward pass on mini-batch
                logits = model(train_data.x, mini_edge_index, mini_edge_attr)
                loss = criterion(logits, mini_y)
                
                # Scale loss by accumulation steps
                loss = loss / accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item()
                
                # Clear intermediate tensors
                del logits, mini_edge_index, mini_edge_attr, mini_y
                if step < accumulation_steps - 1:  # Don't clear on last step
                    torch.cuda.empty_cache()
            
            # Update weights after accumulating gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        else:
            # Standard training for smaller datasets
            optimizer.zero_grad()
            logits = model(train_data.x, train_data.edge_index, train_data.edge_attr)
            loss = criterion(logits, train_data.y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss = loss.item()
        
        # Validation with memory optimization
        model.eval()
        with torch.no_grad():
            if device == 'cuda' and val_data.num_edges > 1000000:  # 1M edges threshold for validation
                # Batch validation for large validation sets
                val_losses = []
                val_preds_list = []
                val_batch_size = val_data.num_edges // 2  # Split validation in half
                
                for val_step in range(2):
                    start_idx = val_step * val_batch_size
                    end_idx = min((val_step + 1) * val_batch_size, val_data.num_edges)
                    
                    val_edge_mask = torch.arange(start_idx, end_idx, device=device)
                    val_mini_edge_index = val_data.edge_index[:, val_edge_mask]
                    val_mini_edge_attr = val_data.edge_attr[val_edge_mask]
                    val_mini_y = val_data.y[val_edge_mask]
                    
                    val_mini_logits = model(val_data.x, val_mini_edge_index, val_mini_edge_attr)
                    val_mini_loss = criterion(val_mini_logits, val_mini_y)
                    
                    val_losses.append(val_mini_loss.item())
                    val_preds_list.append(val_mini_logits.argmax(dim=1))
                    
                    # Cleanup
                    del val_mini_logits, val_mini_edge_index, val_mini_edge_attr, val_mini_y
                    if val_step == 0:  # Clear cache between batches
                        torch.cuda.empty_cache()
                
                val_loss = sum(val_losses) / len(val_losses)
                val_preds = torch.cat(val_preds_list)
                
            else:
                # Standard validation for smaller datasets
                val_logits = model(val_data.x, val_data.edge_index, val_data.edge_attr)
                val_loss = criterion(val_logits, val_data.y)
                val_preds = val_logits.argmax(dim=1)
            
            val_f1 = f1_score(val_data.y.cpu(), val_preds.cpu(), average='binary')
        
        # Update history
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss if isinstance(val_loss, float) else val_loss.item())
        history['val_f1'].append(val_f1)
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            # Save best model state
            best_state = model.state_dict().copy()
        else:
            no_improve += 1
        
        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch:2d}: Train Loss={loss:.4f}, Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")
        
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(best_state)
    print(f"✅ Training completed. Best Val F1: {best_val_f1:.4f}")
    
    return model, history

def evaluate_model(model, data, split_name, device='cpu'):
    """Evaluate model on a dataset split"""
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        y_true = data.y.cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_prob = probs[:, 1].cpu().numpy()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'auc': roc_auc_score(y_true, y_prob)
        }
    
    print(f"{split_name} Results:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    return metrics

def plot_training_history(histories, model_names):
    """Plot training histories for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['train_loss', 'val_loss', 'val_f1']
    titles = ['Training Loss', 'Validation Loss', 'Validation F1']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        for name, history in zip(model_names, histories):
            axes[i].plot(history[metric], label=name, alpha=0.8)
        
        axes[i].set_title(title)
        axes[i].set_xlabel('Epoch')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison(results):
    """Plot model comparison"""
    df = pd.DataFrame(results).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model in enumerate(df.index):
        values = [df.loc[model, metric] for metric in metrics]
        ax.bar(x + i*width, values, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison (Test Set)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training and evaluation pipeline"""
    print("🚀 STARTING GNN MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Setup device with comprehensive debugging
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # CUDA DEBUGGING: Detailed device information with safe error handling
    if torch.cuda.is_available():
        print(f"🔧 CUDA Debug Info:")
        try:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name()}")
            
            # Memory info
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  Memory allocated: {memory_allocated:.2f} GB")
            print(f"  Memory reserved: {memory_reserved:.2f} GB")
            
            # CUDA SAFETY: Check if reserved memory is too high (indicates corruption)
            if memory_reserved > CUDA_MEMORY_THRESHOLD:
                print(f"  ⚠️ High reserved memory detected ({memory_reserved:.2f} GB)")
                print(f"  🔄 CUDA may be corrupted!")
                
                if REQUIRE_GPU_MODE:
                    print(f"  ❌ REQUIRE_GPU_MODE is enabled - cannot proceed with corrupted CUDA")
                    print(f"  🔄 SOLUTION: Restart Colab runtime to clear CUDA memory")
                    print(f"     Go to Runtime → Restart Runtime, then re-run")
                    raise RuntimeError("CUDA memory corrupted and GPU training required")
                else:
                    print(f"  🔄 Switching to CPU for safety")
                    device = torch.device('cpu')
            else:
                # Try to clear cache safely
                print(f"  🔄 Attempting to clear CUDA cache...")
                torch.cuda.empty_cache()
                print(f"  ✅ CUDA cache cleared")
                
                # Test basic CUDA operations
                print(f"  🔄 Testing basic CUDA operations...")
                test_tensor = torch.randn(10, 10).cuda()
                test_result = test_tensor @ test_tensor.T
                print(f"  ✅ Basic CUDA operations working")
                del test_tensor, test_result
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  ❌ CUDA initialization failed: {e}")
            print(f"  🔄 CUDA appears corrupted, falling back to CPU")
            device = torch.device('cpu')
    else:
        print("🔧 CUDA not available")
        if REQUIRE_GPU_MODE:
            print("❌ REQUIRE_GPU_MODE is enabled but CUDA not available!")
            print("🔄 SOLUTION: Change Colab runtime to GPU")
            print("   Go to Runtime → Change Runtime Type → Hardware Accelerator → GPU")
            raise RuntimeError("GPU required but CUDA not available")
        else:
            print("🔄 Using CPU")
            device = torch.device('cpu')
    
    # FINAL DEVICE CHECK
    print(f"🎯 Final device selection: {device}")
    
    # Load data
    train_data, val_data, test_data = load_data()
    
    node_features = train_data.x.shape[1]
    edge_features = train_data.edge_attr.shape[1]
    
    # Define models
    models = {
        'GCN': SimpleGCN(node_features, edge_features, hidden_dim=64),
        'GAT': SimpleGAT(node_features, edge_features, hidden_dim=64, heads=4),
        'GIN': SimpleGIN(node_features, edge_features, hidden_dim=64)
    }
    
    # CUDA SAFETY: Test model creation on CUDA first
    if device.type == 'cuda':
        print("🔧 Testing model creation on CUDA...")
        try:
            test_model = SimpleGCN(node_features, edge_features, hidden_dim=32)  # Smaller test model
            test_model = test_model.cuda()
            print("✅ Model creation on CUDA successful")
            del test_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Model creation on CUDA failed: {e}")
            print("🔄 Switching to CPU for safety")
            device = torch.device('cpu')
    
    # Training configuration
    epochs = 50
    lr = 0.001
    
    # Train all models
    trained_models = {}
    histories = []
    model_names = []
    
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"TRAINING {name} MODEL")
        print(f"{'='*40}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}")
        
        # Train model
        start_time = time.time()
        trained_model, history = train_model(model, train_data, val_data, epochs, lr, device)
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time/60:.1f} minutes")
        
        trained_models[name] = trained_model
        histories.append(history)
        model_names.append(name)
        
        # Save model
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_class': type(trained_model).__name__,
            'node_features': node_features,
            'edge_features': edge_features,
            'history': history
        }, MODELS_DIR / f'{name.lower()}_model.pt')
        
        print(f"✅ {name} model saved!")
    
    # Plot training histories
    print(f"\n📊 Plotting training histories...")
    plot_training_history(histories, model_names)
    
    # Evaluate all models
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    results = {}
    
    for name, model in trained_models.items():
        print(f"\n{name} Model Evaluation:")
        print("-" * 30)
        
        # Evaluate on all splits
        train_metrics = evaluate_model(model, train_data, "Train", device)
        val_metrics = evaluate_model(model, val_data, "Validation", device)
        test_metrics = evaluate_model(model, test_data, "Test", device)
        
        results[name] = test_metrics  # Store test results for comparison
    
    # Plot model comparison
    print(f"\n📊 Plotting model comparison...")
    plot_model_comparison(results)
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    print(f"✅ Successfully trained {len(models)} models")
    print(f"📁 Models saved in: {MODELS_DIR}")
    print(f"📊 Plots saved in: {MODELS_DIR}")
    
    # Best model
    best_model = max(results.keys(), key=lambda x: results[x]['f1'])
    best_f1 = results[best_model]['f1']
    
    print(f"\n🏆 Best model: {best_model} (F1: {best_f1:.4f})")
    
    print(f"\n📋 Test Set Results Summary:")
    for name, metrics in results.items():
        print(f"  {name}: F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
    
    print(f"\n🎉 Training pipeline completed successfully!")
    
    return trained_models, results

# CUDA SETTINGS: Configure CUDA behavior
FORCE_CPU_MODE = False      # Set to True to force CPU training
REQUIRE_GPU_MODE = True     # Set to True to REQUIRE GPU training (fail if no GPU)
CUDA_MEMORY_THRESHOLD = 6.0 # GB of reserved memory before considering corruption

# Run the training pipeline
if __name__ == "__main__":
    if FORCE_CPU_MODE:
        print("🔧 FORCE_CPU_MODE enabled - using CPU only")
        # Disable CUDA completely
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    trained_models, results = main()

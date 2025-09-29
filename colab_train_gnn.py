# ============================================================================
# GOOGLE COLAB SCRIPT - GNN MODEL TRAINING FOR AML TRANSACTION CLASSIFICATION
# Train and evaluate GCN, GAT, and GIN models on preprocessed AML data
# Copy and paste this entire cell into Google Colab
# ============================================================================

# Mount Google Drive and install packages
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install torch torch-geometric scikit-learn matplotlib seaborn tqdm -q

import os
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
        
        # Create edge representations
        row, col = edge_index
        edge_repr = torch.cat([h[row], h[col], edge_feat], dim=1)
        
        # Classify edges
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
        
        row, col = edge_index
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
        
        row, col = edge_index
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
        
        data = torch.load(file_path, map_location='cpu')
        
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
    
    return train_data, val_data, test_data

def calculate_class_weights(train_data):
    """Calculate class weights for imbalanced data"""
    labels = train_data.y.numpy()
    pos_count = np.sum(labels)
    neg_count = len(labels) - pos_count
    
    pos_weight = len(labels) / (2 * pos_count)
    neg_weight = len(labels) / (2 * neg_count)
    
    return torch.tensor([neg_weight, pos_weight], dtype=torch.float32)

def train_model(model, train_data, val_data, epochs=50, lr=0.001, device='cpu'):
    """Train a single model"""
    model = model.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    
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
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        logits = model(train_data.x, train_data.edge_index, train_data.edge_attr)
        loss = criterion(logits, train_data.y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_data.x, val_data.edge_index, val_data.edge_attr)
            val_loss = criterion(val_logits, val_data.y)
            
            val_preds = val_logits.argmax(dim=1)
            val_f1 = f1_score(val_data.y.cpu(), val_preds.cpu(), average='binary')
        
        # Update history
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
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
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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

# Run the training pipeline
if __name__ == "__main__":
    trained_models, results = main()

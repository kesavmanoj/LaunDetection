"""
Streamlined GNN Training Script for AML Detection
Designed to work with preprocessed LI-Small and HI-Small datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
import time
import os
import sys

# Import our models
from gnn_models import EdgeFeatureGCN, EdgeFeatureGAT, EdgeFeatureGIN

# Configuration
BASE_DIR = Path('/content/drive/MyDrive/LaunDetection')
GRAPHS_DIR = BASE_DIR / 'data' / 'graphs'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

def setup_device():
    """Setup and validate CUDA device"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Clear cache and check memory
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
        # Check for corruption
        if memory_reserved > 6.0:
            print("⚠️ High reserved memory detected - potential CUDA corruption")
            print("Consider restarting runtime if training fails")
    
    return device

def load_data(memory_efficient=True):
    """Load preprocessed datasets with optional memory optimization"""
    print("📂 Loading preprocessed datasets...")
    
    if memory_efficient:
        print("🔧 Memory-efficient mode: Loading only LI-Small dataset")
        datasets = ['li-small']  # Start with just one dataset
    else:
        datasets = ['hi-small', 'li-small']
    
    all_train_data = []
    all_val_data = []
    all_test_data = []
    
    for dataset in datasets:
        # Try fixed preprocessing files first, fallback to enhanced
        fixed_file_path = GRAPHS_DIR / f'ibm_aml_{dataset}_fixed_splits.pt'
        enhanced_file_path = GRAPHS_DIR / f'ibm_aml_{dataset}_enhanced_splits.pt'
        
        if fixed_file_path.exists():
            file_path = fixed_file_path
            print(f"Using FIXED preprocessing for {dataset}")
        elif enhanced_file_path.exists():
            file_path = enhanced_file_path
            print(f"Using ENHANCED preprocessing for {dataset} (may have invalid edges)")
        else:
            print(f"❌ No preprocessed file found for {dataset}")
            continue
        
        if file_path.exists():
            print(f"Loading {dataset}...")
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            
            all_train_data.append(data['train'])
            all_val_data.append(data['val'])
            all_test_data.append(data['test'])
            
            print(f"  {dataset}: {data['train'].num_edges:,} train, {data['val'].num_edges:,} val, {data['test'].num_edges:,} test")
        else:
            print(f"❌ Missing {dataset}: {file_path}")
    
    if not all_train_data:
        raise ValueError("No datasets found!")
    
    # Combine datasets
    def combine_data(data_list):
        if len(data_list) == 1:
            return data_list[0]
        
        x = data_list[0].x
        edge_indices = [d.edge_index for d in data_list]
        edge_attrs = [d.edge_attr for d in data_list]
        labels = [d.y for d in data_list]
        
        from torch_geometric.data import Data
        return Data(
            x=x,
            edge_index=torch.cat(edge_indices, dim=1),
            edge_attr=torch.cat(edge_attrs, dim=0),
            y=torch.cat(labels, dim=0)
        )
    
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
    
    # Quick validation check (should pass with fixed preprocessing)
    def quick_validate_edges(data, split_name):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        max_idx = edge_index.max().item()
        min_idx = edge_index.min().item()
        
        if max_idx >= num_nodes or min_idx < 0:
            print(f"⚠️ {split_name}: Invalid edge indices detected! Range: [{min_idx}, {max_idx}], Nodes: {num_nodes}")
            print(f"   This suggests you need to run fixed preprocessing first.")
            return False
        else:
            print(f"✅ {split_name}: All {data.num_edges:,} edges have valid indices")
            return True
    
    # Validate all splits
    train_valid = quick_validate_edges(train_data, "Train")
    val_valid = quick_validate_edges(val_data, "Validation") 
    test_valid = quick_validate_edges(test_data, "Test")
    
    if not (train_valid and val_valid and test_valid):
        print("❌ Invalid edge indices detected!")
        print("🔧 Solution: Run the fixed preprocessing first:")
        print("   1. Use colab_preprocess_fixed.py to create fixed data")
        print("   2. Then run training again")
        raise ValueError("Invalid edge indices - run fixed preprocessing first")
    
    return train_data, val_data, test_data

def calculate_class_weights(train_data):
    """Calculate class weights for imbalanced data"""
    labels = train_data.y.cpu().numpy()
    pos_count = np.sum(labels)
    neg_count = len(labels) - pos_count
    
    pos_weight = len(labels) / (2 * pos_count)
    neg_weight = len(labels) / (2 * neg_count)
    
    print(f"Class distribution: {neg_count:,} negative, {pos_count:,} positive ({pos_count/len(labels)*100:.2f}% positive)")
    
    return torch.tensor([neg_weight, pos_weight], dtype=torch.float32)

def train_model(model, train_data, val_data, device, epochs=50, lr=0.001):
    """Train a single model with memory optimization and debugging"""
    print(f"🚀 Training {model.__class__.__name__} on {device}")
    
    # DEBUGGING: Check data sizes before moving to GPU
    print(f"🔍 Data sizes before GPU transfer:")
    print(f"  Train edges: {train_data.num_edges:,}")
    print(f"  Val edges: {val_data.num_edges:,}")
    print(f"  Node features: {train_data.x.shape}")
    print(f"  Edge features: {train_data.edge_attr.shape}")
    
    # DEBUGGING: Check GPU memory before transfer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1024**3
        print(f"  GPU memory before transfer: {memory_before:.2f} GB")
    
    # Move to device with error handling
    try:
        print(f"🔄 Moving model to {device}...")
        model = model.to(device)
        
        if device.type == 'cuda':
            memory_after_model = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU memory after model: {memory_after_model:.2f} GB")
        
        print(f"🔄 Moving training data to {device}...")
        train_data = train_data.to(device)
        
        if device.type == 'cuda':
            memory_after_train = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU memory after train data: {memory_after_train:.2f} GB")
        
        print(f"🔄 Moving validation data to {device}...")
        val_data = val_data.to(device)
        
        if device.type == 'cuda':
            memory_after_val = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU memory after val data: {memory_after_val:.2f} GB")
            
    except Exception as e:
        print(f"❌ Error moving data to {device}: {e}")
        print(f"🔄 Falling back to CPU...")
        device = torch.device('cpu')
        model = model.cpu()
        train_data = train_data.cpu()
        val_data = val_data.cpu()
    
    # Setup training
    class_weights = calculate_class_weights(train_data).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_val_f1 = 0
    patience = 10
    no_improve = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # DEBUGGING: Memory check before forward pass
        if device.type == 'cuda' and epoch == 0:
            memory_before_forward = torch.cuda.memory_allocated() / 1024**3
            print(f"🔍 GPU memory before first forward pass: {memory_before_forward:.2f} GB")
        
        try:
            # Forward pass with debugging
            logits = model(train_data.x, train_data.edge_index, train_data.edge_attr)
            
            if device.type == 'cuda' and epoch == 0:
                memory_after_forward = torch.cuda.memory_allocated() / 1024**3
                print(f"🔍 GPU memory after forward pass: {memory_after_forward:.2f} GB")
                print(f"🔍 Forward pass used: {memory_after_forward - memory_before_forward:.2f} GB")
            
            loss = criterion(logits, train_data.y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ GPU OOM during training at epoch {epoch}")
                print(f"🔍 Error: {e}")
                
                # Try to recover with CPU fallback
                print(f"🔄 Attempting CPU fallback...")
                device = torch.device('cpu')
                model = model.cpu()
                train_data = train_data.cpu()
                val_data = val_data.cpu()
                
                # Retry forward pass on CPU
                logits = model(train_data.x, train_data.edge_index, train_data.edge_attr)
                loss = criterion(logits, train_data.y)
                loss.backward()
                optimizer.step()
            else:
                raise e
        
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

def evaluate_model(model, data, split_name, device):
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

def plot_results(histories, model_names):
    """Plot training results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training Loss
    for name, history in zip(model_names, histories):
        axes[0].plot(history['train_loss'], label=f'{name} Train', alpha=0.8)
        axes[0].plot(history['val_loss'], label=f'{name} Val', alpha=0.8)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation F1
    for name, history in zip(model_names, histories):
        axes[1].plot(history['val_f1'], label=name, alpha=0.8)
    axes[1].set_title('Validation F1 Score')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Final comparison
    final_f1s = [history['val_f1'][-1] for history in histories]
    bars = axes[2].bar(model_names, final_f1s, alpha=0.7)
    axes[2].set_title('Final Validation F1')
    axes[2].set_ylabel('F1 Score')
    
    # Add value labels on bars
    for bar, f1 in zip(bars, final_f1s):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("🚀 GNN TRAINING FOR AML DETECTION")
    print("="*50)
    
    # Setup
    device = setup_device()
    
    # MEMORY OPTIMIZATION: Start with single dataset if GPU memory is limited
    try:
        train_data, val_data, test_data = load_data(memory_efficient=False)  # Try both datasets first
    except Exception as e:
        print(f"⚠️ Failed to load both datasets, trying memory-efficient mode: {e}")
        train_data, val_data, test_data = load_data(memory_efficient=True)  # Fallback to single dataset
    
    node_features = train_data.x.shape[1]
    edge_features = train_data.edge_attr.shape[1]
    
    # Define models with memory-efficient settings
    models = {
        'GCN': EdgeFeatureGCN(
            node_feature_dim=node_features,
            edge_feature_dim=edge_features,
            hidden_dim=96,
            dropout=0.3
        ),
        'GAT': EdgeFeatureGAT(
            node_feature_dim=node_features,
            edge_feature_dim=edge_features,
            hidden_dim=96,
            num_heads=6,
            dropout=0.3
        ),
        'GIN': EdgeFeatureGIN(
            node_feature_dim=node_features,
            edge_feature_dim=edge_features,
            hidden_dim=96,
            dropout=0.3
        )
    }
    
    print(f"🔧 Training {len(models)} models with memory optimization")
    
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
        
        # Train
        start_time = time.time()
        trained_model, history = train_model(model, train_data, val_data, device)
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time/60:.1f} minutes")
        
        trained_models[name] = trained_model
        histories.append(history)
        model_names.append(name)
        
        # Save model
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_class': trained_model.__class__.__name__,
            'history': history
        }, MODELS_DIR / f'{name.lower()}_model.pt')
    
    # Final evaluation
    print(f"\n{'='*50}")
    print("FINAL EVALUATION")
    print(f"{'='*50}")
    
    results = {}
    for name, model in trained_models.items():
        print(f"\n{name} Model:")
        print("-" * 20)
        
        train_metrics = evaluate_model(model, train_data, "Train", device)
        val_metrics = evaluate_model(model, val_data, "Validation", device)
        test_metrics = evaluate_model(model, test_data, "Test", device)
        
        results[name] = test_metrics
    
    # Plot results
    plot_results(histories, model_names)
    
    # Summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    
    best_model = max(results.keys(), key=lambda x: results[x]['f1'])
    best_f1 = results[best_model]['f1']
    
    print(f"🏆 Best model: {best_model} (Test F1: {best_f1:.4f})")
    print(f"📁 Models saved in: {MODELS_DIR}")
    
    return trained_models, results

if __name__ == "__main__":
    trained_models, results = main()

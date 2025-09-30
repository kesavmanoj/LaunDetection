"""
Production GNN Training Script for AML Detection
Uses fixed preprocessed datasets with no invalid edge indices
Optimized for Google Colab with Tesla T4 GPU
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
        # Use only fixed preprocessing files
        file_path = GRAPHS_DIR / f'ibm_aml_{dataset}_fixed_splits.pt'
        
        if not file_path.exists():
            print(f"❌ Fixed preprocessed file not found for {dataset}: {file_path}")
            print(f"🔧 Run colab_preprocess_fixed.py first to create the fixed data")
            continue
        
        print(f"Loading {dataset}...")
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        all_train_data.append(data['train'])
        all_val_data.append(data['val'])
        all_test_data.append(data['test'])
        
        print(f"  {dataset}: {data['train'].num_edges:,} train, {data['val'].num_edges:,} val, {data['test'].num_edges:,} test")
    
    if not all_train_data:
        raise ValueError("No datasets found!")
    
    # Combine datasets safely
    def combine_or_select_first(data_list, split_name):
        if len(data_list) == 1:
            return data_list[0]

        # Safety: Do NOT naively concatenate across datasets because node sets
        # are independently remapped in preprocessing per dataset. Using the
        # first dataset avoids invalid cross-graph mixing.
        print(f"⚠️ Multiple datasets loaded for {split_name}; using the first dataset only to avoid invalid cross-graph concatenation.")
        return data_list[0]

    train_data = combine_or_select_first(all_train_data, "train")
    val_data = combine_or_select_first(all_val_data, "val")
    test_data = combine_or_select_first(all_test_data, "test")
    
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

def calculate_class_weights(train_data, boost_factor=1.0):
    """Calculate class weights for imbalanced data with zero-division guards"""
    labels = train_data.y.detach().cpu().numpy()
    total = len(labels)
    pos_count = int(np.sum(labels))
    neg_count = int(total - pos_count)

    # Default to equal weights if any class is missing
    if pos_count == 0 or neg_count == 0 or total == 0:
        print("⚠️ Degenerate class distribution detected; using equal class weights [1.0, 1.0].")
        weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
    else:
        pos_weight = total / (2.0 * pos_count)
        neg_weight = total / (2.0 * neg_count)
        # Optionally boost the positive class weight to fight extreme imbalance
        pos_weight *= boost_factor
        weights = torch.tensor([neg_weight, pos_weight], dtype=torch.float32)

    pos_ratio = (pos_count / total * 100.0) if total > 0 else 0.0
    print(f"Class distribution: {neg_count:,} negative, {pos_count:,} positive ({pos_ratio:.2f}% positive)")
    return weights

def train_model_with_memory_management(model, train_data, val_data, epochs=50, lr=0.001, device='cuda', max_edges_per_batch=1000000, use_focal_loss=True):
    """Train a single model with memory optimization and debugging"""
    print(f"🚀 Training {model.__class__.__name__} on {device}")
    
    # DEBUGGING: Check data sizes before moving to GPU
    print(f"🔍 Data sizes before GPU transfer:")
    print(f"  Train edges: {train_data.num_edges:,}")
    print(f"  Val edges: {val_data.num_edges:,}")
    print(f"  Node features: {train_data.x.shape}")
    print(f"  Edge features: {train_data.edge_attr.shape}")
    
    # DEBUGGING: Check GPU memory before transfer
    device_obj = torch.device(device) if isinstance(device, str) else device
    if str(device_obj).startswith('cuda'):
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1024**3
        print(f"  GPU memory before transfer: {memory_before:.2f} GB")
    
    # Move to device with error handling
    try:
        print(f"🔄 Moving model to {device}...")
        model = model.to(device)
        
        if str(device_obj).startswith('cuda'):
            memory_after_model = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU memory after model: {memory_after_model:.2f} GB")
        
        print(f"🔄 Moving training data to {device}...")
        train_data = train_data.to(device)
        
        if str(device_obj).startswith('cuda'):
            memory_after_train = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU memory after train data: {memory_after_train:.2f} GB")
        
        # Keep validation data on CPU to minimize peak GPU memory; we'll move it during evaluation only
        print(f"🔄 Keeping validation data on CPU to reduce GPU memory usage")
            
    except Exception as e:
        print(f"❌ Error moving data to {device}: {e}")
        print(f"🔄 Falling back to CPU...")
        device = torch.device('cpu')
        model = model.cpu()
        train_data = train_data.cpu()
        val_data = val_data.cpu()
    
    # Setup training
    class_weights = calculate_class_weights(train_data, boost_factor=3.0).to(device)
    base_ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def focal_ce_loss(logits, targets, gamma=2.0):
        # per-sample CE
        ce = base_ce(logits, targets)
        # p_t = exp(-ce)
        pt = torch.exp(-ce)
        # focal modulation
        loss = ((1.0 - pt) ** gamma) * ce
        return loss.mean()

    def ce_loss(logits, targets):
        return base_ce(logits, targets).mean()

    criterion = focal_ce_loss if use_focal_loss else ce_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    use_amp = str(device_obj).startswith('cuda') and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_val_f1 = 0
    patience = 10
    no_improve = 0
    # Ensure best_state is always defined
    best_state = model.state_dict().copy()
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # DEBUGGING: Memory check before forward pass
        if str(device_obj).startswith('cuda') and epoch == 0:
            memory_before_forward = torch.cuda.memory_allocated() / 1024**3
            print(f"🔍 GPU memory before first forward pass: {memory_before_forward:.2f} GB")
        
        try:
            # Forward pass with debugging (AMP)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(train_data.x, train_data.edge_index, train_data.edge_attr)
                if str(device_obj).startswith('cuda') and epoch == 0:
                    memory_after_forward = torch.cuda.memory_allocated() / 1024**3
                    print(f"🔍 GPU memory after forward pass: {memory_after_forward:.2f} GB")
                    print(f"🔍 Forward pass used: {memory_after_forward - memory_before_forward:.2f} GB")
                loss = criterion(logits, train_data.y)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Free large tensors promptly
            del logits
            if str(device_obj).startswith('cuda'):
                torch.cuda.empty_cache()

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
                use_amp = False
                scaler = torch.amp.GradScaler('cuda', enabled=False)
                
                # Retry forward pass on CPU
                logits = model(train_data.x, train_data.edge_index, train_data.edge_attr)
                loss = criterion(logits, train_data.y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            else:
                raise e
        
        # Validation
        model.eval()
        with torch.no_grad():
            # Move validation data to device temporarily
            val_data_device = val_data.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                val_logits = model(val_data_device.x, val_data_device.edge_index, val_data_device.edge_attr)
                val_loss = ce_loss(val_logits, val_data_device.y)  # use CE for validation loss reporting
            # Threshold tuning on validation probabilities of positive class
            val_probs = F.softmax(val_logits, dim=1)[:, 1].detach().cpu().numpy()
            y_val = val_data_device.y.cpu().numpy()
            # evaluate a small grid of thresholds for F1
            thresholds = np.linspace(0.01, 0.99, 99)
            f1s = []
            for t in thresholds:
                preds = (val_probs >= t).astype(np.int64)
                f1s.append(f1_score(y_val, preds, average='binary', zero_division=0))
            best_idx = int(np.argmax(f1s))
            best_threshold_epoch = float(thresholds[best_idx])
            val_f1 = float(f1s[best_idx])
            # Free GPU memory used by validation
            del val_data_device, val_logits
            if str(device_obj).startswith('cuda'):
                torch.cuda.empty_cache()
        
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
            best_threshold = best_threshold_epoch
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

    # Persist best threshold discovered during training to history
    if 'best_threshold' not in history:
        history['best_threshold'] = float(locals().get('best_threshold', 0.5))

    return model, history

def evaluate_model(model, data, split_name, device, threshold: float | None = None):
    """Evaluate model on a dataset split"""
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = F.softmax(logits, dim=1)
        if threshold is None:
            preds = logits.argmax(dim=1)
        else:
            preds = (probs[:, 1] >= threshold).long()
        
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
    # Reduce fragmentation before CUDA context creation
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    device = setup_device()
    
    # MEMORY OPTIMIZATION: Start with single dataset if GPU memory is limited
    try:
        train_data, val_data, test_data = load_data(memory_efficient=False)  # Try both datasets first
    except Exception as e:
        print(f"⚠️ Failed to load both datasets, trying memory-efficient mode: {e}")
        train_data, val_data, test_data = load_data(memory_efficient=True)  # Fallback to single dataset
    
    node_features = train_data.x.shape[1]
    edge_features = train_data.edge_attr.shape[1]
    
    # Define models with memory-aware sizing
    # Reduce model size for large graphs to prevent GPU OOM
    if train_data.num_edges > 3000000:  # Large graph
        hidden_dim = 64
        num_heads = 2
        print(f"🔧 Large graph detected ({train_data.num_edges:,} edges), using smaller models")
    else:
        hidden_dim = 128
        num_heads = 4
        print(f"🔧 Standard graph size ({train_data.num_edges:,} edges), using full-size models")
    
    # Further shrink GAT on very large graphs to avoid OOM
    gat_hidden_dim = hidden_dim if train_data.num_edges <= 3000000 else 32
    gat_heads = num_heads if train_data.num_edges <= 3000000 else 1
    gat_layers = (2 if train_data.num_edges > 3000000 else 3)
    if train_data.num_edges > 3000000:
        gat_layers = 1
    # Memory-safe GCN hidden size for large graphs
    gcn_hidden_dim = hidden_dim if train_data.num_edges <= 3000000 else 32

    models = {
        'GCN': EdgeFeatureGCN(
            node_feature_dim=node_features,
            edge_feature_dim=edge_features,
            hidden_dim=gcn_hidden_dim,
            dropout=0.3,
            use_edge_features=True
        ),
        'GAT': EdgeFeatureGAT(
            node_feature_dim=node_features,
            edge_feature_dim=edge_features,
            hidden_dim=gat_hidden_dim,
            num_heads=gat_heads,
            num_layers=gat_layers,
            dropout=0.3,
            use_edge_features=True
        ),
        'GIN': EdgeFeatureGIN(
            node_feature_dim=node_features,
            edge_feature_dim=edge_features,
            hidden_dim=hidden_dim,
            dropout=0.3,
            use_edge_features=True,
            aggregation='sum'
        )
    }
    
    print(f"🔧 Training {len(models)} models with memory optimization")
    
    # Train all models
    trained_models = {}
    histories = []
    model_names = []
    
    for model_name, model in models.items():
        print(f"\n{'='*40}")
        print(f"TRAINING {model_name.upper()} MODEL")
        print(f"{'='*40}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Clear GPU cache before training each model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f" GPU cache cleared before {model_name}")

        trained_model = None
        history = None
        try:
            # Train model
            # Decide per-model device based on free GPU memory
            training_device = 'cpu'
            if torch.cuda.is_available():
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    free_gb = free_bytes / 1024**3
                except Exception:
                    free_gb = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024**3
                # Per-model minimum free memory requirement
                if model_name == 'GAT':
                    min_needed_gb = 2.0 if train_data.num_edges > 3000000 else 1.0
                elif model_name == 'GCN':
                    min_needed_gb = 1.5 if train_data.num_edges > 3000000 else 1.0
                else:
                    min_needed_gb = 1.0
                if free_gb >= min_needed_gb:
                    training_device = 'cuda'
                else:
                    print(f"⚠️ Low free GPU memory ({free_gb:.2f} GB). Training {model_name} on CPU.")

            trained_model, history = train_model_with_memory_management(
                model, train_data, val_data, epochs=30, lr=0.001, device=training_device
            )

            trained_models[model_name] = trained_model
            histories.append(history)
            model_names.append(model_name)

            # Save model in a timestamped directory with richer metadata
            run_id = time.strftime('%Y%m%d_%H%M%S')
            model_dir = MODELS_DIR / f"{model_name.lower()}_{run_id}"
            model_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = model_dir / 'checkpoint.pt'
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'model_class': trained_model.__class__.__name__,
                'hyperparameters': {
                    'node_feature_dim': node_features,
                    'edge_feature_dim': edge_features,
                    'hidden_dim': hidden_dim,
                    'num_heads': num_heads if model_name == 'GAT' else None,
                    'num_layers': (2 if train_data.num_edges > 3000000 else 3) if model_name == 'GAT' else None,
                    'dropout': 0.3,
                    'aggregation': 'sum' if model_name == 'GIN' else None,
                    'lr': 0.001,
                    'epochs': 30
                },
                'history': history,
                'run_id': run_id
            }, checkpoint_path)
            print(f" Saved {model_name} checkpoint to {checkpoint_path}.")
        finally:
            # Move trained model to CPU and free GPU memory before starting next model
            try:
                # Safely obtain local trained_model without referencing an unbound local name
                tm = locals().get('trained_model', None)
                if tm is not None:
                    trained_models[model_name] = tm.cpu()
            except Exception as e:
                print(f"⚠️ Could not move {model_name} to CPU: {e}")
            # Explicitly drop local refs
            try:
                del trained_model
            except Exception:
                pass
            try:
                del history
            except Exception:
                pass
            try:
                del model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                mem_res = torch.cuda.memory_reserved() / 1024**3
                print(f" GPU cache cleared after {model_name} | Alloc: {mem_alloc:.2f} GB, Reserved: {mem_res:.2f} GB")
    
    # Final evaluation
    print(f"\n{'='*50}")
    print("FINAL EVALUATION")
    print(f"{'='*50}")
    
    results = {}
    for name, model in trained_models.items():
        print(f"\n{name} Model:")
        print("-" * 20)
        
        # Force evaluation on CPU to avoid device mismatches after CPU cleanup
        eval_device = torch.device('cpu')
        model_cpu = model.to(eval_device)
        
        # Use tuned threshold from training history if available
        best_threshold = None
        try:
            idx = model_names.index(name)
            h = histories[idx]
            if isinstance(h, dict) and 'best_threshold' in h:
                best_threshold = float(h['best_threshold'])
                print(f" Using tuned threshold for {name}: {best_threshold:.2f}")
        except Exception:
            pass

        train_metrics = evaluate_model(model_cpu, train_data, "Train", eval_device, threshold=best_threshold)
        val_metrics = evaluate_model(model_cpu, val_data, "Validation", eval_device, threshold=best_threshold)
        test_metrics = evaluate_model(model_cpu, test_data, "Test", eval_device, threshold=best_threshold)
        
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

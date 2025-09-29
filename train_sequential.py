"""
Sequential Training Script for AML Detection
Train on one dataset first, then fine-tune on the second dataset
This approach uses transfer learning for better performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
import time
from torch_geometric.data import Data
from gnn_models import EdgeFeatureGCN, EdgeFeatureGAT, EdgeFeatureGIN

# Setup paths
BASE_DIR = Path('/content/drive/MyDrive/LaunDetection')
GRAPHS_DIR = BASE_DIR / 'data' / 'graphs'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

def load_single_dataset(dataset_name):
    """Load a single dataset"""
    print(f"📂 Loading {dataset_name} dataset...")
    
    file_path = GRAPHS_DIR / f'ibm_aml_{dataset_name}_fixed_splits.pt'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    
    train_data = data['train']
    val_data = data['val'] 
    test_data = data['test']
    
    print(f"  {dataset_name}: {train_data.num_edges:,} train, {val_data.num_edges:,} val, {test_data.num_edges:,} test")
    print(f"  Nodes: {train_data.num_nodes:,}, Node features: {train_data.x.shape[1]}, Edge features: {train_data.edge_attr.shape[1]}")
    
    return train_data, val_data, test_data

def validate_edge_indices(data, split_name, dataset_name):
    """Validate edge indices for a dataset"""
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    max_idx = edge_index.max().item()
    min_idx = edge_index.min().item()
    
    if min_idx < 0 or max_idx >= num_nodes:
        print(f"❌ {dataset_name} {split_name}: Invalid edge indices [{min_idx}, {max_idx}] for {num_nodes} nodes")
        return False
    else:
        print(f"✅ {dataset_name} {split_name}: Valid edge indices [0, {max_idx}] for {num_nodes:,} nodes")
        return True

def calculate_class_weights(train_data):
    """Calculate class weights for imbalanced data"""
    labels = train_data.y.cpu().numpy()
    pos_count = np.sum(labels)
    neg_count = len(labels) - pos_count
    
    if pos_count == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    
    pos_weight = len(labels) / (2 * pos_count)
    neg_weight = len(labels) / (2 * neg_count)
    
    print(f"Class distribution: {neg_count:,} negative, {pos_count:,} positive ({pos_count/len(labels)*100:.3f}% positive)")
    
    return torch.tensor([neg_weight, pos_weight], dtype=torch.float32)

def train_model_on_dataset(model, train_data, val_data, epochs=30, lr=0.001, dataset_name="", is_fine_tuning=False):
    """Train a model on a single dataset"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 {'Fine-tuning' if is_fine_tuning else 'Training'} {model.__class__.__name__} on {dataset_name} using {device}")
    
    # Move data to device with error handling
    try:
        model = model.to(device)
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        
        if device.type == 'cuda':
            print(f"  GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"⚠️ GPU OOM, falling back to CPU...")
            device = torch.device('cpu')
            model = model.cpu()
            train_data = train_data.cpu()
            val_data = val_data.cpu()
        else:
            raise e
    
    # Setup training
    class_weights = calculate_class_weights(train_data).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use different learning rates for initial training vs fine-tuning
    learning_rate = lr * 0.1 if is_fine_tuning else lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"  Learning rate: {learning_rate}")
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_val_f1 = 0
    patience = 10
    no_improve = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        try:
            # Forward pass
            logits = model(train_data.x, train_data.edge_index, train_data.edge_attr)
            loss = criterion(logits, train_data.y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(val_data.x, val_data.edge_index, val_data.edge_attr)
                val_loss = criterion(val_logits, val_data.y)
                
                # Calculate metrics
                val_pred = val_logits.argmax(dim=1).cpu().numpy()
                val_true = val_data.y.cpu().numpy()
                
                val_f1 = f1_score(val_true, val_pred)
                val_acc = accuracy_score(val_true, val_pred)
                
            # Update history
            history['train_loss'].append(loss.item())
            history['val_loss'].append(val_loss.item())
            history['val_f1'].append(val_f1)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Progress reporting
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1:2d}/{epochs}: Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
                    
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ GPU OOM during training at epoch {epoch}")
                print(f"🔄 Switching to CPU...")
                device = torch.device('cpu')
                model = model.cpu()
                train_data = train_data.cpu()
                val_data = val_data.cpu()
                class_weights = class_weights.cpu()
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                continue
            else:
                raise e
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"  ✅ Best validation F1: {best_val_f1:.4f}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model, history, best_val_f1

def evaluate_model(model, test_data, dataset_name=""):
    """Evaluate model on test data"""
    device = next(model.parameters()).device
    test_data = test_data.to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(test_data.x, test_data.edge_index, test_data.edge_attr)
        pred = logits.argmax(dim=1).cpu().numpy()
        true = test_data.y.cpu().numpy()
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    
    # Calculate metrics
    acc = accuracy_score(true, pred)
    precision = precision_score(true, pred, zero_division=0)
    recall = recall_score(true, pred, zero_division=0)
    f1 = f1_score(true, pred, zero_division=0)
    
    try:
        auc = roc_auc_score(true, probs)
    except:
        auc = 0.0
    
    print(f"📊 {dataset_name} Test Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def sequential_training():
    """Main sequential training function"""
    
    print("🎯 SEQUENTIAL TRAINING FOR AML DETECTION")
    print("="*60)
    print("Strategy: Train on LI-Small first, then fine-tune on HI-Small")
    print("="*60)
    
    # Load datasets
    try:
        li_train, li_val, li_test = load_single_dataset('li-small')
        hi_train, hi_val, hi_test = load_single_dataset('hi-small')
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("🔧 Run colab_preprocess_fixed.py first")
        return None, None
    
    # Validate edge indices
    print(f"\n🔍 Validating edge indices...")
    datasets_valid = True
    for data, split, dataset in [(li_train, 'train', 'LI-Small'), (li_val, 'val', 'LI-Small'), (li_test, 'test', 'LI-Small'),
                                 (hi_train, 'train', 'HI-Small'), (hi_val, 'val', 'HI-Small'), (hi_test, 'test', 'HI-Small')]:
        if not validate_edge_indices(data, split, dataset):
            datasets_valid = False
    
    if not datasets_valid:
        print("❌ Invalid edge indices detected! Run preprocessing again.")
        return None, None
    
    # Define models - using smaller sizes for memory efficiency
    models = {
        'GCN': EdgeFeatureGCN(
            node_feature_dim=10,
            edge_feature_dim=10,
            hidden_dim=64,
            dropout=0.3,
            use_edge_features=True
        ),
        'GAT': EdgeFeatureGAT(
            node_feature_dim=10,
            edge_feature_dim=10,
            hidden_dim=64,
            num_heads=4,
            dropout=0.3,
            use_edge_features=True
        ),
        'GIN': EdgeFeatureGIN(
            node_feature_dim=10,
            edge_feature_dim=10,
            hidden_dim=64,
            dropout=0.3,
            use_edge_features=True,
            aggregation='sum'
        )
    }
    
    results = {}
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"SEQUENTIAL TRAINING: {model_name.upper()}")
        print(f"{'='*50}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Stage 1: Train on LI-Small
        print(f"\n🎯 Stage 1: Training on LI-Small...")
        model_stage1, history1, best_f1_stage1 = train_model_on_dataset(
            model, li_train, li_val, epochs=25, lr=0.001, 
            dataset_name="LI-Small", is_fine_tuning=False
        )
        
        # Evaluate on LI-Small test set
        li_results = evaluate_model(model_stage1, li_test, "LI-Small")
        
        # Stage 2: Fine-tune on HI-Small
        print(f"\n🎯 Stage 2: Fine-tuning on HI-Small...")
        model_final, history2, best_f1_stage2 = train_model_on_dataset(
            model_stage1, hi_train, hi_val, epochs=15, lr=0.001,
            dataset_name="HI-Small", is_fine_tuning=True
        )
        
        # Evaluate on HI-Small test set
        hi_results = evaluate_model(model_final, hi_test, "HI-Small")
        
        # Test cross-dataset generalization
        print(f"\n🔄 Cross-dataset evaluation...")
        li_cross_results = evaluate_model(model_final, li_test, "LI-Small (after HI-Small fine-tuning)")
        
        # Store results
        results[model_name] = {
            'li_small_results': li_results,
            'hi_small_results': hi_results,
            'li_cross_results': li_cross_results,
            'stage1_best_f1': best_f1_stage1,
            'stage2_best_f1': best_f1_stage2,
            'history_stage1': history1,
            'history_stage2': history2
        }
        
        trained_models[model_name] = model_final
        
        # Save model
        model_path = MODELS_DIR / f'{model_name.lower()}_sequential_model.pt'
        torch.save({
            'model_state_dict': model_final.state_dict(),
            'model_class': model_final.__class__.__name__,
            'results': results[model_name]
        }, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*80}")
    
    for model_name, result in results.items():
        print(f"\n{model_name} Model:")
        print(f"  LI-Small Test F1: {result['li_small_results']['f1']:.4f}")
        print(f"  HI-Small Test F1: {result['hi_small_results']['f1']:.4f}")
        print(f"  Cross-dataset F1: {result['li_cross_results']['f1']:.4f}")
        print(f"  Stage 1 Best Val F1: {result['stage1_best_f1']:.4f}")
        print(f"  Stage 2 Best Val F1: {result['stage2_best_f1']:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), 
                         key=lambda x: (results[x]['hi_small_results']['f1'] + results[x]['li_cross_results']['f1']) / 2)
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Average F1: {(results[best_model_name]['hi_small_results']['f1'] + results[best_model_name]['li_cross_results']['f1']) / 2:.4f}")
    
    return trained_models, results

if __name__ == "__main__":
    trained_models, results = sequential_training()

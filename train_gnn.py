"""
Training Script for GNN Models on AML Transaction Classification

This script provides comprehensive training functionality for GNN models
on the preprocessed IBM AML datasets (LI-Small and HI-Small).

Features:
- Support for all three GNN architectures (GCN, GAT, GIN)
- Advanced training techniques (learning rate scheduling, early stopping)
- Comprehensive evaluation metrics
- Model checkpointing and logging
- Class imbalance handling
- GPU/CPU compatibility
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from gnn_models import get_model, model_summary
from config import Config

class AMLTrainer:
    """
    Comprehensive trainer for GNN models on AML transaction classification
    
    Handles:
    - Model training with advanced optimization techniques
    - Evaluation with comprehensive metrics
    - Class imbalance through weighted loss and sampling
    - Model checkpointing and experiment tracking
    - Visualization of training progress and results
    """
    
    def __init__(self, 
                 model_type='gcn',
                 model_params=None,
                 learning_rate=0.001,
                 weight_decay=1e-4,
                 batch_size=None,  # None for full-batch training
                 epochs=200,
                 patience=20,
                 device=None,
                 class_weights=None,
                 use_scheduler=True,
                 scheduler_type='plateau',
                 log_level='INFO'):
        """
        Initialize AML Trainer
        
        Args:
            model_type: Type of GNN model ('gcn', 'gat', 'gin')
            model_params: Dictionary of model-specific parameters
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            batch_size: Batch size (None for full-batch)
            epochs: Maximum training epochs
            patience: Early stopping patience
            device: Training device
            class_weights: Weights for class imbalance (None for auto-calculation)
            use_scheduler: Whether to use learning rate scheduling
            scheduler_type: Type of scheduler ('plateau', 'cosine')
            log_level: Logging level
        """
        
        self.model_type = model_type.lower()
        self.model_params = model_params or {}
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.class_weights = class_weights
        self.use_scheduler = use_scheduler
        self.scheduler_type = scheduler_type
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'train_acc': [], 'val_acc': [], 'test_acc': [],
            'train_f1': [], 'val_f1': [], 'test_f1': [],
            'train_auc': [], 'val_auc': [], 'test_auc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        self.logger.info(f"Initialized AMLTrainer for {model_type.upper()} model")
    
    def _setup_logging(self, log_level):
        """Setup logging configuration"""
        log_dir = Config.LOGS_DIR
        log_file = log_dir / f"training_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def load_data(self, dataset_names=['LI-Small', 'HI-Small']):
        """
        Load preprocessed data for training
        
        Args:
            dataset_names: List of dataset names to load
            
        Returns:
            Dictionary containing train/val/test data
        """
        self.logger.info(f"Loading datasets: {dataset_names}")
        
        all_data = {'train': [], 'val': [], 'test': []}
        dataset_stats = {}
        
        for dataset_name in dataset_names:
            # Load preprocessed data
            dataset_file = Config.GRAPHS_DIR / f'ibm_aml_{dataset_name.lower()}_enhanced_splits.pt'
            
            if not dataset_file.exists():
                self.logger.error(f"Dataset file not found: {dataset_file}")
                continue
                
            self.logger.info(f"Loading {dataset_name} from {dataset_file}")
            data = torch.load(dataset_file, map_location='cpu')
            
            # Extract splits
            train_data = data['train']
            val_data = data['val']
            test_data = data['test']
            metadata = data['metadata']
            
            # Move to device
            train_data = train_data.to(self.device)
            val_data = val_data.to(self.device)
            test_data = test_data.to(self.device)
            
            all_data['train'].append(train_data)
            all_data['val'].append(val_data)
            all_data['test'].append(test_data)
            
            # Store dataset statistics
            dataset_stats[dataset_name] = {
                'nodes': metadata['num_nodes'],
                'train_edges': train_data.num_edges,
                'val_edges': val_data.num_edges,
                'test_edges': test_data.num_edges,
                'node_features': metadata['node_feature_dim'],
                'edge_features': metadata['edge_feature_dim'],
                'positive_rate_train': train_data.y.float().mean().item(),
                'positive_rate_val': val_data.y.float().mean().item(),
                'positive_rate_test': test_data.y.float().mean().item()
            }
            
            self.logger.info(f"{dataset_name} loaded: {train_data.num_edges:,} train, "
                           f"{val_data.num_edges:,} val, {test_data.num_edges:,} test edges")
        
        if not all_data['train']:
            raise ValueError("No datasets loaded successfully!")
        
        # Combine datasets if multiple
        if len(all_data['train']) > 1:
            self.logger.info("Combining multiple datasets...")
            
            # Combine train data
            combined_train = self._combine_data(all_data['train'])
            combined_val = self._combine_data(all_data['val'])
            combined_test = self._combine_data(all_data['test'])
            
            self.data = {
                'train': combined_train,
                'val': combined_val,
                'test': combined_test
            }
        else:
            self.data = {
                'train': all_data['train'][0],
                'val': all_data['val'][0],
                'test': all_data['test'][0]
            }
        
        # Store feature dimensions
        self.node_feature_dim = self.data['train'].x.shape[1]
        self.edge_feature_dim = self.data['train'].edge_attr.shape[1]
        
        # Calculate class weights if not provided
        if self.class_weights is None:
            self._calculate_class_weights()
        
        # Print dataset summary
        self._print_dataset_summary(dataset_stats)
        
        return self.data
    
    def _combine_data(self, data_list):
        """Combine multiple Data objects into one"""
        if len(data_list) == 1:
            return data_list[0]
        
        # Combine node features (assuming same nodes across datasets)
        x = data_list[0].x
        
        # Combine edge data
        edge_indices = []
        edge_attrs = []
        labels = []
        
        for data in data_list:
            edge_indices.append(data.edge_index)
            edge_attrs.append(data.edge_attr)
            labels.append(data.y)
        
        combined_edge_index = torch.cat(edge_indices, dim=1)
        combined_edge_attr = torch.cat(edge_attrs, dim=0)
        combined_y = torch.cat(labels, dim=0)
        
        return Data(x=x, edge_index=combined_edge_index, 
                   edge_attr=combined_edge_attr, y=combined_y)
    
    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced data"""
        train_labels = self.data['train'].y.cpu().numpy()
        pos_count = np.sum(train_labels)
        neg_count = len(train_labels) - pos_count
        
        # Inverse frequency weighting
        pos_weight = len(train_labels) / (2 * pos_count)
        neg_weight = len(train_labels) / (2 * neg_count)
        
        self.class_weights = torch.tensor([neg_weight, pos_weight], 
                                        dtype=torch.float32, device=self.device)
        
        self.logger.info(f"Calculated class weights: [neg: {neg_weight:.3f}, pos: {pos_weight:.3f}]")
        self.logger.info(f"Class distribution: {neg_count:,} negative, {pos_count:,} positive "
                        f"({pos_count/len(train_labels)*100:.2f}% positive)")
    
    def _print_dataset_summary(self, dataset_stats):
        """Print comprehensive dataset summary"""
        self.logger.info("\n" + "="*60)
        self.logger.info("DATASET SUMMARY")
        self.logger.info("="*60)
        
        total_train = sum(stats['train_edges'] for stats in dataset_stats.values())
        total_val = sum(stats['val_edges'] for stats in dataset_stats.values())
        total_test = sum(stats['test_edges'] for stats in dataset_stats.values())
        
        self.logger.info(f"Combined dataset:")
        self.logger.info(f"  Train edges: {total_train:,}")
        self.logger.info(f"  Val edges: {total_val:,}")
        self.logger.info(f"  Test edges: {total_test:,}")
        self.logger.info(f"  Node features: {self.node_feature_dim}")
        self.logger.info(f"  Edge features: {self.edge_feature_dim}")
        
        self.logger.info(f"\nIndividual datasets:")
        for name, stats in dataset_stats.items():
            self.logger.info(f"  {name}:")
            self.logger.info(f"    Nodes: {stats['nodes']:,}")
            self.logger.info(f"    Train: {stats['train_edges']:,} edges "
                           f"({stats['positive_rate_train']*100:.2f}% positive)")
            self.logger.info(f"    Val: {stats['val_edges']:,} edges "
                           f"({stats['positive_rate_val']*100:.2f}% positive)")
            self.logger.info(f"    Test: {stats['test_edges']:,} edges "
                           f"({stats['positive_rate_test']*100:.2f}% positive)")
        
        self.logger.info("="*60)
    
    def create_model(self):
        """Create and initialize the GNN model"""
        # Default model parameters
        default_params = {
            'node_feature_dim': self.node_feature_dim,
            'edge_feature_dim': self.edge_feature_dim,
            'hidden_dim': 128,
            'num_classes': 2,
            'dropout': 0.3,
            'use_edge_features': True
        }
        
        # Update with user-provided parameters
        model_params = {**default_params, **self.model_params}
        
        # Create model
        self.model = get_model(self.model_type, **model_params)
        self.model = self.model.to(self.device)
        
        # Print model summary
        model_summary(self.model, f"{self.model_type.upper()} Model")
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Setup loss function with class weights
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Setup scheduler
        if self.use_scheduler:
            if self.scheduler_type == 'plateau':
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
                )
            elif self.scheduler_type == 'cosine':
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=self.epochs, eta_min=1e-6
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
        
        self.logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} parameters")
        
        return self.model
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        train_data = self.data['train']
        
        # Forward pass
        self.optimizer.zero_grad()
        logits = self.model(train_data.x, train_data.edge_index, train_data.edge_attr)
        loss = self.criterion(logits, train_data.y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            acc = accuracy_score(train_data.y.cpu(), preds.cpu())
            f1 = f1_score(train_data.y.cpu(), preds.cpu(), average='binary')
            auc = roc_auc_score(train_data.y.cpu(), probs[:, 1].cpu())
        
        return loss.item(), acc, f1, auc
    
    def evaluate(self, split='val'):
        """Evaluate model on validation or test set"""
        self.model.eval()
        
        data = self.data[split]
        
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index, data.edge_attr)
            loss = self.criterion(logits, data.y)
            
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # Calculate metrics
            acc = accuracy_score(data.y.cpu(), preds.cpu())
            precision = precision_score(data.y.cpu(), preds.cpu(), average='binary')
            recall = recall_score(data.y.cpu(), preds.cpu(), average='binary')
            f1 = f1_score(data.y.cpu(), preds.cpu(), average='binary')
            auc = roc_auc_score(data.y.cpu(), probs[:, 1].cpu())
        
        return {
            'loss': loss.item(),
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': preds.cpu(),
            'probabilities': probs.cpu(),
            'labels': data.y.cpu()
        }
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.epochs} epochs...")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc, train_f1, train_auc = self.train_epoch()
            
            # Validation
            val_metrics = self.evaluate('val')
            
            # Test evaluation (for monitoring)
            test_metrics = self.evaluate('test')
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['test_loss'].append(test_metrics['loss'])
            
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['test_acc'].append(test_metrics['accuracy'])
            
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['test_f1'].append(test_metrics['f1'])
            
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['test_auc'].append(test_metrics['auc'])
            
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            if self.scheduler:
                if self.scheduler_type == 'plateau':
                    self.scheduler.step(val_metrics['f1'])
                else:
                    self.scheduler.step()
            
            # Early stopping check
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint('best_model.pt')
            else:
                self.epochs_without_improvement += 1
            
            # Logging
            epoch_time = time.time() - epoch_start
            
            if epoch % 10 == 0 or epoch < 10:
                self.logger.info(
                    f"Epoch {epoch:3d}/{self.epochs} | "
                    f"Train: Loss={train_loss:.4f}, F1={train_f1:.4f} | "
                    f"Val: Loss={val_metrics['loss']:.4f}, F1={val_metrics['f1']:.4f} | "
                    f"Test: F1={test_metrics['f1']:.4f} | "
                    f"LR={self.optimizer.param_groups[0]['lr']:.6f} | "
                    f"Time={epoch_time:.1f}s"
                )
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch} (patience={self.patience})")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.1f} minutes")
        self.logger.info(f"Best validation F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        
        # Load best model for final evaluation
        self.load_checkpoint('best_model.pt')
        
        return self.history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = Config.MODELS_DIR / filename
        Config.MODELS_DIR.mkdir(exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.best_epoch,
            'best_val_f1': self.best_val_f1,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint_path = Config.MODELS_DIR / filename
        
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_f1 = checkpoint['best_val_f1']
        self.best_epoch = checkpoint['epoch']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def final_evaluation(self):
        """Comprehensive final evaluation"""
        self.logger.info("\n" + "="*60)
        self.logger.info("FINAL EVALUATION")
        self.logger.info("="*60)
        
        results = {}
        
        for split in ['train', 'val', 'test']:
            metrics = self.evaluate(split)
            results[split] = metrics
            
            self.logger.info(f"\n{split.upper()} Results:")
            self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            self.logger.info(f"  Precision: {metrics['precision']:.4f}")
            self.logger.info(f"  Recall:    {metrics['recall']:.4f}")
            self.logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
            self.logger.info(f"  AUC-ROC:   {metrics['auc']:.4f}")
        
        self.logger.info("="*60)
        
        return results
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train', alpha=0.8)
        axes[0, 0].plot(self.history['val_loss'], label='Validation', alpha=0.8)
        axes[0, 0].plot(self.history['test_loss'], label='Test', alpha=0.8)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train', alpha=0.8)
        axes[0, 1].plot(self.history['val_acc'], label='Validation', alpha=0.8)
        axes[0, 1].plot(self.history['test_acc'], label='Test', alpha=0.8)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score plot
        axes[1, 0].plot(self.history['train_f1'], label='Train', alpha=0.8)
        axes[1, 0].plot(self.history['val_f1'], label='Validation', alpha=0.8)
        axes[1, 0].plot(self.history['test_f1'], label='Test', alpha=0.8)
        axes[1, 0].axvline(x=self.best_epoch, color='red', linestyle='--', alpha=0.7, label='Best Epoch')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC plot
        axes[1, 1].plot(self.history['train_auc'], label='Train', alpha=0.8)
        axes[1, 1].plot(self.history['val_auc'], label='Validation', alpha=0.8)
        axes[1, 1].plot(self.history['test_auc'], label='Test', alpha=0.8)
        axes[1, 1].set_title('AUC-ROC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC-ROC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved: {save_path}")
        
        plt.show()

# Example usage and training script
def main():
    """Main training function"""
    
    # Training configuration
    config = {
        'model_type': 'gcn',  # Change to 'gat' or 'gin' for other models
        'model_params': {
            'hidden_dim': 128,
            'dropout': 0.3,
            'use_edge_features': True
        },
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 200,
        'patience': 20,
        'use_scheduler': True,
        'scheduler_type': 'plateau'
    }
    
    # Initialize trainer
    trainer = AMLTrainer(**config)
    
    # Load data
    trainer.load_data(['LI-Small', 'HI-Small'])
    
    # Create model
    trainer.create_model()
    
    # Train model
    history = trainer.train()
    
    # Final evaluation
    results = trainer.final_evaluation()
    
    # Plot training history
    plot_path = Config.MODELS_DIR / f"training_history_{config['model_type']}.png"
    trainer.plot_training_history(save_path=plot_path)
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()

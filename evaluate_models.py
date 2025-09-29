"""
Model Evaluation and Comparison Script for GNN-based AML Detection

This script provides comprehensive evaluation and comparison of different GNN models
on the AML transaction classification task. It includes detailed performance analysis,
visualization, and statistical significance testing.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from scipy import stats
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from train_gnn import AMLTrainer
from config import Config

class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison framework
    
    Features:
    - Multi-model comparison
    - Statistical significance testing
    - Detailed performance analysis
    - Visualization of results
    - Error analysis and interpretability
    """
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.models = {}
        self.data = None
        
    def load_data(self, dataset_names=['LI-Small', 'HI-Small']):
        """Load evaluation data"""
        print(f"Loading datasets: {dataset_names}")
        
        # Use trainer to load data (reuse existing functionality)
        temp_trainer = AMLTrainer(device=self.device)
        self.data = temp_trainer.load_data(dataset_names)
        
        print(f"Data loaded successfully:")
        print(f"  Train: {self.data['train'].num_edges:,} edges")
        print(f"  Val: {self.data['val'].num_edges:,} edges") 
        print(f"  Test: {self.data['test'].num_edges:,} edges")
        
        return self.data
    
    def load_trained_model(self, model_type, checkpoint_path=None):
        """Load a trained model for evaluation"""
        if checkpoint_path is None:
            checkpoint_path = Config.MODELS_DIR / f"best_model_{model_type}.pt"
        
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return None
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create trainer and model
        trainer = AMLTrainer(
            model_type=model_type,
            model_params=checkpoint.get('model_params', {}),
            device=self.device
        )
        
        # Load data if not already loaded
        if self.data is None:
            trainer.load_data(['LI-Small', 'HI-Small'])
            self.data = trainer.data
        else:
            trainer.data = self.data
            trainer.node_feature_dim = self.data['train'].x.shape[1]
            trainer.edge_feature_dim = self.data['train'].edge_attr.shape[1]
        
        # Create and load model
        model = trainer.create_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.models[model_type] = {
            'model': model,
            'trainer': trainer,
            'checkpoint': checkpoint
        }
        
        print(f"Loaded {model_type.upper()} model from {checkpoint_path}")
        return model
    
    def evaluate_model(self, model_type, splits=['train', 'val', 'test']):
        """Evaluate a single model on specified splits"""
        if model_type not in self.models:
            print(f"Model {model_type} not loaded!")
            return None
        
        model = self.models[model_type]['model']
        results = {}
        
        for split in splits:
            data = self.data[split]
            
            with torch.no_grad():
                # Forward pass
                logits = model(data.x, data.edge_index, data.edge_attr)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                # Convert to numpy for sklearn metrics
                y_true = data.y.cpu().numpy()
                y_pred = preds.cpu().numpy()
                y_prob = probs[:, 1].cpu().numpy()
                
                # Calculate comprehensive metrics
                metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='binary'),
                    'recall': recall_score(y_true, y_pred, average='binary'),
                    'f1': f1_score(y_true, y_pred, average='binary'),
                    'auc_roc': roc_auc_score(y_true, y_prob),
                    'auc_pr': average_precision_score(y_true, y_prob),
                    'confusion_matrix': confusion_matrix(y_true, y_pred),
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'y_prob': y_prob
                }
                
                # Additional metrics
                tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
                metrics.update({
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive Predictive Value
                    'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
                    'true_positives': tp,
                    'true_negatives': tn,
                    'false_positives': fp,
                    'false_negatives': fn
                })
                
                results[split] = metrics
        
        self.results[model_type] = results
        return results
    
    def compare_models(self, model_types=None, split='test'):
        """Compare multiple models on a specific split"""
        if model_types is None:
            model_types = list(self.models.keys())
        
        comparison_data = []
        
        for model_type in model_types:
            if model_type not in self.results:
                self.evaluate_model(model_type)
            
            metrics = self.results[model_type][split]
            
            comparison_data.append({
                'Model': model_type.upper(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC-ROC': metrics['auc_roc'],
                'AUC-PR': metrics['auc_pr'],
                'Specificity': metrics['specificity']
            })
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def statistical_significance_test(self, model1, model2, split='test', metric='f1'):
        """Test statistical significance between two models using bootstrap"""
        if model1 not in self.results or model2 not in self.results:
            print("Both models must be evaluated first!")
            return None
        
        # Get predictions
        y_true = self.results[model1][split]['y_true']
        y_pred1 = self.results[model1][split]['y_pred']
        y_pred2 = self.results[model2][split]['y_pred']
        
        # Bootstrap test
        n_bootstrap = 1000
        metric_func = {
            'accuracy': accuracy_score,
            'precision': lambda y_t, y_p: precision_score(y_t, y_p, average='binary'),
            'recall': lambda y_t, y_p: recall_score(y_t, y_p, average='binary'),
            'f1': lambda y_t, y_p: f1_score(y_t, y_p, average='binary')
        }[metric]
        
        differences = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            y_true_boot = y_true[indices]
            y_pred1_boot = y_pred1[indices]
            y_pred2_boot = y_pred2[indices]
            
            # Calculate metric difference
            score1 = metric_func(y_true_boot, y_pred1_boot)
            score2 = metric_func(y_true_boot, y_pred2_boot)
            differences.append(score1 - score2)
        
        differences = np.array(differences)
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * min(np.mean(differences >= 0), np.mean(differences <= 0))
        
        # Calculate confidence interval
        ci_lower = np.percentile(differences, 2.5)
        ci_upper = np.percentile(differences, 97.5)
        
        result = {
            'model1': model1,
            'model2': model2,
            'metric': metric,
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < 0.05
        }
        
        return result
    
    def plot_confusion_matrices(self, model_types=None, split='test', figsize=(15, 5)):
        """Plot confusion matrices for multiple models"""
        if model_types is None:
            model_types = list(self.models.keys())
        
        n_models = len(model_types)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for i, model_type in enumerate(model_types):
            cm = self.results[model_type][split]['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Legitimate', 'Laundering'],
                       yticklabels=['Legitimate', 'Laundering'],
                       ax=axes[i])
            
            axes[i].set_title(f'{model_type.upper()} - {split.title()} Set')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self, model_types=None, split='test', figsize=(10, 8)):
        """Plot ROC curves for multiple models"""
        if model_types is None:
            model_types = list(self.models.keys())
        
        plt.figure(figsize=figsize)
        
        for model_type in model_types:
            y_true = self.results[model_type][split]['y_true']
            y_prob = self.results[model_type][split]['y_prob']
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = self.results[model_type][split]['auc_roc']
            
            plt.plot(fpr, tpr, label=f'{model_type.upper()} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {split.title()} Set')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_precision_recall_curves(self, model_types=None, split='test', figsize=(10, 8)):
        """Plot Precision-Recall curves for multiple models"""
        if model_types is None:
            model_types = list(self.models.keys())
        
        plt.figure(figsize=figsize)
        
        # Calculate baseline (random classifier performance)
        y_true_sample = self.results[list(self.models.keys())[0]][split]['y_true']
        baseline = np.mean(y_true_sample)
        
        for model_type in model_types:
            y_true = self.results[model_type][split]['y_true']
            y_prob = self.results[model_type][split]['y_prob']
            
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            auc_pr = self.results[model_type][split]['auc_pr']
            
            plt.plot(recall, precision, label=f'{model_type.upper()} (AUC = {auc_pr:.3f})', linewidth=2)
        
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                   label=f'Random Classifier (AUC = {baseline:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - {split.title()} Set')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_model_comparison(self, split='test', figsize=(12, 8)):
        """Create comprehensive model comparison visualization"""
        df = self.compare_models(split=split)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Metrics to plot
        metrics = ['Accuracy', 'F1-Score', 'AUC-ROC', 'AUC-PR']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            bars = ax.bar(df['Model'], df[metric], alpha=0.7, 
                         color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(df)])
            
            # Add value labels on bars
            for bar, value in zip(bars, df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title(metric)
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def error_analysis(self, model_type, split='test', n_examples=10):
        """Analyze prediction errors for interpretability"""
        if model_type not in self.results:
            print(f"Model {model_type} not evaluated!")
            return None
        
        results = self.results[model_type][split]
        y_true = results['y_true']
        y_pred = results['y_pred']
        y_prob = results['y_prob']
        
        # Find different types of errors
        false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
        false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
        true_positives = np.where((y_true == 1) & (y_pred == 1))[0]
        true_negatives = np.where((y_true == 0) & (y_pred == 0))[0]
        
        analysis = {
            'false_positives': {
                'count': len(false_positives),
                'high_confidence': false_positives[y_prob[false_positives] > 0.8],
                'examples': false_positives[:n_examples] if len(false_positives) > 0 else []
            },
            'false_negatives': {
                'count': len(false_negatives),
                'high_confidence': false_negatives[y_prob[false_negatives] < 0.2],
                'examples': false_negatives[:n_examples] if len(false_negatives) > 0 else []
            },
            'true_positives': {
                'count': len(true_positives),
                'high_confidence': true_positives[y_prob[true_positives] > 0.8],
                'examples': true_positives[:n_examples] if len(true_positives) > 0 else []
            },
            'true_negatives': {
                'count': len(true_negatives),
                'high_confidence': true_negatives[y_prob[true_negatives] < 0.2],
                'examples': true_negatives[:n_examples] if len(true_negatives) > 0 else []
            }
        }
        
        return analysis
    
    def generate_report(self, output_path=None):
        """Generate comprehensive evaluation report"""
        if output_path is None:
            output_path = Config.MODELS_DIR / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GNN MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models evaluated: {list(self.models.keys())}\n")
            f.write(f"Device: {self.device}\n\n")
            
            # Dataset information
            f.write("DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Train edges: {self.data['train'].num_edges:,}\n")
            f.write(f"Validation edges: {self.data['val'].num_edges:,}\n")
            f.write(f"Test edges: {self.data['test'].num_edges:,}\n")
            f.write(f"Node features: {self.data['train'].x.shape[1]}\n")
            f.write(f"Edge features: {self.data['train'].edge_attr.shape[1]}\n\n")
            
            # Model comparison
            for split in ['train', 'val', 'test']:
                f.write(f"{split.upper()} SET RESULTS\n")
                f.write("-" * 40 + "\n")
                
                df = self.compare_models(split=split)
                f.write(df.to_string(index=False))
                f.write("\n\n")
            
            # Statistical significance tests
            if len(self.models) > 1:
                f.write("STATISTICAL SIGNIFICANCE TESTS (Test Set)\n")
                f.write("-" * 40 + "\n")
                
                model_types = list(self.models.keys())
                for i in range(len(model_types)):
                    for j in range(i+1, len(model_types)):
                        result = self.statistical_significance_test(
                            model_types[i], model_types[j], split='test', metric='f1'
                        )
                        
                        f.write(f"{result['model1'].upper()} vs {result['model2'].upper()}:\n")
                        f.write(f"  Mean F1 difference: {result['mean_difference']:.4f}\n")
                        f.write(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]\n")
                        f.write(f"  P-value: {result['p_value']:.4f}\n")
                        f.write(f"  Significant: {'Yes' if result['significant'] else 'No'}\n\n")
        
        print(f"Evaluation report saved: {output_path}")
        return output_path

def train_and_evaluate_all_models():
    """Train and evaluate all three GNN models"""
    
    model_configs = {
        'gcn': {
            'model_type': 'gcn',
            'model_params': {'hidden_dim': 128, 'dropout': 0.3},
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 15
        },
        'gat': {
            'model_type': 'gat', 
            'model_params': {'hidden_dim': 128, 'num_heads': 8, 'dropout': 0.3},
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 15
        },
        'gin': {
            'model_type': 'gin',
            'model_params': {'hidden_dim': 128, 'dropout': 0.3, 'aggregation': 'sum'},
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 15
        }
    }
    
    # Train all models
    trained_models = {}
    
    for model_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"TRAINING {model_name.upper()} MODEL")
        print(f"{'='*60}")
        
        # Initialize trainer
        trainer = AMLTrainer(**config)
        
        # Load data
        trainer.load_data(['LI-Small', 'HI-Small'])
        
        # Create and train model
        trainer.create_model()
        history = trainer.train()
        
        # Save model with specific name
        checkpoint_path = Config.MODELS_DIR / f"best_model_{model_name}.pt"
        trainer.save_checkpoint(f"best_model_{model_name}.pt")
        
        trained_models[model_name] = trainer
        
        print(f"{model_name.upper()} training completed!")
    
    # Comprehensive evaluation
    print(f"\n{'='*60}")
    print("COMPREHENSIVE MODEL EVALUATION")
    print(f"{'='*60}")
    
    evaluator = ModelEvaluator()
    evaluator.load_data(['LI-Small', 'HI-Small'])
    
    # Load all trained models
    for model_name in model_configs.keys():
        evaluator.load_trained_model(model_name)
        evaluator.evaluate_model(model_name)
    
    # Generate comparison plots
    fig1 = evaluator.plot_confusion_matrices()
    fig1.savefig(Config.MODELS_DIR / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    
    fig2 = evaluator.plot_roc_curves()
    fig2.savefig(Config.MODELS_DIR / "roc_curves.png", dpi=300, bbox_inches='tight')
    
    fig3 = evaluator.plot_precision_recall_curves()
    fig3.savefig(Config.MODELS_DIR / "pr_curves.png", dpi=300, bbox_inches='tight')
    
    fig4 = evaluator.plot_model_comparison()
    fig4.savefig(Config.MODELS_DIR / "model_comparison.png", dpi=300, bbox_inches='tight')
    
    # Generate comprehensive report
    evaluator.generate_report()
    
    print("\nEvaluation completed! Check the models directory for results.")
    
    return evaluator

if __name__ == "__main__":
    evaluator = train_and_evaluate_all_models()

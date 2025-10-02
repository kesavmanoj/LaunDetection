#!/usr/bin/env python3
"""
Individual Dataset Training
==========================

This script trains on each dataset individually (no combined dataset).
Each dataset is trained separately for better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
import gc
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

print("🚀 Individual Dataset Training")
print("=" * 50)
print("📊 Training on each dataset individually")
print("📊 No combined dataset - better performance")
print()

class IndividualDatasetGNN(nn.Module):
    """GNN for individual dataset training"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, dropout=0.3):
        super(IndividualDatasetGNN, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Dynamic edge classifier
        self.edge_classifier = None
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
    def forward(self, x, edge_index, edge_attr=None):
        # Input validation
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # GNN layers
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = self.dropout(x3)
        
        if torch.isnan(x3).any():
            x3 = torch.nan_to_num(x3, nan=0.0)
        
        # Edge-level classification
        src_features = x3[edge_index[0]]
        tgt_features = x3[edge_index[1]]
        
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

class IndividualDatasetTrainer:
    """Trainer for individual datasets"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.training_results = {}
        
    def load_processed_dataset(self, dataset_name, processed_dir="/content/drive/MyDrive/LaunDetection/data/processed"):
        """Load a single processed dataset"""
        print(f"📊 Loading {dataset_name} dataset...")
        
        graph_path = os.path.join(processed_dir, f"{dataset_name}_graph.pkl")
        features_path = os.path.join(processed_dir, f"{dataset_name}_features.pkl")
        metadata_path = os.path.join(processed_dir, f"{dataset_name}_metadata.pkl")
        
        if all(os.path.exists(p) for p in [graph_path, features_path, metadata_path]):
            print(f"   ✅ Loading {dataset_name}...")
            
            # Load graph
            with open(graph_path, 'rb') as f:
                graph = pickle.load(f)
            
            # Load features
            with open(features_path, 'rb') as f:
                features = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"      📊 {dataset_name}: {metadata['num_nodes']:,} nodes, {metadata['num_edges']:,} edges")
            print(f"      🚨 AML rate: {metadata['aml_rate']*100:.2f}%")
            
            return {
                'graph': graph,
                'node_features': features['node_features'],
                'edge_features': features['edge_features'],
                'edge_labels': features['edge_labels'],
                'metadata': metadata
            }
        else:
            print(f"   ❌ {dataset_name} not found or incomplete")
            return None
    
    def create_pytorch_data(self, dataset_name, data):
        """Convert dataset to PyTorch Geometric format"""
        print(f"🔄 Converting {dataset_name} to PyTorch format...")
        
        graph = data['graph']
        node_features = data['node_features']
        edge_features = data['edge_features']
        edge_labels = data['edge_labels']
        
        # Create node mapping
        nodes = list(graph.nodes())
        node_to_int = {node: i for i, node in enumerate(nodes)}
        
        # Create node features tensor
        x_list = []
        for node in nodes:
            if node in node_features:
                features = node_features[node]
                features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
                x_list.append(features)
            else:
                # Use actual feature dimension from existing features
                if x_list:
                    default_features = [0.0] * len(x_list[0])
                else:
                    default_features = [0.0] * 25  # Fallback to 25
                x_list.append(default_features)
        
        x = torch.tensor(x_list, dtype=torch.float32).to(self.device)
        
        # Create edge index and features
        edge_index_list = []
        edge_attr_list = []
        y_list = []
        
        for edge in graph.edges(data=True):
            from_node, to_node, edge_data = edge
            if from_node in node_to_int and to_node in node_to_int:
                edge_index_list.append([node_to_int[from_node], node_to_int[to_node]])
                
                if 'features' in edge_data:
                    features = edge_data['features']
                    features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
                    edge_attr_list.append(features)
                else:
                    edge_attr_list.append([0.0] * 13)  # Default edge features
                
                y_list.append(edge_data.get('label', 0))
        
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(self.device)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_list, dtype=torch.long).to(self.device)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        print(f"   ✅ PyTorch data created: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
        print(f"   🚨 AML edges: {y.sum().item():,}")
        print(f"   ✅ Non-AML edges: {(y == 0).sum().item():,}")
        
        return data
    
    def train_individual_model(self, dataset_name, data, epochs=200, learning_rate=0.001):
        """Train model on individual dataset"""
        print(f"🚀 Training {dataset_name} model...")
        
        # Create model
        actual_input_dim = data.x.shape[1]
        print(f"   📊 Input dimension: {actual_input_dim}")
        
        model = IndividualDatasetGNN(
            input_dim=actual_input_dim,
            hidden_dim=128,
            output_dim=2,
            dropout=0.3
        ).to(self.device)
        
        print(f"✅ {dataset_name} model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        # Class weights
        class_counts = torch.bincount(data.y)
        class_weights = len(data.y) / (len(class_counts) * class_counts.float())
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        
        # Boost AML class
        class_weights[1] = class_weights[1] * 3.0  # 3x boost for AML class
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"   📊 Class weights: {class_weights}")
        print(f"   🎯 Training for {epochs} epochs...")
        
        best_f1 = 0.0
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index, data.edge_attr)
            
            if torch.isnan(out).any():
                out = torch.nan_to_num(out, nan=0.0)
            
            loss = criterion(out, data.y)
            
            if torch.isnan(loss):
                print(f"   ⚠️ NaN loss at epoch {epoch+1}")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation every 10 epochs
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index, data.edge_attr)
                    
                    if torch.isnan(out).any():
                        out = torch.nan_to_num(out, nan=0.0)
                    
                    preds = torch.argmax(out, dim=1)
                    
                    # Calculate metrics
                    val_f1 = f1_score(data.y.cpu().numpy(), preds.cpu().numpy(), average='weighted')
                    val_precision = precision_score(data.y.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=0)
                    val_recall = recall_score(data.y.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=0)
                    
                    # AML-specific metrics
                    aml_f1 = f1_score(data.y.cpu().numpy(), preds.cpu().numpy(), average='binary', pos_label=1)
                    aml_precision = precision_score(data.y.cpu().numpy(), preds.cpu().numpy(), average='binary', pos_label=1, zero_division=0)
                    aml_recall = recall_score(data.y.cpu().numpy(), preds.cpu().numpy(), average='binary', pos_label=1, zero_division=0)
                    
                    print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")
                    print(f"   Overall - F1={val_f1:.4f}, Precision={val_precision:.4f}, Recall={val_recall:.4f}")
                    print(f"   AML - F1={aml_f1:.4f}, Precision={aml_precision:.4f}, Recall={aml_recall:.4f}")
                    
                    # Learning rate scheduling
                    scheduler.step(val_f1)
                    
                    # Early stopping
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        patience_counter = 0
                        
                        # Save best model
                        torch.save(model.state_dict(), f'/content/drive/MyDrive/LaunDetection/{dataset_name}_model.pth')
                        print(f"   💾 Saved best {dataset_name} model (F1={best_f1:.4f})")
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"   🛑 Early stopping at epoch {epoch+1}")
                        break
            
            # Memory cleanup every 50 epochs
            if epoch % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        print(f"✅ {dataset_name} training completed! Best F1: {best_f1:.4f}")
        return model, best_f1
    
    def evaluate_individual_model(self, dataset_name, model, data):
        """Evaluate individual model"""
        print(f"📊 Evaluating {dataset_name} model...")
        
        if model is None:
            print("❌ No model loaded!")
            return
        
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            probabilities = torch.softmax(out, dim=1)
            predictions = torch.argmax(out, dim=1)
        
        # Calculate comprehensive metrics
        y_true = data.y.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        y_prob = probabilities.cpu().numpy()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_prob[:, 1]),
        }
        
        # AML-specific metrics
        aml_metrics = {
            'aml_f1': f1_score(y_true, y_pred, average='binary', pos_label=1),
            'aml_precision': precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0),
            'aml_recall': recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\n📊 {dataset_name} Model Evaluation Results:")
        print("=" * 50)
        print(f"🎯 Overall Performance:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1 (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"   F1 (Macro): {metrics['f1_macro']:.4f}")
        print(f"   Precision: {metrics['precision_weighted']:.4f}")
        print(f"   Recall: {metrics['recall_weighted']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        
        print(f"\n🚨 AML Detection Performance:")
        print(f"   AML F1-Score: {aml_metrics['aml_f1']:.4f}")
        print(f"   AML Precision: {aml_metrics['aml_precision']:.4f}")
        print(f"   AML Recall: {aml_metrics['aml_recall']:.4f}")
        
        print(f"\n📈 Confusion Matrix:")
        print(f"   True Negative: {cm[0][0]:,}")
        print(f"   False Positive: {cm[0][1]:,}")
        print(f"   False Negative: {cm[1][0]:,}")
        print(f"   True Positive: {cm[1][1]:,}")
        
        return metrics, aml_metrics, cm
    
    def run_individual_training(self):
        """Run individual training for all datasets"""
        print("🚀 Starting Individual Dataset Training...")
        
        available_datasets = ['HI-Small', 'LI-Small', 'HI-Medium', 'LI-Medium', 'HI-Large', 'LI-Large']
        
        for dataset_name in available_datasets:
            print(f"\n🔍 Processing {dataset_name} dataset...")
            
            # Load dataset
            data = self.load_processed_dataset(dataset_name)
            
            if data is None:
                print(f"   ❌ {dataset_name} not found, skipping...")
                continue
            
            # Convert to PyTorch format
            pytorch_data = self.create_pytorch_data(dataset_name, data)
            
            # Train model
            model, best_f1 = self.train_individual_model(dataset_name, pytorch_data)
            
            # Evaluate model
            metrics, aml_metrics, cm = self.evaluate_individual_model(dataset_name, model, pytorch_data)
            
            # Store results
            self.models[dataset_name] = model
            self.training_results[dataset_name] = {
                'best_f1': best_f1,
                'metrics': metrics,
                'aml_metrics': aml_metrics,
                'confusion_matrix': cm
            }
            
            print(f"✅ {dataset_name} training and evaluation complete!")
        
        print(f"\n🎉 Individual training completed!")
        print(f"📊 Trained {len(self.models)} models")
        print(f"💾 All models saved individually")
        
        return self.models, self.training_results

def main():
    """Run individual training"""
    trainer = IndividualDatasetTrainer()
    models, results = trainer.run_individual_training()
    
    if models:
        print("\n🎉 Individual training successful!")
        print("📊 All datasets trained individually")
        print("📊 Each model optimized for its specific dataset")
        print("🚀 Ready for individual inference!")
    else:
        print("\n❌ Individual training failed!")

if __name__ == "__main__":
    main()

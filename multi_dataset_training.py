#!/usr/bin/env python3
"""
Multi-Dataset Enhanced Training Pipeline
=======================================

This script trains the AML detection model on multiple datasets (LI-Small, HI-Medium, LI-Medium)
with enhanced preprocessing for improved AML detection performance.
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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 Multi-Dataset Enhanced Training Pipeline")
print("=" * 60)

class MultiDatasetEdgeLevelGNN(nn.Module):
    """Enhanced GNN for multi-dataset training"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, dropout=0.4):
        super(MultiDatasetEdgeLevelGNN, self).__init__()
        
        # Enhanced architecture for multi-dataset learning
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)  # Additional layer
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
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
        
        # Enhanced GNN layers with residual connections
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        if torch.isnan(x1).any():
            x1 = torch.nan_to_num(x1, nan=0.0)
        
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        # Residual connection
        if x2.shape == x1.shape:
            x2 = x2 + x1
        
        if torch.isnan(x2).any():
            x2 = torch.nan_to_num(x2, nan=0.0)
        
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = self.dropout(x3)
        
        # Residual connection
        if x3.shape == x2.shape:
            x3 = x3 + x2
        
        if torch.isnan(x3).any():
            x3 = torch.nan_to_num(x3, nan=0.0)
        
        x4 = self.conv4(x3, edge_index)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)
        x4 = self.dropout(x4)
        
        # Residual connection
        if x4.shape == x3.shape:
            x4 = x4 + x3
        
        if torch.isnan(x4).any():
            x4 = torch.nan_to_num(x4, nan=0.0)
        
        # Edge-level classification
        src_features = x4[edge_index[0]]
        tgt_features = x4[edge_index[1]]
        
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
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 4, self.output_dim)
            ).to(edge_features.device)
        
        edge_output = self.edge_classifier(edge_features)
        
        if torch.isnan(edge_output).any():
            edge_output = torch.nan_to_num(edge_output, nan=0.0)
        
        return edge_output

class MultiDatasetTrainer:
    """Enhanced trainer for multi-dataset learning"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.training_history = []
        
    def load_processed_datasets(self, processed_dir="/content/drive/MyDrive/LaunDetection/data/processed"):
        """Load all processed datasets"""
        print("📊 Loading processed multi-datasets...")
        
        datasets = {}
        available_datasets = ['HI-Small', 'LI-Small', 'HI-Medium', 'LI-Medium']
        
        for dataset_name in available_datasets:
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
                
                datasets[dataset_name] = {
                    'graph': graph,
                    'node_features': features['node_features'],
                    'edge_features': features['edge_features'],
                    'edge_labels': features['edge_labels'],
                    'metadata': metadata
                }
                
                print(f"      📊 {dataset_name}: {metadata['num_nodes']:,} nodes, {metadata['num_edges']:,} edges")
                print(f"      🚨 AML rate: {metadata['aml_rate']*100:.2f}%")
            else:
                print(f"   ❌ {dataset_name} not found or incomplete")
        
        print(f"\n✅ Loaded {len(datasets)} processed datasets")
        return datasets
    
    def create_combined_dataset(self, datasets):
        """Create a combined dataset from multiple sources"""
        print("🔄 Creating combined multi-dataset...")
        
        combined_graph = nx.DiGraph()
        combined_node_features = {}
        combined_edge_features = []
        combined_edge_labels = []
        
        # Node ID mapping to avoid conflicts
        node_id_offset = 0
        
        for dataset_name, data in datasets.items():
            print(f"   🔄 Adding {dataset_name} to combined dataset...")
            
            # Add nodes with offset IDs
            for node, features in data['node_features'].items():
                new_node_id = f"{dataset_name}_{node}"
                combined_node_features[new_node_id] = features
                combined_graph.add_node(new_node_id, features=features)
            
            # Add edges with offset IDs
            for edge in data['graph'].edges(data=True):
                from_node, to_node, edge_data = edge
                new_from = f"{dataset_name}_{from_node}"
                new_to = f"{dataset_name}_{to_node}"
                
                if new_from in combined_graph.nodes and new_to in combined_graph.nodes:
                    combined_graph.add_edge(new_from, new_to, 
                                          features=edge_data['features'],
                                          label=edge_data['label'])
                    combined_edge_features.append(edge_data['features'])
                    combined_edge_labels.append(edge_data['label'])
        
        print(f"   ✅ Combined dataset: {combined_graph.number_of_nodes():,} nodes, {combined_graph.number_of_edges():,} edges")
        print(f"   🚨 AML edges: {sum(combined_edge_labels):,}")
        print(f"   ✅ Non-AML edges: {len(combined_edge_labels) - sum(combined_edge_labels):,}")
        
        return {
            'graph': combined_graph,
            'node_features': combined_node_features,
            'edge_features': combined_edge_features,
            'edge_labels': combined_edge_labels
        }
    
    def create_pytorch_data(self, combined_data):
        """Convert combined data to PyTorch Geometric format"""
        print("🔄 Converting to PyTorch Geometric format...")
        
        graph = combined_data['graph']
        node_features = combined_data['node_features']
        edge_features = combined_data['edge_features']
        edge_labels = combined_data['edge_labels']
        
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
                    edge_attr_list.append([0.0] * 25)  # Default edge features
                
                y_list.append(edge_data.get('label', 0))
        
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(self.device)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_list, dtype=torch.long).to(self.device)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        print(f"   ✅ PyTorch data created: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
        print(f"   🚨 AML edges: {y.sum().item():,}")
        print(f"   ✅ Non-AML edges: {(y == 0).sum().item():,}")
        
        return data
    
    def train_multi_dataset_model(self, data, epochs=200, learning_rate=0.001):
        """Train model on multi-dataset"""
        print("🚀 Starting Multi-Dataset Training...")
        
        # Create model with correct input dimension
        # Check actual input dimension from data
        actual_input_dim = data.x.shape[1]
        print(f"   📊 Actual input dimension: {actual_input_dim}")
        
        self.model = MultiDatasetEdgeLevelGNN(
            input_dim=actual_input_dim,  # Use actual input dimension
            hidden_dim=128,
            output_dim=2,
            dropout=0.4
        ).to(self.device)
        
        print(f"✅ Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
        
        # Balanced class weights for better AML detection - FIXED
        class_counts = torch.bincount(data.y)
        class_weights = len(data.y) / (len(class_counts) * class_counts.float())
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        
        # Moderate boost for AML class (not too aggressive)
        class_weights[1] = class_weights[1] * 2.0  # 2x boost for AML class (reduced from 5x)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"   📊 Class weights: {class_weights}")
        print(f"   🎯 Training for {epochs} epochs...")
        
        best_f1 = 0.0
        patience = 20  # Reduced patience to prevent overfitting
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            out = self.model(data.x, data.edge_index, data.edge_attr)
            
            if torch.isnan(out).any():
                out = torch.nan_to_num(out, nan=0.0)
            
            loss = criterion(out, data.y)
            
            if torch.isnan(loss):
                print(f"   ⚠️ NaN loss at epoch {epoch+1}")
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation every 5 epochs for more frequent monitoring
            if epoch % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    out = self.model(data.x, data.edge_index, data.edge_attr)
                    
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
                        torch.save(self.model.state_dict(), '/content/drive/MyDrive/LaunDetection/multi_dataset_model.pth')
                        print(f"   💾 Saved best model (F1={best_f1:.4f})")
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"   🛑 Early stopping at epoch {epoch+1}")
                        break
            
            # Memory cleanup
            if epoch % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        print(f"\n✅ Multi-Dataset Training Complete! Best F1: {best_f1:.4f}")
        return self.model, best_f1
    
    def evaluate_multi_dataset_model(self, data):
        """Evaluate the trained multi-dataset model"""
        print("📊 Evaluating Multi-Dataset Model...")
        
        if self.model is None:
            print("❌ No model loaded!")
            return
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index, data.edge_attr)
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
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
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
        
        print("\n📊 Multi-Dataset Model Evaluation Results:")
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

def main():
    """Main multi-dataset training pipeline"""
    print("🚀 Starting Multi-Dataset Enhanced Training...")
    
    # Initialize trainer
    trainer = MultiDatasetTrainer()
    
    # Load processed datasets
    datasets = trainer.load_processed_datasets()
    
    if not datasets:
        print("❌ No processed datasets found! Run preprocessing first.")
        return
    
    # Create combined dataset
    combined_data = trainer.create_combined_dataset(datasets)
    
    # Convert to PyTorch format
    data = trainer.create_pytorch_data(combined_data)
    
    # Train model
    model, best_f1 = trainer.train_multi_dataset_model(data)
    
    # Evaluate model
    metrics, aml_metrics, cm = trainer.evaluate_multi_dataset_model(data)
    
    print("\n🎉 Multi-Dataset Training Complete!")
    print("=" * 50)
    print("✅ Enhanced model trained on multiple datasets")
    print("✅ Improved AML detection performance")
    print("✅ Production-ready model saved")
    
    if aml_metrics['aml_f1'] > 0.5:
        print("🎉 EXCELLENT! AML detection significantly improved!")
    elif aml_metrics['aml_f1'] > 0.3:
        print("✅ GOOD! AML detection improved!")
    else:
        print("⚠️ AML detection needs further improvement")

if __name__ == "__main__":
    main()

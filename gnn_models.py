"""
Graph Neural Network Models for Anti-Money Laundering Transaction Classification

This module implements three state-of-the-art GNN architectures optimized for 
transaction-level binary classification on financial networks:

1. Graph Convolutional Network (GCN) - Efficient spectral convolutions
2. Graph Attention Network (GAT) - Multi-head attention mechanisms  
3. Graph Isomorphism Network (GIN) - Powerful graph representation learning

All models are designed to handle both node features and edge features for
comprehensive transaction analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.nn import LayerNorm
from torch_geometric.utils import to_dense_batch
import math

class EdgeFeatureGCN(nn.Module):
    """
    Graph Convolutional Network with Edge Feature Integration
    
    Architecture Design Choices:
    - 3-layer GCN for sufficient depth without over-smoothing
    - Edge features integrated via attention mechanism
    - Batch normalization for stable training
    - Dropout for regularization
    - Skip connections to preserve information flow
    
    This model is particularly effective for:
    - Large-scale transaction networks
    - When computational efficiency is important
    - Scenarios with homophilic graph structure
    """
    
    def __init__(self, 
                 node_feature_dim=10, 
                 edge_feature_dim=10, 
                 hidden_dim=128,
                 num_classes=2,
                 dropout=0.3,
                 use_edge_features=True):
        """
        Initialize EdgeFeatureGCN
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features  
            hidden_dim: Hidden layer dimensions
            num_classes: Number of output classes (2 for binary classification)
            dropout: Dropout probability
            use_edge_features: Whether to incorporate edge features
        """
        super(EdgeFeatureGCN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        
        # Node feature transformation layers
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge feature processing (if enabled)
        if use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # Edge attention mechanism for feature integration
            self.edge_attention = nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        # GCN layers with batch normalization
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(3)
        ])
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(3)
        ])
        
        # Skip connection projections
        self.skip_connections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for edge representation
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch assignment for graph-level tasks
            
        Returns:
            Edge-level predictions [num_edges, num_classes]
        """
        # Encode node features
        h = self.node_encoder(x)  # [num_nodes, hidden_dim]
        
        # Store initial representation for skip connections
        h_initial = h.clone()
        
        # Apply GCN layers with skip connections and batch normalization
        for i, (gcn, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            h_prev = h.clone()
            
            # GCN convolution
            h = gcn(h, edge_index)
            
            # Batch normalization
            h = bn(h)
            
            # Activation
            h = F.relu(h)
            
            # Skip connection (except for first layer)
            if i > 0:
                h = h + self.skip_connections[i-1](h_prev)
            
            # Dropout
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # MEMORY OPTIMIZATION: Create edge representations with chunking for large graphs
        row, col = edge_index
        num_edges = edge_index.shape[1]
        
        if num_edges > 500000:  # lower threshold for chunking to reduce peak memory
            chunk_size = 250000  # smaller chunks to limit activation memory
            edge_logits = []
            
            for i in range(0, num_edges, chunk_size):
                end_idx = min(i + chunk_size, num_edges)
                
                # Get chunk indices
                chunk_row = row[i:end_idx]
                chunk_col = col[i:end_idx]
                
                # Create chunk edge representation
                chunk_edge_repr = torch.cat([h[chunk_row], h[chunk_col]], dim=1)
                
                # Integrate edge features if available
                if self.use_edge_features and edge_attr is not None:
                    chunk_edge_attr = edge_attr[i:end_idx]
                    
                    # Encode edge features
                    chunk_edge_feat = self.edge_encoder(chunk_edge_attr)
                    
                    # Compute attention weights for edge feature integration
                    attention_input = torch.cat([h[chunk_row], h[chunk_col], chunk_edge_attr], dim=1)
                    attention_weights = self.edge_attention(attention_input)
                    
                    # Apply attention to edge features
                    weighted_edge_feat = attention_weights * chunk_edge_feat
                    
                    # Combine with edge representation
                    chunk_edge_repr = chunk_edge_repr + torch.cat([weighted_edge_feat, weighted_edge_feat], dim=1)
                
                # Classify chunk
                chunk_logits = self.classifier(chunk_edge_repr)
                edge_logits.append(chunk_logits)
                
                # Clean up intermediate tensors
                del chunk_edge_repr, chunk_row, chunk_col
                if self.use_edge_features and edge_attr is not None:
                    del chunk_edge_attr, chunk_edge_feat, attention_input, attention_weights, weighted_edge_feat
            
            return torch.cat(edge_logits, dim=0)
        else:
            # Standard processing for smaller edge sets
            edge_repr = torch.cat([h[row], h[col]], dim=1)  # [num_edges, hidden_dim * 2]
            
            # Integrate edge features if available
            if self.use_edge_features and edge_attr is not None:
                # Encode edge features
                edge_feat = self.edge_encoder(edge_attr)  # [num_edges, hidden_dim]
                
                # Compute attention weights for edge feature integration
                attention_input = torch.cat([h[row], h[col], edge_attr], dim=1)
                attention_weights = self.edge_attention(attention_input)  # [num_edges, 1]
                
                # Apply attention to edge features
                weighted_edge_feat = attention_weights * edge_feat  # [num_edges, hidden_dim]
                
                # Combine with edge representation
                edge_repr = edge_repr + torch.cat([weighted_edge_feat, weighted_edge_feat], dim=1)
            
            # Final classification
            logits = self.classifier(edge_repr)  # [num_edges, num_classes]
            
            return logits

class EdgeFeatureGAT(nn.Module):
    """
    Graph Attention Network with Edge Feature Integration
    
    Architecture Design Choices:
    - Multi-head attention for capturing diverse relationship patterns
    - Edge features integrated through attention computation
    - Layer normalization for stable attention training
    - Attention dropout to prevent overfitting to specific attention patterns
    - Residual connections to maintain gradient flow
    
    This model excels at:
    - Heterophilic graphs with diverse relationship types
    - Capturing complex attention patterns in financial networks
    - Handling varying importance of different transaction types
    """
    
    def __init__(self,
                 node_feature_dim=10,
                 edge_feature_dim=10,
                 hidden_dim=128,
                 num_classes=2,
                 num_heads=8,
                 num_layers=3,
                 dropout=0.3,
                 attention_dropout=0.1,
                 use_edge_features=True):
        """
        Initialize EdgeFeatureGAT
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden layer dimensions (must be divisible by num_heads)
            num_classes: Number of output classes
            num_heads: Number of attention heads
            num_layers: Number of GAT layers
            dropout: General dropout probability
            attention_dropout: Attention-specific dropout
            use_edge_features: Whether to use edge features
        """
        super(EdgeFeatureGAT, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_edge_features = use_edge_features
        
        # Ensure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge feature processing
        if use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Edge feature integration via gating mechanism
            self.edge_gate = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),  # src + dst + edge
                nn.Sigmoid()
            )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # For the last layer, use single head and output to hidden_dim
            if i == num_layers - 1:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=1, dropout=attention_dropout, concat=False)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, self.head_dim, heads=num_heads, dropout=attention_dropout, concat=True)
                )
            
            self.layer_norms.append(LayerNorm(hidden_dim))
        
        # Residual connection projections
        self.residual_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Edge-level classifier
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenated src + dst features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate scaling for attention networks"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass with attention mechanisms
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch assignment
            
        Returns:
            Edge-level predictions [num_edges, num_classes]
        """
        # Encode node features
        h = self.node_encoder(x)  # [num_nodes, hidden_dim]
        
        # Process edge features if available
        edge_feat = None
        if self.use_edge_features and edge_attr is not None:
            edge_feat = self.edge_encoder(edge_attr)  # [num_edges, hidden_dim]
        
        # Apply GAT layers with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_prev = h.clone()
            
            # GAT convolution with attention
            h = gat(h, edge_index)
            
            # Layer normalization
            h = norm(h)
            
            # Residual connection (except for first layer)
            if i > 0:
                h = h + self.residual_projections[i-1](h_prev)
            
            # Activation and dropout
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Chunked edge classification to avoid peak memory
        row, col = edge_index
        num_edges = edge_index.shape[1]
        if num_edges > 500000:
            chunk_size = 250000
            logits_chunks = []
            for i in range(0, num_edges, chunk_size):
                end_idx = min(i + chunk_size, num_edges)
                src_features = h[row[i:end_idx]]
                dst_features = h[col[i:end_idx]]

                if self.use_edge_features and edge_attr is not None:
                    # Encode edge features per chunk
                    edge_feat_chunk = self.edge_encoder(edge_attr[i:end_idx])
                    gate_input = torch.cat([src_features, dst_features, edge_feat_chunk], dim=1)
                    gate = self.edge_gate(gate_input)
                    gated_edge_feat = gate * edge_feat_chunk
                    src_features = src_features + gated_edge_feat
                    dst_features = dst_features + gated_edge_feat

                edge_repr = torch.cat([src_features, dst_features], dim=1)
                logits_chunks.append(self.edge_classifier(edge_repr))

                # explicit cleanup
                del src_features, dst_features, edge_repr
                if self.use_edge_features and edge_attr is not None:
                    del edge_feat_chunk, gate_input, gate, gated_edge_feat

            return torch.cat(logits_chunks, dim=0)
        else:
            src_features = h[row]
            dst_features = h[col]
            if self.use_edge_features and edge_attr is not None:
                edge_feat = self.edge_encoder(edge_attr)
                gate_input = torch.cat([src_features, dst_features, edge_feat], dim=1)
                gate = self.edge_gate(gate_input)
                gated_edge_feat = gate * edge_feat
                src_features = src_features + gated_edge_feat
                dst_features = dst_features + gated_edge_feat
            edge_repr = torch.cat([src_features, dst_features], dim=1)
            logits = self.edge_classifier(edge_repr)
            return logits

class EdgeFeatureGIN(nn.Module):
    """
    Graph Isomorphism Network with Edge Feature Integration
    
    Architecture Design Choices:
    - MLP aggregators for powerful representation learning
    - Learnable epsilon parameters for enhanced expressiveness
    - Edge features integrated through message passing modification
    - Batch normalization for training stability
    - Multiple aggregation strategies (sum, mean, max) for robustness
    
    This model is optimal for:
    - Complex graph structures requiring high expressiveness
    - Scenarios where graph isomorphism testing is important
    - Learning hierarchical representations of transaction patterns
    """
    
    def __init__(self,
                 node_feature_dim=10,
                 edge_feature_dim=10,
                 hidden_dim=128,
                 num_classes=2,
                 num_layers=3,
                 dropout=0.3,
                 eps_learnable=True,
                 use_edge_features=True,
                 aggregation='sum'):
        """
        Initialize EdgeFeatureGIN
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden layer dimensions
            num_classes: Number of output classes
            num_layers: Number of GIN layers
            dropout: Dropout probability
            eps_learnable: Whether epsilon parameters are learnable
            use_edge_features: Whether to use edge features
            aggregation: Aggregation strategy ('sum', 'mean', 'max', 'concat')
        """
        super(EdgeFeatureGIN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.eps_learnable = eps_learnable
        self.use_edge_features = use_edge_features
        self.aggregation = aggregation
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge feature processing
        if use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Edge-node feature fusion
            self.edge_node_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # GIN layers with MLP aggregators
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # MLP for each GIN layer
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            
            # GIN convolution with learnable epsilon
            self.gin_layers.append(GINConv(mlp, eps=0.0, train_eps=eps_learnable))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Multi-scale feature aggregation
        if aggregation == 'concat':
            classifier_input_dim = hidden_dim * num_layers * 2  # *2 for edge representation
        else:
            classifier_input_dim = hidden_dim * 2
        
        # Edge-level classifier with multi-scale features
        self.edge_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable GIN training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass with isomorphism-aware message passing
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch assignment
            
        Returns:
            Edge-level predictions [num_edges, num_classes]
        """
        # Encode node features
        h = self.node_encoder(x)  # [num_nodes, hidden_dim]
        
        # Process and integrate edge features with chunked aggregation to reduce memory
        if self.use_edge_features and edge_attr is not None:
            row, col = edge_index
            num_edges_total = edge_index.shape[1]
            node_edge_agg = torch.zeros_like(h)
            chunk_size = 250000 if num_edges_total > 500000 else num_edges_total
            for i in range(0, num_edges_total, chunk_size):
                end_idx = min(i + chunk_size, num_edges_total)
                edge_feat_chunk = self.edge_encoder(edge_attr[i:end_idx])
                row_chunk = row[i:end_idx]
                col_chunk = col[i:end_idx]
                node_edge_agg.index_add_(0, row_chunk, edge_feat_chunk)
                node_edge_agg.index_add_(0, col_chunk, edge_feat_chunk)
                del edge_feat_chunk, row_chunk, col_chunk
            edge_count = torch.zeros(h.size(0), device=h.device)
            edge_count.index_add_(0, row, torch.ones(row.size(0), device=h.device))
            edge_count.index_add_(0, col, torch.ones(col.size(0), device=h.device))
            edge_count = edge_count.clamp(min=1).unsqueeze(1)
            node_edge_agg = node_edge_agg / edge_count
            h = self.edge_node_fusion(torch.cat([h, node_edge_agg], dim=1))
        
        # Store representations from all layers for multi-scale aggregation
        layer_representations = []
        
        # Apply GIN layers
        for i, (gin, bn) in enumerate(zip(self.gin_layers, self.batch_norms)):
            h = gin(h, edge_index)
            h = bn(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Store representation for multi-scale aggregation
            if self.aggregation == 'concat':
                layer_representations.append(h)
        
        # Multi-scale feature aggregation
        if self.aggregation == 'concat' and layer_representations:
            h = torch.cat(layer_representations, dim=1)
        
        # Chunked edge representation and classification
        row, col = edge_index
        num_edges_total = edge_index.shape[1]
        if num_edges_total > 500000:
            chunk_size = 250000
            logits_chunks = []
            for i in range(0, num_edges_total, chunk_size):
                end_idx = min(i + chunk_size, num_edges_total)
                src_features = h[row[i:end_idx]]
                dst_features = h[col[i:end_idx]]
                if self.aggregation == 'sum':
                    edge_repr = src_features + dst_features
                    edge_repr = torch.cat([edge_repr, src_features - dst_features], dim=1)
                elif self.aggregation == 'mean':
                    edge_repr = (src_features + dst_features) / 2
                    edge_repr = torch.cat([edge_repr, torch.abs(src_features - dst_features)], dim=1)
                elif self.aggregation == 'max':
                    edge_repr = torch.max(src_features, dst_features)
                    edge_repr = torch.cat([edge_repr, torch.min(src_features, dst_features)], dim=1)
                else:
                    edge_repr = torch.cat([src_features, dst_features], dim=1)
                logits_chunks.append(self.edge_classifier(edge_repr))
                del src_features, dst_features, edge_repr
            return torch.cat(logits_chunks, dim=0)
        else:
            src_features = h[row]
            dst_features = h[col]
            if self.aggregation == 'sum':
                edge_repr = src_features + dst_features
                edge_repr = torch.cat([edge_repr, src_features - dst_features], dim=1)
            elif self.aggregation == 'mean':
                edge_repr = (src_features + dst_features) / 2
                edge_repr = torch.cat([edge_repr, torch.abs(src_features - dst_features)], dim=1)
            elif self.aggregation == 'max':
                edge_repr = torch.max(src_features, dst_features)
                edge_repr = torch.cat([edge_repr, torch.min(src_features, dst_features)], dim=1)
            else:
                edge_repr = torch.cat([src_features, dst_features], dim=1)
            logits = self.edge_classifier(edge_repr)
            return logits

def get_model(model_type, **kwargs):
    """
    Factory function to create GNN models
    
    Args:
        model_type: Type of model ('gcn', 'gat', 'gin')
        **kwargs: Model-specific parameters
        
    Returns:
        Initialized model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'gcn':
        return EdgeFeatureGCN(**kwargs)
    elif model_type == 'gat':
        return EdgeFeatureGAT(**kwargs)
    elif model_type == 'gin':
        return EdgeFeatureGIN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'gcn', 'gat', 'gin'")

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, model_name="Model"):
    """Print a summary of the model architecture"""
    print(f"\n{model_name} Architecture Summary:")
    print("=" * 50)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print(f"Model size (MB): {count_parameters(model) * 4 / 1024 / 1024:.2f}")
    print("=" * 50)
    
    for name, module in model.named_children():
        if hasattr(module, '__len__'):
            print(f"{name}: {len(module)} layers")
        else:
            print(f"{name}: {type(module).__name__}")
    print()

# Example usage and testing
if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    node_feature_dim = 10
    edge_feature_dim = 10
    hidden_dim = 128
    num_classes = 2
    
    # Create test data
    num_nodes = 1000
    num_edges = 5000
    
    x = torch.randn(num_nodes, node_feature_dim).to(device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges)).to(device)
    edge_attr = torch.randn(num_edges, edge_feature_dim).to(device)
    
    # Test all models
    models = {
        'GCN': get_model('gcn', node_feature_dim=node_feature_dim, 
                        edge_feature_dim=edge_feature_dim, hidden_dim=hidden_dim),
        'GAT': get_model('gat', node_feature_dim=node_feature_dim,
                        edge_feature_dim=edge_feature_dim, hidden_dim=hidden_dim),
        'GIN': get_model('gin', node_feature_dim=node_feature_dim,
                        edge_feature_dim=edge_feature_dim, hidden_dim=hidden_dim)
    }
    
    for name, model in models.items():
        model = model.to(device)
        model_summary(model, name)
        
        # Test forward pass
        with torch.no_grad():
            output = model(x, edge_index, edge_attr)
            print(f"{name} output shape: {output.shape}")
            print(f"{name} output range: [{output.min():.3f}, {output.max():.3f}]")
            print()

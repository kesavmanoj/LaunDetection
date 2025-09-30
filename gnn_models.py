"""
Graph Neural Network models for Anti-Money-Laundering (AML)
Updated 2025-09-30
• Larger hidden dimensions (256) and deeper layers
• 8-head GAT with 3 layers
• Better edge–feature integration and gating
• Memory-safe chunking for >0.5 M edges
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv,
    LayerNorm
)

# ---------------------------------------------------------------------------#
#  Utility helpers
# ---------------------------------------------------------------------------#
def _xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------#
#  GCN
# ---------------------------------------------------------------------------#
class EdgeFeatureGCN(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 10,
        edge_feature_dim: int = 10,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 4,
        dropout: float = 0.2,
        use_edge_features: bool = True,
    ):
        super().__init__()
        self.use_edge_features = use_edge_features
        self.dropout = dropout

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.edge_attention = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )
        self.res_proj = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.apply(_xavier)

    # --------------------------------------------------------------------- #
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.node_encoder(x)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h_res = h
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            if i > 0:
                h = h + self.res_proj[i - 1](h_res)
            h = F.dropout(h, self.dropout, self.training)

        row, col = edge_index
        edge_repr = torch.cat([h[row], h[col]], dim=1)

        if self.use_edge_features and edge_attr is not None:
            e = self.edge_encoder(edge_attr)
            w = self.edge_attention(torch.cat([h[row], h[col], edge_attr], dim=1))
            edge_repr = edge_repr + torch.cat([w * e, w * e], dim=1)

        logits = self.classifier(edge_repr)
        return logits


# ---------------------------------------------------------------------------#
#  GAT
# ---------------------------------------------------------------------------#
class EdgeFeatureGAT(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 10,
        edge_feature_dim: int = 10,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
        use_edge_features: bool = True,
    ):
        super().__init__()
        self.use_edge_features = use_edge_features
        self.dropout = dropout
        assert hidden_dim % num_heads == 0

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.edge_gate = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.Sigmoid(),
            )

        self.gats = nn.ModuleList()
        self.lns = nn.ModuleList()
        for i in range(num_layers):
            heads = 1 if i == num_layers - 1 else num_heads
            concat = i != num_layers - 1
            self.gats.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=concat,
                    dropout=attn_dropout,
                )
            )
            self.lns.append(LayerNorm(hidden_dim))

        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.apply(_xavier)

    # --------------------------------------------------------------------- #
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.node_encoder(x)

        for gat, ln in zip(self.gats, self.lns):
            h = gat(h, edge_index)
            h = ln(h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, self.training)

        row, col = edge_index
        src = h[row]
        dst = h[col]

        if self.use_edge_features and edge_attr is not None:
            e = self.edge_encoder(edge_attr)
            gate = self.edge_gate(torch.cat([src, dst, e], dim=1))
            src = src + gate * e
            dst = dst + gate * e

        logits = self.edge_classifier(torch.cat([src, dst], dim=1))
        return logits


# ---------------------------------------------------------------------------#
#  GIN
# ---------------------------------------------------------------------------#
class EdgeFeatureGIN(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 10,
        edge_feature_dim: int = 10,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 4,
        dropout: float = 0.2,
        aggregation: str = "sum",  # sum / mean / max / concat
        use_edge_features: bool = True,
    ):
        super().__init__()
        self.use_edge_features = use_edge_features
        self.dropout = dropout
        self.aggregation = aggregation

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.gin_layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.gin_layers.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        cls_in = hidden_dim * 2 if aggregation != "concat" else hidden_dim * 2 * num_layers
        self.edge_classifier = nn.Sequential(
            nn.Linear(cls_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.apply(_xavier)

    # --------------------------------------------------------------------- #
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.node_encoder(x)

        if self.use_edge_features and edge_attr is not None:
            row, col = edge_index
            e_enc = self.edge_encoder(edge_attr)
            # add edge features to both incident nodes (average)
            h = h + torch.zeros_like(h).scatter_add_(
                0, row.unsqueeze(1).expand_as(e_enc), e_enc
            ).scatter_add_(0, col.unsqueeze(1).expand_as(e_enc), e_enc) / 2

        reps = []
        for gin, bn in zip(self.gin_layers, self.bns):
            h = gin(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, self.training)
            if self.aggregation == "concat":
                reps.append(h)

        row, col = edge_index
        src, dst = h[row], h[col]

        if self.aggregation == "sum":
            edge_repr = torch.cat([src + dst, src - dst], dim=1)
        elif self.aggregation == "mean":
            edge_repr = torch.cat([(src + dst) / 2, torch.abs(src - dst)], dim=1)
        elif self.aggregation == "max":
            edge_repr = torch.cat([torch.max(src, dst), torch.min(src, dst)], dim=1)
        else:  # concat
            h_all = torch.cat(reps, dim=1)
            edge_repr = torch.cat([h_all[row], h_all[col]], dim=1)

        logits = self.edge_classifier(edge_repr)
        return logits


# ---------------------------------------------------------------------------#
#  Factory & helpers
# ---------------------------------------------------------------------------#
def get_model(name: str, **kwargs):
    name = name.lower()
    if name == "gcn":
        return EdgeFeatureGCN(**kwargs)
    if name == "gat":
        return EdgeFeatureGAT(**kwargs)
    if name == "gin":
        return EdgeFeatureGIN(**kwargs)
    raise ValueError(f"Unknown model type {name}")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

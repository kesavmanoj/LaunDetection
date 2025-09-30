"""
Production GNN training script – AML project
LAST UPDATED 2025-09-30
------------------------------------------------------------
• Fixes PyTorch 2.6 UnpicklingError  (weights_only default)
• Registers torch_geometric classes as safe globals
• Uses balanced-batch training, focal loss, larger models
"""

import os, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)
import torch.serialization as ts
from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr

from gnn_models import get_model, count_parameters

# ---------------------------------------------------------------------------#
#  --- 0.  SAFE-GLOBAL REGISTRATION  ---
# ---------------------------------------------------------------------------#
# PyTorch 2.6 blocks un-registered globals when weights_only=False.
# We explicitly allow list the two Data subclasses used in our split file.
ts.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
# ---------------------------------------------------------------------------#

BASE   = Path("/content/drive/MyDrive/LaunDetection")
GRAPHS = BASE / "data" / "graphs"
MODELS = BASE / "models"; MODELS.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"✔ CUDA: {torch.cuda.get_device_name()} | {torch.version.cuda}")

# ---------------------------------------------------------------------------#
#  1. Load split file  (weights_only=False)
# ---------------------------------------------------------------------------#
def load_split(name: str):
    fp = GRAPHS / f"ibm_aml_{name}_fixed_splits.pt"
    if not fp.exists():
        raise FileNotFoundError(f"{fp}  not found; run preprocessing first")

    d = torch.load(fp, map_location="cpu", weights_only=False)
    return d["train"], d["val"], d["test"]


train_d, val_d, test_d = load_split("hi-small")   # default

# ---------------------------------------------------------------------------#
#  2. Extreme-imbalance helpers
# ---------------------------------------------------------------------------#
def make_loss(alpha=0.99, gamma=5.0):
    ce = nn.CrossEntropyLoss(reduction="none")
    def _loss(logits, tgt):
        ce_val = ce(logits, tgt)
        pt = torch.exp(-ce_val)
        return (alpha * (1 - pt) ** gamma * ce_val).mean()
    return _loss


def balanced_sampler(data, batch=2000):
    pos = (data.y == 1).nonzero(as_tuple=False).squeeze()
    neg = (data.y == 0).nonzero(as_tuple=False).squeeze()
    while True:
        i_pos = pos[torch.randperm(len(pos))[: batch // 2]]
        i_neg = neg[torch.randperm(len(neg))[: batch // 2]]
        yield torch.cat([i_pos, i_neg])


# ---------------------------------------------------------------------------#
#  3. Model definitions
# ---------------------------------------------------------------------------#
node_dim  = train_d.x.size(1)
edge_dim  = train_d.edge_attr.size(1)
PARAMS = dict(node_feature_dim=node_dim, edge_feature_dim=edge_dim,
              hidden_dim=256, dropout=0.2)

models = {
    "GCN": get_model("gcn", **PARAMS),
    "GAT": get_model("gat", **PARAMS, num_heads=8),
    "GIN": get_model("gin", **PARAMS, aggregation="sum"),
}
for n, m in models.items():
    m.to(device)
    print(f"{n}: {count_parameters(m):,} params")

# ---------------------------------------------------------------------------#
#  4. Training utilities
# ---------------------------------------------------------------------------#
def train_epoch(m, data, loss_fn, opt, scaler, sampler):
    m.train(); tot = 0.
    steps = max(1, data.num_edges // 4000)
    for _ in range(steps):
        idx = next(sampler).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=device.type=='cuda'):
            out = m(data.x[idx], data.edge_index[:, idx], data.edge_attr[idx])
            loss = loss_fn(out, data.y[idx])
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tot += loss.item()
    return tot / steps


def evaluate(m, data, thresh=0.5):
    m.eval(); idx = torch.arange(data.num_edges, device=device)
    with torch.no_grad():
        prob = F.softmax(
            m(data.x[idx], data.edge_index[:, idx], data.edge_attr[idx]), 1
        )[:, 1].cpu().numpy()
    y = data.y.cpu().numpy()
    pred = (prob >= thresh).astype(int)
    return {
        "acc": accuracy_score(y, pred),
        "prec": precision_score(y, pred, zero_division=0),
        "rec": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, prob),
        "ap": average_precision_score(y, prob),
    }, prob


def best_threshold(p, y):
    th = np.logspace(-4, -1, 100)
    f1 = [f1_score(y, p >= t) for t in th]
    return th[int(np.argmax(f1))]


# ---------------------------------------------------------------------------#
#  5. Training loop
# ---------------------------------------------------------------------------#
hist, final = {}, {}
for name, model in models.items():
    print(f"\n===============  {name}  ===============")
    loss_fn = make_loss()
    opt = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    sampler = balanced_sampler(train_d.to(device))

    best_f1, bad = 0., 0
    for epoch in range(25):
        tl = train_epoch(model, train_d.to(device), loss_fn, opt, scaler, sampler)
        vm, pv = evaluate(model, val_d.to(device))
        if vm["f1"] > best_f1:
            best_f1, bad = vm["f1"], 0
            best_w = model.state_dict()
            best_th = best_threshold(pv, val_d.y.cpu().numpy())
        else:
            bad += 1
        print(f"ep{epoch:02d}  loss={tl:.3f}  valf1={vm['f1']:.4f}")
        if bad == 6: break

    model.load_state_dict(best_w)
    final[name] = (model, best_th)

# ---------------------------------------------------------------------------#
#  6. Test evaluation & saving
# ---------------------------------------------------------------------------#
print("\n===========  TEST  ===========")
for n, (m, th) in final.items():
    tm, _ = evaluate(m, test_d.to(device), th)
    print(f"{n:>4}  F1={tm['f1']:.4f}  AUCPR={tm['ap']:.4f}")
    fn = MODELS / f"{n.lower()}_{int(time.time())}.pt"
    torch.save({"state": m.state_dict(), "thresh": th}, fn)

print("✅ Training complete")

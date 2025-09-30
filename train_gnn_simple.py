"""
Production GNN training script – fixed for extreme class imbalance
Updated 2025-09-30
• boost_factor = 50.0
• focal loss with alpha = 0.99 & gamma = 5.0
• log-spaced threshold search 1e-4–1e-1
• balanced batch sampling (50/50) for rare class
• larger models imported from new gnn_models.py
"""

from pathlib import Path
import os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from gnn_models import get_model, count_parameters

# ---------------------------------------------------------------------------#
#  Paths & device
# ---------------------------------------------------------------------------#
BASE = Path("/content/drive/MyDrive/LaunDetection")
GRAPHS = BASE / "data" / "graphs"
MODELS = BASE / "models"; MODELS.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"✔ CUDA: {torch.cuda.get_device_name()}  |  {torch.version.cuda}")

# ---------------------------------------------------------------------------#
#  Data loading (assumes fixed preprocessing script already run)
# ---------------------------------------------------------------------------#
def load_split(name: str):
    fp = GRAPHS / f"ibm_aml_{name}_fixed_splits.pt"
    if not fp.exists():
        raise FileNotFoundError(f"{fp} not found; run preprocessing first")
    d = torch.load(fp, map_location="cpu")
    return d["train"], d["val"], d["test"]

train_d, val_d, test_d = load_split("hi-small")  # use HI-Small by default

# ---------------------------------------------------------------------------#
#  Extreme-imbalance helpers
# ---------------------------------------------------------------------------#
def class_weights(data, boost=50.0):
    y = data.y.cpu().numpy()
    pos = y.sum(); neg = len(y) - pos
    w_pos = len(y) / (2 * pos + 1e-6) * boost
    w_neg = len(y) / (2 * neg + 1e-6)
    print(f"⛔ imbalance  pos={pos:,}  neg={neg:,}  boost={boost}")
    return torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)

def focal_loss_fn(alpha=0.99, gamma=5.0):
    ce = nn.CrossEntropyLoss(reduction="none")
    def _loss(logits, tgt):
        ce_val = ce(logits, tgt)
        pt = torch.exp(-ce_val)
        return (alpha * (1 - pt)**gamma * ce_val).mean()
    return _loss

# ---------------------------------------------------------------------------#
#  Balanced batch sampler
# ---------------------------------------------------------------------------#
def balanced_edge_indices(data, batch_sz=2000):
    pos_idx = (data.y == 1).nonzero().squeeze()
    neg_idx = (data.y == 0).nonzero().squeeze()
    while True:
        p = pos_idx[torch.randperm(len(pos_idx))[: batch_sz // 2]]
        n = neg_idx[torch.randperm(len(neg_idx))[: batch_sz // 2]]
        yield torch.cat([p, n])

# ---------------------------------------------------------------------------#
#  Train one epoch
# ---------------------------------------------------------------------------#
def train_epoch(model, data, loss_fn, opt, scaler, sampler):
    model.train()
    tot_loss = 0
    for _ in range(max(1, data.num_edges // 4000)):  # ~900 steps/epoch
        idx = next(sampler)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            out = model(data.x[idx], data.edge_index[:, idx], data.edge_attr[idx])
            loss = loss_fn(out, data.y[idx])
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tot_loss += loss.item()
    return tot_loss

# ---------------------------------------------------------------------------#
#  Validation metrics
# ---------------------------------------------------------------------------#
def eval_split(model, data, thresh=None):
    model.eval(); idx = torch.arange(data.num_edges)
    with torch.no_grad():
        out = model(data.x[idx], data.edge_index[:, idx], data.edge_attr[idx])
        prob = F.softmax(out, 1)[:, 1].cpu().numpy()
    y = data.y.cpu().numpy()
    if thresh is None:
        thresh = 0.5
    pred = (prob >= thresh).astype(int)
    return {
        "acc": accuracy_score(y, pred),
        "prec": precision_score(y, pred, zero_division=0),
        "rec": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, prob),
        "ap": average_precision_score(y, prob),
    }, prob

def find_threshold(probs, y):
    ths = np.logspace(-4, -1, 100)
    f1s = [f1_score(y, probs >= t) for t in ths]
    return ths[int(np.argmax(f1s))]

# ---------------------------------------------------------------------------#
#  Model definitions
# ---------------------------------------------------------------------------#
node_dim = train_d.x.size(1); edge_dim = train_d.edge_attr.size(1)
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
#  Train loop
# ---------------------------------------------------------------------------#
hist, bests = {}, {}
for name, model in models.items():
    print(f"\n=== TRAIN {name} ===")
    loss_fn = focal_loss_fn()
    opt = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    sampler = balanced_edge_indices(train_d.to(device))
    train_d = train_d.to(device); val_d = val_d.to(device)

    best_f1 = 0; patience = 5; bad = 0
    for epoch in range(25):
        tl = train_epoch(model, train_d, loss_fn, opt, scaler, sampler)
        val_metrics, val_prob = eval_split(model, val_d, thresh=0.5)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]; bad = 0
            best_state = model.state_dict()
            best_thresh = find_threshold(val_prob, val_d.y.cpu().numpy())
        else:
            bad += 1
        print(f"ep{epoch:02d}  loss={tl:.3f}  vf1={val_metrics['f1']:.4f}")
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    hist[name] = best_f1; bests[name] = (model, best_thresh)

# ---------------------------------------------------------------------------#
#  Final evaluation
# ---------------------------------------------------------------------------#
print("\n=== FINAL TEST ===")
for name, (model, th) in bests.items():
    metrics, _ = eval_split(model, test_d.to(device), th)
    print(f"{name:>4}  F1={metrics['f1']:.4f}  AUCPR={metrics['ap']:.4f}")
    # save
    fn = MODELS / f"{name.lower()}_{int(time.time())}.pt"
    torch.save({"state": model.state_dict(), "thresh": th}, fn)

print("✅ Complete")

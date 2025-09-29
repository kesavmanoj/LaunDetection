## AI Agent Context: AML Graph Neural Network Pipeline

Use this document as your operating manual to run, extend, or debug the AML detection pipeline. It is optimized for autonomous agents executing in Google Colab with files stored under `/content/drive/MyDrive/LaunDetection`.

### Mission
- Ingest IBM AML datasets, build transaction graphs, and train edge-level binary classifiers using GNNs to detect laundering transactions.
- Guarantee: No invalid edge indices after preprocessing; robust handling of column name variations; memory-aware training and evaluation.

### Operating Environment Assumptions
- Runtime: Python 3.10+, PyTorch + PyTorch Geometric on Google Colab with Tesla T4 (or CPU fallback).
- Base path: `/content/drive/MyDrive/LaunDetection`.
- Raw inputs expected in: `data/raw/`
  - `HI-Small_accounts.csv`, `HI-Small_Trans.csv`
  - `LI-Small_accounts.csv`, `LI-Small_Trans.csv`

### System Overview (Modules and Responsibilities)
- `preprocessing_fixed.py`
  - Class `FixedAMLPreprocessor`: robust preprocessing with early filtering of invalid accounts and remapping to contiguous node ids.
  - Outputs: `ibm_aml_<dataset>_fixed_splits.pt` and `ibm_aml_<dataset>_fixed_complete.pt` in `data/graphs/`.
- `gnn_models.py`
  - Models: `EdgeFeatureGCN`, `EdgeFeatureGAT`, `EdgeFeatureGIN`.
  - All operate on node features `x`, edge index `edge_index`, edge features `edge_attr`; predict edge-level logits for 2 classes.
  - Includes `get_model(model_type)` and utilities `count_parameters`, `model_summary`.
- `train_gnn_simple.py`
  - Production trainer. Loads fixed splits, configures device, trains GCN/GAT/GIN with memory-aware sizes, CPU fallback, early stopping, and plotting.
  - Saves models under `models/` as `<name>_model.pt`.
- `train_sequential.py`
  - Two-stage transfer pipeline: train on `LI-Small`, fine-tune on `HI-Small`; evaluates cross-dataset generalization.
- `validate_data.py`
  - `AMLDataValidator` for comprehensive pre-training checks (structure, indices, consistency, class distribution, feature quality, splits).
- `colab_preprocess_fixed.py`, `colab_simple.py`, `colab_validate_data_fixed.py`, `colab_training_ready.py`
  - One-cell Colab entry points for preprocessing, training, validation, and readiness checks.

### Data Contracts
- Splits file `ibm_aml_<dataset>_fixed_splits.pt` is a dict with keys: `train`, `val`, `test`, `complete` (Data), `metadata`.
- `Data` object has: `x` [num_nodes, 10], `edge_index` [2, num_edges], `edge_attr` [num_edges, 10], `y` [num_edges].
- Invariants:
  - Edge indices are within `[0, num_nodes)`; no negatives.
  - Node/edge feature dimensions are consistent across splits.
  - Labels are binary {0,1}.

### Features Modelled
- Node features (10 dims): existence flag, hashed bank id, hashed entity id, entity type heuristic, plus default fillers.
- Edge features (10 dims): log amounts, time-of-day/week, currency mismatch, bitcoin flag, large/round amounts, clipped exchange rate, self-transaction flag.

### Runbook
1) Preprocess (MANDATORY before training)
   - Preferred: Execute the Colab cell in `colab_preprocess_fixed.py`.
   - Programmatic API:
     ```python
     from preprocessing_fixed import process_datasets
     process_datasets(['HI-Small', 'LI-Small'])
     ```
   - Outputs saved to: `data/graphs/` with `_fixed_splits.pt` suffix.

2) Validate
   - Quick: run the cell in `colab_validate_data_fixed.py`.
   - Full: use `validate_data.AMLDataValidator().validate_dataset(dataset, 'fixed')` and `generate_validation_report()`.

3) Train
   - One-shot, multi-model: run the cell in `colab_simple.py`, which calls `train_gnn_simple.main()`.
   - Transfer learning: run `train_sequential.sequential_training()`.
   - Models saved to `models/` as `gcn_model.pt`, `gat_model.pt`, `gin_model.pt` (or `_sequential_model.pt`).

4) Inference (edge classification)
   ```python
   import torch
   from gnn_models import EdgeFeatureGAT
   data = torch.load('/content/drive/MyDrive/LaunDetection/data/graphs/ibm_aml_hi-small_fixed_splits.pt', map_location='cpu', weights_only=False)
   test = data['test']
   model = EdgeFeatureGAT(node_feature_dim=test.x.shape[1], edge_feature_dim=test.edge_attr.shape[1], hidden_dim=128)
   state = torch.load('/content/drive/MyDrive/LaunDetection/models/gat_model.pt', map_location='cpu')
   model.load_state_dict(state['model_state_dict'])
   model.eval();
   with torch.no_grad():
       logits = model(test.x, test.edge_index, test.edge_attr)
       preds = logits.argmax(dim=1)
   ```

### Critical Controls and Defaults
- Device: Auto CUDA or CPU fallback (`train_gnn_simple.setup_device`).
- Memory guards:
  - Preprocessing uses chunked CSV reads (50k) with early filtering.
  - `EdgeFeatureGCN` chunks edge classification if edges > 2M.
  - Trainer downsizes `hidden_dim` and `num_heads` if edges > 3M.
- Class imbalance: Weighted `CrossEntropyLoss` based on split labels.
- Early stopping: patience=10 on validation F1.

### Failure Modes and Agent Actions
- Missing files in `data/graphs/`:
  - Action: run preprocessing first (`colab_preprocess_fixed.py` or `process_datasets`).
- Invalid edge indices detected anywhere:
  - Action: re-run fixed preprocessing; never attempt to "fix" at training time.
- Column mismatch in CSVs:
  - Action: rely on `_find_column()` in preprocessor; do not hardcode names.
- GPU OOM:
  - Action: accept CPU fallback; optionally reduce `hidden_dim`, `num_heads`, or set `memory_efficient=True` in loader.

### Extending the System (Guidelines)
- Add features: extend `_extract_edge_features` and `create_node_features` keeping 10-dim contract unless you propagate new dims through preprocessing, metadata, and models.
- New models: implement class in `gnn_models.py` with signature `forward(x, edge_index, edge_attr, batch=None)` returning edge logits; add case to `get_model`.
- New datasets: place CSVs in `data/raw/`, call `process_datasets([<name>])`; filenames follow `<Name>_accounts.csv` and `<Name>_Trans.csv`.

### Key File Paths
- Raw: `/content/drive/MyDrive/LaunDetection/data/raw/`
- Graphs: `/content/drive/MyDrive/LaunDetection/data/graphs/`
- Models: `/content/drive/MyDrive/LaunDetection/models/`
- Logs: `/content/drive/MyDrive/LaunDetection/logs/`

### Quick Capability Map
- Detection targets: edge-level laundering classification (binary).
- Architectures: GCN (3-layer, BN, skip), GAT (multi-head, layer norm, residual), GIN (MLP aggregators, multi-scale aggregation).
- Evaluation: accuracy, precision, recall, F1, AUC; comparison plots saved as `training_results.png`.

### Minimal Agent Checklists
- Pre-train: ensure `_fixed_splits.pt` exists for each dataset; run quick validator.
- Train: pick model(s), verify feature dims match data; handle device selection.
- Post-train: save state dict, run test evaluation, persist metrics and plots.

This document is designed for autonomous operation. Follow the runbook and invariants to avoid data integrity issues and ensure reproducible training.

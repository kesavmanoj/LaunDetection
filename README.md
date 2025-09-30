# Anti-Money Laundering Detection using Multi-GNN Architecture

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/kesavmanoj/LaunDetection.git)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red)](https://pytorch.org)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Compatible-orange)](https://colab.research.google.com)

## Project Overview

This project implements a **Multi-View Graph Neural Network (Multi-GNN)** architecture for detecting money laundering patterns in financial transaction networks. The system is designed to run on Google Colab with Tesla T4 GPU and uses the IBM AML Synthetic Dataset for training and evaluation.

### Key Features

- **Multi-GNN Architecture**: Two-way message passing for directed graphs
- **Class Imbalance Handling**: Weighted loss functions and sampling techniques
- **Google Colab Integration**: Complete setup and execution in Colab environment
- **Research Focus**: Overall detection performance with F1-score > 0.60 target
- **Scalable Design**: Handles graphs with 1M+ nodes and 10M+ edges

## Project Context

- **Dataset**: IBM AML Synthetic Dataset (HI-Small: 515K nodes, 5M edges)
- **Implementation Strategy**: Start with simpler Multi-GNN, gradually add complexity
- **Performance Focus**: Overall detection performance (F1-score, precision, recall)
- **Processing Mode**: Batch processing for research purposes
- **Environment**: Google Colab with Tesla T4 GPU
- **Repository**: [https://github.com/kesavmanoj/LaunDetection.git](https://github.com/kesavmanoj/LaunDetection.git)

## Quick Start

### 1. Google Colab Setup

```python
# Clone repository in Colab
!git clone https://github.com/kesavmanoj/LaunDetection.git
%cd LaunDetection

# Setup environment
!python colab/setup_colab.py

# Setup complete Colab environment
!python colab/colab_utils.py
```

### 2. Local Development Setup

```bash
# Clone repository
git clone https://github.com/kesavmanoj/LaunDetection.git
cd LaunDetection

# Install requirements
pip install -r requirements.txt

# Setup environment
python colab/setup_colab.py
```

### 3. Download Dataset

```python
# In Colab or local environment
from colab.colab_utils import download_dataset_in_colab
download_dataset_in_colab()
```

## Project Structure

```
AML_MultiGNN/
├── data/                    # Dataset storage
│   ├── raw/                 # Original dataset
│   ├── processed/          # Preprocessed graphs
│   └── splits/             # Train/val/test splits
├── models/                 # Model implementations
│   ├── mvgnn.py           # Multi-GNN implementation
│   ├── baselines.py       # Baseline models
│   └── utils.py           # Model utilities
├── utils/                  # Utility functions
│   ├── gpu_utils.py       # GPU management
│   ├── logging_utils.py   # Logging configuration
│   ├── random_utils.py    # Reproducibility
│   └── data_utils.py      # Data processing
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
├── config/                 # Configuration files
│   ├── config.yaml        # Hyperparameters
│   └── model_configs.py   # Model configurations
├── colab/                  # Colab integration
│   ├── setup_colab.py     # Environment setup
│   ├── clone_repo.py      # Repository cloning
│   └── colab_utils.py     # Colab utilities
├── results/               # Experiment results
│   ├── experiments/       # Experiment logs
│   ├── models/           # Saved models
│   └── visualizations/   # Generated plots
└── tests/                 # Unit tests
```

## Development Phases

### Phase 1: Project Setup and Environment Configuration ✅
- [x] Complete project structure
- [x] Environment setup scripts
- [x] Configuration files
- [x] Basic utility functions
- [x] GitHub repository integration
- [x] Google Colab setup automation

### Phase 2: Data Acquisition and Initial Exploration
- [ ] Download IBM AML dataset
- [ ] Data exploration notebooks
- [ ] Visualization functions
- [ ] Data quality report
- [ ] Google Colab integration scripts

### Phase 3: Graph Construction and Preprocessing
- [ ] Graph construction pipeline
- [ ] Feature engineering functions
- [ ] Preprocessing utilities
- [ ] Temporal splitting implementation
- [ ] Graph statistics and validation

### Phase 4: Multi-GNN Architecture Implementation
- [ ] Basic Multi-GNN implementation
- [ ] Model variants (MVGNN-basic, MVGNN-add)
- [ ] Training utilities
- [ ] Model analysis tools
- [ ] Architecture documentation

### Phase 5: Training Pipeline and Optimization
- [ ] Complete training pipeline
- [ ] Class imbalance handling methods
- [ ] Evaluation framework
- [ ] Optimization utilities
- [ ] Monitoring and logging system

### Phase 6: Baseline Implementation and Comparison
- [ ] Baseline model implementations
- [ ] Comparison framework
- [ ] Performance analysis tools
- [ ] Visualization utilities
- [ ] Comparative results report

### Phase 7: Experimental Analysis and Ablation Studies
- [ ] Ablation study implementation
- [ ] Overall performance analysis
- [ ] Hyperparameter optimization
- [ ] Scalability testing
- [ ] Statistical analysis tools

### Phase 8: Results Analysis and Documentation
- [ ] Comprehensive results analysis
- [ ] Complete documentation suite
- [ ] Research-ready code
- [ ] Quality assurance reports
- [ ] Final research deliverables

## Usage Examples

### Basic Usage

```python
# Import utilities
from utils.gpu_utils import get_device, print_system_info
from utils.logging_utils import setup_logging
from utils.random_utils import set_random_seed

# Setup environment
set_random_seed(42)
logger = setup_logging()
device = get_device()
print_system_info()

# Your Multi-GNN implementation here
```

### Data Loading

```python
from utils.data_utils import load_ibm_aml_dataset, preprocess_transaction_data

# Load dataset
data = load_ibm_aml_dataset("data/raw")

# Preprocess data
processed_data = preprocess_transaction_data(data)

# Create graph features
from utils.data_utils import create_graph_features
node_features, edge_features, labels = create_graph_features(processed_data)
```

### Model Training

```python
# Import model and training utilities
from models.mvgnn import MultiGNN
from utils.logging_utils import ExperimentLogger

# Setup experiment
config = {
    "model": {"hidden_dim": 64, "num_layers": 3},
    "training": {"epochs": 100, "learning_rate": 0.001}
}

with ExperimentLogger("aml_experiment", config) as logger:
    # Initialize model
    model = MultiGNN(node_features.shape[1], edge_features.shape[1], 64)
    
    # Training loop
    # ... your training code here
```

## Configuration

The project uses YAML configuration files for easy parameter management:

```yaml
# config/config.yaml
model:
  name: "MVGNN-basic"
  hidden_dim: 64
  num_layers: 3
  dropout: 0.3

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  class_weights: true

data:
  dataset: "IBM_AML_HI_Small"
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
```

## Performance Targets

- **Primary Metric**: F1-score > 0.60 for illicit class detection
- **Baseline Improvement**: 15-20% improvement over standard GNN methods
- **Overall Performance**: Strong detection performance (not pattern-specific)
- **Scalability**: Handle 1M+ nodes and 10M+ edges
- **Efficiency**: Training time < 2 hours on Tesla T4 GPU

## Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (Tesla T4 for Colab)
- 16GB+ RAM
- 50GB+ storage for dataset and models

### Python Packages
- PyTorch 1.12+ with CUDA support
- PyTorch Geometric 2.0+
- NetworkX, DGL for graph processing
- Pandas, NumPy for data manipulation
- Matplotlib, Plotly for visualization
- Scikit-learn for evaluation metrics

## Google Colab Integration

### Quick Colab Setup

1. **Open Google Colab**
2. **Clone Repository**:
   ```python
   !git clone https://github.com/kesavmanoj/LaunDetection.git
   %cd LaunDetection
   ```

3. **Setup Environment**:
   ```python
   !python colab/setup_colab.py
   ```

4. **Download Dataset**:
   ```python
   from colab.colab_utils import download_dataset_in_colab
   download_dataset_in_colab()
   ```

5. **Start Development**:
   ```python
   # Run Phase 1 setup
   !python colab/colab_utils.py
   ```

### Colab-Specific Features

- **GPU Memory Management**: Automatic memory optimization for Tesla T4
- **Data Persistence**: Google Drive integration for data storage
- **Environment Setup**: Automated package installation and configuration
- **Repository Integration**: Seamless GitHub workflow

## Development Workflow

1. **Local Development**: Code development in your local environment
2. **GitHub Push**: Push code to [https://github.com/kesavmanoj/LaunDetection.git](https://github.com/kesavmanoj/LaunDetection.git)
3. **Google Colab Execution**: Clone repository in Colab and run experiments
4. **Data Persistence**: Use Google Drive for data and model storage
5. **Results Management**: Save results back to GitHub repository

## Contributing

This is a research project focused on AML detection using Multi-GNN architecture. Contributions are welcome for:

- Model architecture improvements
- Data preprocessing enhancements
- Evaluation metric additions
- Documentation improvements
- Bug fixes and optimizations

## License

This project is for research purposes. Please cite appropriately if used in academic work.

## Contact

For questions or collaboration, please contact the project maintainer.

## Acknowledgments

- IBM AML Synthetic Dataset
- PyTorch Geometric team
- Google Colab team
- Open source community

---

**Note**: This project is designed for research and demonstration purposes. All development and experimentation should be conducted in Google Colab with Tesla T4 GPU for optimal performance.

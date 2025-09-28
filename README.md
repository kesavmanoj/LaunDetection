# Enhanced IBM AML Dataset Preprocessing

A comprehensive preprocessing pipeline for the IBM Anti-Money Laundering (AML) dataset, designed for large-scale graph neural network training with advanced features and memory-efficient processing.

## 🚀 Features

### Core Capabilities
- **Memory-efficient chunked processing** for large CSV files (millions of transactions)
- **Comprehensive temporal feature extraction** (hour, day, month, quarter, business hours, weekends)
- **Advanced graph construction** with proper node/edge mappings
- **Chronological data splitting** based on transaction timestamps
- **Enhanced logging and progress tracking** with detailed memory monitoring
- **Robust error handling** and data validation

### Feature Engineering
- **Node Features (25 dimensions)**:
  - Entity type and bank information
  - Transaction count statistics (incoming/outgoing)
  - Amount statistics (mean, std, min, max for both directions)
  - Suspicious activity rates
  - Temporal patterns (average hours, variability)
  - Activity diversity (payment formats, currencies)
  - Network behavior (self-transactions, imbalance, dominance)

- **Edge Features (16 dimensions)**:
  - Log-transformed transaction amounts
  - Temporal features (hour, day, month, quarter, weekend, business hours)
  - Encoded categorical variables (currencies, payment formats)
  - Financial patterns (exchange rates, round amounts, cross-currency)
  - Relationship indicators (self-transactions)

### Data Management
- **Temporal splits**: Train/validation/test based on chronological order
- **Metadata preservation**: Encoder classes, statistics, and configuration
- **PyTorch Geometric compatibility**: Ready-to-use Data objects
- **Configurable processing**: Support for all IBM AML dataset variants

## 📁 Project Structure

```
LaunDetection/
├── config.py                    # Configuration and paths
├── preprocessing_enhanced.py     # Main preprocessing pipeline
├── preprocessing_final.py        # Original preprocessing (for comparison)
├── example_usage.py             # Usage examples and tutorials
├── README.md                    # This file
├── data/
│   ├── raw/                     # Raw CSV files (not in git)
│   ├── processed/               # Intermediate processed data
│   ├── graphs/                  # Final PyTorch Geometric data
├── logs/                        # Processing logs
└── models/                      # Trained models (future)
```

## 🛠️ Installation

### Requirements
```bash
pip install pandas numpy torch torch-geometric scikit-learn tqdm psutil
```

### Dataset Setup
1. Download IBM AML dataset files
2. Place CSV files in `data/raw/` directory:
   - `HI-Small_accounts.csv`
   - `HI-Small_Trans.csv`
   - `HI-Medium_accounts.csv` (optional)
   - `HI-Medium_Trans.csv` (optional)
   - etc.

## 🚀 Quick Start

### Basic Usage
```python
from preprocessing_enhanced import EnhancedAMLPreprocessor

# Initialize preprocessor
preprocessor = EnhancedAMLPreprocessor(
    dataset_name='HI-Small',
    chunk_size=50000,
    log_level='INFO'
)

# Run preprocessing
complete_data, splits_data = preprocessor.run_enhanced_preprocessing()
```

### Command Line Usage
```bash
# Process HI-Small dataset
python preprocessing_enhanced.py --dataset HI-Small

# Process with custom chunk size
python preprocessing_enhanced.py --dataset HI-Medium --chunk_size 25000

# Enable debug logging
python preprocessing_enhanced.py --dataset HI-Large --log_level DEBUG
```

### Loading Processed Data
```python
import torch
from config import Config

# Load processed splits
splits_data = torch.load(Config.GRAPHS_DIR / 'ibm_aml_hi-small_enhanced_splits.pt')

train_data = splits_data['train']
val_data = splits_data['val']
test_data = splits_data['test']
metadata = splits_data['metadata']
```

## 📊 Dataset Information

### Supported Datasets
- **HI-Small**: High illicit ratio, small scale
- **HI-Medium**: High illicit ratio, medium scale  
- **HI-Large**: High illicit ratio, large scale
- **LI-Small**: Low illicit ratio, small scale
- **LI-Medium**: Low illicit ratio, medium scale
- **LI-Large**: Low illicit ratio, large scale

### Data Characteristics
- **Highly imbalanced**: Laundering transactions are typically <1% of total
- **Large scale**: Millions to hundreds of millions of transactions
- **Temporal**: Transactions span multiple days/months
- **Multi-entity**: Individuals, companies, and banks
- **Multi-currency**: Various currencies and payment formats

## 🔧 Configuration

### Key Parameters (config.py)
```python
# Processing parameters
DEFAULT_CHUNK_SIZE = 50000
MAX_MEMORY_USAGE_GB = 8
GC_FREQUENCY = 10

# Data splits
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature dimensions
NODE_FEATURE_DIM = 25
EDGE_FEATURE_DIM = 16
```

### Memory Management
- Automatic chunk size adjustment based on available memory
- Garbage collection every N chunks
- Memory usage monitoring and warnings
- Progress bars with memory statistics

## 📈 Performance

### Processing Times (Approximate)
- **HI-Small** (~100K transactions): 2-5 minutes
- **HI-Medium** (~1M transactions): 15-30 minutes  
- **HI-Large** (~10M+ transactions): 1-3 hours

### Memory Requirements
- **Minimum**: 4GB RAM for small datasets
- **Recommended**: 8GB+ RAM for medium/large datasets
- **Chunked processing**: Handles datasets larger than available RAM

## 🔍 Feature Details

### Node Features (25 dimensions)
1. **Entity Information** (0-1): Entity type, bank ID
2. **Transaction Counts** (2-4): Total, outgoing, incoming
3. **Amount Statistics** (5-12): Mean, std, min, max for both directions
4. **Suspicious Rates** (13-15): Overall, outgoing, incoming
5. **Temporal Patterns** (16-17): Average hour, hour variability
6. **Activity Diversity** (18-21): Payment formats, currencies, cross-currency rate
7. **Network Behavior** (22-24): Self-transactions, imbalance, dominance

### Edge Features (16 dimensions)
1. **Amounts** (0-1): Log-transformed received/paid amounts
2. **Temporal** (2-7): Hour, day, month, quarter, weekend, business hours
3. **Categorical** (8-10): Receiving currency, payment currency, payment format
4. **Financial** (11-13): Amount difference, exchange rate, round amounts
5. **Behavioral** (14-15): Cross-currency, self-transaction indicators

## 🔄 Comparison with Original

### Improvements over `preprocessing_final.py`
- ✅ **Better memory management**: Smarter chunking and garbage collection
- ✅ **Enhanced logging**: Comprehensive logging with file output
- ✅ **Temporal splitting**: Chronological train/val/test splits
- ✅ **More features**: 25 node features vs 20, 16 edge features vs 14
- ✅ **Better error handling**: Robust validation and error recovery
- ✅ **Metadata preservation**: Encoder classes and processing statistics
- ✅ **Configuration management**: Centralized config system
- ✅ **Progress tracking**: Detailed progress bars with memory info

## 🐛 Troubleshooting

### Common Issues

**FileNotFoundError**: CSV files not found
```bash
# Solution: Check file paths in data/raw/
ls data/raw/
```

**MemoryError**: Insufficient RAM
```bash
# Solution: Reduce chunk size
python preprocessing_enhanced.py --chunk_size 10000
```

**Empty results**: No valid transactions
```bash
# Solution: Check CSV file format and required columns
```

### Debug Mode
```bash
python preprocessing_enhanced.py --log_level DEBUG
```

## 📝 Example Workflows

### 1. Basic Preprocessing
```python
# Run example
python example_usage.py
```

### 2. Custom Processing
```python
from preprocessing_enhanced import EnhancedAMLPreprocessor

preprocessor = EnhancedAMLPreprocessor('HI-Medium', chunk_size=25000)
complete_data, splits_data = preprocessor.run_enhanced_preprocessing()

# Access training data
train_data = splits_data['train']
print(f"Training edges: {train_data.num_edges}")
print(f"Node features: {train_data.x.shape}")
print(f"Edge features: {train_data.edge_attr.shape}")
```

### 3. Analysis and Visualization
```python
import torch
from torch_geometric.utils import degree

# Load data
splits_data = torch.load('data/graphs/ibm_aml_hi-small_enhanced_splits.pt')
data = splits_data['complete']

# Compute degree statistics
degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
print(f"Average degree: {degrees.float().mean():.2f}")
```

## 🎯 Next Steps

1. **GNN Training**: Use processed data with PyTorch Geometric models
2. **Hyperparameter Tuning**: Experiment with different architectures
3. **Evaluation**: Test on chronologically split test set
4. **Deployment**: Scale to production environments

## 📄 License

This preprocessing pipeline is designed for research and educational purposes with the IBM AML dataset.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Ready to detect money laundering with graph neural networks! 🕵️‍♂️💰**

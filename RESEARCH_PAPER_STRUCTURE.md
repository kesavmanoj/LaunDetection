# AML Detection Research Paper - Clean Project Structure

## 🎯 **Project Cleanup Complete!**

**Removed 37 unnecessary files** - Project is now optimized for research paper development.

---

## 📁 **Final Project Structure**

### **Core Scripts (Essential for Research)**
```
├── model_evaluation_final_fix.py          # Main evaluation script
├── comprehensive_chunked_training.py      # Best training script  
├── multi_dataset_training.py             # Multi-dataset training
└── advanced_aml_detection.py             # Advanced AML training
```

### **Research Notebooks (Paper Development)**
```
notebooks/
├── 01_data_exploration.ipynb             # Data analysis
├── 02_graph_construction.ipynb           # Graph methodology
├── 03_multi_gnn_architecture.ipynb      # Model architecture
├── 04_training_pipeline.ipynb            # Training process
├── 05_model_training.ipynb               # Model training
└── 10_data_visualization.ipynb          # Results visualization
```

### **Documentation**
```
├── README.md                              # Project overview
├── AML_MultiGNN_Technical_Documentation.md # Technical docs
└── RESEARCH_PAPER_STRUCTURE.md           # This file
```

### **Research Paper Organization**
```
research_paper/
├── figures/                               # All research figures
├── tables/                                # Performance tables
├── models/                                # Trained model files
├── data/                                  # Processed datasets
├── results/                               # Evaluation results
└── README.md                              # Research structure guide
```

### **Supporting Files**
```
├── config/config.yaml                     # Configuration
├── requirements.txt                       # Dependencies
├── utils/                                 # Utility functions
├── data/                                  # Raw datasets
└── results/                               # Model outputs
```

---

## 🚀 **Research Paper Workflow**

### **1. Data Processing**
```bash
python comprehensive_chunked_training.py
```
- Processes HI-Small, LI-Small, HI-Medium, LI-Medium datasets
- Creates balanced datasets with 5% AML rate
- Generates 100,000+ AML cases per dataset

### **2. Model Evaluation**
```bash
python model_evaluation_final_fix.py
```
- Evaluates model across all datasets
- Generates comprehensive performance metrics
- Creates visualizations for research paper

### **3. Research Development**
- Use notebooks for analysis and visualization
- Generate figures in `research_paper/figures/`
- Create tables in `research_paper/tables/`
- Store results in `research_paper/results/`

---

## 📊 **Key Research Components**

### **Datasets**
- **HI-Small**: High-intensity small dataset
- **LI-Small**: Low-intensity small dataset  
- **HI-Medium**: High-intensity medium dataset
- **LI-Medium**: Low-intensity medium dataset

### **Model Architecture**
- **Multi-View Graph Neural Networks**
- **Dual-branch architecture** for robust learning
- **Edge-level classification** for transaction detection
- **Advanced regularization** techniques

### **Evaluation Metrics**
- **F1-Score** (weighted, macro, binary)
- **Precision & Recall** for AML detection
- **ROC-AUC** for model discrimination
- **Matthews Correlation Coefficient**
- **Cohen's Kappa Score**

### **Performance Features**
- **Large-scale evaluation** (100K+ AML cases)
- **Multi-dataset comparison**
- **Advanced visualizations**
- **Production readiness assessment**

---

## 📝 **Research Paper Sections**

### **1. Introduction**
- AML detection challenges in financial systems
- Graph Neural Networks for fraud detection
- Multi-view learning approaches

### **2. Related Work**
- Traditional AML detection methods
- Graph-based fraud detection
- Multi-view learning in finance

### **3. Methodology**
- Data preprocessing pipeline
- Graph construction methodology
- Multi-view GNN architecture
- Training and evaluation protocols

### **4. Experiments**
- Dataset descriptions and characteristics
- Experimental setup and parameters
- Evaluation metrics and methodology

### **5. Results**
- Performance across multiple datasets
- Comparison with baseline methods
- Ablation studies and analysis

### **6. Conclusion**
- Key findings and contributions
- Limitations and future work
- Practical implications

---

## ✅ **Project Status: Research Ready**

- ✅ **37 unnecessary files removed**
- ✅ **Core scripts preserved**
- ✅ **Research notebooks organized**
- ✅ **Documentation updated**
- ✅ **Research paper structure created**

**Your project is now clean and optimized for research paper development! 🚀**

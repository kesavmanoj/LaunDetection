# ============================================================================
# PRODUCTION GOOGLE COLAB CELL - GNN TRAINING FOR AML DETECTION
# Uses fixed preprocessed data with optimized models
# Copy and paste this entire cell into Google Colab
# ============================================================================

# Mount Google Drive and install packages
from google.colab import drive
import os

# Check if drive is already mounted
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
else:
    print("Drive already mounted")

# Install required packages
# !pip install torch torch-geometric scikit-learn matplotlib seaborn tqdm -q

# Change to project directory
import sys
sys.path.append('/content/drive/MyDrive/LaunDetection')

# Import and run the training
from train_gnn_simple import main

# Run training
print("🚀 Starting GNN Training for AML Detection")
print("="*60)

# Run the main training function
trained_models, results = main()

print("\n🎉 Training completed successfully!")
print("Check the models directory for saved models and plots.")

# List the most recent timestamped checkpoint directories for convenience
from pathlib import Path
MODELS_DIR = Path('/content/drive/MyDrive/LaunDetection/models')
if MODELS_DIR.exists():
    ckpts = sorted([p for p in MODELS_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    to_show = [p for p in ckpts if any(tag in p.name.lower() for tag in ['gat_', 'gin_'])][:4]
    if to_show:
        print("\nSaved checkpoints:")
        for p in to_show:
            ckpt_file = p / 'checkpoint.pt'
            print(f" - {p.name}: {ckpt_file}")
    else:
        print("No GAT/GIN checkpoints found yet in models directory.")
else:
    print("Models directory not found at /content/drive/MyDrive/LaunDetection/models")

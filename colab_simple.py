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

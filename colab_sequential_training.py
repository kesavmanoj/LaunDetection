# ============================================================================
# GOOGLE COLAB CELL - SEQUENTIAL TRAINING FOR AML DETECTION
# Train on LI-Small first, then fine-tune on HI-Small for better performance
# ============================================================================

# Mount Google Drive
from google.colab import drive
import os

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
else:
    print("Drive already mounted")

# Install required packages if needed
# !pip install torch torch-geometric scikit-learn matplotlib seaborn tqdm -q

# Change to project directory
import sys
sys.path.append('/content/drive/MyDrive/LaunDetection')

# Import and run sequential training
from train_sequential import sequential_training

print("🎯 SEQUENTIAL TRAINING FOR AML DETECTION")
print("="*60)
print("🔄 Strategy:")
print("  1. Train models on LI-Small dataset (4.8M edges)")
print("  2. Fine-tune the trained models on HI-Small dataset (5.1M edges)")
print("  3. Evaluate cross-dataset generalization")
print("="*60)
print("📊 Expected Benefits:")
print("  ✓ Better feature learning from larger LI-Small dataset")
print("  ✓ Improved performance through transfer learning")
print("  ✓ Better generalization across different data distributions")
print("  ✓ More robust models")
print("="*60)

# Run sequential training
trained_models, results = sequential_training()

if trained_models and results:
    print("\n🎉 SEQUENTIAL TRAINING COMPLETED SUCCESSFULLY!")
    print("📊 All models trained on both datasets")
    print("💾 Models saved with results")
    print("🏆 Best model identified based on cross-dataset performance")
else:
    print("\n❌ Sequential training failed!")
    print("🔧 Check the error messages above")

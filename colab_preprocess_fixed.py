# ============================================================================
# GOOGLE COLAB CELL - FIXED PREPROCESSING FOR AML DATASETS
# Run this FIRST to create properly preprocessed data without invalid edge indices
# ============================================================================

# Mount Google Drive
from google.colab import drive
import os

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
else:
    print("Drive already mounted")

# Install required packages
!pip install torch torch-geometric pandas numpy tqdm -q

# Change to project directory
import sys
sys.path.append('/content/drive/MyDrive/LaunDetection')

# Import and run the fixed preprocessing
from preprocessing_fixed import process_datasets

print("🔧 RUNNING FIXED PREPROCESSING")
print("This will create new preprocessed files without invalid edge indices")
print("="*60)

# Process both datasets with fixed preprocessing
results = process_datasets(['HI-Small', 'LI-Small'])

print("\n✅ Fixed preprocessing completed!")
print("📁 New files created:")
print("  - ibm_aml_hi-small_fixed_splits.pt")
print("  - ibm_aml_li-small_fixed_splits.pt")
print("\n🎯 These files will have NO invalid edge indices!")
print("🚀 You can now run training without edge validation issues.")

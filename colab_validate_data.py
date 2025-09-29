# ============================================================================
# GOOGLE COLAB CELL - VALIDATE PREPROCESSED DATA
# Run this to thoroughly validate your data before training
# ============================================================================

# Mount Google Drive
from google.colab import drive
import os

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
else:
    print("Drive already mounted")

# Install required packages if needed
# !pip install torch torch-geometric pandas numpy matplotlib seaborn -q

# Change to project directory
import sys
sys.path.append('/content/drive/MyDrive/LaunDetection')

# Import and run validation
from validate_data import validate_all_datasets

print("🔍 COMPREHENSIVE DATA VALIDATION")
print("="*60)
print("This will validate:")
print("✓ Basic data structure")
print("✓ Edge indices validity")
print("✓ Data consistency")
print("✓ Class distributions")
print("✓ Feature quality")
print("✓ Temporal splits")
print("✓ Complete data consistency")
print("="*60)

# Run comprehensive validation
is_ready = validate_all_datasets()

if is_ready:
    print("\n🎉 VALIDATION PASSED - READY TO TRAIN!")
    print("🚀 You can now run colab_simple.py with confidence")
else:
    print("\n⚠️ VALIDATION FAILED - DO NOT TRAIN YET")
    print("🔧 Fix the issues above before proceeding")

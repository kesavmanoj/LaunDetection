# ============================================================================
# GOOGLE COLAB CELL - DEBUG COLUMN NAMES
# Run this to check the actual column names in the raw data
# ============================================================================

# Import and run the column checker
import sys
sys.path.append('/content/drive/MyDrive/LaunDetection')

from debug_columns import check_column_names

print("🔍 CHECKING ACTUAL COLUMN NAMES IN RAW DATA")
print("="*50)

check_column_names()

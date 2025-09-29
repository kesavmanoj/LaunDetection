# ============================================================================
# GOOGLE COLAB CELL - UPDATE FILES FROM GITHUB
# Run this FIRST to ensure you have the latest preprocessing fixes
# ============================================================================

# Mount Google Drive
from google.colab import drive
import os

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
else:
    print("Drive already mounted")

# Change to project directory
os.chdir('/content/drive/MyDrive/LaunDetection')

# Pull latest changes from GitHub
print("🔄 Pulling latest changes from GitHub...")
!git pull origin main

print("✅ Files updated!")
print("📁 You now have the latest preprocessing fixes")
print("🚀 Now run colab_preprocess_fixed.py")

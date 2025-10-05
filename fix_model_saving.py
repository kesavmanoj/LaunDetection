#!/usr/bin/env python3
"""
Fix Model Saving for Google Colab
==================================

This script ensures models are properly saved in Google Colab environment
and creates the necessary directory structure.
"""

import os
import sys
import torch
from pathlib import Path

def setup_colab_model_directories():
    """Setup model directories for Google Colab"""
    print("🔧 Setting up Google Colab Model Directories")
    print("=" * 50)
    
    # Define paths
    base_path = '/content/drive/MyDrive/LaunDetection'
    models_path = os.path.join(base_path, 'models')
    checkpoints_path = os.path.join(base_path, 'checkpoints')
    
    print(f"📁 Base path: {base_path}")
    print(f"📁 Models path: {models_path}")
    print(f"📁 Checkpoints path: {checkpoints_path}")
    
    # Create directories if they don't exist
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    
    print(f"✅ Created models directory: {models_path}")
    print(f"✅ Created checkpoints directory: {checkpoints_path}")
    
    # Verify directories exist
    if os.path.exists(models_path):
        print(f"✅ Models directory verified: {models_path}")
    else:
        print(f"❌ Failed to create models directory: {models_path}")
        return False
    
    if os.path.exists(checkpoints_path):
        print(f"✅ Checkpoints directory verified: {checkpoints_path}")
    else:
        print(f"❌ Failed to create checkpoints directory: {checkpoints_path}")
        return False
    
    return True

def check_existing_models():
    """Check for existing models in the project"""
    print("\n🔍 Checking for existing models...")
    
    # Check multiple possible locations
    possible_paths = [
        '/content/drive/MyDrive/LaunDetection/models',
        '/content/drive/MyDrive/LaunDetection/checkpoints',
        '/content/drive/MyDrive/LaunDetection',
        '/content/LaunDetection/models',
        '/content/LaunDetection'
    ]
    
    found_models = []
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"📁 Checking: {path}")
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(('.pth', '.pt', '.pkl')):
                        full_path = os.path.join(root, file)
                        size = os.path.getsize(full_path)
                        found_models.append((full_path, size))
                        print(f"  📄 {file} ({size:,} bytes)")
    
    if found_models:
        print(f"\n✅ Found {len(found_models)} model files:")
        for path, size in found_models:
            print(f"  📄 {path} ({size:,} bytes)")
    else:
        print("\n❌ No model files found")
    
    return found_models

def create_model_saving_script():
    """Create a script to properly save models in Colab"""
    script_content = '''#!/usr/bin/env python3
"""
Enhanced Model Saving for Google Colab
======================================

This script provides robust model saving functionality for Google Colab.
"""

import os
import torch
import pickle
from datetime import datetime

class ColabModelSaver:
    """Enhanced model saver for Google Colab environment"""
    
    def __init__(self, base_path='/content/drive/MyDrive/LaunDetection'):
        self.base_path = base_path
        self.models_path = os.path.join(base_path, 'models')
        self.checkpoints_path = os.path.join(base_path, 'checkpoints')
        
        # Create directories
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        
        print(f"🔧 ColabModelSaver initialized")
        print(f"📁 Models path: {self.models_path}")
        print(f"📁 Checkpoints path: {checkpoints_path}")
    
    def save_model(self, model, model_name, additional_info=None):
        """Save model with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model weights
        model_path = os.path.join(self.models_path, f"{model_name}_{timestamp}.pth")
        torch.save(model.state_dict(), model_path)
        
        # Save full model
        full_model_path = os.path.join(self.models_path, f"{model_name}_full_{timestamp}.pth")
        torch.save(model, full_model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'model_path': model_path,
            'full_model_path': full_model_path,
            'additional_info': additional_info or {}
        }
        
        metadata_path = os.path.join(self.models_path, f"{model_name}_metadata_{timestamp}.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Model saved: {model_name}")
        print(f"  📄 Weights: {model_path}")
        print(f"  📄 Full model: {full_model_path}")
        print(f"  📄 Metadata: {metadata_path}")
        
        return model_path, full_model_path, metadata_path
    
    def save_checkpoint(self, model, optimizer, epoch, loss, model_name):
        """Save training checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': timestamp
        }
        
        checkpoint_path = os.path.join(self.checkpoints_path, f"{model_name}_checkpoint_{timestamp}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        print(f"✅ Checkpoint saved: {model_name} (epoch {epoch})")
        print(f"  📄 Path: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_model(self, model_path, model_class=None):
        """Load model from path"""
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None
        
        try:
            if model_class:
                # Load state dict
                model = model_class()
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"✅ Model loaded: {model_path}")
                return model
            else:
                # Load full model
                model = torch.load(model_path, map_location='cpu')
                print(f"✅ Full model loaded: {model_path}")
                return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def list_models(self):
        """List all available models"""
        print("📋 Available Models:")
        print("=" * 30)
        
        for root, dirs, files in os.walk(self.models_path):
            for file in files:
                if file.endswith('.pth'):
                    full_path = os.path.join(root, file)
                    size = os.path.getsize(full_path)
                    print(f"  📄 {file} ({size:,} bytes)")

# Usage example
if __name__ == "__main__":
    saver = ColabModelSaver()
    saver.list_models()
'''
    
    script_path = '/content/drive/MyDrive/LaunDetection/colab_model_saver.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created model saving script: {script_path}")
    return script_path

def main():
    """Main function to fix model saving"""
    print("🚀 Fixing Model Saving for Google Colab")
    print("=" * 50)
    
    # Setup directories
    if not setup_colab_model_directories():
        print("❌ Failed to setup directories")
        return
    
    # Check existing models
    found_models = check_existing_models()
    
    # Create model saving script
    script_path = create_model_saving_script()
    
    print("\n🎯 Next Steps:")
    print("1. Run the model saving script in your training code")
    print("2. Use ColabModelSaver for robust model persistence")
    print("3. Check the models directory for saved files")
    
    print(f"\n✅ Model saving setup complete!")
    print(f"📁 Models will be saved to: /content/drive/MyDrive/LaunDetection/models")

if __name__ == "__main__":
    main()

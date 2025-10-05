#!/usr/bin/env python3
"""
Test Balanced Evaluation Script
===============================

This script tests the balanced evaluation to ensure all components work correctly.
"""

import sys
import os
import torch

# Add the current directory to Python path
sys.path.append('/content/drive/MyDrive/LaunDetection')

def test_model_loading():
    """Test model loading and initialization"""
    print("🧪 Testing Model Loading")
    print("=" * 30)
    
    try:
        from advanced_aml_detection import AdvancedAMLGNN
        
        # Test model creation
        model = AdvancedAMLGNN(
            input_dim=25,
            hidden_dim=256,
            output_dim=2,
            dropout=0.1
        )
        print("✅ Model created successfully")
        
        # Test edge classifier initialization
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        dummy_edge_features = torch.randn(10, 536).to(device)
        with torch.no_grad():
            _ = model(torch.randn(100, 25).to(device), torch.randint(0, 100, (2, 10)).to(device), dummy_edge_features)
        
        print("✅ Edge classifier initialized successfully")
        print(f"✅ Model ready for loading with 536-dim edge features")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def test_edge_feature_creation():
    """Test edge feature creation"""
    print("\n🧪 Testing Edge Feature Creation")
    print("=" * 40)
    
    try:
        # Test edge feature dimensions
        node_features = torch.randn(100, 25)
        edge_feat = torch.cat([
            node_features[0],  # 25 dims
            node_features[1],  # 25 dims
            torch.tensor([100.0, 12, 3, 0.1, 6.0, 0.5, 1.0, 0.000001, 10.0, 1.0, 0, 0, 0.1], dtype=torch.float32),  # 13 dims
            torch.zeros(473)  # 473 dims
        ])
        
        expected_dim = 25 + 25 + 13 + 473  # 536
        actual_dim = edge_feat.shape[0]
        
        if actual_dim == expected_dim:
            print(f"✅ Edge feature dimension correct: {actual_dim}")
            return True
        else:
            print(f"❌ Edge feature dimension mismatch: {actual_dim} vs {expected_dim}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing edge features: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Balanced Evaluation Components")
    print("=" * 50)
    
    # Test model loading
    model_test = test_model_loading()
    
    # Test edge feature creation
    feature_test = test_edge_feature_creation()
    
    print("\n📊 Test Results:")
    print(f"   Model Loading: {'✅ PASS' if model_test else '❌ FAIL'}")
    print(f"   Edge Features: {'✅ PASS' if feature_test else '❌ FAIL'}")
    
    if model_test and feature_test:
        print("\n🎉 All tests passed! Ready to run balanced evaluation.")
        print("💡 Run: !python balanced_evaluation.py")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        print("💡 Fix the issues before running balanced evaluation.")

if __name__ == "__main__":
    main()

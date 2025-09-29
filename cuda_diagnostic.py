# ============================================================================
# CUDA DIAGNOSTIC AND RESET SCRIPT FOR GOOGLE COLAB
# Run this cell FIRST to diagnose and fix CUDA issues
# ============================================================================

import torch
import gc
import os
import subprocess
import time

def diagnose_cuda():
    """Comprehensive CUDA diagnostics"""
    print("🔧 COMPREHENSIVE CUDA DIAGNOSTICS")
    print("="*60)
    
    # Basic CUDA availability
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - check Colab runtime type")
        return False
    
    # CUDA version info
    print(f"📊 CUDA Version: {torch.version.cuda}")
    print(f"📊 PyTorch Version: {torch.__version__}")
    print(f"📊 Device Count: {torch.cuda.device_count()}")
    print(f"📊 Current Device: {torch.cuda.current_device()}")
    print(f"📊 Device Name: {torch.cuda.get_device_name()}")
    
    # Memory diagnostics
    print(f"\n💾 MEMORY DIAGNOSTICS:")
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    memory_cached = torch.cuda.memory_cached() / 1024**3 if hasattr(torch.cuda, 'memory_cached') else 0
    
    print(f"  Allocated: {memory_allocated:.2f} GB")
    print(f"  Reserved: {memory_reserved:.2f} GB")
    print(f"  Cached: {memory_cached:.2f} GB")
    
    # Check for corruption
    if memory_reserved > 6.0:
        print(f"  ⚠️ HIGH RESERVED MEMORY DETECTED!")
        print(f"  This indicates CUDA memory corruption")
        return False
    
    # Test basic operations
    print(f"\n🧪 TESTING BASIC CUDA OPERATIONS:")
    try:
        # Test 1: Simple tensor creation
        print("  Test 1: Tensor creation...", end=" ")
        x = torch.randn(100, 100).cuda()
        print("✅")
        
        # Test 2: Basic math operations
        print("  Test 2: Math operations...", end=" ")
        y = x @ x.T
        print("✅")
        
        # Test 3: Memory cleanup
        print("  Test 3: Memory cleanup...", end=" ")
        del x, y
        torch.cuda.empty_cache()
        print("✅")
        
        # Test 4: Neural network operations
        print("  Test 4: Neural network ops...", end=" ")
        import torch.nn as nn
        model = nn.Linear(100, 50).cuda()
        x = torch.randn(32, 100).cuda()
        output = model(x)
        del model, x, output
        torch.cuda.empty_cache()
        print("✅")
        
        print(f"\n✅ ALL CUDA TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def reset_cuda():
    """Attempt to reset CUDA state"""
    print(f"\n🔄 ATTEMPTING CUDA RESET...")
    
    try:
        # Method 1: Clear all cached memory
        print("  Step 1: Clearing CUDA cache...", end=" ")
        torch.cuda.empty_cache()
        print("✅")
        
        # Method 2: Force garbage collection
        print("  Step 2: Force garbage collection...", end=" ")
        gc.collect()
        print("✅")
        
        # Method 3: Reset CUDA context (if available)
        print("  Step 3: Reset CUDA context...", end=" ")
        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
            torch.cuda.reset_accumulated_memory_stats()
        if hasattr(torch.cuda, 'reset_max_memory_allocated'):
            torch.cuda.reset_max_memory_allocated()
        if hasattr(torch.cuda, 'reset_max_memory_cached'):
            torch.cuda.reset_max_memory_cached()
        print("✅")
        
        # Method 4: Synchronize device
        print("  Step 4: Synchronize device...", end=" ")
        torch.cuda.synchronize()
        print("✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return False

def restart_runtime_instructions():
    """Provide instructions for runtime restart"""
    print(f"\n🔄 CUDA RESET FAILED - RUNTIME RESTART REQUIRED")
    print("="*60)
    print("To fix the CUDA corruption:")
    print("1. Go to Runtime → Restart Runtime")
    print("2. Wait for restart to complete")
    print("3. Re-run your training script")
    print("4. The CUDA memory will be clean after restart")
    print("="*60)

def check_colab_gpu_type():
    """Check what GPU type is allocated"""
    print(f"\n🖥️ GPU ALLOCATION CHECK:")
    
    try:
        # Get GPU info from nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            print(f"  GPU: {gpu_info[0]}")
            print(f"  Total Memory: {gpu_info[1]} MB")
            print(f"  Used Memory: {gpu_info[2]} MB") 
            print(f"  Free Memory: {gpu_info[3]} MB")
            
            # Check if it's a good GPU for training
            gpu_name = gpu_info[0].lower()
            if 't4' in gpu_name:
                print(f"  ✅ Tesla T4 - Good for training (16GB VRAM)")
            elif 'k80' in gpu_name:
                print(f"  ⚠️ Tesla K80 - Older GPU, may have issues")
            elif 'p100' in gpu_name:
                print(f"  ✅ Tesla P100 - Excellent for training")
            elif 'v100' in gpu_name:
                print(f"  ✅ Tesla V100 - Excellent for training")
            else:
                print(f"  ❓ Unknown GPU type")
        else:
            print(f"  ❌ Could not get GPU info")
            
    except Exception as e:
        print(f"  ❌ Error checking GPU: {e}")

def main():
    """Main diagnostic function"""
    print("🚀 CUDA DIAGNOSTIC TOOL")
    print("Run this BEFORE your training script to check CUDA health")
    print()
    
    # Check GPU allocation
    check_colab_gpu_type()
    
    # Diagnose CUDA
    cuda_healthy = diagnose_cuda()
    
    if not cuda_healthy:
        print(f"\n❌ CUDA IS NOT HEALTHY")
        
        # Attempt reset
        reset_success = reset_cuda()
        
        if reset_success:
            print(f"\n🔄 Testing CUDA after reset...")
            cuda_healthy = diagnose_cuda()
            
            if cuda_healthy:
                print(f"\n✅ CUDA RESET SUCCESSFUL!")
                print(f"You can now run your training script.")
            else:
                restart_runtime_instructions()
        else:
            restart_runtime_instructions()
    else:
        print(f"\n✅ CUDA IS HEALTHY!")
        print(f"You can proceed with GPU training.")
    
    return cuda_healthy

# Run diagnostics
if __name__ == "__main__":
    cuda_healthy = main()
    
    if cuda_healthy:
        print(f"\n🎯 RECOMMENDATION: Proceed with GPU training")
    else:
        print(f"\n🎯 RECOMMENDATION: Restart runtime, then try again")

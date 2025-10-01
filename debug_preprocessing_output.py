#!/usr/bin/env python3
"""
Debug Preprocessing Output
=========================

This script helps debug where the preprocessing files were actually saved.
"""

import os
import glob
from pathlib import Path

print("🔍 Debugging Preprocessing Output")
print("=" * 60)

def find_preprocessing_files():
    """Find where preprocessing files were actually saved"""
    print("🔍 Searching for preprocessing files...")
    
    # Check common locations
    search_paths = [
        "/content/drive/MyDrive/LaunDetection/data/processed",
        "/content/drive/MyDrive/LaunDetection/data",
        "/content/drive/MyDrive/LaunDetection",
        "/content",
        "/tmp"
    ]
    
    # Look for common preprocessing file patterns
    file_patterns = [
        "*.pkl",
        "*checkpoint*",
        "*node_features*",
        "*edge_features*",
        "*graph*",
        "*weights*",
        "*imbalanced*"
    ]
    
    found_files = {}
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            print(f"\n📁 Checking: {search_path}")
            try:
                files = os.listdir(search_path)
                print(f"   Found {len(files)} files")
                
                # Check for preprocessing files
                for pattern in file_patterns:
                    matching_files = glob.glob(os.path.join(search_path, pattern))
                    if matching_files:
                        found_files[search_path] = matching_files
                        print(f"   ✅ {pattern}: {len(matching_files)} files")
                        for file in matching_files[:5]:  # Show first 5
                            print(f"      - {os.path.basename(file)}")
                        if len(matching_files) > 5:
                            print(f"      ... and {len(matching_files) - 5} more")
            except Exception as e:
                print(f"   ❌ Error accessing {search_path}: {e}")
        else:
            print(f"❌ Path not found: {search_path}")
    
    return found_files

def check_notebook_output():
    """Check if the notebook actually ran successfully"""
    print("\n📓 Checking Notebook Execution")
    print("-" * 40)
    
    # Check if the notebook was actually executed
    notebook_path = "/content/drive/MyDrive/LaunDetection/notebooks/08_simple_enhanced_preprocessing.ipynb"
    
    if os.path.exists(notebook_path):
        print(f"✅ Notebook exists: {notebook_path}")
        
        # Try to read the notebook to see if it has output
        try:
            import json
            with open(notebook_path, 'r') as f:
                notebook = json.load(f)
            
            # Check for execution outputs
            output_cells = 0
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'code':
                    if cell.get('outputs'):
                        output_cells += 1
            
            print(f"   Notebook has {output_cells} cells with outputs")
            
            if output_cells > 0:
                print("   ✅ Notebook appears to have been executed")
            else:
                print("   ⚠️  Notebook may not have been executed")
                
        except Exception as e:
            print(f"   ❌ Error reading notebook: {e}")
    else:
        print(f"❌ Notebook not found: {notebook_path}")

def check_memory_and_permissions():
    """Check memory usage and permissions"""
    print("\n💾 Checking System Resources")
    print("-" * 40)
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   Available memory: {memory.available / 1024 / 1024 / 1024:.2f} GB")
        print(f"   Used memory: {memory.used / 1024 / 1024 / 1024:.2f} GB")
        print(f"   Memory usage: {memory.percent}%")
        
        if memory.percent > 90:
            print("   ⚠️  High memory usage - may have caused crashes")
        else:
            print("   ✅ Memory usage looks good")
            
    except ImportError:
        print("   ⚠️  psutil not available - cannot check memory")
    
    # Check disk space
    try:
        import shutil
        disk_usage = shutil.disk_usage("/content")
        free_gb = disk_usage.free / 1024 / 1024 / 1024
        print(f"   Available disk space: {free_gb:.2f} GB")
        
        if free_gb < 1:
            print("   ⚠️  Low disk space - may have caused write failures")
        else:
            print("   ✅ Disk space looks good")
            
    except Exception as e:
        print(f"   ❌ Error checking disk space: {e}")

def suggest_solutions():
    """Suggest solutions based on findings"""
    print("\n💡 Suggested Solutions")
    print("-" * 40)
    
    print("Based on the findings, here are the recommended solutions:")
    print()
    print("1. 🔄 Re-run Enhanced Preprocessing:")
    print("   %run notebooks/08_simple_enhanced_preprocessing.ipynb")
    print("   - Make sure to run ALL cells in the notebook")
    print("   - Watch for any error messages")
    print("   - Check memory usage during execution")
    print()
    print("2. 🔧 Run with Smaller Sample Size:")
    print("   - Modify the sample_size parameter to 1000 or 5000")
    print("   - This will reduce memory usage and processing time")
    print()
    print("3. 📊 Check Notebook Execution:")
    print("   - Open the notebook directly in Colab")
    print("   - Run each cell manually to see where it fails")
    print("   - Look for error messages in the output")
    print()
    print("4. 🚀 Alternative: Use the Working 4-Node Model:")
    print("   - Your 4-node model is working perfectly")
    print("   - You can scale it up gradually")
    print("   - Start with 100 nodes, then 1000, then 10000")

def main():
    """Main debugging function"""
    print("🔍 Debugging Preprocessing Output")
    print("=" * 60)
    
    # Find preprocessing files
    found_files = find_preprocessing_files()
    
    # Check notebook execution
    check_notebook_output()
    
    # Check system resources
    check_memory_and_permissions()
    
    # Summary
    print("\n📋 Summary")
    print("-" * 40)
    
    if found_files:
        print("✅ Found preprocessing files in:")
        for path, files in found_files.items():
            print(f"   {path}: {len(files)} files")
    else:
        print("❌ No preprocessing files found anywhere")
        print("   This suggests the preprocessing didn't complete successfully")
    
    # Suggest solutions
    suggest_solutions()

if __name__ == "__main__":
    main()

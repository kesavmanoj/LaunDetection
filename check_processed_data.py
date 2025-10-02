#!/usr/bin/env python3
"""
Check Processed Data in Google Colab
====================================

This script checks what processed data exists in Google Drive
and helps diagnose path issues.
"""

import os
import sys

def check_google_drive_structure():
    """Check the Google Drive structure for processed data"""
    print("🔍 Checking Google Drive Structure...")
    print("=" * 50)
    
    # Check if Google Drive is mounted
    drive_paths = [
        "/content/drive/MyDrive",
        "/content/drive",
        "/content"
    ]
    
    mounted_drive = None
    for path in drive_paths:
        if os.path.exists(path):
            mounted_drive = path
            print(f"✅ Found mounted drive at: {path}")
            break
    
    if not mounted_drive:
        print("❌ Google Drive not mounted!")
        print("💡 Run: from google.colab import drive; drive.mount('/content/drive')")
        return
    
    # Check for LaunDetection folder
    laun_detection_paths = [
        f"{mounted_drive}/LaunDetection",
        f"{mounted_drive}/LaunDetection/",
        "/content/LaunDetection",
        "/content/LaunDetection/"
    ]
    
    found_laun_detection = None
    for path in laun_detection_paths:
        if os.path.exists(path):
            found_laun_detection = path
            print(f"✅ Found LaunDetection folder at: {path}")
            break
    
    if not found_laun_detection:
        print("❌ LaunDetection folder not found!")
        print("💡 Checked paths:")
        for path in laun_detection_paths:
            print(f"   - {path}")
        return
    
    # Check for data folder
    data_paths = [
        f"{found_laun_detection}/data",
        f"{found_laun_detection}/data/",
        f"{found_laun_detection}/data/processed",
        f"{found_laun_detection}/data/processed/"
    ]
    
    found_data = None
    for path in data_paths:
        if os.path.exists(path):
            found_data = path
            print(f"✅ Found data folder at: {path}")
            break
    
    if not found_data:
        print("❌ Data folder not found!")
        print("💡 Checked paths:")
        for path in data_paths:
            print(f"   - {path}")
        return
    
    # Check for processed folder
    processed_paths = [
        f"{found_data}/processed",
        f"{found_data}/processed/",
        f"{found_laun_detection}/data/processed",
        f"{found_laun_detection}/data/processed/"
    ]
    
    found_processed = None
    for path in processed_paths:
        if os.path.exists(path):
            found_processed = path
            print(f"✅ Found processed folder at: {path}")
            break
    
    if not found_processed:
        print("❌ Processed folder not found!")
        print("💡 Checked paths:")
        for path in processed_paths:
            print(f"   - {path}")
        print("💡 Run preprocessing first: python working_preprocessing.py")
        return
    
    # List contents of processed folder
    print(f"\n📁 Contents of {found_processed}:")
    try:
        files = os.listdir(found_processed)
        if files:
            for file in files:
                file_path = os.path.join(found_processed, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"   📄 {file} ({size:,} bytes)")
                else:
                    print(f"   📁 {file}/")
        else:
            print("   (empty)")
    except Exception as e:
        print(f"   ❌ Error listing contents: {str(e)}")
    
    # Check for specific processed files
    expected_files = [
        "HI-Small_processed.pkl",
        "LI-Small_processed.pkl", 
        "HI-Medium_processed.pkl",
        "LI-Medium_processed.pkl"
    ]
    
    print(f"\n🔍 Checking for expected processed files:")
    found_files = []
    for file in expected_files:
        file_path = os.path.join(found_processed, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {file} ({size:,} bytes)")
            found_files.append(file)
        else:
            print(f"   ❌ {file} (not found)")
    
    if found_files:
        print(f"\n✅ Found {len(found_files)} processed files!")
        print("💡 You can now run visualization scripts")
    else:
        print(f"\n❌ No processed files found!")
        print("💡 Run preprocessing first: python working_preprocessing.py")
    
    return found_processed, found_files

def check_alternative_locations():
    """Check alternative locations for processed data"""
    print("\n🔍 Checking Alternative Locations...")
    print("=" * 50)
    
    # Check common alternative locations
    alt_locations = [
        "/content/drive/MyDrive/LaunDetection/data/processed",
        "/content/drive/MyDrive/LaunDetection/data/processed/",
        "/content/LaunDetection/data/processed",
        "/content/LaunDetection/data/processed/",
        "/content/drive/MyDrive/data/processed",
        "/content/drive/MyDrive/data/processed/",
        "/content/data/processed",
        "/content/data/processed/",
        "/content/processed",
        "/content/processed/"
    ]
    
    found_locations = []
    for location in alt_locations:
        if os.path.exists(location):
            try:
                files = os.listdir(location)
                if files:
                    found_locations.append((location, files))
                    print(f"✅ {location}: {len(files)} files")
                else:
                    print(f"📁 {location}: (empty)")
            except Exception as e:
                print(f"❌ {location}: Error - {str(e)}")
        else:
            print(f"❌ {location}: Not found")
    
    return found_locations

def main():
    """Main diagnostic function"""
    print("🔍 Google Colab Processed Data Diagnostic")
    print("=" * 60)
    
    # Check Google Drive structure
    processed_path, found_files = check_google_drive_structure()
    
    # Check alternative locations
    alt_locations = check_alternative_locations()
    
    # Summary
    print("\n📊 Diagnostic Summary:")
    print("=" * 30)
    
    if processed_path and found_files:
        print(f"✅ Processed data found at: {processed_path}")
        print(f"✅ Found {len(found_files)} processed files")
        print("💡 You can now run visualization scripts")
    elif alt_locations:
        print(f"✅ Found processed data in {len(alt_locations)} alternative locations")
        for location, files in alt_locations:
            print(f"   - {location}: {len(files)} files")
        print("💡 Update visualization scripts to use these paths")
    else:
        print("❌ No processed data found anywhere!")
        print("💡 Run preprocessing first: python working_preprocessing.py")
        print("💡 Make sure Google Drive is properly mounted")

if __name__ == "__main__":
    main()

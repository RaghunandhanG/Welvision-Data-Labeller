#!/usr/bin/env python3
"""
Test script to verify dataset name validation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolo_labeler_app import DatabaseManager

def test_dataset_validation():
    """Test the dataset validation logic"""
    print("🔍 Testing dataset validation logic...")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Get existing datasets
    datasets = db_manager.get_datasets()
    print(f"\n📁 Found {len(datasets)} datasets in database:")
    
    for i, dataset in enumerate(datasets, 1):
        name = dataset['name']
        path = dataset['path']
        print(f"   {i}. '{name}' -> {path}")
        print(f"      Name length: {len(name)}")
        print(f"      Name repr: {repr(name)}")
        print()
    
    # Test some sample names
    test_names = [
        "sample1",
        "Sample1", 
        "SAMPLE1",
        "sample 2",
        "Sample 2",
        "newdataset",
        "test123",
        "sample10"
    ]
    
    print("\n🧪 Testing name collision detection:")
    for test_name in test_names:
        exists = check_name_exists(datasets, test_name)
        status = "❌ EXISTS" if exists else "✅ AVAILABLE"
        print(f"   '{test_name}' -> {status}")

def check_name_exists(datasets, dataset_name):
    """Replicate the validation logic"""
    try:
        if not dataset_name or not dataset_name.strip():
            return False
            
        clean_name = dataset_name.strip().lower()
        
        for dataset in datasets:
            db_name = dataset['name'].strip().lower()
            if db_name == clean_name:
                return True
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    test_dataset_validation()

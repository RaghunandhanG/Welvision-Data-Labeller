#!/usr/bin/env python3
"""
Test script for the new workspace.upload_dataset method.
This demonstrates the efficient bulk upload approach now implemented in the app.
"""

import os
import json
from datetime import datetime

def test_workspace_upload():
    """Test the workspace upload method that's now implemented in the app."""
    
    print("🧪 Testing workspace.upload_dataset method...")
    print("📖 This is the same method now used in the YOLO Data Labeller app")
    print()
    
    # Get API key
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        api_key = input("Enter your Roboflow API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided.")
        return False
    
    # Get dataset path
    dataset_path = input("Enter path to your COCO dataset folder (with images/ and annotations.json): ").strip()
    if not dataset_path:
        dataset_path = "./datasets/sample 2"  # Default path
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Check dataset structure
    images_dir = os.path.join(dataset_path, "images")
    annotations_file = os.path.join(dataset_path, "annotations.json")
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return False
        
    if not os.path.exists(annotations_file):
        print(f"❌ Annotations file not found: {annotations_file}")
        return False
    
    # Get project ID
    project_id = input("Enter Roboflow project ID (or press Enter for 'test-workspace-upload'): ").strip()
    if not project_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_id = f"test-workspace-upload-{timestamp}"
    
    print(f"\n📋 Upload Configuration:")
    print(f"   🔑 API Key: {api_key[:10]}...")
    print(f"   📂 Dataset: {dataset_path}")
    print(f"   📦 Project: {project_id}")
    print(f"   ⚡ Workers: 12")
    print(f"   📋 License: MIT")
    print(f"   🎯 Type: object-detection")
    print()
    
    try:
        # Load and show dataset info
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        num_images = len(coco_data.get('images', []))
        num_annotations = len(coco_data.get('annotations', []))
        num_categories = len(coco_data.get('categories', []))
        
        print(f"📊 Dataset Statistics:")
        print(f"   🖼️ Images: {num_images}")
        print(f"   🏷️ Annotations: {num_annotations}")
        print(f"   📈 Categories: {num_categories}")
        print()
        
        if num_images == 0:
            print("❌ No images found in dataset")
            return False
        
        # Confirm upload
        confirm = input("🚀 Start workspace upload? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ Upload cancelled")
            return False
        
        # Import roboflow
        try:
            import roboflow
            print("✅ Roboflow library loaded")
        except ImportError:
            print("❌ Roboflow library not found. Please install: pip install roboflow")
            return False
        
        # Initialize Roboflow
        print("🔄 Initializing Roboflow connection...")
        rf = roboflow.Roboflow(api_key=api_key)
        
        # Get workspace
        workspace = rf.workspace()
        workspace_name = getattr(workspace, 'name', 'default')
        print(f"✅ Connected to workspace: {workspace_name}")
        
        print("🚀 Starting workspace.upload_dataset()...")
        print("-" * 50)
        
        # Upload dataset using workspace method (same as in the app)
        result = workspace.upload_dataset(
            dataset_path,           # Dataset directory path
            project_id,             # Project ID (will create or use existing)
            num_workers=12,         # High performance with 12 workers
            project_license="MIT",  # License type
            project_type="object-detection",  # Project type
            batch_name=None,        # No specific batch name
            num_retries=0          # No retries for faster upload
        )
        
        print("-" * 50)
        print("🎉 Workspace upload completed!")
        print(f"📊 Upload result: {result}")
        print(f"📁 Dataset uploaded to project: {project_id}")
        print(f"🌐 Check your Roboflow dashboard to view the uploaded data")
        print()
        print("✅ The workspace upload method works correctly!")
        print("🚀 Your YOLO Data Labeller app now uses this efficient method!")
        
        return True
        
    except Exception as e:
        print(f"❌ Workspace upload failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Workspace Upload Test")
    print("=" * 50)
    print("This tests the workspace.upload_dataset() method that's now")
    print("implemented in your YOLO Data Labeller app for efficient bulk uploads.")
    print()
    
    success = test_workspace_upload()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("✅ Your app is ready for efficient dataset uploads!")
    else:
        print("\n💔 Test failed.")
        print("❌ Check the error messages above for details.")

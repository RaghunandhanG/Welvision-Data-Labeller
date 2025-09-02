#!/usr/bin/env python3
"""
Debug script to test Roboflow upload and see what response we get
"""

import os
import json
import sys

def debug_roboflow_upload():
    """Debug the roboflow upload to see what response we get"""
    try:
        # Import roboflow
        import roboflow
        print("✅ Roboflow library imported successfully")
        
        # Test dataset path (using sample8)
        dataset_path = os.path.join(os.getcwd(), "datasets", "sample8")
        coco_file = os.path.join(dataset_path, "annotations.json")
        
        print(f"📂 Testing with dataset: {dataset_path}")
        print(f"🔍 COCO file exists: {os.path.exists(coco_file)}")
        
        if not os.path.exists(coco_file):
            print("❌ COCO file not found. Cannot test upload.")
            return
        
        # Load COCO data for info
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        images_count = len(coco_data.get('images', []))
        annotations_count = len(coco_data.get('annotations', []))
        categories_count = len(coco_data.get('categories', []))
        
        print(f"📊 Dataset info:")
        print(f"   Images: {images_count}")
        print(f"   Annotations: {annotations_count}")
        print(f"   Categories: {categories_count}")
        
        # Ask for API key (don't hardcode it)
        api_key = input("🔑 Enter your Roboflow API key: ").strip()
        if not api_key:
            print("❌ No API key provided")
            return
        
        project_id = input("📦 Enter project ID/slug: ").strip()
        if not project_id:
            print("❌ No project ID provided")
            return
        
        print(f"\n🚀 Testing upload with:")
        print(f"   API Key: {'*' * (len(api_key) - 4) + api_key[-4:]}")
        print(f"   Project: {project_id}")
        print(f"   Dataset: {dataset_path}")
        
        # Initialize Roboflow
        print("\n🔄 Initializing Roboflow...")
        rf = roboflow.Roboflow(api_key=api_key)
        
        # Get workspace
        print("🔄 Getting workspace...")
        workspace = rf.workspace()
        workspace_name = getattr(workspace, 'name', 'Unknown')
        workspace_id = getattr(workspace, 'id', 'Unknown')
        
        print(f"✅ Connected to workspace:")
        print(f"   Name: {workspace_name}")
        print(f"   ID: {workspace_id}")
        
        # Test the upload method
        print(f"\n🚀 Starting upload_dataset test...")
        print(f"   Method: workspace.upload_dataset()")
        print(f"   Parameters:")
        print(f"     dataset_path: {dataset_path}")
        print(f"     project_id: {project_id}")
        print(f"     num_workers: 4")
        print(f"     project_license: MIT")
        print(f"     project_type: object-detection")
        print(f"     num_retries: 1")
        
        # Perform the upload
        try:
            result = workspace.upload_dataset(
                dataset_path,
                project_id,
                num_workers=4,
                project_license="MIT",
                project_type="object-detection",
                batch_name=None,
                num_retries=1
            )
            
            print(f"\n📊 Upload Result Analysis:")
            print(f"   Result type: {type(result)}")
            print(f"   Result value: {result}")
            print(f"   Result bool: {bool(result)}")
            print(f"   Result is None: {result is None}")
            print(f"   Result == False: {result == False}")
            print(f"   Result == True: {result == True}")
            
            if result:
                print("✅ Upload appears successful based on truthy result")
            else:
                print("❌ Upload appears failed based on falsy result")
                
            # Try to inspect the result more
            if hasattr(result, '__dict__'):
                print(f"   Result attributes: {vars(result)}")
            
            if hasattr(result, 'json'):
                try:
                    print(f"   Result JSON: {result.json()}")
                except:
                    pass
            
            print(f"\n🎯 Conclusion:")
            if result:
                print("   The upload_dataset method returned a truthy value - upload likely succeeded")
            else:
                print("   The upload_dataset method returned a falsy value - this is why app shows 'Upload Failed'")
                print("   This might be a false negative - check your Roboflow dashboard to verify")
                
        except Exception as upload_error:
            print(f"❌ Upload exception: {upload_error}")
            print(f"   Exception type: {type(upload_error)}")
            import traceback
            traceback.print_exc()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please install roboflow: pip install roboflow")
        
    except Exception as e:
        print(f"❌ General error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 Roboflow Upload Debug Tool")
    print("=" * 50)
    debug_roboflow_upload()

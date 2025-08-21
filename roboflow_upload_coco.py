"""
Roboflow Upload Script for COCO Format Datasets
This script uploads a dataset with COCO JSON annotations to Roboflow
"""

import roboflow
import os
import json

def upload_coco_dataset_to_roboflow(dataset_path, project_name, api_key, workspace_slug=""):
    """
    Upload a COCO format dataset to Roboflow
    
    Args:
        dataset_path (str): Path to dataset folder containing images/ folder and annotations.json
        project_name (str): Name of the Roboflow project
        api_key (str): Your Roboflow API key
        workspace_slug (str): Your workspace slug (optional)
    """
    
    try:
        # Initialize Roboflow
        rf = roboflow.Roboflow(api_key=api_key)
        workspace = rf.workspace(workspace_slug)
        
        # Check if dataset path exists
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return False
        
        # Check for annotations.json
        annotations_path = os.path.join(dataset_path, "annotations.json")
        if not os.path.exists(annotations_path):
            print(f"❌ annotations.json not found in {dataset_path}")
            print("Please export your dataset to COCO format first using the YOLO Labeler App")
            return False
        
        # Load and validate COCO JSON
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        print(f"📊 Dataset Statistics:")
        print(f"   • Images: {len(coco_data.get('images', []))}")
        print(f"   • Annotations: {len(coco_data.get('annotations', []))}")
        print(f"   • Categories: {len(coco_data.get('categories', []))}")
        
        # Upload dataset
        print(f"🚀 Uploading dataset to Roboflow project: {project_name}")
        print("This may take a few minutes depending on dataset size...")
        
        result = workspace.upload_dataset(
            dataset_path=dataset_path,
            project_name=project_name,
            dataset_format="coco",  # COCO format
            project_type="object-detection"
        )
        
        print("✅ Dataset uploaded successfully!")
        print(f"🔗 Project URL: https://app.roboflow.com/{workspace_slug}/{project_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading dataset: {str(e)}")
        return False

def main():
    """Main function - update these values with your information"""
    
    # Configuration - UPDATE THESE VALUES
    API_KEY = "RKVbzglmD4K4cBfDFXRJ"  # Your Roboflow API key
    WORKSPACE_SLUG = ""  # Your workspace slug (leave empty if default)
    
    # Dataset configuration
    DATASET_PATH = r"C:\Users\raghu\OneDrive\Desktop\sample test"  # Path to your dataset folder
    PROJECT_NAME = "chatter-scratch-damage-coco"  # New project name for COCO format
    
    print("🤖 Roboflow COCO Dataset Upload Tool")
    print("=" * 50)
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Project Name: {PROJECT_NAME}")
    print(f"Format: COCO JSON")
    print("=" * 50)
    
    # Upload dataset
    success = upload_coco_dataset_to_roboflow(
        dataset_path=DATASET_PATH,
        project_name=PROJECT_NAME,
        api_key=API_KEY,
        workspace_slug=WORKSPACE_SLUG
    )
    
    if success:
        print("\n🎉 Upload completed successfully!")
        print("You can now view your dataset in the Roboflow web interface.")
    else:
        print("\n💥 Upload failed. Please check the error messages above.")

if __name__ == "__main__":
    main()

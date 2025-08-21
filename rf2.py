import json
import os
from roboflow import Roboflow

def upload_coco_to_roboflow(annotation_file, api_key, workspace_id, project_id):
    """
    Upload COCO format dataset to Roboflow using COCO JSON file
    Uses full image paths from the COCO JSON file
    """
    
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_id).project(project_id)
    
    # Load the COCO JSON file
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"📊 Dataset Statistics:")
    print(f"   • Images: {len(coco_data.get('images', []))}")
    print(f"   • Annotations: {len(coco_data.get('annotations', []))}")
    print(f"   • Categories: {len(coco_data.get('categories', []))}")
    print(f"   • Categories: {', '.join([cat['name'] for cat in coco_data.get('categories', [])])}")
    print(f"   • Upload Mode: Predictions (will go to 'Unassigned')")
    print("-" * 50)
    
    # Upload images using the COCO JSON format
    successful_uploads = 0
    failed_uploads = 0
    
    for i, image_info in enumerate(coco_data['images'], 1):
        # Use full path from COCO JSON if available
        if 'file_path' in image_info and os.path.exists(image_info['file_path']):
            image_path = image_info['file_path']
        else:
            # Fallback: try to construct path from annotation file location
            annotation_dir = os.path.dirname(annotation_file)
            images_dir = os.path.join(annotation_dir, "images")
            image_path = os.path.join(images_dir, image_info['file_name'])
        
        filename = image_info['file_name']
        print(f"[{i}/{len(coco_data['images'])}] Uploading: {filename}")
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            failed_uploads += 1
            continue
        
        try:
            # Upload with COCO JSON annotations as predictions
            response = project.single_upload(
                image_path=image_path,
                annotation_path=annotation_file,  # COCO JSON with all annotations
                is_prediction=True,  # Upload as predictions (goes to Unassigned)
                num_retry_uploads=3
            )
            print(f"✅ Success (Prediction): {filename}")
            successful_uploads += 1
            
        except Exception as e:
            print(f"❌ Error uploading {filename}: {e}")
            failed_uploads += 1
    
    print("-" * 50)
    print(f"🎉 Upload Summary:")
    print(f"   ✅ Successful: {successful_uploads}")
    print(f"   ❌ Failed: {failed_uploads}")
    print(f"   📊 Total: {len(coco_data['images'])}")
    print(f"   📁 Destination: Unassigned (Predictions for Review)")
    print("🎉 Upload process completed!")
    print("📋 Check the 'Unassigned' section in your Roboflow project to review the uploaded images")

if __name__ == "__main__":
    # Configuration
    API_KEY = "RKVbzglmD4K4cBfDFXRJ"
    WORKSPACE_ID = 'verify-workspace-od-1'
    PROJECT_ID = 'chatter-scratch-damage-2'
    
    # Path to your COCO JSON file (exported from the app)
    ANNOTATION_FILE = r"C:\Users\raghu\OneDrive\Desktop\sample2\annotations.json"
    
    # Upload to Roboflow
    upload_coco_to_roboflow(ANNOTATION_FILE, API_KEY, WORKSPACE_ID, PROJECT_ID)

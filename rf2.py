import json
import os
from roboflow import Roboflow

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key="RKVbzglmD4K4cBfDFXRJ")

# Specify workspace and project
workspaceId = 'verify-workspace-od-1'
projectId = 'chatter-scratch-damage-2'
project = rf.workspace(workspaceId).project(projectId)

# Path to annotation file
annotation_file = r"C:\Users\raghu\OneDrive\Desktop\sample test\annotations.json"

# Load the COCO JSON file
with open(annotation_file, 'r') as f:
    coco_data = json.load(f)

print(f"📊 Dataset Statistics:")
print(f"   • Images: {len(coco_data.get('images', []))}")
print(f"   • Annotations: {len(coco_data.get('annotations', []))}")
print(f"   • Categories: {len(coco_data.get('categories', []))}")

# Upload images using paths from COCO JSON
for image_info in coco_data['images']:
    # Use full path if available, otherwise construct from images folder
    if 'file_path' in image_info:
        image_path = image_info['file_path']
    else:
        # Fallback: construct path from images folder
        images_dir = r"C:\Users\raghu\OneDrive\Desktop\sample test\images"
        image_path = os.path.join(images_dir, image_info['file_name'])
    
    print(f"Uploading: {image_path}")
    
    try:
        response = project.single_upload(
            image_path=image_path,
            annotation_path=annotation_file,  # your COCO JSON
            is_prediction=False  # since these are ground truth annotations
        )
        print(f"✅ Success: {response}")
    except Exception as e:
        print(f"❌ Error uploading {image_path}: {e}")

print("🎉 Upload process completed!")

"""
Sample COCO JSON structure with full image paths
This shows what the updated export will generate
"""

sample_coco_json = {
    "info": {
        "description": "Dataset: sample_dataset",
        "version": "1.0",
        "year": 2025,
        "contributor": "YOLO Labeler App",
        "date_created": "2025-08-21T10:30:00"
    },
    "licenses": [
        {
            "id": 1,
            "name": "Attribution License",
            "url": "http://creativecommons.org/licenses/by/2.0/"
        }
    ],
    "images": [
        {
            "id": 1,
            "file_name": "image1.bmp",
            "width": 640,
            "height": 480,
            "file_path": "C:\\Users\\raghu\\OneDrive\\Desktop\\sample test\\images\\image1.bmp"
        },
        {
            "id": 2,
            "file_name": "image2.bmp", 
            "width": 800,
            "height": 600,
            "file_path": "C:\\Users\\raghu\\OneDrive\\Desktop\\sample test\\images\\image2.bmp"
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 50, 200, 150],
            "area": 30000,
            "iscrowd": 0
        },
        {
            "id": 2,
            "image_id": 2,
            "category_id": 2,
            "bbox": [150, 75, 180, 120],
            "area": 21600,
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "scratch",
            "supercategory": "damage"
        },
        {
            "id": 2,
            "name": "chatter",
            "supercategory": "damage"
        }
    ]
}

print("📋 Sample COCO JSON Structure with Full Paths:")
print("=" * 60)

import json
print(json.dumps(sample_coco_json, indent=2))

print("\n✅ Key Features:")
print("• Each image now includes 'file_path' with full absolute path")
print("• Compatible with your Roboflow upload script")
print("• Maintains all standard COCO format requirements")
print("• Ready for single_upload() method in Roboflow API")

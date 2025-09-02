"""
Dataset Manager Module
Handles dataset operations, COCO format conversion, and file management
"""

import os
import json
import yaml
import shutil
from datetime import datetime
from pathlib import Path


class DatasetManager:
    """Handles dataset operations and format conversions"""
    
    def __init__(self):
        self.datasets_dir = "datasets"
        self.ensure_datasets_directory()
    
    def ensure_datasets_directory(self):
        """Ensure datasets directory exists"""
        if not os.path.exists(self.datasets_dir):
            os.makedirs(self.datasets_dir)
    
    def create_dataset_structure(self, dataset_name):
        """Create standard dataset directory structure"""
        dataset_path = os.path.join(self.datasets_dir, dataset_name)
        
        # Create directories
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "labels"), exist_ok=True)
        
        return dataset_path
    
    def create_data_yaml(self, dataset_path, classes, description=""):
        """Create data.yaml file for YOLOv8 format"""
        data_yaml_content = {
            'path': dataset_path,
            'train': 'images',
            'val': 'images',
            'nc': len(classes),
            'names': classes
        }
        
        yaml_path = os.path.join(dataset_path, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f, default_flow_style=False)
        
        return yaml_path
    
    def create_classes_json(self, dataset_path, classes):
        """Create classes.json file"""
        classes_data = {
            "classes": classes,
            "created_at": datetime.now().isoformat()
        }
        
        json_path = os.path.join(dataset_path, "classes.json")
        with open(json_path, 'w') as f:
            json.dump(classes_data, f, indent=2)
        
        return json_path
    
    def validate_dataset_structure(self, dataset_path):
        """Validate dataset has proper structure"""
        required_files = ["data.yaml", "classes.json"]
        required_dirs = ["images", "labels"]
        
        for file_name in required_files:
            if not os.path.exists(os.path.join(dataset_path, file_name)):
                return False, f"Missing required file: {file_name}"
        
        for dir_name in required_dirs:
            if not os.path.exists(os.path.join(dataset_path, dir_name)):
                return False, f"Missing required directory: {dir_name}"
        
        return True, "Dataset structure is valid"
    
    def get_dataset_info(self, dataset_path):
        """Get dataset information"""
        info = {
            'name': os.path.basename(dataset_path),
            'path': dataset_path,
            'images_count': 0,
            'labels_count': 0,
            'classes': [],
            'valid': False
        }
        
        # Check if dataset structure is valid
        is_valid, message = self.validate_dataset_structure(dataset_path)
        info['valid'] = is_valid
        
        if not is_valid:
            info['error'] = message
            return info
        
        # Count images and labels
        images_dir = os.path.join(dataset_path, "images")
        labels_dir = os.path.join(dataset_path, "labels")
        
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            info['images_count'] = len(image_files)
        
        if os.path.exists(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            info['labels_count'] = len(label_files)
        
        # Get classes from data.yaml
        data_yaml_path = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(data_yaml_path):
            try:
                with open(data_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    info['classes'] = data.get('names', [])
            except Exception as e:
                info['error'] = f"Error reading data.yaml: {e}"
        
        return info
    
    def scan_datasets_directory(self):
        """Scan datasets directory for available datasets"""
        datasets = []
        
        if not os.path.exists(self.datasets_dir):
            return datasets
        
        for item in os.listdir(self.datasets_dir):
            item_path = os.path.join(self.datasets_dir, item)
            if os.path.isdir(item_path):
                dataset_info = self.get_dataset_info(item_path)
                datasets.append(dataset_info)
        
        return datasets
    
    def convert_yolo_to_coco(self, dataset_path, output_file=None):
        """Convert YOLOv8 format to COCO format"""
        if output_file is None:
            output_file = os.path.join(dataset_path, "annotations.json")
        
        # Load data.yaml
        data_yaml_path = os.path.join(dataset_path, "data.yaml")
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError("data.yaml not found")
        
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        class_names = data_config.get('names', [])
        images_dir = os.path.join(dataset_path, "images")
        labels_dir = os.path.join(dataset_path, "labels")
        
        # Initialize COCO structure
        coco_data = {
            "info": {
                "description": f"Dataset converted from YOLOv8 format",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "WelVision Data Labeller",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        for idx, class_name in enumerate(class_names):
            coco_data["categories"].append({
                "id": idx,
                "name": class_name,
                "supercategory": "object"
            })
        
        annotation_id = 1
        
        # Process images and annotations
        for image_file in os.listdir(images_dir):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            image_path = os.path.join(images_dir, image_file)
            
            # Get image dimensions
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception:
                continue
            
            image_id = len(coco_data["images"]) + 1
            
            # Add image info
            coco_data["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": image_file
            })
            
            # Process corresponding label file
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            
                            # Convert YOLO format to COCO format
                            bbox_width = w * width
                            bbox_height = h * height
                            bbox_x = (x_center * width) - (bbox_width / 2)
                            bbox_y = (y_center * height) - (bbox_height / 2)
                            
                            area = bbox_width * bbox_height
                            
                            coco_data["annotations"].append({
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": class_id,
                                "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                                "area": area,
                                "iscrowd": 0
                            })
                            
                            annotation_id += 1
        
        # Save COCO file
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        return output_file
    
    def copy_image_to_dataset(self, source_image_path, dataset_path, new_filename=None):
        """Copy image to dataset images directory"""
        if new_filename is None:
            new_filename = os.path.basename(source_image_path)
        
        dest_path = os.path.join(dataset_path, "images", new_filename)
        shutil.copy2(source_image_path, dest_path)
        return dest_path
    
    def save_yolo_annotation(self, dataset_path, image_filename, annotations, image_width, image_height):
        """Save annotation in YOLO format"""
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(dataset_path, "labels", label_filename)
        
        with open(label_path, 'w') as f:
            for annotation in annotations:
                class_id = annotation['class_id']
                x_center = annotation['x_center']
                y_center = annotation['y_center'] 
                width = annotation['width']
                height = annotation['height']
                
                # Normalize coordinates
                x_center_norm = x_center / image_width
                y_center_norm = y_center / image_height
                width_norm = width / image_width
                height_norm = height / image_height
                
                f.write(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")
        
        return label_path
    
    def delete_dataset(self, dataset_path):
        """Delete entire dataset directory"""
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
            return True
        return False

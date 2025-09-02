"""
Roboflow Manager Module
Handles Roboflow API operations, project management, and dataset uploads
"""

import os
import json
import tempfile
import shutil
from datetime import datetime


class RoboflowManager:
    """Handles Roboflow API operations and dataset uploads"""
    
    def __init__(self):
        self.api_key = None
        self.workspace = None
        self.rf = None
    
    def initialize(self, api_key):
        """Initialize Roboflow connection"""
        try:
            from roboflow import Roboflow
            self.api_key = api_key
            self.rf = Roboflow(api_key=api_key)
            self.workspace = self.rf.workspace()
            return True
        except ImportError:
            raise Exception("Roboflow library not found. Please install: pip install roboflow")
        except Exception as e:
            raise Exception(f"Failed to initialize Roboflow: {e}")
    
    def test_connection(self):
        """Test Roboflow API connection"""
        if not self.rf or not self.workspace:
            return False, "Not initialized"
        
        try:
            # Try to get workspace info
            workspace_info = {
                'id': getattr(self.workspace, 'id', 'Unknown'),
                'name': getattr(self.workspace, 'name', 'Unknown')
            }
            return True, workspace_info
        except Exception as e:
            return False, str(e)
    
    def list_projects(self):
        """List all projects in the workspace"""
        if not self.workspace:
            return []
        
        try:
            projects = self.workspace.list_projects()
            project_list = []
            
            for project in projects:
                project_info = {
                    'id': getattr(project, 'id', 'Unknown'),
                    'name': getattr(project, 'name', 'Unknown'),
                    'slug': getattr(project, 'id', 'Unknown')  # In Roboflow, id is often the slug
                }
                project_list.append(project_info)
            
            return project_list
        except Exception as e:
            print(f"Error listing projects: {e}")
            return []
    
    def create_project(self, project_name, project_type="object-detection"):
        """Create a new Roboflow project"""
        if not self.workspace:
            raise Exception("Workspace not initialized")
        
        try:
            project = self.workspace.create_project(
                project_name=project_name,
                project_type=project_type
            )
            
            return {
                'id': getattr(project, 'id', project_name),
                'name': getattr(project, 'name', project_name),
                'slug': getattr(project, 'id', project_name)
            }
        except Exception as e:
            raise Exception(f"Failed to create project: {e}")
    
    def get_project(self, project_id):
        """Get project by ID/slug"""
        if not self.workspace:
            return None
        
        try:
            project = self.workspace.project(project_id)
            return project
        except Exception as e:
            print(f"Error getting project: {e}")
            return None
    
    def upload_dataset_individual(self, data_yaml_file, project_id, progress_callback=None):
        """Upload dataset using individual image upload method"""
        try:
            import yaml
            from PIL import Image
            
            # Load dataset configuration
            with open(data_yaml_file, 'r') as f:
                data_config = yaml.safe_load(f)
            
            dataset_dir = os.path.dirname(data_yaml_file)
            images_dir = os.path.join(dataset_dir, "images")
            labels_dir = os.path.join(dataset_dir, "labels")
            
            if not os.path.exists(images_dir):
                raise Exception(f"Images directory not found: {images_dir}")
            
            # Get project
            project = self.get_project(project_id)
            if not project:
                raise Exception(f"Project not found: {project_id}")
            
            # Get image files
            image_files = [f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not image_files:
                raise Exception("No image files found")
            
            uploaded_count = 0
            total_files = len(image_files)
            
            if progress_callback:
                progress_callback(f"Starting upload of {total_files} images...")
            
            for i, image_file in enumerate(image_files):
                try:
                    image_path = os.path.join(images_dir, image_file)
                    
                    # Check if corresponding label file exists
                    label_file = os.path.splitext(image_file)[0] + '.txt'
                    label_path = os.path.join(labels_dir, label_file)
                    
                    if os.path.exists(label_path):
                        # Convert YOLO annotation to Roboflow format
                        annotations = self._convert_yolo_annotation_to_roboflow(
                            label_path, image_path, data_config.get('names', [])
                        )
                        
                        # Upload with annotations
                        project.upload(
                            image_path=image_path,
                            annotation_path=label_path,
                            annotation_format="yolo",
                            split="train"
                        )
                    else:
                        # Upload without annotations
                        project.upload(
                            image_path=image_path,
                            split="train"
                        )
                    
                    uploaded_count += 1
                    
                    if progress_callback:
                        progress = (i + 1) / total_files * 100
                        progress_callback(f"Uploaded {uploaded_count}/{total_files} images ({progress:.1f}%)")
                
                except Exception as e:
                    print(f"Error uploading {image_file}: {e}")
                    continue
            
            if progress_callback:
                progress_callback(f"Upload complete! {uploaded_count}/{total_files} images uploaded successfully.")
            
            return uploaded_count == total_files
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Upload failed: {e}")
            raise e
    
    def upload_dataset_workspace(self, dataset_path, project_id, progress_callback=None):
        """Upload dataset using workspace upload method"""
        try:
            if not self.workspace:
                raise Exception("Workspace not initialized")
            
            if progress_callback:
                progress_callback("Starting workspace dataset upload...")
            
            # Validate dataset structure
            coco_file = os.path.join(dataset_path, "annotations.json")
            images_dir = os.path.join(dataset_path, "images")
            
            if not os.path.exists(coco_file):
                raise Exception(f"COCO annotations file not found: {coco_file}")
            
            if not os.path.exists(images_dir):
                raise Exception(f"Images directory not found: {images_dir}")
            
            # Create unique batch name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_name = f"upload_{timestamp}"
            
            if progress_callback:
                progress_callback(f"Uploading with batch name: {batch_name}")
            
            # Upload dataset
            result = self.workspace.upload_dataset(
                dataset_format="coco",
                dataset_location=dataset_path,
                project_id=project_id,
                batch_name=batch_name
            )
            
            if progress_callback:
                progress_callback("Workspace upload completed successfully!")
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Workspace upload failed: {e}")
            raise e
    
    def _convert_yolo_annotation_to_roboflow(self, label_path, image_path, class_names):
        """Convert YOLO annotation format to Roboflow format"""
        try:
            from PIL import Image
            
            # Get image dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            annotations = []
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert normalized coordinates to absolute
                        abs_x = x_center * img_width
                        abs_y = y_center * img_height
                        abs_width = width * img_width
                        abs_height = height * img_height
                        
                        # Calculate bounding box corners
                        x_min = abs_x - (abs_width / 2)
                        y_min = abs_y - (abs_height / 2)
                        x_max = abs_x + (abs_width / 2)
                        y_max = abs_y + (abs_height / 2)
                        
                        annotation = {
                            'class': class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                            'x': x_min,
                            'y': y_min,
                            'width': abs_width,
                            'height': abs_height
                        }
                        
                        annotations.append(annotation)
            
            return annotations
            
        except Exception as e:
            print(f"Error converting annotation: {e}")
            return []
    
    def download_dataset(self, project_id, version=None, format_type="yolov8"):
        """Download dataset from Roboflow"""
        try:
            project = self.get_project(project_id)
            if not project:
                raise Exception(f"Project not found: {project_id}")
            
            if version:
                dataset = project.version(version).download(format_type)
            else:
                dataset = project.download(format_type)
            
            return dataset.location
            
        except Exception as e:
            raise Exception(f"Failed to download dataset: {e}")
    
    def get_project_info(self, project_id):
        """Get detailed project information"""
        try:
            project = self.get_project(project_id)
            if not project:
                return None
            
            info = {
                'id': getattr(project, 'id', 'Unknown'),
                'name': getattr(project, 'name', 'Unknown'),
                'type': getattr(project, 'type', 'Unknown'),
                'images': getattr(project, 'images', 0),
                'classes': getattr(project, 'classes', [])
            }
            
            return info
            
        except Exception as e:
            print(f"Error getting project info: {e}")
            return None

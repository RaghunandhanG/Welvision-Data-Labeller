"""
Image Processing Utilities Module
Common image processing functions and utilities
"""

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import os


class ImageProcessor:
    """Image processing utilities"""
    
    @staticmethod
    def load_image(image_path):
        """Load image using OpenCV"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return image
    
    @staticmethod
    def load_image_pil(image_path):
        """Load image using PIL"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        return Image.open(image_path)
    
    @staticmethod
    def resize_image(image, target_width=None, target_height=None, maintain_aspect=True):
        """Resize image while optionally maintaining aspect ratio"""
        if isinstance(image, np.ndarray):  # OpenCV image
            h, w = image.shape[:2]
        else:  # PIL image
            w, h = image.size
        
        if target_width and target_height:
            if maintain_aspect:
                # Calculate scaling factor
                scale_w = target_width / w
                scale_h = target_height / h
                scale = min(scale_w, scale_h)
                
                new_w = int(w * scale)
                new_h = int(h * scale)
            else:
                new_w = target_width
                new_h = target_height
        elif target_width:
            scale = target_width / w
            new_w = target_width
            new_h = int(h * scale)
        elif target_height:
            scale = target_height / h
            new_w = int(w * scale)
            new_h = target_height
        else:
            return image
        
        if isinstance(image, np.ndarray):  # OpenCV image
            return cv2.resize(image, (new_w, new_h))
        else:  # PIL image
            return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    @staticmethod
    def convert_cv2_to_pil(cv2_image):
        """Convert OpenCV image to PIL image"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    
    @staticmethod
    def convert_pil_to_cv2(pil_image):
        """Convert PIL image to OpenCV image"""
        # Convert to numpy array
        rgb_array = np.array(pil_image)
        # Convert RGB to BGR
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def create_thumbnail(image_path, max_size=(200, 200)):
        """Create thumbnail of image"""
        try:
            image = Image.open(image_path)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            return None
    
    @staticmethod
    def get_image_info(image_path):
        """Get basic image information"""
        try:
            with Image.open(image_path) as img:
                info = {
                    'format': img.format,
                    'mode': img.mode,
                    'width': img.width,
                    'height': img.height,
                    'size_bytes': os.path.getsize(image_path)
                }
                return info
        except Exception as e:
            print(f"Error getting image info: {e}")
            return None
    
    @staticmethod
    def validate_image_file(file_path):
        """Check if file is a valid image"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in valid_extensions:
            return False, f"Unsupported file type: {ext}"
        
        # Try to open the image
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify it's a valid image
            return True, "Valid image file"
        except Exception as e:
            return False, f"Invalid image file: {e}"
    
    @staticmethod
    def draw_bounding_box(image, bbox, label=None, color=(255, 0, 0), thickness=2):
        """Draw bounding box on image (OpenCV format)"""
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # Draw label if provided
        if label:
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
            cv2.rectangle(
                image,
                (int(x1), int(y1) - label_size[1] - 10),
                (int(x1) + label_size[0], int(y1)),
                color,
                -1
            )
            cv2.putText(
                image,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness
            )
        
        return image
    
    @staticmethod
    def draw_bounding_box_pil(image, bbox, label=None, color="red", width=2):
        """Draw bounding box on PIL image"""
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        # Draw label if provided
        if label:
            try:
                from PIL import ImageFont
                font = ImageFont.load_default()
            except:
                font = None
            
            draw.text((x1, y1 - 15), label, fill=color, font=font)
        
        return image
    
    @staticmethod
    def enhance_image(image, brightness=1.0, contrast=1.0, saturation=1.0):
        """Enhance image brightness, contrast, and saturation"""
        try:
            from PIL import ImageEnhance
            
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            # Apply enhancements
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness)
            
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)
            
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(saturation)
            
            return image
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return image
    
    @staticmethod
    def create_image_grid(image_paths, grid_size=(3, 3), cell_size=(200, 200)):
        """Create a grid of images"""
        try:
            rows, cols = grid_size
            cell_width, cell_height = cell_size
            
            # Create blank canvas
            grid_width = cols * cell_width
            grid_height = rows * cell_height
            grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
            
            # Place images in grid
            for i, image_path in enumerate(image_paths[:rows * cols]):
                row = i // cols
                col = i % cols
                
                if os.path.exists(image_path):
                    try:
                        img = Image.open(image_path)
                        img.thumbnail(cell_size, Image.Resampling.LANCZOS)
                        
                        # Center image in cell
                        x_offset = col * cell_width + (cell_width - img.width) // 2
                        y_offset = row * cell_height + (cell_height - img.height) // 2
                        
                        grid_image.paste(img, (x_offset, y_offset))
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
            
            return grid_image
        except Exception as e:
            print(f"Error creating image grid: {e}")
            return None


class AnnotationUtils:
    """Utilities for handling annotations"""
    
    @staticmethod
    def yolo_to_bbox(yolo_coords, image_width, image_height):
        """Convert YOLO format to bounding box coordinates"""
        x_center, y_center, width, height = yolo_coords
        
        # Convert normalized coordinates to absolute
        x_center_abs = x_center * image_width
        y_center_abs = y_center * image_height
        width_abs = width * image_width
        height_abs = height * image_height
        
        # Calculate bounding box corners
        x1 = x_center_abs - width_abs / 2
        y1 = y_center_abs - height_abs / 2
        x2 = x_center_abs + width_abs / 2
        y2 = y_center_abs + height_abs / 2
        
        return [x1, y1, x2, y2]
    
    @staticmethod
    def bbox_to_yolo(bbox_coords, image_width, image_height):
        """Convert bounding box coordinates to YOLO format"""
        x1, y1, x2, y2 = bbox_coords
        
        # Calculate center and dimensions
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        
        # Normalize coordinates
        x_center_norm = x_center / image_width
        y_center_norm = y_center / image_height
        width_norm = width / image_width
        height_norm = height / image_height
        
        return [x_center_norm, y_center_norm, width_norm, height_norm]
    
    @staticmethod
    def coco_to_bbox(coco_bbox):
        """Convert COCO format [x, y, width, height] to [x1, y1, x2, y2]"""
        x, y, width, height = coco_bbox
        return [x, y, x + width, y + height]
    
    @staticmethod
    def bbox_to_coco(bbox_coords):
        """Convert [x1, y1, x2, y2] to COCO format [x, y, width, height]"""
        x1, y1, x2, y2 = bbox_coords
        return [x1, y1, x2 - x1, y2 - y1]
    
    @staticmethod
    def calculate_iou(bbox1, bbox2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0  # No intersection
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def filter_overlapping_boxes(boxes, iou_threshold=0.5):
        """Remove overlapping bounding boxes using Non-Maximum Suppression"""
        if not boxes:
            return []
        
        # Sort boxes by confidence (if available) or area
        if len(boxes[0]) > 4:  # Has confidence score
            boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        else:
            # Sort by area (larger boxes first)
            boxes = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
        
        filtered_boxes = []
        
        while boxes:
            current_box = boxes.pop(0)
            filtered_boxes.append(current_box)
            
            # Remove overlapping boxes
            boxes = [box for box in boxes 
                    if AnnotationUtils.calculate_iou(current_box[:4], box[:4]) < iou_threshold]
        
        return filtered_boxes

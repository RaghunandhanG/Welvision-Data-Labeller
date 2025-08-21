"""
Configuration file for WelVision YOLO Data Labeller
Modify these settings according to your environment
"""

# Database Configuration
# IMPORTANT: Update the password field with your MySQL root password before running
DATABASE_CONFIG = {
    'host': 'localhost',  # Change to your MySQL server IP if on another PC
    'user': 'root',
    'password': '1469',  # ⚠️ ENTER YOUR MYSQL PASSWORD HERE
    'database': 'welvision_db',
    'port': 3306
}

# Application Settings
APP_CONFIG = {
    'title': 'WelVision Data Labeller',
    'geometry': '1600x1000',
    'bg_color': '#0a2158',
    'version': 'v2.0'
}

# File Settings
FILE_CONFIG = {
    'supported_formats': [
        ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'),
        ('JPEG files', '*.jpg *.jpeg'),
        ('PNG files', '*.png'),
        ('BMP files', '*.bmp'),
        ('TIFF files', '*.tiff *.tif'),
        ('All files', '*.*')
    ],
    'dataset_base_path': 'datasets',
    'max_preview_size': (800, 600)
}

# YOLO Settings
YOLO_CONFIG = {
    'confidence_threshold': 0.25,
    'iou_threshold': 0.45,
    'max_detections': 1000,
    'device': 'auto'  # 'auto', 'cpu', 'cuda', or specific GPU like 'cuda:0'
}

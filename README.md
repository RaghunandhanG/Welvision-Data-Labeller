# WelVision Data Labeller

A comprehensive Tkinter application for automated image labeling using YOLO v8 models with MySQL database integration and enhanced features.

## ✨ Enhanced Features

### Core Features
- **Interactive Confidence Control**: Real-time slider to adjust YOLO confidence threshold
- **Simplified Settings**: Streamlined YOLO configuration without complex IOU settings
- **Roller Class Filtering**: Only processes images that contain "roller" class detections
- **Smart Image Rejection**: Automatically skips images without the required "roller" class
- **Two-Panel Layout**: Controls on left (60%), image preview on right (40%)
- **Image Preview Panel**: 2-column grid of larger thumbnails (160×120px) with file info
- **Full-Width Interface**: Utilizes 90% of screen width for optimal workspace
- **Smart Device Detection**: Automatic GPU/CPU detection with fallback support
- **Scrollable Interface**: Full application scrolling with mouse wheel support
- **Model Management**: Load YOLO models from database or add new models from filesystem
- **Dataset Management**: Create new datasets or add to existing ones with custom save locations
- **Batch Image Processing**: Upload individual images or entire folders
- **Automated Labeling**: Generate YOLO format labels automatically
- **Database Integration**: Store and manage both models and datasets in MySQL
- **Status Updates**: Real-time status updates during processing
- **WelVision Styling**: Professional dark theme interface

### Latest Updates
- ✅ **Simplified Interface**: Removed progress bar for cleaner, streamlined experience
- ✅ **Interactive Confidence Slider**: Real-time confidence threshold adjustment (0.01-0.95)
- ✅ **Simplified YOLO Settings**: Removed IOU threshold, focused on confidence control
- ✅ **Roller Class Filtering**: Only processes images containing "roller" class detections
- ✅ **Smart Image Rejection**: Automatically rejects images without roller class
- ✅ **Enhanced Image Preview**: 2-column grid layout with larger thumbnails (160×120px)
- ✅ **Optimized Layout**: Controls on left (60%), image previews on right (40%)
- ✅ **Image Preview Panel**: Right-side panel showing thumbnails of uploaded images
- ✅ **Smart Database Management**: Auto-creates database and tables if not exists
- ✅ **Available Models Only**: Shows only models with existing files in dropdown
- ✅ **Database Settings UI**: Configure MySQL credentials from within the app
- ✅ **AI Models Table**: Renamed to `ai_models` for better organization
- ✅ **Full-Width Layout**: Utilizes 90% of screen width for better workspace
- ✅ **Smart Device Detection**: Auto-detects GPU/CPU with intelligent fallback
- ✅ **Streamlined Title**: Clean "WelVision Data Labeller" branding
- ✅ **Consistent UI**: Standardized button sizes and layout
- ✅ **Scrollable Interface**: Navigate through content with scrollbars
- ✅ **Dataset Dropdown**: Select from existing datasets with image counts
- ✅ **Model Loading**: Browse and add new YOLO models (.pt, .pth, .onnx)
- ✅ **Folder Upload**: Upload entire directories of images
- ✅ **Custom Dataset Location**: Choose where to save datasets with folder browser
- ✅ **Enhanced Database**: Separate tables for models and datasets

## Prerequisites

- Python 3.8 or higher
- MySQL Server
- YOLO v8 models (`.pt`, `.pth`, or `.onnx` files)

## Installation

1. **Clone or download the application files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure database connection**:
   - Update `config.py` with your MySQL credentials:
   ```python
   DATABASE_CONFIG = {
       'host': 'localhost',
       'user': 'root',
       'password': 'your_mysql_password',  # ⚠️ UPDATE THIS
       'database': 'welvision_db',
       'port': 3306
   }
   ```

4. **Database Setup**:
   
   **Option 1 - Use Configuration Tool (Recommended):**
   ```bash
   python db_settings_standalone.py
   ```
   Or double-click `configure_database.bat` on Windows.
   
   **Option 2 - Automatic Setup:**
   The app will automatically create the database and tables on first run.
   
   **Option 3 - Manual Setup:**
   ```bash
   python setup_database.py
   ```

## Usage

### 1. Start the Application
```bash
python yolo_labeler_app.py
```

### 2. Model Management
- **Select Existing Model**: Choose from dropdown and click "Load Selected Model"
- **Add New Model**: Click "Browse Model File" → select model → "Add to Database"
- **Model Types**: Supports .pt, .pth, and .onnx YOLO models

### 3. Dataset Management
- **Create New Dataset**: Select radio button → enter dataset name → choose save location
- **Add to Existing**: Select radio button → choose from dropdown
- **Dataset Location**: Click "Browse Folder" to select custom save location (default: datasets folder)
- **Dataset Tracking**: View image counts for existing datasets

### 4. Upload Images
- **Individual Images**: Click "Upload Individual Images" → select multiple files
- **Folder Upload**: Click "Upload Image Folder" → select directory
- **Supported Formats**: JPG, JPEG, PNG, BMP, TIFF

### 5. Start Labeling
- Click "Start Labeling" to begin automatic annotation
- Monitor progress with real-time updates
- View completion summary with statistics

### 6. Access Results
- **Images**: `datasets/[dataset_name]/images/`
- **Labels**: `datasets/[dataset_name]/labels/` (YOLO format)

## Database Schema

### AI Models Table
```sql
CREATE TABLE ai_models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    path VARCHAR(500) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

### Datasets Table
```sql
CREATE TABLE datasets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    path VARCHAR(500) NOT NULL,
    description TEXT,
    image_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

## Configuration

### YOLO Settings
Modify `config.py` to adjust YOLO inference parameters:
```python
YOLO_CONFIG = {
    'confidence_threshold': 0.25,  # Detection confidence threshold
    'iou_threshold': 0.45,         # IoU threshold for NMS
    'max_detections': 1000,        # Maximum detections per image
    'device': 'cpu'                # 'cpu', 'cuda', 'auto', or 'cuda:0'
}
```

### Device Configuration
Run the device check utility to determine your system capabilities:
```bash
python check_device.py
```

**Device Options:**
- `'cpu'` - CPU-only processing (compatible with all systems)
- `'cuda'` - GPU acceleration (requires CUDA-compatible GPU)
- `'auto'` - Automatic device selection (may cause errors on systems without GPU)
- `'cuda:0'` - Specific GPU device selection

### Application Settings
```python
APP_CONFIG = {
    'title': 'WelVision YOLO Data Labeller',
    'geometry': '1400x900',
    'bg_color': '#0a2158',
    'version': 'v2.0'
}
```

## File Structure

```
WELVISION DATA LABELLER/
├── yolo_labeler_app.py        # Main application
├── config.py                  # Configuration settings
├── setup_database.py          # Database setup script
├── create_database.sql        # SQL schema
├── requirements.txt           # Python dependencies
├── run_labeler.bat           # Windows batch launcher
└── datasets/                  # Generated datasets
    └── [dataset_name]/
        ├── images/            # Original images
        └── labels/            # YOLO format labels
```

## YOLO Label Format

Each `.txt` file contains one line per detected object:
```
class_id center_x center_y width height
```

Where all coordinates are normalized (0-1):
- `class_id`: Object class identifier (integer)
- `center_x`, `center_y`: Normalized center coordinates
- `width`, `height`: Normalized bounding box dimensions

## Interface Guide

### Scrollable Navigation
- Use mouse wheel to scroll through the interface
- Scrollbar on the right for manual navigation
- All sections are accessible via scrolling

### Model Section
- **Model Dropdown**: Select from database models
- **Load Model**: Initialize selected YOLO model
- **Browse Model File**: Add new models from filesystem
- **Add to Database**: Save new models for future use

### Dataset Section
- **Radio Buttons**: Choose new vs existing dataset
- **Dataset Name**: Enter name for new datasets
- **Save Location**: Browse and select custom folder for dataset storage
- **Dataset Dropdown**: Select existing datasets with image counts
- **Refresh**: Update dataset list from database

### Upload Section
- **Individual Images**: Multi-select file dialog
- **Image Folder**: Directory selection with recursive scanning
- **Format Support**: All common image formats

### Actions Section
- **Start Labeling**: Begin automated annotation process
- **Clear All**: Reset uploaded images and progress
- **DB Settings**: Configure MySQL database connection settings
- **YOLO Settings**: View current configuration parameters

## Troubleshooting

### Database Connection Issues
1. **Check MySQL Service**: Ensure MySQL server is running
   ```bash
   net start | findstr -i mysql  # Windows
   ```
2. **Use Configuration Tool**: Run database settings tool
   ```bash
   python db_settings_standalone.py
   ```
3. **Manual Configuration**: Update `config.py` with correct credentials
4. **Test Connection**: Use "DB Settings" button in the app
5. **Create Database**: Use the configuration tool to create database and tables

### Device/CUDA Issues
1. **Auto-Detection**: App automatically detects and uses optimal device (GPU/CPU)
2. **GPU Fallback**: Automatically falls back to CPU if GPU fails
3. **Check Capabilities**: Run `python check_device.py` to see available options
4. **Manual Override**: Set specific device in `config.py` if needed

### Model Loading Issues
1. **File Path**: Ensure model file exists and is accessible
2. **Model Format**: Use compatible YOLO models (.pt, .pth, .onnx)
3. **Permissions**: Check file read permissions
4. **Dependencies**: Ensure ultralytics package is installed

### Application Errors
1. **Tkinter Errors**: Ensure Python has tkinter support
2. **Memory Issues**: Close other applications for large datasets
3. **Device Conflicts**: Use CPU mode if experiencing GPU-related crashes

### Performance Optimization
- **GPU Usage**: Set `device: 'cuda'` in config for GPU acceleration
- **Batch Size**: Process smaller batches for memory-constrained systems
- **Model Size**: Use smaller models (yolov8n) for faster processing

## Development Notes

### Code Structure
- **DatabaseManager**: Handles all MySQL operations
- **YOLOLabelerApp**: Main Tkinter application class
- **Scrollable Interface**: Canvas-based scrolling implementation
- **Error Handling**: Graceful degradation when database unavailable

### Styling Standards
- **WelVision Theme**: Dark blue color scheme (#0a2158)
- **Typography**: Arial font family with size hierarchy
- **Layout**: Organized sections with consistent spacing
- **Responsive**: Adapts to different screen sizes

## Version History

### v2.0 (Current)
- Added scrollable interface
- Enhanced dataset management with dropdown
- Model loading from filesystem
- Folder upload functionality
- Removed image preview for streamlined workflow
- Improved error handling and database management

### v1.0
- Basic YOLO labeling functionality
- MySQL database integration
- Single image upload
- Basic progress tracking

## License

This application is part of the WelVision ecosystem developed by WelVision Innovations Private Limited.

## Support

For support and issues, contact the WelVision development team or refer to the troubleshooting section above.
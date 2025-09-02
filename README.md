# WelVision Data Labeller

A comprehensive computer vision data labelling application that combines YOLO model inference with seamless Roboflow integration. Create, label, and upload datasets with an intuitive workflow designed for efficiency and professional results.

## 🎯 Application Overview

The WelVision Data Labeller streamlines the complete computer vision workflow from image annotation to dataset deployment. Whether you're creating training data for object detection, managing datasets, or uploading to Roboflow for model training, this application provides a unified interface for all your data labelling needs.

### Key Capabilities
- **🤖 Automated Labeling**: Use pre-trained YOLO models to automatically generate annotations
- **🗄️ Database Management**: Store and organize models and datasets in MySQL
- **☁️ Roboflow Integration**: Create projects and upload datasets directly to Roboflow
- **📊 Smart Upload Control**: Intelligent upload methods with cancellation support
- **🎨 Professional Interface**: Clean, intuitive UI with real-time progress tracking

---

## 📋 Complete Application Workflow

### Phase 1: Initial Setup & Configuration

#### Step 1.1: Database Configuration
```bash
# Start the application
python yolo_labeler_app.py

# First-time setup: Configure database
# Click "DB Settings" in the main interface
```

**What happens here:**
- Configure MySQL connection settings
- Create database and required tables automatically
- Test connection and verify setup

#### Step 1.2: System Verification
**Automatic Process:**
- GPU/CPU detection and optimization
- YOLO dependencies verification
- Roboflow SDK availability check

---

### Phase 2: Model Management

#### Step 2.1: Load or Add YOLO Models

**Option A: Use Existing Model**
1. **Select Model**: Choose from the "Select Model" dropdown
2. **Load Model**: Click "🔄 Load Selected Model"
3. **Verification**: Wait for "✅ Model loaded successfully" message

**Option B: Add New Model**
1. **Browse**: Click "📁 Browse Model File"
2. **Select**: Choose your YOLO model file (.pt, .pth, .onnx)
3. **Add**: Click "➕ Add to Database"
4. **Load**: Select and load the newly added model

**Workflow Result:** ✅ YOLO model is ready for inference

---

### Phase 3: Dataset Preparation

#### Step 3.1: Create or Select Dataset

**Option A: Create New Dataset**
1. **Select**: Choose "🆕 Create New Dataset" radio button
2. **Name**: Enter descriptive dataset name (e.g., "factory_inspection_v1")
3. **Location**: Click "📁 Browse Folder" to choose save location
4. **Confirm**: Dataset structure is created automatically

**Option B: Add to Existing Dataset**
1. **Select**: Choose "📂 Add to Existing Dataset" radio button
2. **Choose**: Select dataset from dropdown (shows image counts)
3. **Confirm**: Ready to add more images to existing dataset

#### Step 3.2: Upload Images

**Method A: Individual Images**
```
Click "📤 Upload Individual Images"
→ Multi-select images (Ctrl+click)
→ Supported: JPG, JPEG, PNG, BMP, TIFF
→ Images appear in preview panel
```

**Method B: Folder Upload**
```
Click "📁 Upload Image Folder"
→ Select directory containing images
→ Recursive scanning of subdirectories
→ All valid images loaded automatically
```

**Workflow Result:** ✅ Images are uploaded and ready for labeling

---

### Phase 4: Automated Labeling

#### Step 4.1: Configure Detection Parameters
1. **Confidence Threshold**: Adjust slider (0.01 - 0.95)
   - Higher values = fewer, more confident detections
   - Lower values = more detections, potentially less accurate
2. **Model Settings**: Verify loaded model and device (GPU/CPU)

#### Step 4.2: Start Labeling Process
```
Click "🚀 Start Labeling"
```

**What happens during labeling:**
1. **Image Processing**: Each image is analyzed by YOLO model
2. **Detection Generation**: Bounding boxes and class predictions created
3. **Format Conversion**: Annotations saved in YOLO format (.txt files)
4. **Progress Tracking**: Real-time updates and statistics
5. **Quality Control**: Only processes images meeting confidence criteria

**Output Structure:**
```
datasets/your_dataset_name/
├── images/           # Original images
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── labels/           # YOLO format annotations
    ├── image001.txt
    ├── image002.txt
    └── ...
```

**Workflow Result:** ✅ Dataset with images and annotations is ready

---

### Phase 5: Roboflow Integration

#### Step 5.1: Setup Roboflow Connection
1. **Navigate**: Click "☁️ Roboflow Projects" tab
2. **API Key**: Enter your Roboflow API key
3. **Connect**: Click "🔍 Test API & Load Projects"
4. **Verification**: See list of existing projects

#### Step 5.2: Create New Project (Optional)
```
Enter Project Details:
├── Project Name: "Factory Defect Detection"
├── Project Type: Object Detection
├── License: Private/MIT/CC BY 4.0/Public Domain
└── Click "🆕 Create New Project"
```

#### Step 5.3: Upload Dataset to Roboflow

**Step 5.3.1: Select Target Project**
- Choose from dropdown of existing projects
- Newly created projects appear automatically

**Step 5.3.2: Choose Dataset**
- Select dataset from "Choose Dataset" dropdown
- Only datasets with COCO format annotations shown

**Step 5.3.3: Initiate Upload**
```
Click "🚀 Upload Dataset to Roboflow"
```

**Smart Upload Process:**
- **Small Datasets (≤50 images)**: Individual upload with full cancellation support
- **Large Datasets (>50 images)**: Bulk upload with optimization warnings
- **Progress Monitoring**: Real-time upload status and file counts
- **Error Handling**: Automatic retries and detailed error messages

**Workflow Result:** ✅ Dataset is uploaded to Roboflow and ready for training

---

### Phase 6: Upload Management & Control

#### Step 6.1: Monitor Upload Progress
- **Real-time Status**: Live progress updates in status panel
- **File Tracking**: Current file being uploaded
- **Success Metrics**: Uploaded vs total file counts

#### Step 6.2: Upload Cancellation (if needed)

**During Upload:**
- **Cancel Button**: Click "🛑 Cancel Upload" for graceful stopping
- **Force Close**: Options when closing app during upload

**Cancellation Behavior:**
- **Small Datasets**: Immediate cancellation between file uploads
- **Large Datasets**: Limited cancellation (bulk upload constraints)
- **Clean Exit**: No orphaned processes or corrupted data

---

## 🔄 Typical Workflow Examples

### Example 1: New Project from Scratch
```
1. Setup → Database Configuration
2. Models → Add new YOLO model → Load model
3. Dataset → Create new dataset "quality_control_v1"
4. Upload → Upload folder of 200 factory images
5. Label → Start labeling with 0.5 confidence
6. Roboflow → Create project "QC Detection"
7. Upload → Upload dataset to Roboflow
8. Result → Ready for model training in Roboflow
```

### Example 2: Expanding Existing Dataset
```
1. Models → Load existing model from database
2. Dataset → Add to existing "quality_control_v1"
3. Upload → Add 50 more images to dataset
4. Label → Process new images only
5. Roboflow → Upload updated dataset
6. Result → Enhanced dataset with more training data
```

### Example 3: Multiple Dataset Management
```
1. Create dataset "defects_batch1" → Process 100 images
2. Create dataset "defects_batch2" → Process 150 images
3. Upload both datasets to different Roboflow projects
4. Compare results and annotation quality
5. Merge best practices for future datasets
```

---

## 🎛️ Interface Navigation Guide

### Main Tabs
1. **📊 Main Labeling**: Core labeling functionality
2. **☁️ Roboflow Projects**: Project creation and dataset upload

### Main Labeling Tab Sections
1. **🤖 Model Management**
   - Model selection dropdown
   - Load/add model controls
   - Model status display

2. **📁 Dataset Management**
   - New vs existing dataset options
   - Dataset name input and location selection
   - Existing dataset dropdown with counts

3. **📤 Image Upload**
   - Individual image upload button
   - Folder upload button
   - Image preview panel (right side)

4. **⚙️ Labeling Controls**
   - Confidence threshold slider
   - Start labeling button
   - Progress and status display

5. **🔧 Settings & Actions**
   - Database settings
   - Clear all function
   - Status and device information

### Roboflow Tab Sections
1. **🔑 API Configuration**
   - API key input
   - Connection testing
   - Project loading

2. **🆕 Project Creation**
   - Project details form
   - Type and license selection
   - Creation controls

3. **📤 Dataset Upload**
   - Project selection
   - Dataset selection
   - Upload controls and progress

---

## 📊 Status Indicators & Feedback

### Visual Feedback System
- **🟢 Green**: Success states and completed operations
- **🟡 Yellow**: Warning states and in-progress operations
- **🔴 Red**: Error states and failed operations
- **🔵 Blue**: Information and neutral states

### Status Messages
- **Model Status**: "✅ Model loaded: yolov8n.pt (GPU)"
- **Dataset Status**: "📁 Dataset: factory_v1 (125 images)"
- **Upload Status**: "📤 Uploading 5/50: image_batch_001.jpg"
- **Completion Status**: "🎉 Dataset uploaded successfully to Roboflow!"

### Progress Tracking
- **Real-time Updates**: Live progress for all operations
- **File Counts**: Current/total for batch operations
- **Time Estimates**: Automatic calculation for long operations
- **Error Reporting**: Detailed messages for troubleshooting

---

## 🔧 Advanced Features

### Smart Device Management
- **Automatic Detection**: GPU/CPU capabilities assessed on startup
- **Intelligent Fallback**: Automatic CPU fallback if GPU issues occur
- **Performance Optimization**: Device-specific settings applied automatically

### Database Integration
- **Model Storage**: Persistent model registry with metadata
- **Dataset Tracking**: Image counts and creation timestamps
- **Settings Persistence**: Configuration stored across sessions

### Error Recovery
- **Graceful Degradation**: Application continues working if components fail
- **Automatic Retries**: Network operations retry with exponential backoff
- **Clean Shutdown**: Proper cleanup of resources and processes

---

## 🚀 Getting Started Quick Guide

### 1. First Time Setup (5 minutes)
```bash
# Install dependencies
pip install -r requirements.txt

# Start application
python yolo_labeler_app.py

# Configure database (click "DB Settings")
# Add your first YOLO model
# Create your first dataset
```

### 2. Your First Dataset (10 minutes)
```bash
# Load a model → Upload 10-20 test images
# Adjust confidence to 0.5 → Start labeling
# Check generated annotations in datasets folder
# Verify quality and adjust parameters if needed
```

### 3. Roboflow Integration (5 minutes)
```bash
# Get API key from Roboflow dashboard
# Switch to Roboflow tab → Enter API key
# Create test project → Upload your dataset
# Verify upload in Roboflow dashboard
```

**Total time to full workflow: ~20 minutes**

---

## 📁 Project Structure & Output

### Application Files
```
WELVISION DATA LABELLER/
├── 📄 yolo_labeler_app.py      # Main application
├── 📄 config.py                # Configuration settings
├── 📄 database_manager.py      # Database operations
├── 📄 dataset_manager.py       # Dataset handling
├── 📄 roboflow_manager.py      # Roboflow integration
├── 📄 yolo_model_manager.py    # Model management
├── 📄 requirements.txt         # Dependencies
├── 📄 README.md               # This documentation
└── 📁 datasets/               # Generated datasets
```

### Generated Dataset Structure
```
datasets/your_dataset_name/
├── 📁 images/                 # Original images
│   ├── 🖼️ frame_001.jpg
│   ├── 🖼️ frame_002.jpg
│   └── 🖼️ ...
├── 📁 labels/                 # YOLO annotations
│   ├── 📄 frame_001.txt
│   ├── 📄 frame_002.txt
│   └── 📄 ...
├── 📄 classes.json           # Class mapping
├── 📄 data.yaml             # YOLO dataset config
└── 📄 annotations.json      # COCO format (for Roboflow)
```

This workflow ensures a seamless transition from raw images to production-ready datasets, making the WelVision Data Labeller your complete solution for computer vision data preparation.

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.8+**: Download from [python.org](https://python.org)
- **MySQL Server**: Required for data persistence
- **YOLO Models**: Pre-trained models (.pt, .pth, .onnx files)

### Quick Installation
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start application
python yolo_labeler_app.py

# 3. Configure database on first run
# Click "DB Settings" → Enter MySQL credentials → Test connection
```

### Manual Database Setup (Optional)
```python
# Update config.py with your MySQL settings
DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_mysql_password',
    'database': 'welvision_db',
    'port': 3306
}
```

---

## 📋 Technical Specifications

### Supported File Formats
- **Images**: JPG, JPEG, PNG, BMP, TIFF
- **Models**: .pt (PyTorch), .pth (PyTorch), .onnx (ONNX)
- **Output**: YOLO format (.txt), COCO JSON format

### System Requirements
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 500MB for application, varies by dataset size
- **GPU**: Optional but recommended (CUDA-compatible)
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Performance Specifications
- **Processing Speed**: 1-5 images/second (depends on model and device)
- **Batch Size**: Up to 1000 images per session
- **Upload Speed**: Varies by network (Roboflow uploads)
- **Memory Usage**: ~500MB base + model size + image cache

---

## 🔧 Configuration Options

### YOLO Model Settings
```python
# Adjustable in real-time via interface
confidence_threshold: 0.01 - 0.95    # Detection confidence
device: 'auto', 'cpu', 'cuda'        # Processing device
max_detections: 1000                  # Per-image detection limit
```

### Upload Behavior
- **Small Datasets (≤50 images)**: Individual upload with full cancellation
- **Large Datasets (>50 images)**: Bulk upload with speed optimization
- **Retry Logic**: Automatic retries with exponential backoff
- **Progress Tracking**: Real-time status updates

### Database Schema
```sql
-- AI Models table
CREATE TABLE ai_models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    path VARCHAR(500) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Datasets table  
CREATE TABLE datasets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    path VARCHAR(500) NOT NULL,
    image_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

---

## 🔍 Troubleshooting Guide

### Common Issues & Solutions

#### Database Connection Problems
```bash
❌ Error: "Can't connect to MySQL server"
✅ Solution:
   1. Check MySQL service is running
   2. Verify credentials in "DB Settings"
   3. Ensure database 'welvision_db' exists
   4. Test connection using built-in test feature
```

#### Model Loading Issues
```bash
❌ Error: "Model file not found" or "Invalid model format"
✅ Solution:
   1. Verify model file path is accessible
   2. Ensure model is YOLO-compatible (.pt, .pth, .onnx)
   3. Check file permissions
   4. Try loading directly from file browser
```

#### GPU/CUDA Problems
```bash
❌ Error: CUDA errors or GPU detection issues
✅ Solution:
   1. App automatically falls back to CPU
   2. Check NVIDIA drivers are updated
   3. Verify CUDA installation
   4. Use CPU mode if GPU issues persist
```

#### Roboflow Upload Failures
```bash
❌ Error: Upload timeouts or authentication failures
✅ Solution:
   1. Verify API key is correct and active
   2. Check internet connection stability
   3. Try smaller batch uploads for large datasets
   4. Use "Force Close" only if necessary
```

#### Memory Issues
```bash
❌ Error: "Out of memory" or application crashes
✅ Solution:
   1. Process smaller batches of images
   2. Close other memory-intensive applications
   3. Use CPU mode instead of GPU
   4. Restart application periodically for large datasets
```

### Performance Optimization Tips

#### For Better Speed
- Use GPU mode when available
- Process images in smaller batches
- Use smaller YOLO models (yolov8n vs yolov8x)
- Close unnecessary applications

#### For Better Accuracy
- Use higher confidence thresholds (0.7+)
- Use larger YOLO models when possible
- Manually review generated annotations
- Fine-tune models on your specific data

#### For Large Datasets
- Use bulk upload mode for >50 images
- Process during off-peak hours
- Ensure stable internet for Roboflow uploads
- Monitor system resources during processing

---

## 📖 Additional Resources

### Related Documentation
- `QUICK_START.md` - Fast setup guide
- `STEP_BY_STEP_WORKFLOW.md` - Detailed workflow instructions
- `ROBOFLOW_PROJECT_CREATION_GUIDE.md` - Roboflow integration guide
- `UPLOAD_CANCELLATION_GUIDE.md` - Upload management features
- `GPU_CPU_GUIDE.md` - Device optimization guide

### External Resources
- [YOLO Documentation](https://docs.ultralytics.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [MySQL Installation Guide](https://dev.mysql.com/doc/mysql-installation-excerpt/)

---

## 🤝 Support & Development

### Getting Help
1. **Check Troubleshooting**: Review common issues above
2. **Status Messages**: Read application status messages carefully
3. **Log Files**: Check console output for detailed error information
4. **Documentation**: Refer to additional guides in project folder

### Contributing
This application is part of the WelVision ecosystem. For feature requests, bug reports, or contributions, contact the WelVision development team.

### Version Information
- **Current Version**: v2.1 (Enhanced Workflow)
- **Last Updated**: September 2025
- **Compatibility**: Python 3.8+, MySQL 5.7+

---

## 📄 License

This application is developed by WelVision Innovations Private Limited. All rights reserved.

**Usage Terms:**
- Licensed for use within authorized organizations
- Not for redistribution without explicit permission
- Support and updates provided through official channels

---

**Ready to start labelling? Launch the application and follow the workflow guide above! 🚀**

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

## 🚀 Roboflow Project Creation

The application now supports creating new Roboflow projects directly from the interface, streamlining your computer vision workflow.

### Setup Roboflow Integration
1. **Get API Key**: Visit [Roboflow Settings](https://app.roboflow.com/settings/api) to get your API key
2. **Navigate to Roboflow Tab**: Click on "Roboflow Projects" tab in the application
3. **Enter API Key**: Paste your API key and click "🔍 Load Projects"

### Create New Project
Once authenticated, you can create new projects:

1. **Project Name**: Enter a unique descriptive name
2. **Project Type**: Choose from:
   - **Object Detection**: Bounding boxes around objects (most common)
   - **Classification**: Categorize entire images
   - **Instance Segmentation**: Pixel-perfect object masks
   - **Semantic Segmentation**: Classify every pixel
3. **License**: Select appropriate license:
   - **Private**: Restricted access (recommended for commercial use)
   - **MIT**: Open source with attribution
   - **CC BY 4.0**: Creative Commons with attribution
   - **Public Domain**: No restrictions
4. **Click "🆕 Create New Project"**

### Benefits
- **Seamless Workflow**: Create projects without leaving the labelling app
- **Automatic Selection**: Newly created projects are automatically selected
- **Immediate Upload**: Start uploading datasets right after creation
- **Error Handling**: Comprehensive error messages and suggestions

### Project Type Guide
| Type | Use Case | Annotation Format |
|------|----------|-------------------|
| Object Detection | Vehicle/person detection, quality control | Bounding boxes |
| Classification | Pass/fail, medical diagnosis | Single label per image |
| Instance Segmentation | Medical imaging, precision agriculture | Pixel masks per object |
| Semantic Segmentation | Satellite imagery, scene understanding | Pixel-level labels |

For detailed instructions, see [ROBOFLOW_PROJECT_CREATION_GUIDE.md](ROBOFLOW_PROJECT_CREATION_GUIDE.md)

### Testing
Test the Roboflow integration:
```bash
# Windows
test_roboflow_creation.bat

# Or run Python script directly
python test_roboflow_creation.py
```

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
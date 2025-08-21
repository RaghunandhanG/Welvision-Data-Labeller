# Quick Start Guide - WelVision YOLO Data Labeller

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Database
1. Open `config.py`
2. Update the MySQL password:
```python
DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'YOUR_MYSQL_PASSWORD_HERE',  # ⚠️ Enter your password
    'database': 'welvision_db',
    'port': 3306
}
```

### Step 3: Setup Database
```bash
python setup_database.py
```

### Step 4: Run Application
```bash
python yolo_labeler_app.py
```

## 🎯 Quick Usage

1. **Load Model**: Select a model from dropdown → Click "Load Model"
2. **Create Dataset**: Enter dataset name → Select "Create New Dataset"
3. **Upload Images**: Click "Upload Images" → Select your image files
4. **Start Labeling**: Click "Start Labeling" → Wait for completion

## 📁 Output Location
Your labeled data will be saved in:
```
datasets/
└── [your_dataset_name]/
    ├── images/     # Original images
    └── labels/     # YOLO format labels (.txt)
```

## 🔧 Configuration for Remote Database

If your MySQL database is on another PC, update `config.py`:

```python
DATABASE_CONFIG = {
    'host': '192.168.1.100',  # IP of the PC with MySQL
    'user': 'root',
    'password': 'your_password',
    'database': 'welvision_db',
    'port': 3306
}
```

## 📝 Adding Your Own Models

1. **Option 1: Direct Database Insert**
```sql
INSERT INTO models (name, path, description) VALUES 
('My Model', 'C:/path/to/my/model.pt', 'My custom model description');
```

2. **Option 2: Update setup_database.py**
Add your model to the `sample_models` list and re-run the setup script.

## ⚠️ Troubleshooting

### Database Connection Issues
- ✅ Ensure MySQL server is running
- ✅ Check password in `config.py`
- ✅ Verify database user permissions

### Model Loading Issues
- ✅ Check model file path exists
- ✅ Ensure model is compatible with ultralytics YOLO
- ✅ Verify file permissions

### Image Processing Issues
- ✅ Supported formats: JPG, PNG, BMP, TIFF
- ✅ Check available disk space
- ✅ Ensure write permissions in application directory

## 🎨 Features Included

- ✅ **WelVision Styling**: Dark theme matching your application
- ✅ **Database Integration**: MySQL model management
- ✅ **Batch Processing**: Multiple image upload and processing
- ✅ **Progress Tracking**: Real-time progress updates
- ✅ **Dataset Management**: Create new or append to existing datasets
- ✅ **YOLO Format**: Standard YOLO v8 label format output
- ✅ **Configuration**: Easy configuration through config.py
- ✅ **Error Handling**: Comprehensive error handling and user feedback

## 📞 Support

For issues or questions, refer to the main README.md or contact the WelVision development team. 
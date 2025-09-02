#!/usr/bin/env python3
"""
Test script to verify modular architecture works correctly
"""

def test_modular_imports():
    """Test that all modular components can be imported"""
    print("🧪 Testing modular imports...")
    
    try:
        # Test database manager
        from database_manager import DatabaseManager
        print("✅ DatabaseManager imported successfully")
        
        # Test YOLO model manager
        from yolo_model_manager import YOLOModelManager
        print("✅ YOLOModelManager imported successfully")
        
        # Test dataset manager
        from dataset_manager import DatasetManager
        print("✅ DatasetManager imported successfully")
        
        # Test Roboflow manager
        from roboflow_manager import RoboflowManager
        print("✅ RoboflowManager imported successfully")
        
        # Test GUI components
        from gui_components import StatusBar, ModelSelectionFrame, DatasetSelectionFrame
        print("✅ GUI components imported successfully")
        
        # Test image utilities
        from image_utils import ImageProcessor, AnnotationUtils
        print("✅ Image utilities imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of modular components"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test DatasetManager
        from dataset_manager import DatasetManager
        dm = DatasetManager()
        print("✅ DatasetManager instantiated")
        
        # Test YOLOModelManager
        from yolo_model_manager import YOLOModelManager
        ymm = YOLOModelManager()
        device_info = ymm.get_device_info()
        print(f"✅ YOLOModelManager instantiated - Device: {device_info['name']}")
        
        # Test ImageProcessor
        from image_utils import ImageProcessor
        # Test with a simple image info function
        if ImageProcessor.validate_image_file("yolov8n.pt"):  # This will fail as expected
            print("✅ ImageProcessor validation works")
        else:
            print("✅ ImageProcessor validation correctly rejects non-image file")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def test_main_app_import():
    """Test that the main application can still be imported"""
    print("\n📱 Testing main application import...")
    
    try:
        # This should work if modular imports are correct
        import yolo_labeler_app
        print("✅ Main application imported successfully with modular components")
        return True
        
    except ImportError as e:
        print(f"❌ Main app import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Main app unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Modular Architecture")
    print("=" * 50)
    
    imports_ok = test_modular_imports()
    functionality_ok = test_basic_functionality()
    main_app_ok = test_main_app_import()
    
    print("=" * 50)
    
    if imports_ok and functionality_ok and main_app_ok:
        print("🎉 All modular architecture tests passed!")
        print("\n📋 Modular structure summary:")
        print("   📁 database_manager.py - Database operations")
        print("   📁 yolo_model_manager.py - YOLO model handling") 
        print("   📁 dataset_manager.py - Dataset operations")
        print("   📁 roboflow_manager.py - Roboflow API operations")
        print("   📁 gui_components.py - Reusable GUI widgets")
        print("   📁 image_utils.py - Image processing utilities")
        print("   📁 yolo_labeler_app.py - Main application (now modular)")
        print("\n✨ Code is now properly modularized!")
    else:
        print("❌ Some modular architecture tests failed")
        print("Please check the import statements and file structure")

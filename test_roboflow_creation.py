#!/usr/bin/env python3
"""
Test script for Roboflow project creation functionality
This script tests the project creation feature before adding it to the main app.
"""

def test_roboflow_project_creation():
    """Test creating a new project in Roboflow"""
    print("🧪 Testing Roboflow Project Creation")
    print("=" * 40)
    
    try:
        # Test import
        from roboflow import Roboflow
        print("✅ Roboflow SDK imported successfully")
        
        # Test API key (you would need to replace with your actual key)
        api_key = "YOUR_API_KEY_HERE"  # Replace with actual API key for testing
        
        if api_key == "YOUR_API_KEY_HERE":
            print("⚠️  Please set your actual API key in the script to test")
            print("   Find your API key at: https://app.roboflow.com/settings/api")
            return False
        
        print("🔍 Testing API connection...")
        rf = Roboflow(api_key=api_key)
        
        # Test workspace access
        print("🏢 Getting workspace...")
        workspace = rf.workspace()
        print(f"✅ Connected to workspace: {getattr(workspace, 'name', 'default')}")
        
        # Test project creation (with a test project name)
        test_project_name = "test-project-creation-" + str(int(time.time()))
        print(f"🆕 Creating test project: {test_project_name}")
        
        # Get workspace and create project
        workspace = rf.workspace()
        new_project = workspace.create_project(
            project_name=test_project_name,
            project_type="object-detection",
            project_license="private",
            annotation=""  # Required parameter
        )
        
        print(f"✅ Project created successfully!")
        print(f"   Project Name: {test_project_name}")
        print(f"   Project ID: {getattr(new_project, 'id', 'N/A')}")
        print(f"   Project URL: {getattr(new_project, 'url', 'N/A')}")
        
        # Test project listing to verify creation
        print("📋 Verifying project appears in project list...")
        workspace = rf.workspace()
        project_list = workspace.project_list
        
        project_found = False
        for project_info in project_list:
            if project_info.get('name', '').lower() == test_project_name.lower():
                project_found = True
                print(f"✅ Project found in list: {project_info}")
                break
        
        if project_found:
            print("✅ Project creation test: PASSED")
        else:
            print("⚠️  Project creation test: PARTIAL (created but not found in list)")
        
        return True
        
    except ImportError:
        print("❌ Roboflow SDK not installed")
        print("   Install with: pip install roboflow")
        return False
        
    except Exception as e:
        print(f"❌ Project creation test failed: {e}")
        return False

def show_project_type_info():
    """Show information about different project types"""
    print("\n🔍 Roboflow Project Types:")
    print("=" * 40)
    
    project_types = {
        "object-detection": {
            "description": "Detect and locate objects with bounding boxes",
            "use_cases": ["Object counting", "Quality inspection", "Security monitoring"],
            "annotation_format": "Bounding boxes (x, y, width, height)"
        },
        "classification": {
            "description": "Classify entire images into categories",
            "use_cases": ["Quality control", "Medical diagnosis", "Content moderation"],
            "annotation_format": "Single label per image"
        },
        "instance-segmentation": {
            "description": "Detect objects and create pixel-perfect masks",
            "use_cases": ["Medical imaging", "Autonomous vehicles", "Precision agriculture"],
            "annotation_format": "Polygons or masks for each object instance"
        },
        "semantic-segmentation": {
            "description": "Classify every pixel in the image",
            "use_cases": ["Satellite imagery", "Medical scans", "Scene understanding"],
            "annotation_format": "Pixel-level class labels"
        }
    }
    
    for project_type, info in project_types.items():
        print(f"\n📊 {project_type.upper().replace('-', ' ')}")
        print(f"   Description: {info['description']}")
        print(f"   Use Cases: {', '.join(info['use_cases'])}")
        print(f"   Format: {info['annotation_format']}")

def show_license_info():
    """Show information about different license types"""
    print("\n📄 License Types:")
    print("=" * 40)
    
    licenses = {
        "private": "Only you and collaborators can access this project",
        "MIT": "Open source license allowing commercial and private use",
        "CC BY 4.0": "Creative Commons - requires attribution",
        "Public Domain": "No restrictions, completely open"
    }
    
    for license_type, description in licenses.items():
        print(f"   {license_type}: {description}")

if __name__ == "__main__":
    import time
    
    print("WelVision YOLO Data Labeller - Roboflow Project Creation Test")
    print("This script tests the new project creation functionality.")
    print()
    
    # Show project information
    show_project_type_info()
    show_license_info()
    
    print("\n🧪 Ready to test project creation")
    print("⚠️  To run the actual test, edit this script and add your Roboflow API key")
    
    # Uncomment the line below to run the actual test (after adding your API key)
    # test_roboflow_project_creation()

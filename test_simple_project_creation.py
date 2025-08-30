#!/usr/bin/env python3
"""
Simple test for the correct Roboflow project creation method.
"""

import os
from datetime import datetime

def test_correct_project_creation():
    """Test the correct way to create a Roboflow project."""
    
    print("🧪 Testing correct Roboflow project creation method...")
    
    # Get API key
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        api_key = input("Enter your Roboflow API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided.")
        return False
    
    try:
        from roboflow import Roboflow
        
        print("📡 Connecting to Roboflow...")
        rf = Roboflow(api_key=api_key)
        
        # Generate unique project name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"test_project_{timestamp}"
        
        print(f"🆕 Creating project: {project_name}")
        
        # Method 1: Direct creation (from GitHub docs)
        try:
            print("   Method 1: rf.create_project() - Direct method")
            new_project = rf.create_project(
                project_name=project_name,
                project_type="object-detection",
                license="private"  # Use private for testing
            )
            
            print(f"✅ SUCCESS! Project created using direct method")
            print(f"   Project: {getattr(new_project, 'name', 'N/A')}")
            print(f"   ID: {getattr(new_project, 'id', 'N/A')}")
            print(f"   URL: {getattr(new_project, 'url', 'N/A')}")
            
            return True
            
        except Exception as direct_error:
            print(f"❌ Direct method failed: {direct_error}")
            
            # Method 2: Workspace creation
            try:
                print("   Method 2: workspace.create_project() - Workspace method")
                workspace = rf.workspace()
                
                new_project = workspace.create_project(
                    project_name=project_name + "_ws",
                    project_type="object-detection",
                    project_license="private"
                )
                
                print(f"✅ SUCCESS! Project created using workspace method")
                print(f"   Project: {getattr(new_project, 'name', 'N/A')}")
                print(f"   ID: {getattr(new_project, 'id', 'N/A')}")
                print(f"   URL: {getattr(new_project, 'url', 'N/A')}")
                
                return True
                
            except Exception as workspace_error:
                print(f"❌ Workspace method failed: {workspace_error}")
                return False
    
    except ImportError:
        print("❌ Roboflow package not installed. Run: pip install roboflow")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_correct_project_creation()
    if success:
        print("\n🎉 Project creation test passed! The app should now work correctly.")
    else:
        print("\n💔 Project creation test failed. Check your API key and permissions.")

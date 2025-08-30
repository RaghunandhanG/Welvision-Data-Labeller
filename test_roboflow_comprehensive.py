#!/usr/bin/env python3
"""
Comprehensive test for Roboflow project creation functionality.
This tests both SDK and REST API methods for creating projects.
"""

import os
import sys
import traceback
from datetime import datetime

def test_roboflow_creation():
    """Test Roboflow project creation with multiple methods."""
    
    print("🧪 Starting comprehensive Roboflow project creation test...")
    
    # Get API key from environment or user input
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        api_key = input("Enter your Roboflow API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return False
    
    # Generate unique project name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"test_project_{timestamp}"
    
    print(f"📝 Testing with project name: {project_name}")
    
    # Test 1: SDK Method
    print("\n🔧 Test 1: Using Roboflow Python SDK...")
    sdk_success = test_sdk_method(api_key, project_name + "_sdk")
    
    # Test 2: REST API Method
    print("\n🌐 Test 2: Using REST API...")
    rest_success = test_rest_method(api_key, project_name + "_rest")
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"   SDK Method: {'✅ Success' if sdk_success else '❌ Failed'}")
    print(f"   REST API Method: {'✅ Success' if rest_success else '❌ Failed'}")
    
    if sdk_success or rest_success:
        print("\n✅ At least one method worked! Project creation should work in the app.")
        return True
    else:
        print("\n❌ Both methods failed. Check your API key and network connection.")
        return False

def test_sdk_method(api_key, project_name):
    """Test project creation using Roboflow Python SDK."""
    try:
        from roboflow import Roboflow
        
        print("   🔗 Connecting to Roboflow...")
        rf = Roboflow(api_key=api_key)
        
        print("   🏢 Getting workspace...")
        workspace = rf.workspace()
        print(f"   📂 Workspace: {getattr(workspace, 'name', 'default')}")
        
        print(f"   🆕 Creating project: {project_name}")
        
        # Try multiple parameter combinations
        param_sets = [
            {
                "project_name": project_name,
                "project_type": "object-detection",
                "project_license": "MIT",
                "annotation": ""
            },
            {
                "project_name": project_name,
                "project_type": "object-detection",
                "license": "MIT",
                "annotation": ""
            },
            {
                "project_name": project_name,
                "project_type": "object-detection"
            }
        ]
        
        for i, params in enumerate(param_sets):
            try:
                print(f"   🔧 Trying parameter set {i+1}: {list(params.keys())}")
                new_project = workspace.create_project(**params)
                print(f"   ✅ SDK method succeeded with parameter set {i+1}")
                print(f"   📋 Project ID: {getattr(new_project, 'id', 'unknown')}")
                return True
            except Exception as e:
                print(f"   ⚠️ Parameter set {i+1} failed: {e}")
                continue
        
        print("   ❌ All parameter sets failed")
        return False
        
    except ImportError:
        print("   ❌ Roboflow package not installed. Run: pip install roboflow")
        return False
    except Exception as e:
        print(f"   ❌ SDK method failed: {e}")
        print(f"   📋 Full error: {traceback.format_exc()}")
        return False

def test_rest_method(api_key, project_name):
    """Test project creation using REST API."""
    try:
        import requests
        
        print("   🌐 Testing REST API connection...")
        
        # Get workspace info
        workspace_url = f"https://api.roboflow.com/?api_key={api_key}"
        workspace_response = requests.get(workspace_url)
        
        if workspace_response.status_code != 200:
            print(f"   ❌ Failed to get workspace info: {workspace_response.status_code}")
            return False
        
        workspace_data = workspace_response.json()
        workspace_id = workspace_data.get('workspace', 'default')
        print(f"   📂 Workspace ID: {workspace_id}")
        
        # Create project
        create_url = f"https://api.roboflow.com/{workspace_id}/projects"
        payload = {
            "name": project_name,
            "type": "object-detection",
            "license": "MIT"
        }
        
        print(f"   🆕 Creating project via REST API...")
        response = requests.post(create_url, 
                               json=payload,
                               params={"api_key": api_key})
        
        if response.status_code in [200, 201]:
            project_data = response.json()
            print(f"   ✅ REST API method succeeded")
            print(f"   📋 Response: {project_data}")
            return True
        else:
            print(f"   ❌ REST API failed: {response.status_code}")
            print(f"   📋 Response: {response.text}")
            return False
            
    except ImportError:
        print("   ❌ Requests package not installed. Run: pip install requests")
        return False
    except Exception as e:
        print(f"   ❌ REST API method failed: {e}")
        print(f"   📋 Full error: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_roboflow_creation()
    sys.exit(0 if success else 1)

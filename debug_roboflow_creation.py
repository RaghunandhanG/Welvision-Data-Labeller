#!/usr/bin/env python3
"""
Debug script to test Roboflow project creation and find the actual issue.
This will help us understand what's going wrong with the project creation.
"""

import os
import requests
import json
from datetime import datetime

def debug_roboflow_creation():
    """Debug Roboflow project creation step by step."""
    
    print("🔍 Debugging Roboflow project creation...")
    
    # Get API key
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        api_key = input("Enter your Roboflow API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return False
    
    print(f"🔑 Using API key: {api_key[:10]}...")
    
    # Step 1: Get workspace information
    print("\n📋 Step 1: Getting workspace information...")
    try:
        workspace_url = f"https://api.roboflow.com/?api_key={api_key}"
        print(f"   URL: {workspace_url}")
        
        response = requests.get(workspace_url)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")
            
            workspace_id = data.get('workspace')
            workspace_name = data.get('name', 'Unknown')
            print(f"   Workspace ID: {workspace_id}")
            print(f"   Workspace Name: {workspace_name}")
        else:
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   Exception: {e}")
        return False
    
    # Step 2: List existing projects
    print(f"\n📂 Step 2: Listing existing projects...")
    try:
        projects_url = f"https://api.roboflow.com/{workspace_id}/projects?api_key={api_key}"
        print(f"   URL: {projects_url}")
        
        response = requests.get(projects_url)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")
        else:
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"   Exception: {e}")
    
    # Step 3: Try to create a test project
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"debug_test_{timestamp}"
    
    print(f"\n🆕 Step 3: Creating test project '{project_name}'...")
    
    # Try different API endpoints and methods
    creation_methods = [
        {
            "name": "Method 1: POST to /projects",
            "url": f"https://api.roboflow.com/{workspace_id}/projects",
            "method": "POST",
            "data": {
                "name": project_name,
                "type": "object-detection",
                "license": "MIT"
            },
            "params": {"api_key": api_key}
        },
        {
            "name": "Method 2: POST with different data structure",
            "url": f"https://api.roboflow.com/{workspace_id}/projects",
            "method": "POST",
            "data": {
                "project": {
                    "name": project_name,
                    "type": "object-detection",
                    "license": "MIT"
                }
            },
            "params": {"api_key": api_key}
        },
        {
            "name": "Method 3: Form data instead of JSON",
            "url": f"https://api.roboflow.com/{workspace_id}/projects",
            "method": "POST",
            "form_data": {
                "name": project_name,
                "type": "object-detection",
                "license": "MIT",
                "api_key": api_key
            }
        }
    ]
    
    for method in creation_methods:
        print(f"\n   🧪 {method['name']}")
        print(f"      URL: {method['url']}")
        
        try:
            if 'form_data' in method:
                # Use form data
                response = requests.post(method['url'], data=method['form_data'])
            else:
                # Use JSON data
                response = requests.post(
                    method['url'], 
                    json=method['data'],
                    params=method.get('params', {})
                )
            
            print(f"      Status: {response.status_code}")
            print(f"      Response: {response.text}")
            
            if response.status_code in [200, 201]:
                print(f"      ✅ {method['name']} succeeded!")
                try:
                    response_data = response.json()
                    print(f"      Data: {json.dumps(response_data, indent=2)}")
                except:
                    pass
                return True
            else:
                print(f"      ❌ {method['name']} failed")
                
        except Exception as e:
            print(f"      Exception: {e}")
    
    # Step 4: Check if we can find any project creation endpoint documentation
    print(f"\n📖 Step 4: Checking API endpoints...")
    
    test_endpoints = [
        f"https://api.roboflow.com/{workspace_id}/project/create",
        f"https://api.roboflow.com/{workspace_id}/new-project",
        f"https://api.roboflow.com/projects",
        f"https://api.roboflow.com/create-project"
    ]
    
    for endpoint in test_endpoints:
        try:
            print(f"   Testing: {endpoint}")
            response = requests.post(endpoint, 
                                   json={"name": f"test_{timestamp}", "type": "object-detection"},
                                   params={"api_key": api_key})
            print(f"      Status: {response.status_code}")
            if response.status_code != 404:
                print(f"      Response: {response.text}")
        except Exception as e:
            print(f"      Exception: {e}")
    
    print("\n❌ All project creation methods failed.")
    print("🔍 This suggests the REST API method for project creation might not be available")
    print("    or requires different authentication/parameters.")
    
    return False

if __name__ == "__main__":
    debug_roboflow_creation()

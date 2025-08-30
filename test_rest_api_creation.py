#!/usr/bin/env python3
"""
Test the new REST API implementation for Roboflow project creation.
This tests the exact same method now implemented in the app.
"""

import os
import requests
import json
import re
from datetime import datetime

def test_rest_api_project_creation():
    """Test project creation using the REST API approach from the documentation."""
    
    print("🧪 Testing REST API project creation (same as app implementation)...")
    
    # Get API key
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        api_key = input("Enter your Roboflow API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided.")
        return False
    
    try:
        # Step 1: Get workspace information
        print("🔍 Getting workspace information...")
        workspace_response = requests.get(f"https://api.roboflow.com/?api_key={api_key}")
        
        if workspace_response.status_code != 200:
            print(f"❌ Failed to get workspace info: {workspace_response.status_code}")
            print(f"Response: {workspace_response.text}")
            return False
        
        workspace_data = workspace_response.json()
        workspace_id = workspace_data.get('workspace', 'default')
        print(f"✅ Workspace ID: {workspace_id}")
        print(f"📋 Workspace data: {json.dumps(workspace_data, indent=2)}")
        
        # Step 2: Create test project
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"REST API Test {timestamp}"
        
        print(f"\n🆕 Creating project: {project_name}")
        
        # Create annotation name (alphanumeric only as per docs)
        annotation_name = re.sub(r'[^a-zA-Z0-9]', '', project_name.lower())[:20]
        if not annotation_name:
            annotation_name = "annotation1"
        
        print(f"📝 Annotation group: {annotation_name}")
        
        # Prepare payload according to documentation
        payload = {
            "name": project_name,
            "type": "object-detection",
            "annotation": annotation_name
        }
        
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        # Make POST request
        create_url = f"https://api.roboflow.com/{workspace_id}/projects"
        headers = {
            'Content-Type': 'application/json'
        }
        params = {
            'api_key': api_key
        }
        
        print(f"🌐 POST {create_url}")
        print(f"🔑 Headers: {headers}")
        print(f"🎛️ Params: {params}")
        
        create_response = requests.post(
            create_url,
            json=payload,
            headers=headers,
            params=params
        )
        
        print(f"\n📋 Response Status: {create_response.status_code}")
        print(f"📋 Response Headers: {dict(create_response.headers)}")
        print(f"📋 Response Text: {create_response.text}")
        
        if create_response.status_code == 200 or create_response.status_code == 201:
            # Success!
            try:
                response_data = create_response.json()
                print(f"\n✅ PROJECT CREATED SUCCESSFULLY!")
                print(f"📋 Response Data: {json.dumps(response_data, indent=2)}")
                
                project_id = response_data.get('id', '')
                print(f"🆔 Project ID: {project_id}")
                
                return True
                
            except Exception as parse_error:
                print(f"⚠️ Could not parse JSON response: {parse_error}")
                print("But HTTP status indicates success!")
                return True
                
        else:
            # Failed
            print(f"\n❌ PROJECT CREATION FAILED")
            
            try:
                error_data = create_response.json()
                print(f"Error data: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw error response: {create_response.text}")
            
            return False
            
    except requests.exceptions.RequestException as req_error:
        print(f"❌ Network error: {req_error}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🔬 This tests the exact same REST API method implemented in the app")
    print("📖 Based on: https://docs.roboflow.com/developer/create-a-project")
    print()
    
    success = test_rest_api_project_creation()
    
    if success:
        print("\n🎉 REST API project creation works!")
        print("✅ The app implementation should now create real projects!")
    else:
        print("\n💔 REST API project creation failed.")
        print("❌ Check your API key and permissions.")
        print("🔍 Check the error messages above for details.")

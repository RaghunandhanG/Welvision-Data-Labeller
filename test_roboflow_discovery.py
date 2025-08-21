"""
Roboflow Workspace and Project Discovery Script
Tests API key and discovers available workspaces and projects
"""
import sys
from roboflow import Roboflow

def test_roboflow_discovery(api_key):
    """Test Roboflow API and discover workspaces/projects"""
    
    print("🔍 Roboflow Discovery Test")
    print("=" * 50)
    
    try:
        # Initialize Roboflow
        print("🔄 Initializing Roboflow...")
        rf = Roboflow(api_key=api_key)
        print("✅ Roboflow instance created")
        
        # Test 1: Try to access default workspace
        print("\n📋 Test 1: Accessing default workspace...")
        try:
            default_workspace = rf.workspace()
            
            print(f"✅ Default workspace found")
            print(f"🔍 WORKSPACE OBJECT DETAILS:")
            print(f"   Type: {type(default_workspace)}")
            print(f"   Dir: {dir(default_workspace)}")
            
            # Print all non-private attributes and methods
            for attr in dir(default_workspace):
                if not attr.startswith('_'):
                    try:
                        value = getattr(default_workspace, attr)
                        if callable(value):
                            print(f"   {attr}(): <method>")
                        else:
                            print(f"   {attr}: {value}")
                    except Exception as attr_error:
                        print(f"   {attr}: <error accessing: {attr_error}>")
            
            workspace_id = getattr(default_workspace, 'id', 'default')
            workspace_list = [workspace_id]
                
        except Exception as e:
            print(f"❌ Default workspace access failed: {e}")
            # Try known workspace IDs
            workspace_list = ["verify-workspace-od-1"]
        
        # Test 2: For each workspace, try to access known projects
        for workspace_id in workspace_list:
            print(f"\n🏢 Test 2: Accessing workspace '{workspace_id}'...")
            try:
                workspace = rf.workspace(workspace_id)
                print(f"✅ Successfully accessed workspace: {workspace_id}")
                
                print(f"🔍 WORKSPACE '{workspace_id}' OBJECT DETAILS:")
                print(f"   Type: {type(workspace)}")
                print(f"   Dir: {dir(workspace)}")
                
                # Print all non-private attributes and methods
                for attr in dir(workspace):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(workspace, attr)
                            if callable(value):
                                print(f"   {attr}(): <method>")
                            else:
                                print(f"   {attr}: {value}")
                        except Exception as attr_error:
                            print(f"   {attr}: <error accessing: {attr_error}>")
                
                # Test 3: Try to access known projects (SDK doesn't provide project listing)
                print(f"\n📂 Test 3: Trying known projects in '{workspace_id}'...")
                known_projects = ["chatter-scratch-damage-2"]
                
                for project_id in known_projects:
                    try:
                        project = workspace.project(project_id)
                        print(f"✅ Successfully accessed project: {project_id}")
                        
                        print(f"🔍 PROJECT '{project_id}' OBJECT DETAILS:")
                        print(f"   Type: {type(project)}")
                        print(f"   Dir: {dir(project)}")
                        
                        # Print all non-private attributes and methods
                        for attr in dir(project):
                            if not attr.startswith('_'):
                                try:
                                    value = getattr(project, attr)
                                    if callable(value):
                                        print(f"   {attr}(): <method>")
                                    else:
                                        print(f"   {attr}: {value}")
                                except Exception as attr_error:
                                    print(f"   {attr}: <error accessing: {attr_error}>")
                            
                    except Exception as proj_error:
                        print(f"❌ Failed to access project {project_id}: {proj_error}")
                            
            except Exception as e:
                print(f"❌ Workspace access failed: {e}")
        
        print("\n" + "=" * 50)
        print("🎯 Summary:")
        print("✅ API key is valid and working")
        print("✅ Can access Roboflow services")
        print("💡 Note: Roboflow SDK doesn't provide workspace/project listing methods")
        print("💡 You need to know your workspace ID and project ID in advance")
        print("💡 Check your Roboflow dashboard for the correct IDs")
        
        return True
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your API key is correct")
        print("2. Ensure you have internet connection")
        print("3. Verify your Roboflow account has access to workspaces")
        return False

def main():
    """Main function"""
    # Your API key
    API_KEY = "RKVbzglmD4K4cBfDFXRJ"
    
    print("🚀 Starting Roboflow Discovery...")
    print(f"🔑 Using API Key: {API_KEY[:10]}...")
    print()
    
    success = test_roboflow_discovery(API_KEY)
    
    if success:
        print("\n🎉 Discovery completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Note the workspace and project names from above")
        print("2. Use these in your YOLO Labeler app")
        print("3. Upload your datasets with confidence!")
    else:
        print("\n❌ Discovery failed. Please check your API key and connection.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()

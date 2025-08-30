# Roboflow Project Creation Guide
## WelVision YOLO Data Labeller

### Overview
The WelVision YOLO Data Labeller now supports creating new Roboflow projects directly from the application. This feature streamlines your workflow by allowing you to create projects without leaving the labelling interface.

### How to Create a New Roboflow Project

#### Step 1: API Key Setup
1. **Navigate to the "Roboflow Projects" tab**
2. **Enter your Roboflow API key** in the API Key field
   - Find your API key at: [Roboflow Settings](https://app.roboflow.com/settings/api)
3. **Click "🔍 Load Projects"** to authenticate and load existing projects

#### Step 2: Create New Project (Optional)
Once authenticated, the project creation section becomes available:

1. **Project Name**: Enter a unique name for your project
   - Example: "vehicle-detection", "quality-control-v1"
   - Avoid special characters except hyphens and underscores

2. **Project Type**: Select the type that matches your use case:
   - **Object Detection**: For detecting and locating objects with bounding boxes
   - **Classification**: For categorizing entire images
   - **Instance Segmentation**: For pixel-perfect object outlines
   - **Semantic Segmentation**: For pixel-level scene understanding

3. **License**: Choose the appropriate license:
   - **Private**: Only you and collaborators can access (recommended for most users)
   - **MIT**: Open source, allows commercial use
   - **CC BY 4.0**: Creative Commons with attribution requirement
   - **Public Domain**: No restrictions

4. **Click "🆕 Create New Project"** to create the project

#### Step 3: Automatic Project Selection
- The newly created project is automatically selected in the project dropdown
- You can immediately start uploading datasets to this project
- The project appears in your Roboflow workspace instantly

### Project Types Explained

#### 🎯 Object Detection
**Best for**: Most common computer vision tasks
- **Use Cases**: 
  - Vehicle detection in traffic
  - Product detection in retail
  - Defect detection in manufacturing
  - Person/animal detection in security
- **Annotation Format**: Bounding boxes (rectangles) around objects
- **Output**: Object class + location coordinates

#### 🏷️ Classification  
**Best for**: Categorizing entire images
- **Use Cases**:
  - Quality control (pass/fail)
  - Medical diagnosis (healthy/disease)
  - Content moderation (appropriate/inappropriate)
  - Emotion recognition (happy/sad/angry)
- **Annotation Format**: Single label per image
- **Output**: Image class/category

#### 🎨 Instance Segmentation
**Best for**: Precise object boundaries
- **Use Cases**:
  - Medical imaging (organ segmentation)
  - Autonomous vehicles (object boundaries)
  - Agriculture (crop/weed identification)
  - Manufacturing (precise part inspection)
- **Annotation Format**: Pixel-perfect masks for each object
- **Output**: Object class + precise pixel boundaries

#### 🗺️ Semantic Segmentation
**Best for**: Understanding every pixel
- **Use Cases**:
  - Satellite imagery analysis
  - Medical scan analysis
  - Road scene understanding
  - Environmental monitoring
- **Annotation Format**: Every pixel labeled with a class
- **Output**: Complete scene understanding

### License Types Guide

#### 🔒 Private (Recommended)
- **Access**: Only you and invited collaborators
- **Best for**: Commercial projects, sensitive data
- **Cost**: May have limits based on your Roboflow plan
- **Sharing**: Controlled access only

#### 📜 MIT License
- **Access**: Open source
- **Best for**: Research projects, open collaboration
- **Commercial Use**: Allowed
- **Attribution**: Required in source code

#### 🌐 Creative Commons BY 4.0
- **Access**: Public
- **Best for**: Educational content, research
- **Commercial Use**: Allowed with attribution
- **Attribution**: Must credit original creator

#### 🌍 Public Domain
- **Access**: Completely open
- **Best for**: Unrestricted sharing
- **Commercial Use**: No restrictions
- **Attribution**: Not required

### Best Practices

#### Project Naming
- Use descriptive names: `vehicle-detection-v2` not `project1`
- Include version numbers for iterations
- Use hyphens instead of spaces
- Keep names under 50 characters

#### Project Type Selection
- **Start with Object Detection** if unsure - it's the most versatile
- **Classification** for simple yes/no or category decisions
- **Segmentation** only if you need precise boundaries

#### License Selection
- **Choose Private** for most commercial applications
- **MIT/CC BY 4.0** for research or educational projects
- **Consider your data sensitivity** and sharing requirements

### Workflow Integration

#### After Creating a Project
1. **Upload your labeled dataset** using the upload section
2. **Configure preprocessing** (if needed)
3. **Generate dataset versions** for training
4. **Train models** directly in Roboflow
5. **Deploy models** for inference

#### Project Management
- **Created projects appear immediately** in the project dropdown
- **Switch between projects** using the project selection dropdown
- **Upload different datasets** to the same project
- **Create multiple versions** for experimentation

### Troubleshooting

#### Common Errors

**"Project already exists"**
- **Solution**: Use a different project name
- **Tip**: Add version numbers or dates to make names unique

**"Unauthorized" or "Forbidden"**
- **Solution**: Check your API key permissions
- **Tip**: Ensure your API key has project creation rights

**"Project limit reached"**
- **Solution**: Upgrade your Roboflow plan or delete unused projects
- **Tip**: Check your current plan limits

**"Invalid project type"**
- **Solution**: Use one of the supported types: object-detection, classification, instance-segmentation, semantic-segmentation
- **Tip**: Object detection is the most common choice

#### API Key Issues
- **Get your API key** from [Roboflow Settings](https://app.roboflow.com/settings/api)
- **Keep API keys secure** - don't share them publicly
- **Test API connection** before creating projects

#### Network Issues
- **Check internet connection**
- **Verify Roboflow service status**
- **Try again after a few minutes**

### Advanced Features

#### Batch Project Creation
- Create multiple projects with similar settings
- Use consistent naming conventions
- Set up project templates for repeated use

#### Integration with Existing Workflow
- **Import existing datasets** into new projects
- **Migrate annotations** from other formats
- **Collaborate with team members** using appropriate licenses

#### Project Organization
- **Group related projects** by naming convention
- **Use consistent project types** within teams
- **Document project purposes** in Roboflow descriptions

### Security Considerations

#### API Key Security
- **Never commit API keys** to version control
- **Use environment variables** in production
- **Rotate keys regularly** for security

#### Data Privacy
- **Choose Private license** for sensitive data
- **Review collaborator access** regularly
- **Consider data location** and compliance requirements

### Support Resources

#### Getting Help
- **Test your setup** using the included test script
- **Check Roboflow documentation** for API details
- **Contact Roboflow support** for account issues
- **Use app console output** for debugging

#### Useful Links
- [Roboflow Documentation](https://docs.roboflow.com/)
- [API Reference](https://docs.roboflow.com/api-reference)
- [Project Types Guide](https://docs.roboflow.com/getting-started/project-types)
- [Pricing Plans](https://roboflow.com/pricing)

### Example Workflow

1. **Start the WelVision YOLO Data Labeller**
2. **Go to "Roboflow Projects" tab**
3. **Enter API key and load existing projects**
4. **Create new project**: "car-damage-detection"
   - Type: Object Detection
   - License: Private
5. **Label images** using the labelling tools
6. **Upload dataset** to the new project
7. **Train model** in Roboflow
8. **Deploy and use** the trained model

This streamlined workflow eliminates the need to switch between applications and keeps your entire computer vision pipeline in one place.

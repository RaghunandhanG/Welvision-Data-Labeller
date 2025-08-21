import roboflow

rf = roboflow.Roboflow(api_key="RKVbzglmD4K4cBfDFXRJ")
workspace = rf.workspace("")   # your workspace slug

workspace.upload_dataset(
    dataset_path=r"C:\Users\raghu\OneDrive\Desktop\sample test",         # folder path with images + txt
    project_name="chatter-scratch-damage-2",   # project slug
    dataset_format="yolov8",          # tells Roboflow your labels are YOLOv8 format
    project_type="object-detection"   # type of project
)

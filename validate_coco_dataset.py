#!/usr/bin/env python3
"""
COCO Dataset Validation Tool for Roboflow Upload
Validates your COCO dataset before upload to ensure annotations work correctly
"""

import os
import json
import argparse

def validate_coco_dataset(dataset_path):
    """Comprehensive COCO dataset validation"""
    print(f"🔍 Validating COCO dataset: {dataset_path}")
    print("=" * 50)
    
    # Check basic structure
    coco_file = os.path.join(dataset_path, "annotations.json")
    images_dir = os.path.join(dataset_path, "images")
    
    if not os.path.exists(coco_file):
        print(f"❌ COCO annotations file not found: {coco_file}")
        return False
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    print(f"✅ Basic structure valid")
    print(f"   📄 COCO file: {coco_file}")
    print(f"   📁 Images dir: {images_dir}")
    
    # Load and validate COCO data
    try:
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in COCO file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading COCO file: {e}")
        return False
    
    # Check required fields
    required_fields = ['images', 'annotations', 'categories']
    for field in required_fields:
        if field not in coco_data:
            print(f"❌ Missing required field: {field}")
            return False
    
    images_data = coco_data['images']
    annotations_data = coco_data['annotations']
    categories_data = coco_data['categories']
    
    print(f"✅ COCO structure valid")
    print(f"   📊 Images: {len(images_data)}")
    print(f"   🏷️ Annotations: {len(annotations_data)}")
    print(f"   📋 Categories: {len(categories_data)}")
    
    # Validate categories
    print(f"\n📋 Category Validation:")
    category_ids = set()
    category_names = set()
    
    for cat in categories_data:
        if 'id' not in cat:
            print(f"❌ Category missing 'id': {cat}")
            return False
        if 'name' not in cat:
            print(f"❌ Category missing 'name': {cat}")
            return False
        
        cat_id = cat['id']
        cat_name = cat['name']
        
        if cat_id in category_ids:
            print(f"❌ Duplicate category ID: {cat_id}")
            return False
        
        if cat_name in category_names:
            print(f"❌ Duplicate category name: {cat_name}")
            return False
        
        category_ids.add(cat_id)
        category_names.add(cat_name)
    
    print(f"✅ Categories valid: {list(category_names)}")
    
    # Validate images
    print(f"\n📊 Image Validation:")
    image_ids = set()
    missing_images = []
    image_files = set()
    
    for img in images_data:
        if 'id' not in img:
            print(f"❌ Image missing 'id': {img}")
            return False
        if 'file_name' not in img:
            print(f"❌ Image missing 'file_name': {img}")
            return False
        
        img_id = img['id']
        img_filename = img['file_name']
        
        if img_id in image_ids:
            print(f"❌ Duplicate image ID: {img_id}")
            return False
        
        image_ids.add(img_id)
        image_files.add(img_filename)
        
        # Check if file exists
        img_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(img_path):
            missing_images.append(img_filename)
    
    if missing_images:
        print(f"❌ Missing {len(missing_images)} image files:")
        for missing in missing_images[:10]:  # Show first 10
            print(f"   📷 {missing}")
        if len(missing_images) > 10:
            print(f"   ... and {len(missing_images) - 10} more")
        return False
    
    print(f"✅ All {len(images_data)} images found on disk")
    
    # Check for extra files in images directory
    actual_files = {f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))}
    extra_files = actual_files - image_files
    
    if extra_files:
        print(f"⚠️ Found {len(extra_files)} extra image files not in COCO:")
        for extra in list(extra_files)[:5]:  # Show first 5
            print(f"   📷 {extra}")
        if len(extra_files) > 5:
            print(f"   ... and {len(extra_files) - 5} more")
    
    # Validate annotations
    print(f"\n🏷️ Annotation Validation:")
    orphan_annotations = []
    invalid_categories = []
    
    for ann in annotations_data:
        required_ann_fields = ['id', 'image_id', 'category_id', 'bbox']
        for field in required_ann_fields:
            if field not in ann:
                print(f"❌ Annotation missing '{field}': {ann}")
                return False
        
        # Check if image_id exists
        if ann['image_id'] not in image_ids:
            orphan_annotations.append(ann['id'])
        
        # Check if category_id exists
        if ann['category_id'] not in category_ids:
            invalid_categories.append(ann['id'])
        
        # Validate bbox format
        bbox = ann['bbox']
        if not isinstance(bbox, list) or len(bbox) != 4:
            print(f"❌ Invalid bbox format in annotation {ann['id']}: {bbox}")
            return False
        
        x, y, w, h = bbox
        if any(v < 0 for v in [x, y, w, h]):
            print(f"❌ Negative bbox values in annotation {ann['id']}: {bbox}")
            return False
    
    if orphan_annotations:
        print(f"❌ Found {len(orphan_annotations)} orphan annotations (no matching image)")
        return False
    
    if invalid_categories:
        print(f"❌ Found {len(invalid_categories)} annotations with invalid category_id")
        return False
    
    print(f"✅ All {len(annotations_data)} annotations valid")
    
    # Summary statistics
    print(f"\n📈 Dataset Statistics:")
    
    # Annotations per image
    annotations_per_image = {}
    for ann in annotations_data:
        img_id = ann['image_id']
        annotations_per_image[img_id] = annotations_per_image.get(img_id, 0) + 1
    
    images_with_annotations = len(annotations_per_image)
    images_without_annotations = len(images_data) - images_with_annotations
    avg_annotations = sum(annotations_per_image.values()) / len(annotations_per_image) if annotations_per_image else 0
    
    print(f"   📊 Images with annotations: {images_with_annotations}")
    print(f"   📊 Images without annotations: {images_without_annotations}")
    print(f"   📊 Average annotations per image: {avg_annotations:.2f}")
    
    # Annotations per category
    annotations_per_category = {}
    for ann in annotations_data:
        cat_id = ann['category_id']
        annotations_per_category[cat_id] = annotations_per_category.get(cat_id, 0) + 1
    
    print(f"   📋 Annotations per category:")
    category_lookup = {cat['id']: cat['name'] for cat in categories_data}
    for cat_id, count in sorted(annotations_per_category.items()):
        cat_name = category_lookup.get(cat_id, f"ID_{cat_id}")
        print(f"      {cat_name}: {count}")
    
    # Create annotation_labelmap for Roboflow
    annotation_labelmap = {cat['id']: cat['name'] for cat in categories_data}
    print(f"\n🏷️ Annotation Label Map for Roboflow:")
    print(f"   {annotation_labelmap}")
    
    print(f"\n✅ Dataset validation completed successfully!")
    print(f"🚀 This dataset is ready for Roboflow upload with annotations.")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate COCO dataset for Roboflow upload")
    parser.add_argument("dataset_path", nargs="?", help="Path to dataset directory containing annotations.json and images/")
    
    args = parser.parse_args()
    
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        # Default to sample8 dataset
        dataset_path = os.path.join(os.getcwd(), "datasets", "sample8")
        if not os.path.exists(dataset_path):
            print("❌ No dataset path provided and default 'datasets/sample8' not found")
            print("Usage: python validate_coco_dataset.py <dataset_path>")
            return
        print(f"📂 Using default dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    success = validate_coco_dataset(dataset_path)
    
    if success:
        print(f"\n🎉 Validation successful! Your dataset is ready for upload.")
    else:
        print(f"\n❌ Validation failed. Please fix the issues above before uploading.")

if __name__ == "__main__":
    main()

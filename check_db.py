#!/usr/bin/env python3
"""
Quick script to check database contents
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolo_labeler_app import DatabaseManager

def main():
    db = DatabaseManager()
    
    try:
        # Connect to database
        if not db.connect():
            print("❌ Cannot connect to database")
            return
        
        cursor = db.connection.cursor(dictionary=True)
        
        # Get all datasets (including inactive)
        cursor.execute("SELECT id, name, path, image_count, is_active FROM datasets ORDER BY id")
        all_datasets = cursor.fetchall()
        
        print(f"📊 Total datasets in database: {len(all_datasets)}")
        print("\n📋 All datasets:")
        for dataset in all_datasets:
            active_status = "✅ Active" if dataset['is_active'] else "❌ Inactive"
            print(f"   ID: {dataset['id']}, Name: {dataset['name']}, Images: {dataset['image_count']}, {active_status}")
            print(f"      Path: {dataset['path']}")
        
        # Get only active datasets (what the app sees)
        cursor.execute("SELECT * FROM datasets WHERE is_active = TRUE ORDER BY name")
        active_datasets = cursor.fetchall()
        
        print(f"\n🔍 Active datasets (what app sees): {len(active_datasets)}")
        for dataset in active_datasets:
            print(f"   {dataset['name']} -> {dataset['path']} ({dataset['image_count']} images)")
        
        cursor.close()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
    finally:
        db.disconnect()

if __name__ == "__main__":
    main()

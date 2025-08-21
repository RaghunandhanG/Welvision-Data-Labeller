"""
Database Setup Script for WelVision YOLO Data Labeller
Run this script to automatically create the database and tables
"""

import mysql.connector
from config import DATABASE_CONFIG
import sys

def create_database():
    """Create the database and tables"""
    try:
        # Connect to MySQL server (without specifying database)
        connection = mysql.connector.connect(
            host=DATABASE_CONFIG['host'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password'],
            port=DATABASE_CONFIG['port']
        )
        
        cursor = connection.cursor()
        
        # Create database
        print("Creating database...")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_CONFIG['database']}")
        print(f"✅ Database '{DATABASE_CONFIG['database']}' created successfully")
        
        # Use the database
        cursor.execute(f"USE {DATABASE_CONFIG['database']}")
        
        # Create ai_models table
        print("Creating ai_models table...")
        create_models_table_query = """
        CREATE TABLE IF NOT EXISTS ai_models (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            path VARCHAR(500) NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
        """
        cursor.execute(create_models_table_query)
        print("✅ AI Models table created successfully")
        
        # Create datasets table
        print("Creating datasets table...")
        create_datasets_table_query = """
        CREATE TABLE IF NOT EXISTS datasets (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            path VARCHAR(500) NOT NULL,
            description TEXT,
            image_count INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
        """
        cursor.execute(create_datasets_table_query)
        print("✅ Datasets table created successfully")
        
        # Insert sample data
        print("Inserting sample model data...")
        sample_models = [
            ('Head Test Aug 12', 
             r'C:\Users\raghu\AppData\Local\Microsoft\Windows\INetCache\IE\FARWN71Z\runs[1].zip\runs\detect\Head Test Aug 12\weights\best.pt', 
             'Head detection model trained in August 2012'),
            ('Default YOLO', 'yolov8n.pt', 'Default YOLOv8 nano model'),
            ('YOLOv8 Small', 'yolov8s.pt', 'YOLOv8 small model for general object detection'),
            ('YOLOv8 Medium', 'yolov8m.pt', 'YOLOv8 medium model for balanced performance'),
            ('Custom Object Detection', 'models/custom_od.pt', 'Custom object detection model')
        ]
        
        insert_query = """
        INSERT IGNORE INTO ai_models (name, path, description) VALUES (%s, %s, %s)
        """
        
        cursor.executemany(insert_query, sample_models)
        connection.commit()
        
        print(f"✅ Inserted {cursor.rowcount} sample models")
        
        # Show created data
        cursor.execute("SELECT * FROM ai_models")
        models = cursor.fetchall()
        
        print("\n📋 Current AI models in database:")
        print("-" * 80)
        for model in models:
            print(f"ID: {model[0]}, Name: {model[1]}")
            print(f"    Path: {model[2]}")
            print(f"    Description: {model[3]}")
            print(f"    Active: {model[6]}")
            print("-" * 80)
        
        cursor.close()
        connection.close()
        
        print("\n🎉 Database setup completed successfully!")
        print(f"📍 Database: {DATABASE_CONFIG['database']}")
        print(f"📍 Host: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}")
        print("\nYou can now run the YOLO labeler application.")
        
    except mysql.connector.Error as e:
        print(f"❌ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

def test_connection():
    """Test database connection"""
    try:
        connection = mysql.connector.connect(
            host=DATABASE_CONFIG['host'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password'],
            database=DATABASE_CONFIG['database'],
            port=DATABASE_CONFIG['port']
        )
        
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM ai_models WHERE is_active = TRUE")
        model_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM datasets WHERE is_active = TRUE")
        dataset_count = cursor.fetchone()[0]
        
        cursor.close()
        connection.close()
        
        print(f"✅ Database connection successful!")
        print(f"📊 Found {model_count} active models in database")
        print(f"📊 Found {dataset_count} active datasets in database")
        return True
        
    except mysql.connector.Error as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 WelVision YOLO Data Labeller - Database Setup")
    print("=" * 50)
    
    print("\n📋 Configuration:")
    print(f"Host: {DATABASE_CONFIG['host']}")
    print(f"Port: {DATABASE_CONFIG['port']}")
    print(f"User: {DATABASE_CONFIG['user']}")
    print(f"Database: {DATABASE_CONFIG['database']}")
    
    print("\n🔧 Setting up database...")
    create_database()
    
    print("\n🧪 Testing connection...")
    if test_connection():
        print("\n✅ Setup completed successfully!")
        print("You can now run: python yolo_labeler_app.py")
    else:
        print("\n❌ Setup failed!")
        print("Please check your database configuration in config.py") 
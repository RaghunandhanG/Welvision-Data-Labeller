"""
Database setup script for WelVision YOLO Data Labeller
This script creates the required database and tables
"""

import mysql.connector
import sys

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '1469',  # Update this with your MySQL password
}

def setup_database():
    """Set up the database and tables"""
    try:
        # Connect to MySQL server (without specifying database)
        print("Connecting to MySQL server...")
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = conn.cursor()
        
        print("Connected successfully!")
        
        # Create database
        print("Creating database 'welvision_db'...")
        cursor.execute("CREATE DATABASE IF NOT EXISTS welvision_db")
        print("Database created successfully!")
        
        # Use the database
        cursor.execute("USE welvision_db")
        
        # Create models table
        print("Creating 'models' table...")
        create_models_table = """
        CREATE TABLE IF NOT EXISTS models (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """
        cursor.execute(create_models_table)
        print("Models table created successfully!")
        
        # Create datasets table
        print("Creating 'datasets' table...")
        create_datasets_table = """
        CREATE TABLE IF NOT EXISTS datasets (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            path TEXT NOT NULL,
            description TEXT,
            image_count INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """
        cursor.execute(create_datasets_table)
        print("Datasets table created successfully!")
        
        # Insert your model data
        print("Inserting your model...")
        insert_model_query = """
        INSERT INTO models (name, path) VALUES 
        (%s, %s)
        ON DUPLICATE KEY UPDATE path = VALUES(path)
        """
        
        model_data = (
            'Head Test Aug 12', 
            r'C:\Users\raghu\AppData\Local\Microsoft\Windows\INetCache\IE\FARWN71Z\runs[1].zip\runs\detect\Head Test Aug 12\weights\best.pt'
        )
        
        cursor.execute(insert_model_query, model_data)
        print("Model data inserted successfully!")
        
        # Commit changes
        conn.commit()
        
        # Display the data
        print("\nCurrent models in database:")
        cursor.execute("SELECT * FROM models")
        models = cursor.fetchall()
        
        print("ID | Name | Path")
        print("-" * 80)
        for model in models:
            print(f"{model[0]} | {model[1]} | {model[2]}")
        
        print("\nCurrent datasets in database:")
        cursor.execute("SELECT * FROM datasets")
        datasets = cursor.fetchall()
        
        if datasets:
            print("ID | Name | Path | Image Count")
            print("-" * 80)
            for dataset in datasets:
                print(f"{dataset[0]} | {dataset[1]} | {dataset[2]} | {dataset[4]}")
        else:
            print("No datasets found.")
        
        cursor.close()
        conn.close()
        print("\nDatabase setup completed successfully!")
        print("You can now run 'python yolo_labeler_fixed.py' to start the application.")
        
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        print("\nPlease check:")
        print("1. MySQL server is running")
        print("2. Username and password are correct")
        print("3. You have permission to create databases")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("WelVision YOLO Data Labeller - Database Setup")
    print("=" * 50)
    setup_database()

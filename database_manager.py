"""
Database Manager Module
Handles all database operations for model and dataset management
"""

import mysql.connector
import os
from config import DATABASE_CONFIG


class DatabaseManager:
    """Handle all database operations for model and dataset management"""
    
    def __init__(self, host=None, user=None, password=None, database=None, port=None):
        self.host = host or DATABASE_CONFIG['host']
        self.user = user or DATABASE_CONFIG['user']
        self.password = password or DATABASE_CONFIG['password']
        self.database = database or DATABASE_CONFIG['database']
        self.port = port or DATABASE_CONFIG['port']
        self.connection = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            return True
        except mysql.connector.Error as e:
            print(f"Database connection error: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
    
    def connect_without_db(self):
        """Connect without specifying database (for database creation)"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                port=self.port
            )
            return True
        except mysql.connector.Error as e:
            print(f"Database connection error: {e}")
            return False
    
    def check_and_create_database(self):
        """Check if database exists and create tables if needed"""
        try:
            # Connect without database first
            if not self.connect_without_db():
                return False
            
            cursor = self.connection.cursor()
            
            # Check if database exists
            cursor.execute("SHOW DATABASES LIKE %s", (self.database,))
            if not cursor.fetchone():
                # Create database
                cursor.execute(f"CREATE DATABASE {self.database}")
                print(f"Created database: {self.database}")
            
            # Switch to the database
            cursor.execute(f"USE {self.database}")
            
            # Check and create ai_models table (renamed from models)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_models (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL UNIQUE,
                    path VARCHAR(500) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Check and create datasets table
            cursor.execute("""
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
            """)
            
            self.connection.commit()
            cursor.close()
            self.disconnect()
            return True
            
        except mysql.connector.Error as e:
            print(f"Database setup error: {e}")
            return False
    
    def get_models(self):
        """Retrieve all active models from database that exist on filesystem"""
        if not self.connect():
            raise Exception("Cannot connect to database. Please check your database configuration")
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM ai_models WHERE is_active = TRUE ORDER BY name")
            all_models = cursor.fetchall()
            cursor.close()
            
            # Filter models to only include those with existing files
            available_models = []
            for model in all_models:
                if os.path.exists(model['path']):
                    available_models.append(model)
                else:
                    print(f"Model file not found: {model['path']} (skipping {model['name']})")
            
            return available_models
        except mysql.connector.Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            self.disconnect()
    
    def get_model_by_name(self, name):
        """Get specific model by name"""
        if not self.connect():
            return None
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM ai_models WHERE name = %s AND is_active = TRUE", (name,))
            model = cursor.fetchone()
            cursor.close()
            return model
        except mysql.connector.Error as e:
            print(f"Error fetching model: {e}")
            return None
        finally:
            self.disconnect()
    
    def add_model(self, name, path, description=""):
        """Add a new model to database"""
        if not self.connect():
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO ai_models (name, path, description) VALUES (%s, %s, %s)",
                (name, path, description)
            )
            self.connection.commit()
            cursor.close()
            return True
        except mysql.connector.Error as e:
            print(f"Error adding model: {e}")
            return False
        finally:
            self.disconnect()
    
    def get_datasets(self):
        """Retrieve all active datasets from database"""
        if not self.connect():
            return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM datasets WHERE is_active = TRUE ORDER BY name")
            datasets = cursor.fetchall()
            cursor.close()
            return datasets
        except mysql.connector.Error as e:
            print(f"Error fetching datasets: {e}")
            return []
        finally:
            self.disconnect()
    
    def add_dataset(self, name, path, description="", image_count=0):
        """Add a new dataset to database"""
        if not self.connect():
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO datasets (name, path, description, image_count) VALUES (%s, %s, %s, %s)",
                (name, path, description, image_count)
            )
            self.connection.commit()
            cursor.close()
            return True
        except mysql.connector.Error as e:
            print(f"Error adding dataset: {e}")
            return False
        finally:
            self.disconnect()
    
    def update_dataset_image_count(self, name, image_count):
        """Update image count for a dataset"""
        if not self.connect():
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "UPDATE datasets SET image_count = %s WHERE name = %s",
                (image_count, name)
            )
            self.connection.commit()
            cursor.close()
            return True
        except mysql.connector.Error as e:
            print(f"Error updating dataset: {e}")
            return False
        finally:
            self.disconnect()
    
    def remove_dataset(self, dataset_id):
        """Remove dataset by ID"""
        if not self.connect():
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("UPDATE datasets SET is_active = FALSE WHERE id = %s", (dataset_id,))
            self.connection.commit()
            cursor.close()
            return True
        except mysql.connector.Error as e:
            print(f"Error removing dataset: {e}")
            return False
        finally:
            self.disconnect()
    
    def remove_dataset_by_name(self, dataset_name):
        """Remove dataset by name"""
        if not self.connect():
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("UPDATE datasets SET is_active = FALSE WHERE name = %s", (dataset_name,))
            self.connection.commit()
            cursor.close()
            return True
        except mysql.connector.Error as e:
            print(f"Error removing dataset: {e}")
            return False
        finally:
            self.disconnect()

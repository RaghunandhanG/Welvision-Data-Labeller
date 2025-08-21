"""
Standalone Database Settings Configuration Tool
Use this to configure database settings for WelVision Data Labeller
"""

import tkinter as tk
from tkinter import ttk, messagebox
import mysql.connector
import os
import re

def update_config_file(host, port, user, password, database):
    """Update the config.py file with new database settings"""
    try:
        # Read the current config file
        with open('config.py', 'r') as f:
            content = f.read()
        
        # Update the DATABASE_CONFIG section
        new_config = f"""DATABASE_CONFIG = {{
    'host': '{host}',  # Change to your MySQL server IP if on another PC
    'user': '{user}',
    'password': '{password}',  # ⚠️ ENTER YOUR MYSQL PASSWORD HERE
    'database': '{database}',
    'port': {port}
}}"""
        
        # Replace the existing DATABASE_CONFIG
        pattern = r'DATABASE_CONFIG\s*=\s*\{[^}]*\}'
        updated_content = re.sub(pattern, new_config, content, flags=re.DOTALL)
        
        # Write back to file
        with open('config.py', 'w') as f:
            f.write(updated_content)
        
        return True
    except Exception as e:
        print(f"Error updating config file: {e}")
        return False

class DatabaseSettingsApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WelVision Data Labeller - Database Settings")
        self.root.geometry("600x500")
        self.root.configure(bg="#0a2158")
        
        # Center window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 600) // 2
        y = (screen_height - 500) // 2
        self.root.geometry(f"600x500+{x}+{y}")
        
        # Load current config
        try:
            from config import DATABASE_CONFIG
            self.current_config = DATABASE_CONFIG.copy()
        except ImportError:
            self.current_config = {
                'host': 'localhost',
                'user': 'root',
                'password': '',
                'database': 'welvision_db',
                'port': 3306
            }
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create the settings interface"""
        # Header
        tk.Label(self.root, text="WELVISION Database Settings", 
                font=("Arial", 18, "bold"), fg="white", bg="#0a2158").pack(pady=20)
        
        # Current settings display
        current_frame = tk.LabelFrame(self.root, text="Current Settings", 
                                     font=("Arial", 12, "bold"), fg="white", bg="#0a2158")
        current_frame.pack(fill=tk.X, padx=20, pady=10)
        
        for key, value in self.current_config.items():
            display_value = "****" if key == 'password' and value else str(value)
            tk.Label(current_frame, text=f"{key.title()}: {display_value}", 
                     font=("Arial", 11), fg="#cccccc", bg="#0a2158").pack(anchor=tk.W, padx=10, pady=2)
        
        # New settings input
        input_frame = tk.LabelFrame(self.root, text="Configure Settings", 
                                   font=("Arial", 12, "bold"), fg="white", bg="#0a2158")
        input_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Create input fields
        self.fields = {}
        field_configs = [
            ("Host", "host", "localhost"),
            ("Port", "port", "3306"),
            ("User", "user", "root"),
            ("Password", "password", ""),
            ("Database", "database", "welvision_db")
        ]
        
        for i, (label, key, default) in enumerate(field_configs):
            frame = tk.Frame(input_frame, bg="#0a2158")
            frame.pack(fill=tk.X, padx=10, pady=5)
            
            tk.Label(frame, text=f"{label}:", font=("Arial", 11), 
                     fg="white", bg="#0a2158", width=12, anchor=tk.W).pack(side=tk.LEFT)
            
            current_value = str(self.current_config.get(key, default))
            var = tk.StringVar(value=current_value)
            
            entry = tk.Entry(frame, textvariable=var, font=("Arial", 11), width=30)
            if key == "password":
                entry.config(show="*")
            entry.pack(side=tk.LEFT, padx=5)
            
            self.fields[key] = var
        
        # Buttons
        button_frame = tk.Frame(self.root, bg="#0a2158")
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Test Connection", 
                 font=("Arial", 12, "bold"), bg="#007bff", fg="white", width=15,
                 command=self.test_connection).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Save Settings", 
                 font=("Arial", 12, "bold"), bg="#28a745", fg="white", width=15,
                 command=self.save_settings).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Create Database", 
                 font=("Arial", 12, "bold"), bg="#ffc107", fg="black", width=15,
                 command=self.create_database).pack(side=tk.LEFT, padx=5)
        
        # Status area
        self.status_text = tk.Text(self.root, height=8, width=70, font=("Arial", 9))
        self.status_text.pack(padx=20, pady=10)
        
        self.log("Database Settings Configuration Tool")
        self.log("=" * 40)
        self.log("1. Configure your MySQL connection settings above")
        self.log("2. Click 'Test Connection' to verify settings")
        self.log("3. Click 'Create Database' to set up database and tables")
        self.log("4. Click 'Save Settings' to update config.py")
        self.log("5. Restart WelVision Data Labeller")
    
    def log(self, message):
        """Add message to status log"""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update()
    
    def test_connection(self):
        """Test database connection"""
        try:
            host = self.fields['host'].get()
            port = int(self.fields['port'].get())
            user = self.fields['user'].get()
            password = self.fields['password'].get()
            database = self.fields['database'].get()
            
            self.log(f"Testing connection to {user}@{host}:{port}...")
            
            # Test server connection first
            connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                port=port
            )
            
            cursor = connection.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()[0]
            cursor.close()
            connection.close()
            
            self.log(f"✅ Server connection successful! MySQL version: {version}")
            
            # Test database connection
            try:
                connection = mysql.connector.connect(
                    host=host,
                    user=user,
                    password=password,
                    port=port,
                    database=database
                )
                connection.close()
                self.log(f"✅ Database '{database}' connection successful!")
                messagebox.showinfo("Success", "Database connection successful!")
            except mysql.connector.Error as db_error:
                if "Unknown database" in str(db_error):
                    self.log(f"⚠️  Database '{database}' does not exist. Click 'Create Database' to create it.")
                    messagebox.showwarning("Database Not Found", f"Database '{database}' does not exist.\nClick 'Create Database' to create it.")
                else:
                    self.log(f"❌ Database connection failed: {db_error}")
                    messagebox.showerror("Error", f"Database connection failed: {db_error}")
            
        except mysql.connector.Error as e:
            self.log(f"❌ Connection failed: {e}")
            messagebox.showerror("Connection Error", f"Connection failed: {e}")
        except ValueError:
            messagebox.showerror("Error", "Port must be a valid number")
    
    def create_database(self):
        """Create database and tables"""
        try:
            host = self.fields['host'].get()
            port = int(self.fields['port'].get())
            user = self.fields['user'].get()
            password = self.fields['password'].get()
            database = self.fields['database'].get()
            
            self.log(f"Creating database '{database}'...")
            
            # Connect to MySQL server
            connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                port=port
            )
            
            cursor = connection.cursor()
            
            # Create database
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
            self.log(f"✅ Database '{database}' created")
            
            # Use the database
            cursor.execute(f"USE {database}")
            
            # Create ai_models table
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
            self.log("✅ ai_models table created")
            
            # Create datasets table
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
            self.log("✅ datasets table created")
            
            connection.commit()
            cursor.close()
            connection.close()
            
            self.log("🎉 Database setup completed successfully!")
            messagebox.showinfo("Success", "Database and tables created successfully!")
            
        except mysql.connector.Error as e:
            self.log(f"❌ Database creation failed: {e}")
            messagebox.showerror("Error", f"Database creation failed: {e}")
        except ValueError:
            messagebox.showerror("Error", "Port must be a valid number")
    
    def save_settings(self):
        """Save settings to config.py"""
        try:
            host = self.fields['host'].get()
            port = int(self.fields['port'].get())
            user = self.fields['user'].get()
            password = self.fields['password'].get()
            database = self.fields['database'].get()
            
            if update_config_file(host, port, user, password, database):
                self.log("✅ Settings saved to config.py")
                messagebox.showinfo("Success", "Settings saved successfully!\nRestart WelVision Data Labeller to use new settings.")
            else:
                self.log("❌ Failed to save settings")
                messagebox.showerror("Error", "Failed to save settings to config.py")
                
        except ValueError:
            messagebox.showerror("Error", "Port must be a valid number")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = DatabaseSettingsApp()
    app.run()

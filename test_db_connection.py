"""
Database Connection Test Utility
Use this to test and configure database connection settings
"""

import mysql.connector
import tkinter as tk
from tkinter import ttk, messagebox
from config import DATABASE_CONFIG

def test_connection(host, user, password, port, database=None):
    """Test database connection with given parameters"""
    try:
        if database:
            connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                port=port,
                database=database
            )
        else:
            connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                port=port
            )
        
        cursor = connection.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        cursor.close()
        connection.close()
        return True, f"Connection successful! MySQL version: {version[0]}"
    
    except mysql.connector.Error as e:
        return False, f"Connection failed: {str(e)}"

def create_test_window():
    """Create a simple test window for database configuration"""
    root = tk.Tk()
    root.title("Database Connection Test")
    root.geometry("500x400")
    root.configure(bg="#0a2158")
    
    # Center window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 500) // 2
    y = (screen_height - 400) // 2
    root.geometry(f"500x400+{x}+{y}")
    
    tk.Label(root, text="Database Connection Test", 
             font=("Arial", 16, "bold"), fg="white", bg="#0a2158").pack(pady=20)
    
    # Current config display
    current_frame = tk.LabelFrame(root, text="Current Config", 
                                 font=("Arial", 12, "bold"), fg="white", bg="#0a2158")
    current_frame.pack(fill=tk.X, padx=20, pady=10)
    
    tk.Label(current_frame, text=f"Host: {DATABASE_CONFIG['host']}", 
             font=("Arial", 11), fg="#cccccc", bg="#0a2158").pack(anchor=tk.W, padx=10, pady=2)
    tk.Label(current_frame, text=f"Port: {DATABASE_CONFIG['port']}", 
             font=("Arial", 11), fg="#cccccc", bg="#0a2158").pack(anchor=tk.W, padx=10, pady=2)
    tk.Label(current_frame, text=f"User: {DATABASE_CONFIG['user']}", 
             font=("Arial", 11), fg="#cccccc", bg="#0a2158").pack(anchor=tk.W, padx=10, pady=2)
    tk.Label(current_frame, text=f"Database: {DATABASE_CONFIG['database']}", 
             font=("Arial", 11), fg="#cccccc", bg="#0a2158").pack(anchor=tk.W, padx=10, pady=2)
    
    # Test buttons
    button_frame = tk.Frame(root, bg="#0a2158")
    button_frame.pack(pady=20)
    
    def test_current_config():
        success, message = test_connection(
            DATABASE_CONFIG['host'],
            DATABASE_CONFIG['user'],
            DATABASE_CONFIG['password'],
            DATABASE_CONFIG['port']
        )
        
        if success:
            messagebox.showinfo("Success", message)
        else:
            messagebox.showerror("Error", message)
    
    def test_with_database():
        success, message = test_connection(
            DATABASE_CONFIG['host'],
            DATABASE_CONFIG['user'],
            DATABASE_CONFIG['password'],
            DATABASE_CONFIG['port'],
            DATABASE_CONFIG['database']
        )
        
        if success:
            messagebox.showinfo("Success", message)
        else:
            messagebox.showerror("Error", message)
    
    def open_config_editor():
        """Open a simple config editor"""
        config_window = tk.Toplevel(root)
        config_window.title("Edit Database Config")
        config_window.geometry("400x300")
        config_window.configure(bg="#0a2158")
        config_window.transient(root)
        config_window.grab_set()
        
        tk.Label(config_window, text="Database Configuration", 
                 font=("Arial", 14, "bold"), fg="white", bg="#0a2158").pack(pady=10)
        
        # Input fields
        fields = {}
        for i, (key, value) in enumerate([
            ("Host", DATABASE_CONFIG['host']),
            ("Port", str(DATABASE_CONFIG['port'])),
            ("User", DATABASE_CONFIG['user']),
            ("Password", DATABASE_CONFIG['password']),
            ("Database", DATABASE_CONFIG['database'])
        ]):
            frame = tk.Frame(config_window, bg="#0a2158")
            frame.pack(fill=tk.X, padx=20, pady=5)
            
            tk.Label(frame, text=f"{key}:", font=("Arial", 11), 
                     fg="white", bg="#0a2158", width=10, anchor=tk.W).pack(side=tk.LEFT)
            
            var = tk.StringVar(value=str(value))
            entry = tk.Entry(frame, textvariable=var, font=("Arial", 11), width=25)
            if key == "Password":
                entry.config(show="*")
            entry.pack(side=tk.LEFT, padx=5)
            fields[key.lower()] = var
        
        def save_and_test():
            # Test the new configuration
            try:
                port = int(fields['port'].get())
                success, message = test_connection(
                    fields['host'].get(),
                    fields['user'].get(),
                    fields['password'].get(),
                    port,
                    fields['database'].get()
                )
                
                if success:
                    messagebox.showinfo("Success", f"Configuration works!\n{message}\n\nUpdate config.py manually with these settings.")
                    config_window.destroy()
                else:
                    messagebox.showerror("Error", message)
            except ValueError:
                messagebox.showerror("Error", "Port must be a number")
        
        tk.Button(config_window, text="Test Configuration", 
                 font=("Arial", 11, "bold"), bg="#28a745", fg="white",
                 command=save_and_test).pack(pady=10)
        
        tk.Button(config_window, text="Cancel", 
                 font=("Arial", 11, "bold"), bg="#6c757d", fg="white",
                 command=config_window.destroy).pack(pady=5)
    
    tk.Button(button_frame, text="Test Server Connection", 
             font=("Arial", 11, "bold"), bg="#007bff", fg="white",
             command=test_current_config).pack(pady=5)
    
    tk.Button(button_frame, text="Test Database Connection", 
             font=("Arial", 11, "bold"), bg="#28a745", fg="white",
             command=test_with_database).pack(pady=5)
    
    tk.Button(button_frame, text="Edit Configuration", 
             font=("Arial", 11, "bold"), bg="#ffc107", fg="black",
             command=open_config_editor).pack(pady=5)
    
    # Status text
    status_text = tk.Text(root, height=6, width=60, font=("Arial", 9))
    status_text.pack(padx=20, pady=10)
    status_text.insert(tk.END, "Instructions:\n")
    status_text.insert(tk.END, "1. Click 'Test Server Connection' to check MySQL server\n")
    status_text.insert(tk.END, "2. Click 'Test Database Connection' to check specific database\n")
    status_text.insert(tk.END, "3. Click 'Edit Configuration' to modify settings\n")
    status_text.insert(tk.END, "4. Update config.py with working settings\n")
    status_text.insert(tk.END, "5. Restart the main application")
    status_text.config(state=tk.DISABLED)
    
    root.mainloop()

if __name__ == "__main__":
    print("🔧 Database Connection Test Utility")
    print("=" * 40)
    
    # Quick command line test
    print("Testing current configuration...")
    success, message = test_connection(
        DATABASE_CONFIG['host'],
        DATABASE_CONFIG['user'],
        DATABASE_CONFIG['password'],
        DATABASE_CONFIG['port']
    )
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
    
    print("\nOpening GUI configuration tool...")
    create_test_window()

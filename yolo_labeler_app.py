"""
WelVision YOLO Data Labeller
A comprehensive image labeling application using YOLO v8 models
Enhanced version with scrollable interface, dataset management, and folder upload
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import shutil
import json
from datetime import datetime
import mysql.connector
from ultralytics import YOLO
import threading
from pathlib import Path
import torch
from config import DATABASE_CONFIG, APP_CONFIG, FILE_CONFIG, YOLO_CONFIG

# Application configuration from config file
APP_TITLE = APP_CONFIG['title']
APP_GEOMETRY = APP_CONFIG['geometry']
APP_BG_COLOR = APP_CONFIG['bg_color']

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
        """Connect to MySQL server without specifying database"""
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
        """Check if database exists and create if not"""
        try:
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
            raise Exception("Cannot connect to database. Please check your database configuration in config.py")
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM datasets WHERE is_active = TRUE ORDER BY name")
            datasets = cursor.fetchall()
            cursor.close()
            return datasets
        except mysql.connector.Error as e:
            raise Exception(f"Database error: {e}")
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
        """Update dataset image count"""
        if not self.connect():
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "UPDATE datasets SET image_count = %s, updated_at = CURRENT_TIMESTAMP WHERE name = %s",
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

class YOLOLabelerApp(tk.Tk):
    """Main application class for YOLO image labeling with enhanced features"""
    
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(APP_GEOMETRY)
        self.configure(bg=APP_BG_COLOR)
        
        # Center window on screen
        self.center_window()
        
        # Initialize variables
        self.db_manager = DatabaseManager()
        self.current_model = None
        self.yolo_model = None
        self.uploaded_images = []
        self.current_dataset_name = ""
        self.current_dataset_path = ""
        self.dataset_save_location = ""

        
        # Detect optimal device
        self.optimal_device = self.detect_optimal_device()
        
        # Initialize database
        self.initialize_database()
        
        # Create main scrollable frame
        self.create_scrollable_interface()
        
        # Create GUI
        self.create_widgets()
        
        # Try to load data from database
        try:
            self.load_models()
            self.load_datasets()
        except Exception as e:
            self.status_var.set(f"Database connection failed: {str(e)}")
            messagebox.showwarning("Database Connection", 
                                 "Could not connect to database. Please run setup_database.py first.\n\n"
                                 "You can still use the app to load models manually.")
        
        # Bind window close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def detect_optimal_device(self):
        """Detect the optimal device for YOLO inference"""
        try:
            if torch.cuda.is_available():
                device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                self.status_var.set(f"GPU detected: {gpu_name} - Using CUDA acceleration") if hasattr(self, 'status_var') else None
                return device
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                self.status_var.set("Apple Silicon GPU detected - Using MPS acceleration") if hasattr(self, 'status_var') else None
                return device
            else:
                device = 'cpu'
                self.status_var.set("No GPU detected - Using CPU processing") if hasattr(self, 'status_var') else None
                return device
        except Exception as e:
            print(f"Device detection error: {e}")
            return 'cpu'
    
    def initialize_database(self):
        """Initialize database and tables"""
        try:
            if self.db_manager.check_and_create_database():
                print("Database initialized successfully")
            else:
                print("Failed to initialize database")
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def center_window(self):
        """Center the application window on screen"""
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # Use 90% of screen width and 85% of screen height for better utilization
        app_width = int(screen_width * 0.9)
        app_height = int(screen_height * 0.85)
        
        x = (screen_width - app_width) // 2
        y = (screen_height - app_height) // 2
        self.geometry(f"{app_width}x{app_height}+{x}+{y}")
    
    def create_scrollable_interface(self):
        """Create scrollable main interface"""
        # Create main canvas and scrollbar
        self.main_canvas = tk.Canvas(self, bg=APP_BG_COLOR, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = tk.Frame(self.main_canvas, bg=APP_BG_COLOR)
        
        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack scrollable components
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel to canvas
        self.bind_mousewheel()
    
    def bind_mousewheel(self):
        """Bind mouse wheel events for scrolling"""
        def _on_mousewheel(event):
            self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            self.main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            self.main_canvas.unbind_all("<MouseWheel>")
        
        self.main_canvas.bind('<Enter>', _bind_to_mousewheel)
        self.main_canvas.bind('<Leave>', _unbind_from_mousewheel)
    
    def create_widgets(self):
        """Create and arrange all GUI widgets"""
        # Header
        self.create_header(self.scrollable_frame)
        
        # Create notebook for tabs
        from tkinter import ttk
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Tab 1: Image Labeling (current functionality)
        self.labeling_frame = tk.Frame(self.notebook, bg=APP_BG_COLOR)
        self.notebook.add(self.labeling_frame, text="📷 Image Labeling")
        
        # Tab 2: Dataset Preview
        self.preview_frame = tk.Frame(self.notebook, bg=APP_BG_COLOR)
        self.notebook.add(self.preview_frame, text="📁 Dataset Preview")
        
        # Setup labeling tab (original functionality)
        self.setup_labeling_tab()
        
        # Setup dataset preview tab (new functionality)
        self.setup_dataset_preview_tab()
    
    def setup_labeling_tab(self):
        """Setup the image labeling tab with original functionality"""
        # Main content container with left and right panels
        main_content = tk.Frame(self.labeling_frame, bg=APP_BG_COLOR)
        main_content.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls (55% width)
        self.left_panel = tk.Frame(main_content, bg=APP_BG_COLOR)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel for image preview (45% width) - increased for better image display
        self.right_panel = tk.Frame(main_content, bg=APP_BG_COLOR, width=650)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(20, 0))
        self.right_panel.pack_propagate(False)  # Maintain fixed width
        
        # Create sections in left panel
        self.create_model_section(self.left_panel)
        self.create_dataset_section(self.left_panel)
        self.create_upload_section(self.left_panel)
        self.create_action_section(self.left_panel)
        self.create_status_panel(self.left_panel)
        
        # Create image preview in right panel
        self.create_image_preview_panel(self.right_panel)
    
    def setup_dataset_preview_tab(self):
        """Setup the dataset preview tab for viewing labeled images"""
        # Top section for dataset selection
        selection_frame = tk.LabelFrame(self.preview_frame, text="📂 Dataset Selection", 
                                       font=("Arial", 16, "bold"), 
                                       fg="white", bg=APP_BG_COLOR, 
                                       relief="raised", bd=3)
        selection_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Dataset selection controls
        controls_frame = tk.Frame(selection_frame, bg=APP_BG_COLOR)
        controls_frame.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(controls_frame, text="Select Dataset:", 
                font=("Arial", 14, "bold"), fg="white", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        
        self.preview_dataset_var = tk.StringVar()
        self.preview_dataset_dropdown = ttk.Combobox(controls_frame, textvariable=self.preview_dataset_var, 
                                                    font=("Arial", 12), width=50, state="readonly")
        self.preview_dataset_dropdown.pack(side=tk.LEFT, padx=(10, 10))
        self.preview_dataset_dropdown.bind('<<ComboboxSelected>>', self.on_preview_dataset_selected)
        
        tk.Button(controls_frame, text="🔄 Refresh Datasets", 
                 font=("Arial", 12, "bold"), bg="#17a2b8", fg="white",
                 command=self.refresh_preview_datasets).pack(side=tk.LEFT, padx=5)
        
        tk.Button(controls_frame, text="📁 Browse Dataset Folder", 
                 font=("Arial", 12, "bold"), bg="#28a745", fg="white",
                 command=self.browse_dataset_folder).pack(side=tk.LEFT, padx=5)
        
        # Dataset info section
        info_frame = tk.Frame(selection_frame, bg=APP_BG_COLOR)
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        self.dataset_info_var = tk.StringVar(value="No dataset selected")
        tk.Label(info_frame, textvariable=self.dataset_info_var, 
                font=("Arial", 11), fg="#ffc107", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        
        # Image preview section
        preview_main_frame = tk.LabelFrame(self.preview_frame, text="🖼️ Dataset Images", 
                                          font=("Arial", 16, "bold"), 
                                          fg="white", bg=APP_BG_COLOR, 
                                          relief="raised", bd=3)
        preview_main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Create horizontal layout: thumbnails on left, preview on right
        main_preview_container = tk.Frame(preview_main_frame, bg=APP_BG_COLOR)
        main_preview_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Left side: Thumbnail grid (60% width)
        thumbnail_frame = tk.LabelFrame(main_preview_container, text="📁 Image Gallery", 
                                       font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR)
        thumbnail_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create scrollable canvas for image grid
        self.dataset_canvas_frame = tk.Frame(thumbnail_frame, bg=APP_BG_COLOR)
        self.dataset_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.dataset_canvas = tk.Canvas(self.dataset_canvas_frame, bg="#1a1a1a", highlightthickness=0)
        dataset_scrollbar_v = ttk.Scrollbar(self.dataset_canvas_frame, orient=tk.VERTICAL, command=self.dataset_canvas.yview)
        self.dataset_canvas.configure(yscrollcommand=dataset_scrollbar_v.set)
        
        self.dataset_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dataset_scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Frame inside canvas for thumbnails
        self.dataset_inner_frame = tk.Frame(self.dataset_canvas, bg="#1a1a1a")
        self.dataset_canvas_window = self.dataset_canvas.create_window((0, 0), window=self.dataset_inner_frame, anchor="nw")
        
        # Right side: Image preview (40% width)
        preview_panel = tk.LabelFrame(main_preview_container, text="🔍 Image Preview", 
                                     font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR)
        preview_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        preview_panel.pack_propagate(False)
        
        # Preview controls
        preview_controls = tk.Frame(preview_panel, bg=APP_BG_COLOR)
        preview_controls.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(preview_controls, text="⬅️ Previous", 
                 font=("Arial", 10, "bold"), bg="#6c757d", fg="white",
                 command=self.prev_preview_image).pack(side=tk.LEFT, padx=2)
        
        tk.Button(preview_controls, text="➡️ Next", 
                 font=("Arial", 10, "bold"), bg="#6c757d", fg="white",
                 command=self.next_preview_image).pack(side=tk.LEFT, padx=2)
        
        # Preview info
        self.preview_image_info_var = tk.StringVar(value="No image selected")
        tk.Label(preview_controls, textvariable=self.preview_image_info_var, 
                font=("Arial", 9), fg="#ffc107", bg=APP_BG_COLOR).pack(side=tk.RIGHT)
        
        # Preview canvas (smaller size)
        self.preview_image_canvas = tk.Canvas(preview_panel, width=300, height=300, 
                                             bg="#1a1a1a", highlightthickness=0)
        self.preview_image_canvas.pack(padx=10, pady=(0, 10))
        
        # Add keyboard bindings for navigation
        self.bind('<Left>', lambda e: self.prev_preview_image())
        self.bind('<Right>', lambda e: self.next_preview_image())
        self.bind('<Up>', lambda e: self.prev_preview_image())
        self.bind('<Down>', lambda e: self.next_preview_image())
        
        # Focus the window to enable keyboard navigation
        self.focus_set()
        
        # Bind canvas events
        self.dataset_canvas.bind('<Configure>', self.on_dataset_canvas_configure)
        self.dataset_inner_frame.bind('<Configure>', self.on_dataset_frame_configure)
        
        # Mouse wheel binding for dataset canvas
        def _on_dataset_mousewheel(event):
            self.dataset_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.dataset_canvas.bind_all("<MouseWheel>", _on_dataset_mousewheel)
        
        # Initialize dataset preview variables
        self.dataset_images = []
        self.dataset_thumbnails = []
        self.current_preview_index = 0
        self.current_preview_image = None
        
        # Load datasets after UI is created
        self.after(100, self.refresh_preview_datasets)  # Delay to ensure UI is ready
    
    def on_dataset_canvas_configure(self, event):
        """Handle dataset canvas resize"""
        self.dataset_canvas.configure(scrollregion=self.dataset_canvas.bbox("all"))
        # Update the inner frame width to match canvas width
        canvas_width = event.width
        self.dataset_canvas.itemconfig(self.dataset_canvas_window, width=canvas_width)
    
    def on_dataset_frame_configure(self, event):
        """Handle dataset frame resize"""
        self.dataset_canvas.configure(scrollregion=self.dataset_canvas.bbox("all"))
    
    def refresh_preview_datasets(self):
        """Refresh the dataset dropdown for preview"""
        try:
            # Check if the preview dropdown exists yet
            if not hasattr(self, 'preview_dataset_dropdown'):
                print("Preview dropdown not created yet, skipping refresh")
                return
                
            datasets = self.db_manager.get_datasets()
            print(f"Found {len(datasets)} datasets in database")
            
            dataset_names = [f"{ds['name']} ({ds['image_count']} images)" for ds in datasets]  # Use dictionary keys
            self.preview_dataset_dropdown['values'] = dataset_names
            
            if dataset_names:
                self.preview_dataset_dropdown.set(dataset_names[0])
                self.dataset_info_var.set(f"📊 {len(datasets)} dataset(s) available")
                print(f"Set dropdown values: {dataset_names}")
            else:
                self.preview_dataset_dropdown.set("")
                self.dataset_info_var.set("No datasets found")
                print("No datasets found in database")
                
        except Exception as e:
            print(f"Error refreshing datasets: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'dataset_info_var'):
                self.dataset_info_var.set("Error loading datasets")
    
    def browse_dataset_folder(self):
        """Browse for an existing dataset folder to preview"""
        from tkinter import filedialog
        folder_path = filedialog.askdirectory(title="Select Dataset Folder to Preview")
        
        if folder_path:
            self.load_dataset_images_from_folder(folder_path)
    
    def on_preview_dataset_selected(self, event=None):
        """Handle dataset selection in preview tab"""
        selected = self.preview_dataset_var.get()
        if not selected:
            return
        
        # Extract dataset name from selection (remove image count part)
        dataset_name = selected.split(' (')[0]
        
        try:
            # Get dataset info from database
            datasets = self.db_manager.get_datasets()
            selected_dataset = None
            for ds in datasets:
                if ds['name'] == dataset_name:  # Use dictionary key
                    selected_dataset = ds
                    break
            
            if selected_dataset:
                dataset_path = selected_dataset['path']  # Use dictionary key
                self.dataset_info_var.set(f"📁 Loading: {dataset_name} from {dataset_path}")
                self.load_dataset_images_from_folder(dataset_path)
            else:
                self.dataset_info_var.set("❌ Dataset not found")
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.dataset_info_var.set(f"❌ Error loading dataset: {str(e)}")
    
    def load_dataset_images_from_folder(self, folder_path):
        """Load and display images from a dataset folder"""
        try:
            # Clear previous images
            for widget in self.dataset_inner_frame.winfo_children():
                widget.destroy()
            self.dataset_images.clear()
            self.dataset_thumbnails.clear()
            
            # Find image files in the folder - prioritize labeled images
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            
            if os.path.exists(folder_path):
                # First, look for images in the images subfolder (original structure)
                images_folder = os.path.join(folder_path, "images")
                labels_folder = os.path.join(folder_path, "labels")
                
                if os.path.exists(images_folder):
                    # Look for original images in images folder
                    for file in os.listdir(images_folder):
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_path = os.path.join(images_folder, file)
                            
                            # Check if there's a corresponding labeled image
                            base_name = os.path.splitext(file)[0]
                            labeled_image_path = os.path.join(folder_path, f"{base_name}_labeled.jpg")
                            
                            # Prefer labeled image if it exists, otherwise use original
                            if os.path.exists(labeled_image_path):
                                image_files.append(labeled_image_path)
                            else:
                                image_files.append(image_path)
                else:
                    # Look in the main folder for any images (including labeled ones)
                    for file in os.listdir(folder_path):
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_files.append(os.path.join(folder_path, file))
            
            if not image_files:
                self.dataset_info_var.set(f"📂 No images found in {os.path.basename(folder_path)}")
                return
            
            # Sort files for consistent ordering
            image_files.sort()
            self.dataset_images = image_files
            
            # Create image grid (4 columns for better layout in smaller space)
            columns = 4
            thumbnail_size = (120, 120)  # Smaller thumbnails
            
            self.dataset_info_var.set(f"📊 Loading {len(image_files)} images from {os.path.basename(folder_path)}...")
            self.update()
            
            for i, image_path in enumerate(image_files):
                try:
                    # Calculate grid position
                    row = i // columns
                    col = i % columns
                    
                    # Load and create thumbnail
                    pil_image = Image.open(image_path)
                    pil_image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(pil_image)
                    self.dataset_thumbnails.append(photo)  # Keep reference
                    
                    # Create thumbnail container
                    thumb_frame = tk.Frame(self.dataset_inner_frame, bg="#2a2a2a", relief=tk.RAISED, bd=2)
                    thumb_frame.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
                    
                    # Image label
                    img_label = tk.Label(thumb_frame, image=photo, bg="#2a2a2a", cursor="hand2")
                    img_label.pack(pady=3)
                    
                    # Filename label (shortened)
                    filename = os.path.basename(image_path)
                    if len(filename) > 15:
                        filename = filename[:12] + "..."
                    
                    # Add indicator if it's a labeled image
                    if "_labeled" in filename:
                        filename = "🏷️ " + filename.replace("_labeled", "")
                    
                    name_label = tk.Label(thumb_frame, text=filename, 
                                        font=("Arial", 7), fg="white", bg="#2a2a2a")
                    name_label.pack(pady=(0, 3))
                    
                    # Bind click event for preview
                    def make_click_handler(path, idx):
                        def handler(event=None):
                            self.show_dataset_image_preview(idx)
                        return handler
                    
                    click_handler = make_click_handler(image_path, i)
                    thumb_frame.bind("<Button-1>", click_handler)
                    img_label.bind("<Button-1>", click_handler)
                    name_label.bind("<Button-1>", click_handler)
                    
                    # Hover effects
                    def on_enter(e, frame=thumb_frame):
                        frame.config(bg="#3a3a3a", relief=tk.RAISED, bd=3)
                    
                    def on_leave(e, frame=thumb_frame):
                        frame.config(bg="#2a2a2a", relief=tk.RAISED, bd=2)
                    
                    thumb_frame.bind("<Enter>", on_enter)
                    thumb_frame.bind("<Leave>", on_leave)
                    img_label.bind("<Enter>", on_enter)
                    img_label.bind("<Leave>", on_leave)
                    name_label.bind("<Enter>", on_enter)
                    name_label.bind("<Leave>", on_leave)
                    
                except Exception as e:
                    print(f"Error creating thumbnail for {image_path}: {e}")
            
            # Configure grid weights for responsive layout
            for i in range(columns):
                self.dataset_inner_frame.grid_columnconfigure(i, weight=1)
            
            # Update canvas scroll region
            self.dataset_inner_frame.update_idletasks()
            self.dataset_canvas.configure(scrollregion=self.dataset_canvas.bbox("all"))
            
            # Update info and initialize preview
            labeled_count = sum(1 for f in image_files if "_labeled" in f)
            self.dataset_info_var.set(f"✅ Loaded {len(image_files)} images ({labeled_count} labeled) - Click for preview")
            
            # Initialize preview variables
            self.current_preview_index = 0
            
            # Show first image if available
            if image_files:
                self.show_dataset_image_preview(0)
            
        except Exception as e:
            print(f"Error loading dataset images: {e}")
            self.dataset_info_var.set(f"❌ Error loading images: {str(e)}")
    
    def show_dataset_image_preview(self, index):
        """Show selected image in the preview panel"""
        if not self.dataset_images or index >= len(self.dataset_images):
            return
        
        try:
            self.current_preview_index = index
            image_path = self.dataset_images[index]
            
            # Load and display image
            pil_image = Image.open(image_path)
            img_width, img_height = pil_image.size
            
            # Calculate display size for preview canvas (300x300)
            canvas_width, canvas_height = 300, 300
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            display_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_preview_image = ImageTk.PhotoImage(display_image)
            
            # Display in canvas
            self.preview_image_canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            self.preview_image_canvas.create_image(x, y, anchor=tk.NW, image=self.current_preview_image)
            
            # Update info
            filename = os.path.basename(image_path)
            is_labeled = "_labeled" in filename
            label_text = "🏷️ Labeled" if is_labeled else "📷 Original"
            self.preview_image_info_var.set(f"{label_text} | {index + 1}/{len(self.dataset_images)} | {filename}")
            
        except Exception as e:
            print(f"Error showing preview image: {e}")
            self.preview_image_info_var.set("❌ Error loading image")
    
    def prev_preview_image(self):
        """Show previous image in preview"""
        if not self.dataset_images:
            return
        
        self.current_preview_index = (self.current_preview_index - 1) % len(self.dataset_images)
        self.show_dataset_image_preview(self.current_preview_index)
    
    def next_preview_image(self):
        """Show next image in preview"""
        if not self.dataset_images:
            return
        
        self.current_preview_index = (self.current_preview_index + 1) % len(self.dataset_images)
        self.show_dataset_image_preview(self.current_preview_index)
    
    def show_dataset_image_fullsize(self, image_path):
        """Show a dataset image in full size in a new window"""
        try:
            # Create new window for full-size image (smaller than before)
            img_window = tk.Toplevel(self)
            img_window.title(f"Dataset Image: {os.path.basename(image_path)}")
            img_window.configure(bg="#1a1a1a")
            
            # Smaller window size
            img_window.geometry("600x500")
            img_window.transient(self)
            
            # Load and display image
            pil_image = Image.open(image_path)
            img_width, img_height = pil_image.size
            
            # Calculate display size (max 550x400 to fit in smaller window)
            max_width, max_height = 550, 400
            ratio = min(max_width / img_width, max_height / img_height)
            if ratio < 1:  # Only resize if image is larger than max size
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(pil_image)
            
            # Create canvas for image
            canvas = tk.Canvas(img_window, bg="#1a1a1a", highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
            
            # Display image centered in canvas
            canvas.create_image(300, 200, image=photo, anchor=tk.CENTER)
            
            # Info label
            info_text = f"{os.path.basename(image_path)} | {img_width}×{img_height}px"
            is_labeled = "_labeled" in os.path.basename(image_path)
            if is_labeled:
                info_text = "🏷️ " + info_text
            
            info_label = tk.Label(img_window, text=info_text, 
                                font=("Arial", 10), fg="white", bg="#1a1a1a")
            info_label.pack(pady=(0, 10))
            
            # Keep reference to prevent garbage collection
            img_window.photo = photo
            
            # Close button
            close_btn = tk.Button(img_window, text="Close", 
                                font=("Arial", 12, "bold"), bg="#dc3545", fg="white",
                                command=img_window.destroy)
            close_btn.pack(pady=(0, 15))
            
        except Exception as e:
            print(f"Error showing full-size image: {e}")
            messagebox.showerror("Error", f"Could not display image: {str(e)}")
    
    def create_image_preview_panel(self, parent):
        """Create image preview panel on the right side"""
        preview_frame = tk.LabelFrame(parent, text="📷 Image Preview & Annotation", 
                                     font=("Arial", 16, "bold"), 
                                     fg="white", bg=APP_BG_COLOR, 
                                     relief="raised", bd=3)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Preview info and controls
        info_frame = tk.Frame(preview_frame, bg=APP_BG_COLOR)
        info_frame.pack(fill=tk.X, padx=15, pady=10)
        
        self.preview_info_var = tk.StringVar(value="No images loaded")
        info_label = tk.Label(info_frame, textvariable=self.preview_info_var, 
                             font=("Arial", 12, "bold"), fg="#ffc107", bg=APP_BG_COLOR)
        info_label.pack(side=tk.LEFT)
        
        # Preview mode buttons
        mode_frame = tk.Frame(info_frame, bg=APP_BG_COLOR)
        mode_frame.pack(side=tk.RIGHT)
        
        self.preview_mode = tk.StringVar(value="split")
        
        split_btn = tk.Radiobutton(mode_frame, text="Split View", variable=self.preview_mode, value="split",
                                  font=("Arial", 10), fg="white", bg=APP_BG_COLOR,
                                  selectcolor=APP_BG_COLOR, command=self.update_preview_mode)
        split_btn.pack(side=tk.LEFT, padx=5)
        
        raw_btn = tk.Radiobutton(mode_frame, text="Raw Only", variable=self.preview_mode, value="raw",
                                font=("Arial", 10), fg="white", bg=APP_BG_COLOR,
                                selectcolor=APP_BG_COLOR, command=self.update_preview_mode)
        raw_btn.pack(side=tk.LEFT, padx=5)
        
        labeled_btn = tk.Radiobutton(mode_frame, text="Labeled Only", variable=self.preview_mode, value="labeled",
                                    font=("Arial", 10), fg="white", bg=APP_BG_COLOR,
                                    selectcolor=APP_BG_COLOR, command=self.update_preview_mode)
        labeled_btn.pack(side=tk.LEFT, padx=5)
        
        # Auto-detection toggle
        self.auto_detect_var = tk.BooleanVar(value=True)
        auto_detect_cb = tk.Checkbutton(mode_frame, text="Auto-Detect", variable=self.auto_detect_var,
                                       font=("Arial", 10, "bold"), fg="#ffc107", bg=APP_BG_COLOR,
                                       selectcolor=APP_BG_COLOR, command=self.on_auto_detect_changed)
        auto_detect_cb.pack(side=tk.LEFT, padx=(15, 5))
        
        # Main image display container - much larger now
        main_display_frame = tk.LabelFrame(preview_frame, text="Image Comparison", 
                                          font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR)
        main_display_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))
        
        # Split view container
        self.split_container = tk.Frame(main_display_frame, bg=APP_BG_COLOR)
        self.split_container.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left side - Raw image (larger)
        raw_frame = tk.LabelFrame(self.split_container, text="🖼️ Original Image", 
                                 font=("Arial", 12, "bold"), fg="#17a2b8", bg=APP_BG_COLOR)
        raw_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5))
        
        self.raw_image_canvas = tk.Canvas(raw_frame, width=300, height=400, 
                                         bg="#1a1a1a", highlightthickness=0)
        self.raw_image_canvas.pack(pady=10, expand=True, fill=tk.BOTH)
        
        # Right side - Labeled image (larger)
        labeled_frame = tk.LabelFrame(self.split_container, text="🎯 Labeled Image", 
                                     font=("Arial", 12, "bold"), fg="#28a745", bg=APP_BG_COLOR)
        labeled_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10))
        
        self.labeled_image_canvas = tk.Canvas(labeled_frame, width=300, height=400, 
                                             bg="#1a1a1a", highlightthickness=0)
        self.labeled_image_canvas.pack(pady=10, expand=True, fill=tk.BOTH)
        
        # Single view container (initially hidden)
        self.single_container = tk.Frame(main_display_frame, bg=APP_BG_COLOR)
        
        self.single_image_canvas = tk.Canvas(self.single_container, width=600, height=500, 
                                           bg="#1a1a1a", highlightthickness=0)
        self.single_image_canvas.pack(pady=10, expand=True, fill=tk.BOTH)
        
        # Image info label
        self.main_image_info_var = tk.StringVar(value="Upload images and select from thumbnails below")
        main_info_label = tk.Label(main_display_frame, textvariable=self.main_image_info_var,
                                  font=("Arial", 10), fg="#cccccc", bg=APP_BG_COLOR)
        main_info_label.pack(pady=(0, 10))
        
        # Detection info
        self.detection_info_var = tk.StringVar(value="")
        detection_label = tk.Label(main_display_frame, textvariable=self.detection_info_var,
                                  font=("Arial", 9), fg="#ffc107", bg=APP_BG_COLOR)
        detection_label.pack(pady=(0, 5))
        
        # Labeling controls
        control_frame = tk.Frame(main_display_frame, bg=APP_BG_COLOR)
        control_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(control_frame, text="🔄 Run Detection", 
                 font=("Arial", 12, "bold"), bg="#007bff", fg="white", width=15,
                 command=self.run_detection_on_current).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="💾 Save Labeled", 
                 font=("Arial", 12, "bold"), bg="#28a745", fg="white", width=15,
                 command=self.save_current_labeled).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="⬅️ Previous", 
                 font=("Arial", 12, "bold"), bg="#6c757d", fg="white", width=10,
                 command=self.show_previous_image).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="➡️ Next", 
                 font=("Arial", 12, "bold"), bg="#6c757d", fg="white", width=10,
                 command=self.show_next_image).pack(side=tk.LEFT, padx=5)
        
        # Store for current selection
        self.current_main_image = None
        self.current_labeled_image = None
        self.current_selected_path = None
        self.current_detections = None
        self.current_image_index = 0
    
    def on_preview_canvas_configure(self, event):
        """Handle preview canvas resize"""
        canvas_width = event.width
        self.preview_canvas.itemconfig(self.preview_canvas_window, width=canvas_width)
    
    def on_preview_frame_configure(self, event):
        """Handle preview frame resize"""
        self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))
    
    def update_image_previews(self):
        """Update the image preview panel with loaded images"""
        if not self.uploaded_images:
            self.preview_info_var.set("No images loaded")
            self.main_image_info_var.set("Upload images to start labeling")
            self.clear_canvases()
            return
        
        # Update info
        count = len(self.uploaded_images)
        auto_status = "Auto-detection enabled" if self.auto_detect_var.get() else "Auto-detection disabled"
        self.preview_info_var.set(f"📊 {count} image(s) loaded - {auto_status}")
        
        # Show the first image by default if none selected
        if self.current_image_index >= count:
            self.current_image_index = 0
        
        if count > 0:
            self.show_image_at_index(self.current_image_index)
    
    def show_image_at_index(self, index):
        """Show image at the specified index"""
        if not self.uploaded_images or index >= len(self.uploaded_images):
            return
        
        self.current_image_index = index
        image_path = self.uploaded_images[index]
        
        try:
            # Load and display raw image
            image = Image.open(image_path)
            self.current_main_image = image.copy()
            self.current_selected_path = image_path
            
            # Update info
            filename = os.path.basename(image_path)
            self.main_image_info_var.set(f"📷 {filename} ({index + 1}/{len(self.uploaded_images)}) - {image.width}x{image.height}px")
            
            # Clear any existing labeled image
            self.current_labeled_image = None
            self.current_detections = None
            
            # Show raw image in canvas
            self.show_raw_image_in_canvas()
            
            # Automatically run detection if model is loaded and auto-detect is enabled
            if self.yolo_model and self.auto_detect_var.get():
                self.detection_info_var.set("🔄 Auto-running detection...")
                self.update()
                
                try:
                    # Run YOLO inference automatically
                    results = self.yolo_model(
                        self.current_selected_path,
                        conf=YOLO_CONFIG['confidence_threshold'],
                        max_det=YOLO_CONFIG['max_detections'],
                        device=self.optimal_device
                    )
                    
                    self.current_detections = results[0]
                    
                    # Create and show labeled image
                    self.create_labeled_image()
                    
                    # Update detection info
                    detection_count = len(self.current_detections.boxes) if self.current_detections.boxes is not None else 0
                    self.detection_info_var.set(f"✅ Auto-detection complete: {detection_count} objects detected")
                    
                except Exception as e:
                    print(f"Auto-detection failed: {e}")
                    self.detection_info_var.set(f"❌ Auto-detection failed: {str(e)}")
            elif self.yolo_model and not self.auto_detect_var.get():
                self.detection_info_var.set("💡 Auto-detect disabled - Click 'Run Detection' or enable Auto-Detect")
            else:
                self.detection_info_var.set("⚠️ Load a YOLO model to see labeled images")
            
            # Update display based on current mode
            self.update_preview_mode()
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            self.main_image_info_var.set("Error loading image")
    
    def show_raw_image_in_canvas(self):
        """Show the current raw image in the appropriate canvas"""
        if not self.current_main_image:
            return
        
        try:
            img_width, img_height = self.current_main_image.size
            
            # Calculate display size based on preview mode
            if self.preview_mode.get() == "split":
                canvas_width, canvas_height = 300, 400  # Updated to match larger canvas
                canvas = self.raw_image_canvas
            else:
                canvas_width, canvas_height = 600, 500  # Updated to match larger canvas
                canvas = self.single_image_canvas
            
            # Resize while maintaining aspect ratio
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            display_image = self.current_main_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_display_image = ImageTk.PhotoImage(display_image)
            
            # Display in canvas
            canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            canvas.create_image(x, y, anchor=tk.NW, image=self.current_display_image)
            
        except Exception as e:
            print(f"Error showing raw image: {e}")
    
    def show_previous_image(self):
        """Show the previous image in the list"""
        if not self.uploaded_images:
            return
        
        self.current_image_index = (self.current_image_index - 1) % len(self.uploaded_images)
        self.show_image_at_index(self.current_image_index)
    
    def show_next_image(self):
        """Show the next image in the list"""
        if not self.uploaded_images:
            return
        
        self.current_image_index = (self.current_image_index + 1) % len(self.uploaded_images)
        self.show_image_at_index(self.current_image_index)
    
    def select_image_for_preview(self, image_path, index, thumb_frame=None):
        """Select an image for detailed preview in split view"""
        try:
            # Reset previous selection highlight
            for label in self.thumbnail_labels:
                if isinstance(label, tk.Frame) and label.cget('bg') == '#007bff':
                    label.config(bg="#2a2a2a")
            
            # Highlight current selection
            if thumb_frame:
                thumb_frame.config(bg="#007bff")
            
            self.current_selected_path = image_path
            
            # Load and display raw image
            self.show_raw_image(image_path, index)
            
            # Clear labeled image initially
            self.clear_labeled_canvas()
            self.detection_info_var.set("Click 'Run Detection' to generate labeled image")
            
            # Update preview mode display
            self.update_preview_mode()
            
        except Exception as e:
            print(f"Error selecting image {image_path}: {e}")
    
    def show_raw_image(self, image_path, index):
        """Show raw image in the appropriate canvas"""
        try:
            # Load image
            pil_image = Image.open(image_path)
            img_width, img_height = pil_image.size
            
            # Calculate display size based on preview mode
            if self.preview_mode.get() == "split":
                canvas_width, canvas_height = 290, 200
            else:
                canvas_width, canvas_height = 600, 300
            
            # Resize while maintaining aspect ratio
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            display_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_main_image = ImageTk.PhotoImage(display_image)
            
            # Display in appropriate canvas
            if self.preview_mode.get() == "split":
                canvas = self.raw_image_canvas
                canvas_w, canvas_h = 290, 200
            else:
                canvas = self.single_image_canvas
                canvas_w, canvas_h = 600, 300
            
            canvas.delete("all")
            x = (canvas_w - new_width) // 2
            y = (canvas_h - new_height) // 2
            canvas.create_image(x, y, anchor=tk.NW, image=self.current_main_image)
            
            # Update info
            filename = os.path.basename(image_path)
            file_size = os.path.getsize(image_path) / 1024  # KB
            self.main_image_info_var.set(f"📷 {filename} | {img_width}×{img_height} | {file_size:.1f} KB | Image {index + 1} of {len(self.uploaded_images)}")
            
        except Exception as e:
            print(f"Error showing raw image {image_path}: {e}")
            self.main_image_info_var.set("Error loading image")
    
    def run_detection_on_current(self):
        """Run YOLO detection on currently selected image"""
        if not self.yolo_model:
            messagebox.showerror("Error", "Please load a YOLO model first")
            return
        
        if not self.current_selected_path:
            messagebox.showerror("Error", "Please select an image first")
            return
        
        try:
            self.detection_info_var.set("🔄 Running detection...")
            self.update()
            
            # Run YOLO inference
            results = self.yolo_model(
                self.current_selected_path,
                conf=YOLO_CONFIG['confidence_threshold'],
                max_det=YOLO_CONFIG['max_detections'],
                device=self.optimal_device
            )
            
            self.current_detections = results[0]
            
            # Create labeled image
            self.create_labeled_image()
            
            # Update detection info
            detection_count = len(self.current_detections.boxes) if self.current_detections.boxes is not None else 0
            self.detection_info_var.set(f"✅ Detection complete: {detection_count} objects detected")
            
        except Exception as e:
            self.detection_info_var.set(f"❌ Detection failed: {str(e)}")
            messagebox.showerror("Detection Error", f"Failed to run detection: {str(e)}")
    
    def create_labeled_image(self):
        """Create and display labeled image with YOLO detections"""
        try:
            if not self.current_detections:
                return
            
            # Load original image
            import cv2
            import numpy as np
            
            image = cv2.imread(self.current_selected_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Draw detections
            if self.current_detections.boxes is not None:
                for box in self.current_detections.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_name = self.yolo_model.names[class_id]
                    
                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Convert to PIL and display
            pil_image = Image.fromarray(image)
            img_width, img_height = pil_image.size
            
            # Calculate display size based on preview mode and larger canvas sizes
            if self.preview_mode.get() == "split":
                canvas_width, canvas_height = 300, 400  # Updated to match the larger canvas
                canvas = self.labeled_image_canvas
            else:
                canvas_width, canvas_height = 600, 500  # Updated to match the larger canvas
                canvas = self.single_image_canvas
            
            # Resize while maintaining aspect ratio
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            display_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_labeled_image = ImageTk.PhotoImage(display_image)
            
            # Display in appropriate canvas
            canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            canvas.create_image(x, y, anchor=tk.NW, image=self.current_labeled_image)
            
        except Exception as e:
            print(f"Error creating labeled image: {e}")
            self.detection_info_var.set("❌ Error creating labeled image")
    
    def update_preview_mode(self):
        """Update preview mode (split, raw only, or labeled only)"""
        mode = self.preview_mode.get()
        
        if mode == "split":
            self.single_container.pack_forget()
            self.split_container.pack(fill=tk.BOTH, expand=True, pady=10)
            # Show raw image in left canvas
            if self.current_main_image:
                self.show_raw_image_in_canvas()
            # Show labeled image in right canvas if detections exist
            if self.current_detections:
                self.create_labeled_image()
            else:
                self.labeled_image_canvas.delete("all")
        else:
            self.split_container.pack_forget()
            self.single_container.pack(fill=tk.BOTH, expand=True, pady=10)
            if mode == "raw" and self.current_main_image:
                self.show_raw_image_in_canvas()
            elif mode == "labeled" and self.current_detections:
                self.create_labeled_image()
            elif mode == "labeled" and not self.current_detections:
                self.single_image_canvas.delete("all")
    
    def on_auto_detect_changed(self):
        """Handle auto-detect checkbox change"""
        if self.auto_detect_var.get() and self.current_selected_path and self.yolo_model:
            # Auto-detect is now enabled and we have an image and model, run detection
            self.run_detection_on_current()
        elif not self.auto_detect_var.get():
            # Auto-detect disabled
            self.detection_info_var.set("💡 Auto-detect disabled - Click 'Run Detection' to label images manually")
    
    def clear_canvases(self):
        """Clear all image canvases"""
        self.raw_image_canvas.delete("all")
        self.labeled_image_canvas.delete("all")
        self.single_image_canvas.delete("all")
    
    def clear_labeled_canvas(self):
        """Clear only the labeled image canvas"""
        self.labeled_image_canvas.delete("all")
        self.single_image_canvas.delete("all")
    
    def save_current_labeled(self):
        """Save the current labeled image"""
        if not self.current_detections or not self.current_selected_path:
            messagebox.showwarning("Warning", "No labeled image to save. Run detection first.")
            return
        
        try:
            # Get save location
            filename = os.path.splitext(os.path.basename(self.current_selected_path))[0] + "_labeled.jpg"
            save_path = filedialog.asksaveasfilename(
                title="Save Labeled Image",
                defaultextension=".jpg",
                initialvalue=filename,
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
            )
            
            if save_path:
                # Create full-size labeled image
                import cv2
                image = cv2.imread(self.current_selected_path)
                
                if self.current_detections.boxes is not None:
                    for box in self.current_detections.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        # Draw bounding box
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(image, (x1, y1 - label_size[1] - 15), (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                cv2.imwrite(save_path, image)
                messagebox.showinfo("Success", f"Labeled image saved to:\n{save_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save labeled image: {str(e)}")
    
    def create_header(self, parent):
        """Create application header"""
        header_frame = tk.Frame(parent, bg=APP_BG_COLOR, height=100)
        header_frame.pack(fill=tk.X, pady=20, padx=20)
        header_frame.pack_propagate(False)
        
        # Logo and title
        title_label = tk.Label(header_frame, text="WELVISION", 
                              font=("Arial", 32, "bold"), 
                              fg="white", bg=APP_BG_COLOR)
        title_label.pack(side=tk.LEFT, pady=10)
        
        subtitle_label = tk.Label(header_frame, text="Data Labeller", 
                                 font=("Arial", 18), 
                                 fg="#cccccc", bg=APP_BG_COLOR)
        subtitle_label.pack(side=tk.LEFT, padx=(15, 0), pady=10)
    
    def create_model_section(self, parent):
        """Create model management section"""
        model_frame = tk.LabelFrame(parent, text="Model Management", 
                                   font=("Arial", 16, "bold"), 
                                     fg="white", bg=APP_BG_COLOR, 
                                     relief="raised", bd=2)
        model_frame.pack(fill=tk.X, pady=10)
        
        # Model selection row
        selection_frame = tk.Frame(model_frame, bg=APP_BG_COLOR)
        selection_frame.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(selection_frame, text="Select Model:", 
                font=("Arial", 12, "bold"), 
                fg="white", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(selection_frame, textvariable=self.model_var, 
                                          font=("Arial", 11), width=50, state="readonly")
        self.model_dropdown.pack(side=tk.LEFT, padx=(10, 20))
        self.model_dropdown.bind('<<ComboboxSelected>>', self.on_model_selected)
        
        load_model_btn = tk.Button(selection_frame, text="Load Selected Model", 
                                  font=("Arial", 11, "bold"), 
                                  bg="#007bff", fg="white", 
                                  command=self.load_selected_model)
        load_model_btn.pack(side=tk.LEFT, padx=5)
        
        # Model loading row
        loading_frame = tk.Frame(model_frame, bg=APP_BG_COLOR)
        loading_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        tk.Label(loading_frame, text="Load New Model:", 
                font=("Arial", 12, "bold"), 
                fg="white", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        
        browse_model_btn = tk.Button(loading_frame, text="Browse Model File", 
                                    font=("Arial", 11, "bold"), 
                                    bg="#28a745", fg="white", 
                                    command=self.browse_model_file)
        browse_model_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        self.model_path_var = tk.StringVar()
        model_path_entry = tk.Entry(loading_frame, textvariable=self.model_path_var, 
                                   font=("Arial", 10), width=60, state="readonly")
        model_path_entry.pack(side=tk.LEFT, padx=(10, 10))
        
        add_model_btn = tk.Button(loading_frame, text="Add to Database", 
                                 font=("Arial", 11, "bold"), 
                                 bg="#ffc107", fg="black", 
                                 command=self.add_model_to_database)
        add_model_btn.pack(side=tk.LEFT, padx=5)
    
    def create_dataset_section(self, parent):
        """Create dataset management section"""
        dataset_frame = tk.LabelFrame(parent, text="Dataset Management", 
                                     font=("Arial", 16, "bold"), 
                                     fg="white", bg=APP_BG_COLOR, 
                                     relief="raised", bd=2)
        dataset_frame.pack(fill=tk.X, pady=10)
        
        # Dataset options row
        options_frame = tk.Frame(dataset_frame, bg=APP_BG_COLOR)
        options_frame.pack(fill=tk.X, padx=20, pady=15)
        
        # Dataset type selection
        self.dataset_type_var = tk.StringVar(value="new")
        new_dataset_rb = tk.Radiobutton(options_frame, text="Create New Dataset", 
                                       variable=self.dataset_type_var, value="new",
                                       font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR,
                                       selectcolor=APP_BG_COLOR, command=self.on_dataset_type_changed)
        new_dataset_rb.pack(side=tk.LEFT, padx=(0, 30))
        
        existing_dataset_rb = tk.Radiobutton(options_frame, text="Add to Existing Dataset", 
                                            variable=self.dataset_type_var, value="existing",
                                            font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR,
                                            selectcolor=APP_BG_COLOR, command=self.on_dataset_type_changed)
        existing_dataset_rb.pack(side=tk.LEFT)
        
        # New dataset row
        self.new_dataset_frame = tk.Frame(dataset_frame, bg=APP_BG_COLOR)
        self.new_dataset_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        # Dataset name row
        name_row = tk.Frame(self.new_dataset_frame, bg=APP_BG_COLOR)
        name_row.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(name_row, text="Dataset Name:", 
                              font=("Arial", 12, "bold"), 
                fg="white", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        
        self.dataset_name_var = tk.StringVar()
        self.dataset_entry = tk.Entry(name_row, textvariable=self.dataset_name_var, 
                                     font=("Arial", 11), width=40)
        self.dataset_entry.pack(side=tk.LEFT, padx=(10, 0))
        
        # Dataset location row
        location_row = tk.Frame(self.new_dataset_frame, bg=APP_BG_COLOR)
        location_row.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(location_row, text="Save Location:", 
                             font=("Arial", 12, "bold"), 
                fg="white", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        
        browse_location_btn = tk.Button(location_row, text="Browse Folder", 
                                       font=("Arial", 11, "bold"), 
                                       bg="#17a2b8", fg="white", 
                                       command=self.browse_dataset_location)
        browse_location_btn.pack(side=tk.LEFT, padx=(10, 10))
        
        self.location_var = tk.StringVar(value="datasets (default)")
        location_entry = tk.Entry(location_row, textvariable=self.location_var, 
                                 font=("Arial", 10), width=50, state="readonly")
        location_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Existing dataset row
        self.existing_dataset_frame = tk.Frame(dataset_frame, bg=APP_BG_COLOR)
        self.existing_dataset_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        tk.Label(self.existing_dataset_frame, text="Select Dataset:", 
                             font=("Arial", 12, "bold"), 
                fg="white", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        
        self.existing_dataset_var = tk.StringVar()
        self.dataset_dropdown = ttk.Combobox(self.existing_dataset_frame, textvariable=self.existing_dataset_var, 
                                           font=("Arial", 11), width=50, state="readonly")
        self.dataset_dropdown.pack(side=tk.LEFT, padx=(10, 20))
        
        refresh_datasets_btn = tk.Button(self.existing_dataset_frame, text="Refresh", 
                                        font=("Arial", 11, "bold"), 
                                        bg="#17a2b8", fg="white", 
                                        command=self.load_datasets)
        refresh_datasets_btn.pack(side=tk.LEFT, padx=5)
        
        # Initially hide existing dataset frame
        self.existing_dataset_frame.pack_forget()
    
    def create_upload_section(self, parent):
        """Create image upload section"""
        upload_frame = tk.LabelFrame(parent, text="Image Upload", 
                                    font=("Arial", 16, "bold"), 
                                     fg="white", bg=APP_BG_COLOR, 
                                     relief="raised", bd=2)
        upload_frame.pack(fill=tk.X, pady=10)
        
        # Upload options
        options_frame = tk.Frame(upload_frame, bg=APP_BG_COLOR)
        options_frame.pack(fill=tk.X, padx=20, pady=15)
        
        upload_files_btn = tk.Button(options_frame, text="Upload Individual Images", 
                                    font=("Arial", 12, "bold"), 
                                    bg="#28a745", fg="white", width=25,
                                    command=self.upload_images)
        upload_files_btn.pack(side=tk.LEFT, padx=5)
        
        upload_folder_btn = tk.Button(options_frame, text="Upload Image Folder", 
                                     font=("Arial", 12, "bold"), 
                                     bg="#007bff", fg="white", width=25,
                                     command=self.upload_image_folder)
        upload_folder_btn.pack(side=tk.LEFT, padx=5)
        
        # Image info
        info_frame = tk.Frame(upload_frame, bg=APP_BG_COLOR)
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        self.image_info_var = tk.StringVar(value="No images loaded")
        info_label = tk.Label(info_frame, textvariable=self.image_info_var, 
                             font=("Arial", 12), fg="#cccccc", bg=APP_BG_COLOR)
        info_label.pack(side=tk.LEFT)
    
    def create_action_section(self, parent):
        """Create action buttons section"""
        action_frame = tk.LabelFrame(parent, text="Actions", 
                                    font=("Arial", 16, "bold"), 
                                    fg="white", bg=APP_BG_COLOR, 
                                    relief="raised", bd=2)
        action_frame.pack(fill=tk.X, pady=10)
        
        button_frame = tk.Frame(action_frame, bg=APP_BG_COLOR)
        button_frame.pack(fill=tk.X, padx=20, pady=15)
        
        start_btn = tk.Button(button_frame, text="Start Labeling", 
                                 font=("Arial", 12, "bold"), 
                             bg="#ffc107", fg="black", width=15,
                             command=self.start_labeling)
        start_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(button_frame, text="Clear All", 
                             font=("Arial", 12, "bold"), 
                             bg="#dc3545", fg="white", width=15,
                             command=self.clear_all)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        db_settings_btn = tk.Button(button_frame, text="DB Settings", 
                                   font=("Arial", 12, "bold"), 
                                   bg="#17a2b8", fg="white", width=15,
                                   command=self.show_db_settings)
        db_settings_btn.pack(side=tk.LEFT, padx=5)
        
        settings_btn = tk.Button(button_frame, text="YOLO Settings", 
                                font=("Arial", 12, "bold"), 
                                bg="#6c757d", fg="white", width=15,
                                command=self.show_yolo_settings)
        settings_btn.pack(side=tk.LEFT, padx=5)
    
    def create_status_panel(self, parent):
        """Create status panel"""
        status_frame = tk.LabelFrame(parent, text="Status", 
                                    font=("Arial", 16, "bold"), 
                                    fg="white", bg=APP_BG_COLOR, 
                                    relief="raised", bd=2)
        status_frame.pack(fill=tk.X, pady=10)
        
        # Status section
        self.status_var = tk.StringVar(value=f"Ready - Device: {self.optimal_device.upper()} - Load a model and upload images to begin")
        status_label = tk.Label(status_frame, textvariable=self.status_var, 
                               font=("Arial", 11), fg="#cccccc", bg=APP_BG_COLOR,
                               wraplength=800, justify=tk.LEFT)
        status_label.pack(anchor=tk.W, padx=20, pady=15)
        
        # Dataset location info
        self.location_info_var = tk.StringVar(value="Dataset save location: datasets (default)")
        location_info_label = tk.Label(status_frame, textvariable=self.location_info_var, 
                                     font=("Arial", 10), fg="#888888", bg=APP_BG_COLOR,
                                     wraplength=800, justify=tk.LEFT)
        location_info_label.pack(anchor=tk.W, padx=20, pady=(0, 15))
    
    def on_dataset_type_changed(self):
        """Handle dataset type radio button change"""
        if self.dataset_type_var.get() == "new":
            self.existing_dataset_frame.pack_forget()
            self.new_dataset_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        else:
            self.new_dataset_frame.pack_forget()
            self.existing_dataset_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
    
    def load_models(self):
        """Load available models from database"""
        try:
            models = self.db_manager.get_models()
            model_names = [model['name'] for model in models]
        
            if model_names:
                self.model_dropdown['values'] = model_names
                self.model_dropdown.set(model_names[0])  # Select first model by default
                self.status_var.set(f"Loaded {len(model_names)} models from database")
            else:
                self.status_var.set("No models found in database. Add models using 'Load New Model' section.")
        except Exception as e:
            self.model_dropdown['values'] = []
            raise e
    
    def load_datasets(self):
        """Load available datasets from database"""
        try:
            datasets = self.db_manager.get_datasets()
            dataset_names = [f"{dataset['name']} ({dataset['image_count']} images)" for dataset in datasets]
            
            if dataset_names:
                self.dataset_dropdown['values'] = dataset_names
                if dataset_names:
                    self.dataset_dropdown.set(dataset_names[0])
                self.status_var.set(f"Loaded {len(dataset_names)} datasets from database")
            else:
                self.dataset_dropdown['values'] = []
                self.status_var.set("No existing datasets found")
        except Exception as e:
            self.dataset_dropdown['values'] = []
            raise e
    
    def browse_model_file(self):
        """Browse for model file"""
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model File",
            filetypes=[
                ("PyTorch files", "*.pt *.pth"),
                ("ONNX files", "*.onnx"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.model_path_var.set(file_path)
            self.status_var.set(f"Selected model file: {os.path.basename(file_path)}")
    
    def browse_dataset_location(self):
        """Browse for dataset save location"""
        folder_path = filedialog.askdirectory(title="Select Dataset Save Location")
        
        if folder_path:
            self.dataset_save_location = folder_path
            self.location_var.set(folder_path)
            self.location_info_var.set(f"Dataset save location: {folder_path}")
            self.status_var.set(f"Selected dataset save location: {os.path.basename(folder_path)}")
        else:
            # If cancelled, reset to default
            self.dataset_save_location = ""
            self.location_var.set("datasets (default)")
            self.location_info_var.set("Dataset save location: datasets (default)")
    
    def add_model_to_database(self):
        """Add selected model to database"""
        model_path = self.model_path_var.get()
        if not model_path:
            messagebox.showerror("Error", "Please select a model file first")
            return
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Selected model file does not exist")
            return
        
        # Get model name from user
        model_name = tk.simpledialog.askstring("Model Name", 
                                              "Enter a name for this model:",
                                              initialvalue=os.path.splitext(os.path.basename(model_path))[0])
        
        if not model_name:
            return
        
        # Get description
        description = tk.simpledialog.askstring("Model Description", 
                                               "Enter a description for this model (optional):",
                                               initialvalue="")
        
        # Add to database
        if self.db_manager.add_model(model_name, model_path, description or ""):
            messagebox.showinfo("Success", f"Model '{model_name}' added successfully!")
            self.load_models()  # Refresh model list
            self.model_path_var.set("")  # Clear the path
        else:
            messagebox.showerror("Error", "Failed to add model to database. Model name might already exist.")
    
    def on_model_selected(self, event=None):
        """Handle model selection change"""
        selected_model = self.model_var.get()
        self.status_var.set(f"Selected model: {selected_model}")
    
    def load_selected_model(self):
        """Load the selected YOLO model"""
        selected_model_name = self.model_var.get()
        if not selected_model_name:
            messagebox.showerror("Error", "Please select a model first")
            return
        
        # Get model details from database
        model_info = self.db_manager.get_model_by_name(selected_model_name)
        if not model_info:
            messagebox.showerror("Error", "Model not found in database")
            return
        
        model_path = model_info['path']
        
        # Check if model file exists
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return
        
        try:
            # Load YOLO model with optimal device
            self.status_var.set(f"Loading YOLO model on {self.optimal_device.upper()}...")
            self.update()
            
            self.yolo_model = YOLO(model_path)
            
            # Try to move model to optimal device
            try:
                if self.optimal_device == 'cuda':
                    self.yolo_model.to('cuda')
                    device_info = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
                elif self.optimal_device == 'mps':
                    self.yolo_model.to('mps')
                    device_info = "Apple Silicon GPU (MPS)"
                else:
                    self.yolo_model.to('cpu')
                    device_info = "CPU"
                
                self.current_model = model_info
                self.status_var.set(f"Model loaded successfully on {device_info}: {selected_model_name}")
                messagebox.showinfo("Success", f"Model '{selected_model_name}' loaded successfully!\nDevice: {device_info}")
                
            except Exception as device_error:
                # Fallback to CPU if device fails
                print(f"Device error: {device_error}, falling back to CPU")
                self.yolo_model.to('cpu')
                self.optimal_device = 'cpu'
                self.current_model = model_info
                self.status_var.set(f"Model loaded on CPU (GPU fallback): {selected_model_name}")
                messagebox.showinfo("Success", f"Model '{selected_model_name}' loaded successfully!\nDevice: CPU (GPU fallback)")
            
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Failed to load model: {error_msg}")
            self.status_var.set("Failed to load model")
    
    def upload_images(self):
        """Upload individual images for labeling"""
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=FILE_CONFIG['supported_formats']
        )
        
        if file_paths:
            self.uploaded_images = list(file_paths)
            self.image_info_var.set(f"Loaded {len(self.uploaded_images)} images")
            self.status_var.set(f"Uploaded {len(self.uploaded_images)} images")
            # Update image previews
            self.update_image_previews()
            
    def upload_image_folder(self):
        """Upload entire folder of images"""
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        
        if folder_path:
            # Get all image files from folder
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = []
            
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if os.path.splitext(file.lower())[1] in image_extensions:
                        image_files.append(os.path.join(root, file))
            
            if image_files:
                self.uploaded_images = image_files
                self.image_info_var.set(f"Loaded {len(self.uploaded_images)} images from folder")
                self.status_var.set(f"Uploaded {len(self.uploaded_images)} images from folder: {os.path.basename(folder_path)}")
                # Update image previews
                self.update_image_previews()
            else:
                messagebox.showwarning("No Images", "No supported image files found in the selected folder")
    
    def start_labeling(self):
        """Start the labeling process"""
        if not self.yolo_model:
            messagebox.showerror("Error", "Please load a YOLO model first")
            return
        
        if not self.uploaded_images:
            messagebox.showerror("Error", "Please upload images first")
            return
        
        # Get dataset information
        dataset_name = ""
        if self.dataset_type_var.get() == "new":
            dataset_name = self.dataset_name_var.get().strip()
            if not dataset_name:
                messagebox.showerror("Error", "Please enter a dataset name")
                return
        else:
            selected_dataset = self.existing_dataset_var.get()
            if not selected_dataset:
                messagebox.showerror("Error", "Please select an existing dataset")
                return
            # Extract dataset name from "name (count images)" format
            dataset_name = selected_dataset.split(" (")[0]
        
        # Create dataset directory structure
        if not self.create_dataset_structure(dataset_name):
            return
        
        # Start labeling in a separate thread
        threading.Thread(target=self.label_images, daemon=True).start()
    
    def create_dataset_structure(self, dataset_name):
        """Create dataset directory structure"""
        try:
            # Determine base path based on user selection
            if self.dataset_save_location:
                base_path = os.path.join(self.dataset_save_location, dataset_name)
            else:
                # Use default location from config
                base_path = os.path.join(FILE_CONFIG['dataset_base_path'], dataset_name)
            
            if self.dataset_type_var.get() == "new":
                # Create new dataset
                if os.path.exists(base_path):
                    result = messagebox.askyesno("Dataset Exists", 
                                               f"Dataset '{dataset_name}' already exists at:\n{base_path}\n\nOverwrite?")
                    if not result:
                        return False
                    shutil.rmtree(base_path)
                
                os.makedirs(base_path, exist_ok=True)
                os.makedirs(os.path.join(base_path, "images"), exist_ok=True)
                os.makedirs(os.path.join(base_path, "labels"), exist_ok=True)
                
                # Add to database with full path
                self.db_manager.add_dataset(dataset_name, base_path, f"Dataset created on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                
                # Refresh the preview datasets dropdown
                self.refresh_preview_datasets()
                
                self.status_var.set(f"Created new dataset: {dataset_name} at {base_path}")
            else:
                # Add to existing dataset - need to find the existing path
                datasets = self.db_manager.get_datasets()
                existing_dataset = None
                selected_name = self.existing_dataset_var.get().split(" (")[0]  # Extract name from "name (count images)"
                
                for dataset in datasets:
                    if dataset['name'] == selected_name:
                        existing_dataset = dataset
                        break
                
                if not existing_dataset:
                    messagebox.showerror("Error", f"Dataset '{selected_name}' not found in database")
                    return False
                
                base_path = existing_dataset['path']
                if not os.path.exists(base_path):
                    messagebox.showerror("Error", f"Dataset directory does not exist: {base_path}")
                    return False
                
                dataset_name = selected_name  # Use the existing dataset name
                self.status_var.set(f"Adding to existing dataset: {dataset_name} at {base_path}")
            
            self.current_dataset_path = base_path
            self.current_dataset_name = dataset_name
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create dataset structure: {str(e)}")
            return False
    
    def label_images(self):
        """Label images using the loaded YOLO model"""
        try:
            total_images = len(self.uploaded_images)
            labeled_count = 0
            
            for i, image_path in enumerate(self.uploaded_images):
                # Update status
                self.status_var.set(f"Labeling image {i+1}/{total_images}: {os.path.basename(image_path)}")
                self.update()
                
                # Run YOLO inference with optimal device
                try:
                    results = self.yolo_model(
                        image_path,
                        conf=YOLO_CONFIG['confidence_threshold'],
                        max_det=YOLO_CONFIG['max_detections'],
                        device=self.optimal_device
                    )
                except Exception as device_error:
                    # If device error occurs, try with CPU
                    if "CUDA" in str(device_error) or "device" in str(device_error).lower() or "mps" in str(device_error).lower():
                        self.status_var.set(f"Device error, falling back to CPU for image {i+1}/{total_images}")
                        self.update()
                        self.optimal_device = 'cpu'  # Update optimal device
                        results = self.yolo_model(
                            image_path,
                            conf=YOLO_CONFIG['confidence_threshold'],
                            max_det=YOLO_CONFIG['max_detections'],
                            device='cpu'
                        )
                    else:
                        raise device_error
                
                # Process results and create YOLO format labels
                if self.process_yolo_results(image_path, results[0]):
                    labeled_count += 1
            
            # Update dataset image count in database
            current_count = len([f for f in os.listdir(os.path.join(self.current_dataset_path, "images")) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))])
            self.db_manager.update_dataset_image_count(self.current_dataset_name, current_count)
            
            # Complete
            rejected_count = total_images - labeled_count
            self.status_var.set(f"Labeling complete! Added {labeled_count}/{total_images} images with 'roller' class")
            
            # Show completion message
            self.after(0, lambda: messagebox.showinfo("Complete", 
                                                     f"Labeling complete!\n\n"
                                                     f"Dataset: {self.current_dataset_name}\n"
                                                     f"✅ Images added: {labeled_count} (with 'roller' class)\n"
                                                     f"❌ Images rejected: {rejected_count} (no 'roller' class)\n"
                                                     f"📁 Total processed: {total_images}\n\n"
                                                     f"Location: {self.current_dataset_path}"))
            
            # Refresh datasets list
            self.after(0, self.load_datasets)
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Labeling failed: {str(e)}"))
            self.status_var.set("Labeling failed")
    
    def process_yolo_results(self, image_path, results):
        """Process YOLO results and save in YOLO format - only if 'roller' class is detected"""
        try:
            # First check if any "roller" class is detected
            has_roller_class = False
            
            if results.boxes is not None:
                for box in results.boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.yolo_model.names[class_id]
                    
                    # Check if this detection is a "roller" class (case-insensitive)
                    if class_name.lower() == "roller":
                        has_roller_class = True
                        break
            
            # Only process the image if it contains at least one "roller" class
            if not has_roller_class:
                print(f"Skipping image {os.path.basename(image_path)}: No 'roller' class detected")
                return False
            
            # Get image dimensions
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            height, width = image.shape[:2]
            
            # Copy image to dataset
            image_filename = os.path.basename(image_path)
            image_name, image_ext = os.path.splitext(image_filename)
            
            # Create unique filename if necessary
            dest_image_path = os.path.join(self.current_dataset_path, "images", image_filename)
            counter = 1
            while os.path.exists(dest_image_path):
                new_filename = f"{image_name}_{counter}{image_ext}"
                dest_image_path = os.path.join(self.current_dataset_path, "images", new_filename)
                image_filename = new_filename
                counter += 1
            
            shutil.copy2(image_path, dest_image_path)
            
            # Create label file
            label_filename = os.path.splitext(image_filename)[0] + ".txt"
            label_path = os.path.join(self.current_dataset_path, "labels", label_filename)
            
            # Write YOLO format labels
            with open(label_path, 'w') as f:
                if results.boxes is not None:
                    for box in results.boxes:
                        # Get box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Convert to YOLO format (normalized center coordinates and dimensions)
                        center_x = (x1 + x2) / 2 / width
                        center_y = (y1 + y2) / 2 / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        
                        # Get class ID and confidence
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Write YOLO format: class_id center_x center_y width height
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")
            
            print(f"✅ Added image {os.path.basename(image_path)}: Contains 'roller' class")
            return True
            
        except Exception as e:
            print(f"Error processing results for {image_path}: {e}")
            return False
    
    def show_yolo_settings(self):
        """Show YOLO configuration settings dialog with confidence slider"""
        settings_window = tk.Toplevel(self)
        settings_window.title("YOLO Settings")
        settings_window.geometry("450x400")
        settings_window.configure(bg=APP_BG_COLOR)
        settings_window.transient(self)
        settings_window.grab_set()
        
        # Center the window
        settings_window.update_idletasks()
        x = (settings_window.winfo_screenwidth() - 450) // 2
        y = (settings_window.winfo_screenheight() - 400) // 2
        settings_window.geometry(f"450x400+{x}+{y}")
        
        # Settings content
        tk.Label(settings_window, text="YOLO Configuration", 
                font=("Arial", 16, "bold"), fg="white", bg=APP_BG_COLOR).pack(pady=20)
        
        # Confidence threshold slider
        conf_frame = tk.Frame(settings_window, bg=APP_BG_COLOR)
        conf_frame.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(conf_frame, text="Confidence Threshold:", 
                font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR).pack(anchor=tk.W)
        
        # Current confidence value display
        self.conf_value_var = tk.StringVar(value=f"{YOLO_CONFIG['confidence_threshold']:.2f}")
        conf_value_frame = tk.Frame(conf_frame, bg=APP_BG_COLOR)
        conf_value_frame.pack(fill=tk.X, pady=(5, 10))
        
        tk.Label(conf_value_frame, text="Current:", 
                font=("Arial", 10), fg="#cccccc", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        tk.Label(conf_value_frame, textvariable=self.conf_value_var, 
                font=("Arial", 10, "bold"), fg="#17a2b8", bg=APP_BG_COLOR).pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(conf_value_frame, text="(Higher = More Selective)", 
                font=("Arial", 9), fg="#888888", bg=APP_BG_COLOR).pack(side=tk.RIGHT)
        
        # Confidence slider
        def update_confidence(value):
            conf_val = float(value)
            self.conf_value_var.set(f"{conf_val:.2f}")
            YOLO_CONFIG['confidence_threshold'] = conf_val
        
        self.conf_slider = tk.Scale(conf_frame, from_=0.01, to=0.95, resolution=0.01, 
                                   orient=tk.HORIZONTAL, length=350, 
                                   bg="#2a2a2a", fg="white", highlightbackground=APP_BG_COLOR,
                                   troughcolor="#1a1a1a", activebackground="#17a2b8",
                                   command=update_confidence)
        self.conf_slider.set(YOLO_CONFIG['confidence_threshold'])
        self.conf_slider.pack(fill=tk.X, pady=5)
        
        # Confidence guide
        guide_frame = tk.Frame(conf_frame, bg=APP_BG_COLOR)
        guide_frame.pack(fill=tk.X, pady=5)
        tk.Label(guide_frame, text="0.01", font=("Arial", 8), fg="#888888", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        tk.Label(guide_frame, text="Less Selective", font=("Arial", 8), fg="#888888", bg=APP_BG_COLOR).pack(side=tk.LEFT, padx=(10, 0))
        tk.Label(guide_frame, text="More Selective", font=("Arial", 8), fg="#888888", bg=APP_BG_COLOR).pack(side=tk.RIGHT, padx=(0, 10))
        tk.Label(guide_frame, text="0.95", font=("Arial", 8), fg="#888888", bg=APP_BG_COLOR).pack(side=tk.RIGHT)
        
        # Max detections (read-only)
        max_det_frame = tk.Frame(settings_window, bg=APP_BG_COLOR)
        max_det_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(max_det_frame, text="Max Detections:", 
                font=("Arial", 11), fg="white", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        tk.Label(max_det_frame, text=str(YOLO_CONFIG['max_detections']), 
                font=("Arial", 11), fg="#cccccc", bg=APP_BG_COLOR).pack(side=tk.RIGHT)
        
        # Device info
        device_frame = tk.Frame(settings_window, bg=APP_BG_COLOR)
        device_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(device_frame, text="Device:", 
                font=("Arial", 11), fg="white", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        
        device_color = "#cccccc"
        device_text = str(YOLO_CONFIG['device'])
        if YOLO_CONFIG['device'] == 'cpu':
            device_color = "#ffc107"
            device_text += " (CPU-only mode)"
        elif YOLO_CONFIG['device'] == 'auto':
            device_color = "#17a2b8"
            device_text += " (Auto-detect)"
        
        tk.Label(device_frame, text=device_text, 
                font=("Arial", 11), fg=device_color, bg=APP_BG_COLOR).pack(side=tk.RIGHT)
        
        # Current device info
        current_device_frame = tk.Frame(settings_window, bg=APP_BG_COLOR)
        current_device_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(current_device_frame, text="Current Device:", 
                font=("Arial", 11, "bold"), fg="white", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        
        device_color = "#28a745" if self.optimal_device == 'cuda' else "#ffc107" if self.optimal_device == 'mps' else "#17a2b8"
        current_device_text = self.optimal_device.upper()
        if self.optimal_device == 'cuda':
            current_device_text += f" ({torch.cuda.get_device_name(0)})"
        elif self.optimal_device == 'mps':
            current_device_text += " (Apple Silicon)"
        
        tk.Label(current_device_frame, text=current_device_text, 
                font=("Arial", 11), fg=device_color, bg=APP_BG_COLOR).pack(side=tk.RIGHT)
        
        # Separator
        separator = tk.Frame(settings_window, height=2, bg="#444444")
        separator.pack(fill=tk.X, padx=20, pady=15)
        
        # Info text
        info_text = "💡 Confidence Threshold:\n"
        info_text += "• Lower values detect more objects (less selective)\n"
        info_text += "• Higher values detect fewer, more confident objects\n"
        info_text += "• Recommended range: 0.25 - 0.70 for most use cases"
        
        tk.Label(settings_window, text=info_text, 
                font=("Arial", 9), fg="#888888", bg=APP_BG_COLOR, justify=tk.LEFT).pack(pady=10)
        
        # Button frame
        button_frame = tk.Frame(settings_window, bg=APP_BG_COLOR)
        button_frame.pack(pady=20)
        
        # Reset to default button
        def reset_to_default():
            default_conf = 0.25
            self.conf_slider.set(default_conf)
            update_confidence(default_conf)
        
        tk.Button(button_frame, text="Reset to Default", 
                 font=("Arial", 10, "bold"), bg="#ffc107", fg="black", width=15,
                 command=reset_to_default).pack(side=tk.LEFT, padx=5)
        
        # Close button
        tk.Button(button_frame, text="Close", 
                 font=("Arial", 10, "bold"), bg="#6c757d", fg="white", width=10,
                 command=settings_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def show_db_settings(self):
        """Show database settings configuration dialog"""
        settings_window = tk.Toplevel(self)
        settings_window.title("Database Settings")
        settings_window.geometry("650x550")
        settings_window.configure(bg=APP_BG_COLOR)
        settings_window.transient(self)
        settings_window.grab_set()
        settings_window.resizable(True, True)  # Allow resizing
        
        # Center the window
        settings_window.update_idletasks()
        x = (settings_window.winfo_screenwidth() - 650) // 2
        y = (settings_window.winfo_screenheight() - 550) // 2
        settings_window.geometry(f"650x550+{x}+{y}")
        
        # Settings content
        tk.Label(settings_window, text="Database Configuration", 
                font=("Arial", 16, "bold"), fg="white", bg=APP_BG_COLOR).pack(pady=20)
        
        # Current settings frame
        current_frame = tk.LabelFrame(settings_window, text="Current Settings", 
                                     font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR)
        current_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Host
        host_frame = tk.Frame(current_frame, bg=APP_BG_COLOR)
        host_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(host_frame, text="Host:", font=("Arial", 11), fg="white", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        tk.Label(host_frame, text=self.db_manager.host, font=("Arial", 11), fg="#cccccc", bg=APP_BG_COLOR).pack(side=tk.RIGHT)
        
        # Port
        port_frame = tk.Frame(current_frame, bg=APP_BG_COLOR)
        port_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(port_frame, text="Port:", font=("Arial", 11), fg="white", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        tk.Label(port_frame, text=str(self.db_manager.port), font=("Arial", 11), fg="#cccccc", bg=APP_BG_COLOR).pack(side=tk.RIGHT)
        
        # User
        user_frame = tk.Frame(current_frame, bg=APP_BG_COLOR)
        user_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(user_frame, text="User:", font=("Arial", 11), fg="white", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        tk.Label(user_frame, text=self.db_manager.user, font=("Arial", 11), fg="#cccccc", bg=APP_BG_COLOR).pack(side=tk.RIGHT)
        
        # Database
        db_frame = tk.Frame(current_frame, bg=APP_BG_COLOR)
        db_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(db_frame, text="Database:", font=("Arial", 11), fg="white", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        tk.Label(db_frame, text=self.db_manager.database, font=("Arial", 11), fg="#cccccc", bg=APP_BG_COLOR).pack(side=tk.RIGHT)
        
        # New settings frame
        new_frame = tk.LabelFrame(settings_window, text="Update Settings", 
                                 font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR)
        new_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Configure grid weights for proper resizing
        new_frame.grid_columnconfigure(1, weight=1)
        
        # Host input
        tk.Label(new_frame, text="Host:", font=("Arial", 11), fg="white", bg=APP_BG_COLOR).grid(row=0, column=0, sticky=tk.W, padx=15, pady=8)
        host_var = tk.StringVar(value=self.db_manager.host)
        host_entry = tk.Entry(new_frame, textvariable=host_var, font=("Arial", 11), width=40)
        host_entry.grid(row=0, column=1, sticky=tk.EW, padx=15, pady=8)
        
        # Port input
        tk.Label(new_frame, text="Port:", font=("Arial", 11), fg="white", bg=APP_BG_COLOR).grid(row=1, column=0, sticky=tk.W, padx=15, pady=8)
        port_var = tk.StringVar(value=str(self.db_manager.port))
        port_entry = tk.Entry(new_frame, textvariable=port_var, font=("Arial", 11), width=40)
        port_entry.grid(row=1, column=1, sticky=tk.EW, padx=15, pady=8)
        
        # User input
        tk.Label(new_frame, text="User:", font=("Arial", 11), fg="white", bg=APP_BG_COLOR).grid(row=2, column=0, sticky=tk.W, padx=15, pady=8)
        user_var = tk.StringVar(value=self.db_manager.user)
        user_entry = tk.Entry(new_frame, textvariable=user_var, font=("Arial", 11), width=40)
        user_entry.grid(row=2, column=1, sticky=tk.EW, padx=15, pady=8)
        
        # Password input
        tk.Label(new_frame, text="Password:", font=("Arial", 11), fg="white", bg=APP_BG_COLOR).grid(row=3, column=0, sticky=tk.W, padx=15, pady=8)
        password_var = tk.StringVar(value=self.db_manager.password)
        password_entry = tk.Entry(new_frame, textvariable=password_var, font=("Arial", 11), width=40, show="*")
        password_entry.grid(row=3, column=1, sticky=tk.EW, padx=15, pady=8)
        
        # Show/Hide password button
        show_password_var = tk.BooleanVar()
        def toggle_password():
            if show_password_var.get():
                password_entry.config(show="")
            else:
                password_entry.config(show="*")
        
        show_password_cb = tk.Checkbutton(new_frame, text="Show Password", variable=show_password_var,
                                         font=("Arial", 9), fg="white", bg=APP_BG_COLOR,
                                         selectcolor=APP_BG_COLOR, command=toggle_password)
        show_password_cb.grid(row=3, column=2, sticky=tk.W, padx=5, pady=8)
        
        # Database input
        tk.Label(new_frame, text="Database:", font=("Arial", 11), fg="white", bg=APP_BG_COLOR).grid(row=4, column=0, sticky=tk.W, padx=15, pady=8)
        database_var = tk.StringVar(value=self.db_manager.database)
        database_entry = tk.Entry(new_frame, textvariable=database_var, font=("Arial", 11), width=40)
        database_entry.grid(row=4, column=1, sticky=tk.EW, padx=15, pady=8)
        
        # Status display
        status_frame = tk.Frame(new_frame, bg=APP_BG_COLOR)
        status_frame.grid(row=5, column=0, columnspan=3, sticky=tk.EW, padx=15, pady=10)
        
        status_label = tk.Label(status_frame, text="Status: Ready", 
                               font=("Arial", 10), fg="#cccccc", bg=APP_BG_COLOR)
        status_label.pack(anchor=tk.W)
        
        # Buttons frame
        button_frame = tk.Frame(settings_window, bg=APP_BG_COLOR)
        button_frame.pack(pady=20)
        
        def update_status(message, color="#cccccc"):
            status_label.config(text=f"Status: {message}", fg=color)
            settings_window.update()
        
        def test_connection():
            """Test database connection with new settings"""
            try:
                update_status("Testing connection...", "#ffc107")
                test_db = DatabaseManager(
                    host=host_var.get().strip(),
                    user=user_var.get().strip(),
                    password=password_var.get(),
                    database=database_var.get().strip(),
                    port=int(port_var.get().strip())
                )
                if test_db.connect_without_db():
                    test_db.disconnect()
                    update_status("Connection successful!", "#28a745")
                    messagebox.showinfo("Success", "Database connection successful!")
                else:
                    update_status("Connection failed", "#dc3545")
                    messagebox.showerror("Error", "Failed to connect to database")
            except ValueError:
                update_status("Invalid port number", "#dc3545")
                messagebox.showerror("Error", "Port must be a valid number")
            except Exception as e:
                update_status("Connection failed", "#dc3545")
                messagebox.showerror("Error", f"Connection failed: {str(e)}")
        
        def apply_settings():
            """Apply new database settings"""
            try:
                update_status("Applying settings...", "#ffc107")
                
                # Validate inputs
                host = host_var.get().strip()
                user = user_var.get().strip()
                password = password_var.get()
                database = database_var.get().strip()
                port = int(port_var.get().strip())
                
                if not all([host, user, database]):
                    messagebox.showerror("Error", "Host, User, and Database fields are required")
                    update_status("Missing required fields", "#dc3545")
                    return
                
                # Update database manager
                self.db_manager.host = host
                self.db_manager.user = user
                self.db_manager.password = password
                self.db_manager.database = database
                self.db_manager.port = port
                
                # Test connection and initialize database
                update_status("Initializing database...", "#ffc107")
                if self.db_manager.check_and_create_database():
                    update_status("Settings applied successfully!", "#28a745")
                    messagebox.showinfo("Success", "Database settings updated and initialized!")
                    # Reload models and datasets
                    try:
                        self.load_models()
                        self.load_datasets()
                    except:
                        pass
                    settings_window.destroy()
                else:
                    update_status("Failed to initialize database", "#dc3545")
                    messagebox.showerror("Error", "Failed to initialize database with new settings")
            except ValueError:
                update_status("Invalid port number", "#dc3545")
                messagebox.showerror("Error", "Port must be a valid number")
            except Exception as e:
                update_status("Failed to apply settings", "#dc3545")
                messagebox.showerror("Error", f"Failed to apply settings: {str(e)}")
        
        tk.Button(button_frame, text="Test Connection", 
                 font=("Arial", 12, "bold"), bg="#28a745", fg="white", width=15,
                 command=test_connection).pack(side=tk.LEFT, padx=8)
        
        tk.Button(button_frame, text="Apply Settings", 
                 font=("Arial", 12, "bold"), bg="#007bff", fg="white", width=15,
                 command=apply_settings).pack(side=tk.LEFT, padx=8)
        
        tk.Button(button_frame, text="Cancel", 
                 font=("Arial", 12, "bold"), bg="#6c757d", fg="white", width=15,
                 command=settings_window.destroy).pack(side=tk.LEFT, padx=8)
    
    def clear_all(self):
        """Clear all uploaded images and reset interface"""
        self.uploaded_images = []
        self.image_info_var.set("No images loaded")
        self.status_var.set("Cleared all images")
        self.dataset_name_var.set("")
        self.dataset_save_location = ""
        self.location_var.set("datasets (default)")
        self.location_info_var.set("Dataset save location: datasets (default)")
        if hasattr(self, 'existing_dataset_var'):
            self.existing_dataset_var.set("")
        # Clear image previews
        self.update_image_previews()
    
    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.destroy()

# Import simpledialog for model name input
import tkinter.simpledialog

def main():
    """Main function to run the application"""
    app = YOLOLabelerApp()
    app.mainloop()

if __name__ == "__main__":
    main() 
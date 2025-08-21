"""
WelVision YOLO Data Labeller - Fixed Version
A comprehensive image labeling application using YOLO v8 models
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

# Configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '1469',
    'database': 'welvision_db',
    'port': 3306
}

APP_CONFIG = {
    'title': 'WelVision Data Labeller',
    'geometry': '1600x1000',
    'bg_color': '#0a2158',
    'version': 'v2.0'
}

FILE_CONFIG = {
    'supported_formats': [
        ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'),
        ('JPEG files', '*.jpg *.jpeg'),
        ('PNG files', '*.png'),
        ('BMP files', '*.bmp'),
        ('TIFF files', '*.tiff *.tif'),
        ('All files', '*.*')
    ],
    'dataset_base_path': 'datasets',
    'max_preview_size': (800, 600)
}

YOLO_CONFIG = {
    'confidence_threshold': 0.25,
    'iou_threshold': 0.45,
    'max_detections': 1000,
    'device': 'auto'
}

class DatabaseManager:
    """Handle all database operations for model and dataset management"""
    
    def __init__(self):
        self.host = DATABASE_CONFIG['host']
        self.user = DATABASE_CONFIG['user']
        self.password = DATABASE_CONFIG['password']
        self.database = DATABASE_CONFIG['database']
        self.port = DATABASE_CONFIG['port']
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
    
    def get_models(self):
        """Get all available models from database"""
        if not self.connect():
            return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM models ORDER BY name")
            models = cursor.fetchall()
            cursor.close()
            return models
        except mysql.connector.Error as e:
            print(f"Error fetching models: {e}")
            return []
        finally:
            self.disconnect()
    
    def add_model(self, name, path):
        """Add a new model to database"""
        if not self.connect():
            return False
        
        try:
            cursor = self.connection.cursor()
            query = "INSERT INTO models (name, path) VALUES (%s, %s) ON DUPLICATE KEY UPDATE path = VALUES(path)"
            cursor.execute(query, (name, path))
            self.connection.commit()
            cursor.close()
            return True
        except mysql.connector.Error as e:
            print(f"Error adding model: {e}")
            return False
        finally:
            self.disconnect()
    
    def get_datasets(self):
        """Get all available datasets from database"""
        if not self.connect():
            return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM datasets ORDER BY name")
            datasets = cursor.fetchall()
            cursor.close()
            return datasets
        except mysql.connector.Error as e:
            print(f"Error fetching datasets: {e}")
            return []
        finally:
            self.disconnect()
    
    def add_dataset(self, name, path, description=""):
        """Add a new dataset to database"""
        if not self.connect():
            return False
        
        try:
            cursor = self.connection.cursor()
            query = "INSERT INTO datasets (name, path, description) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE path = VALUES(path), description = VALUES(description)"
            cursor.execute(query, (name, path, description))
            self.connection.commit()
            cursor.close()
            return True
        except mysql.connector.Error as e:
            print(f"Error adding dataset: {e}")
            return False
        finally:
            self.disconnect()

class YOLOLabelerApp:
    """Main application class for YOLO Data Labeller"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(APP_CONFIG['title'])
        self.root.geometry(APP_CONFIG['geometry'])
        self.root.configure(bg=APP_CONFIG['bg_color'])
        
        # Initialize variables
        self.db_manager = DatabaseManager()
        self.current_model = None
        self.yolo_model = None
        self.uploaded_images = []
        self.current_image_index = 0
        self.current_dataset_path = None
        
        # Device detection
        self.detect_optimal_device()
        
        # Status variable
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        # Create GUI
        self.create_widgets()
        
        # Try to load data from database
        try:
            self.load_models()
            self.load_datasets()
        except Exception as e:
            self.status_var.set(f"Database connection failed: {str(e)}")
            messagebox.showwarning("Database Connection", 
                                 f"Could not connect to database. Please check your MySQL connection.\nError: {str(e)}")
    
    def detect_optimal_device(self):
        """Detect the best available device for YOLO inference"""
        if torch.cuda.is_available():
            self.optimal_device = 'cuda'
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.optimal_device = 'mps'
            print("Apple Silicon GPU (MPS) available")
        else:
            self.optimal_device = 'cpu'
            print("Using CPU for inference")
    
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        # Create main frame with scrollbar
        self.main_canvas = tk.Canvas(self.root, bg=APP_CONFIG['bg_color'])
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.main_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Model Selection Section
        self.create_model_section()
        
        # Dataset Section
        self.create_dataset_section()
        
        # Image Upload Section
        self.create_upload_section()
        
        # Image Display Section
        self.create_display_section()
        
        # Action Buttons Section
        self.create_action_section()
        
        # Status Bar
        self.create_status_bar()
        
        # Bind mousewheel to canvas
        self.main_canvas.bind("<MouseWheel>", self._on_mousewheel)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def create_model_section(self):
        """Create model selection and loading section"""
        model_frame = ttk.LabelFrame(self.scrollable_frame, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Model dropdown
        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.model_dropdown = ttk.Combobox(model_frame, state="readonly", width=50)
        self.model_dropdown.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        
        # Load model button
        self.load_model_btn = ttk.Button(model_frame, text="Load Model", command=self.load_selected_model)
        self.load_model_btn.grid(row=0, column=2, padx=(10, 0))
        
        # Add new model section
        ttk.Separator(model_frame, orient='horizontal').grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(model_frame, text="Add New Model:").grid(row=2, column=0, sticky="w", padx=(0, 10))
        
        self.new_model_name = tk.StringVar()
        self.new_model_path = tk.StringVar()
        
        ttk.Label(model_frame, text="Model Name:").grid(row=3, column=0, sticky="w", padx=(0, 10))
        ttk.Entry(model_frame, textvariable=self.new_model_name, width=30).grid(row=3, column=1, sticky="w", padx=(0, 10))
        
        ttk.Label(model_frame, text="Model Path:").grid(row=4, column=0, sticky="w", padx=(0, 10))
        ttk.Entry(model_frame, textvariable=self.new_model_path, width=40).grid(row=4, column=1, sticky="ew", padx=(0, 10))
        ttk.Button(model_frame, text="Browse", command=self.browse_model_file).grid(row=4, column=2, padx=(10, 0))
        
        ttk.Button(model_frame, text="Add Model", command=self.add_new_model).grid(row=5, column=1, pady=10)
        
        model_frame.columnconfigure(1, weight=1)
    
    def create_dataset_section(self):
        """Create dataset selection and creation section"""
        dataset_frame = ttk.LabelFrame(self.scrollable_frame, text="Dataset Management", padding="10")
        dataset_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # New dataset section
        ttk.Label(dataset_frame, text="Create New Dataset:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        self.new_dataset_name = tk.StringVar()
        ttk.Label(dataset_frame, text="Dataset Name:").grid(row=1, column=0, sticky="w", padx=(0, 10))
        ttk.Entry(dataset_frame, textvariable=self.new_dataset_name, width=30).grid(row=1, column=1, sticky="w", padx=(0, 10))
        ttk.Button(dataset_frame, text="Create Dataset", command=self.create_new_dataset).grid(row=1, column=2, padx=(10, 0))
        
        # Existing dataset section
        ttk.Separator(dataset_frame, orient='horizontal').grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(dataset_frame, text="Select Existing Dataset:").grid(row=3, column=0, sticky="w", padx=(0, 10))
        
        self.dataset_dropdown = ttk.Combobox(dataset_frame, state="readonly", width=50)
        self.dataset_dropdown.grid(row=3, column=1, sticky="ew", padx=(0, 10))
        ttk.Button(dataset_frame, text="Select Dataset", command=self.select_existing_dataset).grid(row=3, column=2, padx=(10, 0))
        
        dataset_frame.columnconfigure(1, weight=1)
    
    def create_upload_section(self):
        """Create image upload section"""
        upload_frame = ttk.LabelFrame(self.scrollable_frame, text="Image Upload", padding="10")
        upload_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(upload_frame, text="Upload Images", command=self.upload_images).pack(side=tk.LEFT, padx=10)
        ttk.Button(upload_frame, text="Upload Folder", command=self.upload_folder).pack(side=tk.LEFT, padx=10)
        
        self.image_count_label = ttk.Label(upload_frame, text="No images loaded")
        self.image_count_label.pack(side=tk.RIGHT, padx=10)
    
    def create_display_section(self):
        """Create image display section"""
        display_frame = ttk.LabelFrame(self.scrollable_frame, text="Image Preview", padding="10")
        display_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Image display canvas
        self.image_canvas = tk.Canvas(display_frame, width=800, height=600, bg='white')
        self.image_canvas.pack(pady=10)
        
        # Navigation buttons
        nav_frame = ttk.Frame(display_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        self.prev_btn = ttk.Button(nav_frame, text="← Previous", command=self.show_previous_image, state="disabled")
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.image_info_label = ttk.Label(nav_frame, text="No image loaded")
        self.image_info_label.pack(side=tk.LEFT, expand=True)
        
        self.next_btn = ttk.Button(nav_frame, text="Next →", command=self.show_next_image, state="disabled")
        self.next_btn.pack(side=tk.RIGHT, padx=5)
    
    def create_action_section(self):
        """Create action buttons section"""
        action_frame = ttk.LabelFrame(self.scrollable_frame, text="Actions", padding="10")
        action_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.label_btn = ttk.Button(action_frame, text="Label All Images", command=self.label_all_images, state="disabled")
        self.label_btn.pack(side=tk.LEFT, padx=10)
        
        self.label_current_btn = ttk.Button(action_frame, text="Label Current Image", command=self.label_current_image, state="disabled")
        self.label_current_btn.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(action_frame, text="Clear All", command=self.clear_all).pack(side=tk.RIGHT, padx=10)
    
    def create_status_bar(self):
        """Create status bar"""
        status_frame = ttk.Frame(self.scrollable_frame)
        status_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(5, 0))
    
    def load_models(self):
        """Load available models from database"""
        try:
            models = self.db_manager.get_models()
            model_names = [model['name'] for model in models]
            
            if model_names:
                self.model_dropdown['values'] = model_names
                self.model_dropdown.set(model_names[0])
                self.status_var.set(f"Loaded {len(model_names)} models from database")
            else:
                self.status_var.set("No models found in database. Add models using 'Add New Model' section.")
        except Exception as e:
            self.model_dropdown['values'] = []
            raise e
    
    def load_datasets(self):
        """Load available datasets from database"""
        try:
            datasets = self.db_manager.get_datasets()
            dataset_names = [f"{dataset['name']}" for dataset in datasets]
            
            if dataset_names:
                self.dataset_dropdown['values'] = dataset_names
                self.status_var.set(f"Loaded {len(dataset_names)} datasets from database")
        except Exception as e:
            self.dataset_dropdown['values'] = []
            print(f"Error loading datasets: {e}")
    
    def browse_model_file(self):
        """Browse for model file"""
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model File",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            self.new_model_path.set(file_path)
    
    def add_new_model(self):
        """Add new model to database"""
        name = self.new_model_name.get().strip()
        path = self.new_model_path.get().strip()
        
        if not name or not path:
            messagebox.showerror("Error", "Please provide both model name and path")
            return
        
        if not os.path.exists(path):
            messagebox.showerror("Error", "Model file does not exist")
            return
        
        try:
            if self.db_manager.add_model(name, path):
                messagebox.showinfo("Success", "Model added successfully")
                self.new_model_name.set("")
                self.new_model_path.set("")
                self.load_models()
            else:
                messagebox.showerror("Error", "Failed to add model to database")
        except Exception as e:
            messagebox.showerror("Error", f"Error adding model: {str(e)}")
    
    def load_selected_model(self):
        """Load the selected YOLO model"""
        selected_model_name = self.model_dropdown.get()
        if not selected_model_name:
            messagebox.showerror("Error", "Please select a model")
            return
        
        try:
            models = self.db_manager.get_models()
            model_info = next((m for m in models if m['name'] == selected_model_name), None)
            
            if not model_info:
                messagebox.showerror("Error", "Selected model not found in database")
                return
            
            model_path = model_info['path']
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return
            
            self.status_var.set(f"Loading YOLO model on {self.optimal_device.upper()}...")
            self.root.update()
            
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
                
                # Enable labeling buttons if images are loaded
                if self.uploaded_images:
                    self.label_btn.config(state="normal")
                    self.label_current_btn.config(state="normal")
                
            except Exception as device_error:
                # Fallback to CPU if device fails
                print(f"Device error: {device_error}, falling back to CPU")
                self.yolo_model.to('cpu')
                self.optimal_device = 'cpu'
                self.current_model = model_info
                self.status_var.set(f"Model loaded on CPU (GPU fallback): {selected_model_name}")
                messagebox.showinfo("Success", f"Model '{selected_model_name}' loaded successfully!\nDevice: CPU (GPU fallback)")
                
                # Enable labeling buttons if images are loaded
                if self.uploaded_images:
                    self.label_btn.config(state="normal")
                    self.label_current_btn.config(state="normal")
            
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Failed to load model: {error_msg}")
            self.status_var.set("Failed to load model")
    
    def create_new_dataset(self):
        """Create a new dataset"""
        name = self.new_dataset_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Please provide a dataset name")
            return
        
        # Create dataset directory structure
        dataset_path = os.path.join(FILE_CONFIG['dataset_base_path'], name)
        images_path = os.path.join(dataset_path, 'images')
        labels_path = os.path.join(dataset_path, 'labels')
        
        try:
            os.makedirs(images_path, exist_ok=True)
            os.makedirs(labels_path, exist_ok=True)
            
            # Add to database
            if self.db_manager.add_dataset(name, dataset_path):
                self.current_dataset_path = dataset_path
                messagebox.showinfo("Success", f"Dataset '{name}' created successfully")
                self.new_dataset_name.set("")
                self.load_datasets()
                self.dataset_dropdown.set(name)
            else:
                messagebox.showerror("Error", "Failed to add dataset to database")
        except Exception as e:
            messagebox.showerror("Error", f"Error creating dataset: {str(e)}")
    
    def select_existing_dataset(self):
        """Select an existing dataset"""
        selected_dataset = self.dataset_dropdown.get()
        if not selected_dataset:
            messagebox.showerror("Error", "Please select a dataset")
            return
        
        try:
            datasets = self.db_manager.get_datasets()
            dataset_info = next((d for d in datasets if d['name'] == selected_dataset), None)
            
            if dataset_info:
                self.current_dataset_path = dataset_info['path']
                self.status_var.set(f"Selected dataset: {selected_dataset}")
                messagebox.showinfo("Success", f"Dataset '{selected_dataset}' selected")
            else:
                messagebox.showerror("Error", "Dataset not found")
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting dataset: {str(e)}")
    
    def upload_images(self):
        """Upload individual images for labeling"""
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=FILE_CONFIG['supported_formats']
        )
        
        if file_paths:
            self.uploaded_images = list(file_paths)
            self.current_image_index = 0
            self.update_image_display()
            self.update_navigation_buttons()
            
            # Enable labeling buttons if model is loaded
            if self.yolo_model:
                self.label_btn.config(state="normal")
                self.label_current_btn.config(state="normal")
    
    def upload_folder(self):
        """Upload all images from a folder"""
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        
        if folder_path:
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
            image_files = []
            
            for file in os.listdir(folder_path):
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(folder_path, file))
            
            if image_files:
                self.uploaded_images = image_files
                self.current_image_index = 0
                self.update_image_display()
                self.update_navigation_buttons()
                
                # Enable labeling buttons if model is loaded
                if self.yolo_model:
                    self.label_btn.config(state="normal")
                    self.label_current_btn.config(state="normal")
            else:
                messagebox.showinfo("No Images", "No supported image files found in the selected folder")
    
    def update_image_display(self):
        """Update the image display"""
        if not self.uploaded_images:
            self.image_count_label.config(text="No images loaded")
            self.image_info_label.config(text="No image loaded")
            return
        
        current_image_path = self.uploaded_images[self.current_image_index]
        
        try:
            # Load and resize image for display
            image = Image.open(current_image_path)
            
            # Calculate display size while maintaining aspect ratio
            canvas_width, canvas_height = 800, 600
            img_width, img_height = image.size
            
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and display
            self.photo = ImageTk.PhotoImage(image)
            self.image_canvas.delete("all")
            
            # Center the image on canvas
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
            
            # Update labels
            self.image_count_label.config(text=f"{len(self.uploaded_images)} images loaded")
            self.image_info_label.config(text=f"Image {self.current_image_index + 1} of {len(self.uploaded_images)}: {os.path.basename(current_image_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def update_navigation_buttons(self):
        """Update navigation button states"""
        if not self.uploaded_images:
            self.prev_btn.config(state="disabled")
            self.next_btn.config(state="disabled")
            return
        
        self.prev_btn.config(state="normal" if self.current_image_index > 0 else "disabled")
        self.next_btn.config(state="normal" if self.current_image_index < len(self.uploaded_images) - 1 else "disabled")
    
    def show_previous_image(self):
        """Show previous image"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image_display()
            self.update_navigation_buttons()
    
    def show_next_image(self):
        """Show next image"""
        if self.current_image_index < len(self.uploaded_images) - 1:
            self.current_image_index += 1
            self.update_image_display()
            self.update_navigation_buttons()
    
    def label_current_image(self):
        """Label the currently displayed image"""
        if not self.yolo_model:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if not self.uploaded_images:
            messagebox.showerror("Error", "Please upload images first")
            return
        
        if not self.current_dataset_path:
            messagebox.showerror("Error", "Please create or select a dataset first")
            return
        
        current_image_path = self.uploaded_images[self.current_image_index]
        self.process_single_image(current_image_path)
    
    def label_all_images(self):
        """Label all uploaded images"""
        if not self.yolo_model:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if not self.uploaded_images:
            messagebox.showerror("Error", "Please upload images first")
            return
        
        if not self.current_dataset_path:
            messagebox.showerror("Error", "Please create or select a dataset first")
            return
        
        # Process all images
        processed_count = 0
        total_images = len(self.uploaded_images)
        
        for i, image_path in enumerate(self.uploaded_images):
            try:
                self.status_var.set(f"Processing image {i + 1} of {total_images}...")
                self.root.update()
                
                self.process_single_image(image_path)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        self.status_var.set(f"Completed: {processed_count} of {total_images} images processed")
        messagebox.showinfo("Complete", f"Labeling completed!\nProcessed: {processed_count} of {total_images} images")
    
    def process_single_image(self, image_path):
        """Process a single image with YOLO and save results"""
        try:
            # Run YOLO inference
            results = self.yolo_model(image_path, conf=YOLO_CONFIG['confidence_threshold'], iou=YOLO_CONFIG['iou_threshold'])
            
            # Get image dimensions
            image = Image.open(image_path)
            img_width, img_height = image.size
            
            # Prepare YOLO format labels
            yolo_labels = []
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        # Get class and coordinates
                        cls = int(box.cls[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Convert to YOLO format (normalized xywh)
                        x_center = (x1 + x2) / 2 / img_width
                        y_center = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        yolo_labels.append([cls, x_center, y_center, width, height])
            
            # Save image and labels to dataset
            image_name = os.path.basename(image_path)
            dataset_images_path = os.path.join(self.current_dataset_path, 'images')
            dataset_labels_path = os.path.join(self.current_dataset_path, 'labels')
            
            # Copy image to dataset
            dest_image_path = os.path.join(dataset_images_path, image_name)
            shutil.copy2(image_path, dest_image_path)
            
            # Save labels
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(dataset_labels_path, label_name)
            
            with open(label_path, 'w') as f:
                for label in yolo_labels:
                    f.write(' '.join(map(str, label)) + '\n')
            
            print(f"Processed: {image_name} - {len(yolo_labels)} detections")
            
        except Exception as e:
            raise Exception(f"Error processing image {image_path}: {str(e)}")
    
    def clear_all(self):
        """Clear all loaded data"""
        self.uploaded_images = []
        self.current_image_index = 0
        self.image_canvas.delete("all")
        self.image_count_label.config(text="No images loaded")
        self.image_info_label.config(text="No image loaded")
        self.update_navigation_buttons()
        self.label_btn.config(state="disabled")
        self.label_current_btn.config(state="disabled")
        self.status_var.set("Cleared all data")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = YOLOLabelerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

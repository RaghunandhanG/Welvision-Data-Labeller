"""
WelVision YOLO Data Labeller
A comprehensive image labeling application using YOLO v8 models
Enhanced version with scrollable interface, dataset management, and folder upload
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import os
import shutil
import json
from datetime import datetime
import mysql.connector
from ultralytics import YOLO
import threading
import queue
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
        
        # Temporary annotation storage for labeling sessions
        self.temp_annotations = {}  # Store annotations temporarily until dataset creation
        self.is_labeling_session_active = False  # Track if labeling session is active
        self.temp_session_name = ""  # Name for the current temporary session
        
        # Initialize status tracking variables
        self.image_info_var = None  # Will be initialized in create_upload_section
        
        # Threading attributes for Roboflow uploads
        self.upload_queue = queue.Queue()
        self.upload_thread = None
        self.upload_in_progress = False
        
        # Default canvas dimensions (will be updated in create_image_preview_panel)
        self.split_canvas_width = 600  # Increased from 400 to 600
        self.split_canvas_height = 450
        self.single_canvas_width = 1000  # Increased from 800 to 1000
        self.single_canvas_height = 500

        
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
            # Clean up deleted datasets first
            self.cleanup_deleted_datasets()
            
            self.load_models()
            self.load_datasets()
        except Exception as e:
            self.status_var.set(f"Database connection failed: {str(e)}")
            messagebox.showwarning("Database Connection", 
                                 "Could not connect to database. Please run setup_database.py first.\n\n"
                                 "You can still use the app to load models manually.")
        
        # Bind window close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start periodic device status updates (every 10 seconds)
        self.schedule_device_status_update()
    
    def detect_optimal_device(self):
        """Detect the best available device for YOLO inference with enhanced GPU detection"""
        print("🔍 Detecting optimal device for YOLO inference...")
        
        try:
            # Check for NVIDIA CUDA GPU
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                
                # Get GPU memory info
                try:
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    memory_gb = total_memory / (1024**3)
                    memory_info = f"{memory_gb:.1f}GB"
                except:
                    memory_info = "Unknown"
                
                print(f"🎮 NVIDIA GPU detected: {gpu_name}")
                print(f"💾 GPU Memory: {memory_info}")
                print(f"🔢 Available GPUs: {gpu_count}")
                
                # Test CUDA compatibility with comprehensive testing
                try:
                    print("🧪 Testing CUDA compatibility...")
                    
                    # Test basic CUDA operations
                    test_tensor = torch.randn(1, 3, 640, 640, device='cuda')
                    
                    # Test CUDA tensor operations
                    test_boxes = torch.tensor([[10, 10, 50, 50]], device='cuda', dtype=torch.float32)
                    test_scores = torch.tensor([0.9], device='cuda', dtype=torch.float32)
                    
                    # Test torchvision NMS operations (common failure point)
                    import torchvision
                    _ = torchvision.ops.nms(test_boxes, test_scores, iou_threshold=0.5)
                    
                    # Test memory allocation and cleanup
                    torch.cuda.empty_cache()
                    
                    print("✅ CUDA compatibility test passed")
                    device = 'cuda'
                    
                    if hasattr(self, 'status_var'):
                        self.status_var.set(f"🎮 GPU Ready: {gpu_name} ({memory_info})")
                    
                    return device
                    
                except Exception as cuda_test_error:
                    print(f"❌ CUDA compatibility test failed: {cuda_test_error}")
                    print("⚠️ Falling back to CPU due to CUDA incompatibility")
                    device = 'cpu'
                    
                    if hasattr(self, 'status_var'):
                        self.status_var.set("⚠️ GPU detected but incompatible - Using CPU")
                    
                    return device
                    
            # Check for Apple Silicon MPS
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("🍎 Apple Silicon GPU detected")
                
                # Test MPS compatibility
                try:
                    print("🧪 Testing MPS compatibility...")
                    test_tensor = torch.randn(100, 100).to('mps')
                    result = test_tensor.matmul(test_tensor)
                    _ = result.cpu()
                    
                    print("✅ MPS compatibility test passed")
                    device = 'mps'
                    
                    if hasattr(self, 'status_var'):
                        self.status_var.set("🍎 Apple Silicon GPU ready - MPS acceleration")
                    
                    return device
                    
                except Exception as mps_error:
                    print(f"❌ MPS compatibility test failed: {mps_error}")
                    print("⚠️ Falling back to CPU")
                    device = 'cpu'
                    
                    if hasattr(self, 'status_var'):
                        self.status_var.set("⚠️ Apple GPU detected but incompatible - Using CPU")
                    
                    return device
            
            # Default to CPU
            else:
                print("💻 No GPU detected - Using CPU processing")
                device = 'cpu'
                
                if hasattr(self, 'status_var'):
                    self.status_var.set("💻 CPU processing mode")
                
                return device
                
        except Exception as e:
            print(f"❌ Device detection error: {e}")
            print("🔄 Defaulting to CPU for safety")
            
            if hasattr(self, 'status_var'):
                self.status_var.set("⚠️ Device detection failed - Using CPU")
            
            return 'cpu'
    
    def test_cuda_inference(self):
        """Test if CUDA inference will work with the loaded model"""
        try:
            if not hasattr(self, 'yolo_model') or self.yolo_model is None:
                return False
                
            # Create a small test image
            import tempfile
            test_image_path = tempfile.mktemp(suffix='.jpg')
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(test_image_path, test_image)
            
            try:
                # Test inference on CUDA
                _ = self.yolo_model(test_image_path, device='cuda', verbose=False)
                os.unlink(test_image_path)  # Clean up test image
                return True
                
            except Exception as inference_error:
                print(f"CUDA inference test failed: {inference_error}")
                if os.path.exists(test_image_path):
                    os.unlink(test_image_path)  # Clean up test image
                return False
                
        except Exception as test_error:
            print(f"CUDA test setup failed: {test_error}")
            return False
    
    def safe_yolo_inference(self, image_path, conf=None, max_det=None):
        """Perform YOLO inference with enhanced GPU/CPU fallback and device monitoring"""
        if conf is None:
            conf = YOLO_CONFIG['confidence_threshold']
        if max_det is None:
            max_det = YOLO_CONFIG['max_detections']
        
        # Track inference attempts for better logging
        device_used = self.optimal_device
        
        try:
            print(f"🔍 Starting inference on {device_used.upper()}: {os.path.basename(image_path)}")
            
            # First attempt with optimal device
            results = self.yolo_model(
                image_path,
                conf=conf,
                max_det=max_det,
                device=self.optimal_device,
                verbose=False
            )
            
            print(f"✅ Inference successful on {device_used.upper()}")
            return results
            
        except Exception as e:
            # Check if it's a device compatibility issue
            error_str = str(e).lower()
            is_device_error = any(keyword in error_str for keyword in [
                'cuda', 'torchvision::nms', 'backend', 'mps', 'device', 
                'gpu', 'memory', 'out of memory', 'runtime error'
            ])
            
            if is_device_error and self.optimal_device != 'cpu':
                print(f"⚠️ {device_used.upper()} inference failed: {e}")
                print("🔄 Attempting automatic fallback to CPU...")
                
                try:
                    # Force CPU inference
                    results = self.yolo_model(
                        image_path,
                        conf=conf,
                        max_det=max_det,
                        device="cpu",
                        verbose=False
                    )
                    
                    print("✅ CPU inference successful")
                    
                    # Update optimal device to CPU to avoid future issues
                    if self.optimal_device != 'cpu':
                        print("📝 Updating optimal device to CPU due to compatibility issues")
                        self.optimal_device = 'cpu'
                        
                        # Move model to CPU for future inferences
                        try:
                            self.yolo_model.to("cpu")
                            print("🔧 Model moved to CPU")
                            
                            # Update device status display
                            self.update_device_status()
                            
                            # Update UI status if available
                            if hasattr(self, 'model_status_var'):
                                current_status = self.model_status_var.get()
                                if "✅" in current_status:
                                    model_name = current_status.split("✅")[1].split(" - ")[0].strip()
                                    self.model_status_var.set(f"✅ {model_name} - CPU (Auto-fallback)")
                                    
                        except Exception as move_error:
                            print(f"⚠️ Warning: Could not move model to CPU: {move_error}")
                    
                    return results
                    
                except Exception as cpu_error:
                    error_msg = f"Inference failed on both {device_used.upper()} and CPU.\n{device_used.upper()} Error: {e}\nCPU Error: {cpu_error}"
                    print(f"❌ {error_msg}")
                    raise Exception(error_msg)
            else:
                # Re-raise if it's not a device compatibility issue
                print(f"❌ Inference failed with non-device error: {e}")
                raise e
    
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
        
        # Create notebook for tabs with enhanced styling
        from tkinter import ttk
        
        # Use default ttk styles for normal appearance
        self.style = ttk.Style()
        
        # Use default theme for standard appearance
        self.style.theme_use('default')
        
        # Create the notebook with default styling
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=0, pady=15)
        
        # Tab 1: Image Labeling (current functionality)
        self.labeling_frame = tk.Frame(self.notebook, bg=APP_BG_COLOR)
        self.notebook.add(self.labeling_frame, text="Image Labeling")
        
        # Tab 2: Dataset Preview
        self.preview_frame = tk.Frame(self.notebook, bg=APP_BG_COLOR)
        self.notebook.add(self.preview_frame, text="Dataset Preview")
        
        # Tab 3: Roboflow Management
        self.roboflow_frame = tk.Frame(self.notebook, bg=APP_BG_COLOR)
        self.notebook.add(self.roboflow_frame, text="Roboflow Projects")
        
        # Setup labeling tab (original functionality)
        self.setup_labeling_tab()
        
        # Setup dataset preview tab (new functionality)
        self.setup_dataset_preview_tab()
        
        # Setup roboflow upload tab
        self.setup_roboflow_upload_tab()
    
    def setup_labeling_tab(self):
        """Setup the image labeling tab with optimized three-column layout"""
        # Main content container with three columns
        main_content = tk.Frame(self.labeling_frame, bg=APP_BG_COLOR)
        main_content.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Left panel for controls (30% width - model & dataset)
        self.left_panel = tk.Frame(main_content, bg=APP_BG_COLOR, width=350)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        self.left_panel.pack_propagate(False)  # Maintain fixed width
        
        # Middle panel for controls (25% width - upload & actions)
        self.middle_panel = tk.Frame(main_content, bg=APP_BG_COLOR, width=300)
        self.middle_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.middle_panel.pack_propagate(False)  # Maintain fixed width
        
        # Right panel for image preview (45% width - increased for better image display)
        self.right_panel = tk.Frame(main_content, bg=APP_BG_COLOR)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create sections in left panel (Model & Dataset)
        self.create_model_section(self.left_panel)
        self.create_dataset_section(self.left_panel)
        
        # Create sections in middle panel (Upload & Actions)
        self.create_upload_section(self.middle_panel)
        self.create_action_section(self.middle_panel)
        self.create_status_panel(self.middle_panel)
        
        # Create image preview in right panel
        self.create_image_preview_panel(self.right_panel)
    
    def setup_dataset_preview_tab(self):
        """Setup the dataset preview tab for viewing labeled images with annotations"""
        # Top section for dataset selection
        selection_frame = tk.LabelFrame(self.preview_frame, text="📂 Dataset Selection", 
                                       font=("Arial", 16, "bold"), 
                                       fg="white", bg=APP_BG_COLOR, 
                                       relief="raised", bd=3)
        selection_frame.pack(fill=tk.X, padx=0, pady=20)
        
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
                 command=self.refresh_all_datasets).pack(side=tk.LEFT, padx=5)
        
        tk.Button(controls_frame, text="📁 Browse Dataset Folder", 
                 font=("Arial", 12, "bold"), bg="#28a745", fg="white",
                 command=self.browse_dataset_folder).pack(side=tk.LEFT, padx=5)
        
        # Dataset info section
        info_frame = tk.Frame(selection_frame, bg=APP_BG_COLOR)
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        self.dataset_info_var = tk.StringVar(value="No dataset selected")
        tk.Label(info_frame, textvariable=self.dataset_info_var, 
                font=("Arial", 11), fg="#ffc107", bg=APP_BG_COLOR).pack(side=tk.LEFT)
        
        # Single image preview section with annotations
        preview_main_frame = tk.LabelFrame(self.preview_frame, text="🖼️ Image Viewer with Annotations", 
                                          font=("Arial", 16, "bold"), 
                                          fg="white", bg=APP_BG_COLOR, 
                                          relief="raised", bd=3)
        preview_main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 20))
        
        # Navigation controls at the top
        nav_frame = tk.Frame(preview_main_frame, bg=APP_BG_COLOR)
        nav_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # Previous button
        self.prev_btn = tk.Button(nav_frame, text="⬅️ Previous Image", 
                                 font=("Arial", 14, "bold"), bg="#007bff", fg="white",
                                 command=self.prev_preview_image, state="disabled",
                                 padx=20, pady=8)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        # Next button
        self.next_btn = tk.Button(nav_frame, text="Next Image ➡️", 
                                 font=("Arial", 14, "bold"), bg="#007bff", fg="white",
                                 command=self.next_preview_image, state="disabled",
                                 padx=20, pady=8)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # Image counter in the center
        self.image_counter_var = tk.StringVar(value="No images loaded")
        counter_label = tk.Label(nav_frame, textvariable=self.image_counter_var, 
                               font=("Arial", 14, "bold"), fg="#ffc107", bg=APP_BG_COLOR)
        counter_label.pack(side=tk.LEFT, expand=True)
        
        # Toggle annotations button
        self.toggle_annotations_btn = tk.Button(nav_frame, text="� Toggle Annotations", 
                                               font=("Arial", 12, "bold"), bg="#28a745", fg="white",
                                               command=self.toggle_annotations_visibility, state="disabled",
                                               padx=15, pady=6)
        self.toggle_annotations_btn.pack(side=tk.RIGHT, padx=5)
        
        # Image display area
        image_display_frame = tk.Frame(preview_main_frame, bg="#1a1a1a", relief=tk.SUNKEN, bd=2)
        image_display_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        # Large canvas for image with annotations
        self.preview_canvas = tk.Canvas(image_display_frame, bg="#1a1a1a", highlightthickness=0)
        
        # Scrollbars for large images
        h_scrollbar = ttk.Scrollbar(image_display_frame, orient=tk.HORIZONTAL, command=self.preview_canvas.xview)
        v_scrollbar = ttk.Scrollbar(image_display_frame, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        
        self.preview_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Pack scrollbars and canvas
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Image info panel at the bottom
        info_panel = tk.Frame(preview_main_frame, bg=APP_BG_COLOR)
        info_panel.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        self.current_image_info_var = tk.StringVar(value="Load a dataset to view images with annotations")
        tk.Label(info_panel, textvariable=self.current_image_info_var, 
                font=("Arial", 11), fg="#ffffff", bg=APP_BG_COLOR, 
                justify=tk.LEFT, wraplength=800).pack(side=tk.LEFT)
        
        # Keyboard bindings for navigation
        self.bind('<Left>', lambda e: self.prev_preview_image())
        self.bind('<Right>', lambda e: self.next_preview_image())
        self.bind('<space>', lambda e: self.toggle_annotations_visibility())
        
        # Focus the window to enable keyboard navigation
        self.focus_set()
        
        # Bind canvas events
        self.preview_canvas.bind('<Configure>', self.on_preview_canvas_configure)
        
        # Mouse wheel binding for preview canvas
        def _on_preview_mousewheel(event):
            self.preview_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.preview_canvas.bind_all("<MouseWheel>", _on_preview_mousewheel)
        
        # Initialize dataset preview variables
        self.dataset_images = []
        self.dataset_annotations = []  # Store COCO annotations for each image
        self.current_preview_index = 0
        self.current_preview_image = None
        self.show_annotations = True  # Toggle for showing/hiding annotations
        
        # Load datasets after UI is created
        self.after(100, self.refresh_preview_datasets)  # Delay to ensure UI is ready
    
    def on_preview_canvas_configure(self, event):
        """Handle preview canvas resize - refresh current image display"""
        if hasattr(self, 'dataset_images') and self.dataset_images:
            # Refresh the current image display with new canvas size
            self.after_idle(self.show_single_image_with_annotations)
    
    def refresh_all_datasets(self):
        """Refresh all dataset dropdowns across the application"""
        try:
            print("Refreshing all dataset dropdowns...")
            # Clean up deleted datasets first
            self.cleanup_deleted_datasets()
            
            self.refresh_preview_datasets()
            self.refresh_rf_datasets()
            print("All dataset dropdowns refreshed and cleaned up successfully")
        except Exception as e:
            print(f"Error refreshing datasets: {e}")
    
    def refresh_preview_datasets(self):
        """Refresh the dataset dropdown for preview - show only datasets with COCO format"""
        try:
            # Check if the preview dropdown exists yet
            if not hasattr(self, 'preview_dataset_dropdown'):
                print("Preview dropdown not created yet, skipping refresh")
                return
                
            datasets = []
            
            # Get datasets from database
            db_datasets = self.db_manager.get_datasets()
            for ds in db_datasets:
                # First check if dataset folder actually exists
                if not os.path.exists(ds['path']):
                    continue  # Skip datasets with non-existent paths
                    
                # Check if dataset has COCO JSON file
                coco_file = os.path.join(ds['path'], "annotations.json")
                if os.path.exists(coco_file):
                    try:
                        import json
                        with open(coco_file, 'r') as f:
                            coco_data = json.load(f)
                        image_count = len(coco_data.get('images', []))
                        ann_count = len(coco_data.get('annotations', []))
                        datasets.append(f"{ds['name']} (COCO)")
                    except:
                        # Skip datasets with invalid COCO files
                        continue
                # Skip datasets without COCO format
            
            # Also scan default datasets folder
            default_datasets_path = "datasets"
            if os.path.exists(default_datasets_path):
                for folder_name in os.listdir(default_datasets_path):
                    folder_path = os.path.join(default_datasets_path, folder_name)
                    if os.path.isdir(folder_path):
                        # Skip if already added from database
                        if any(item.startswith(folder_name + " ") for item in datasets):
                            continue
                            
                        coco_file = os.path.join(folder_path, "annotations.json")
                        if os.path.exists(coco_file):
                            try:
                                import json
                                with open(coco_file, 'r') as f:
                                    coco_data = json.load(f)
                                image_count = len(coco_data.get('images', []))
                                ann_count = len(coco_data.get('annotations', []))
                                datasets.append(f"{folder_name} (COCO)")
                            except:
                                # Skip datasets with invalid COCO files
                                continue
                        # Skip datasets without COCO format
            
            print(f"Found {len(datasets)} datasets with COCO format")
            
            self.preview_dataset_dropdown['values'] = datasets
            
            if datasets:
                self.preview_dataset_dropdown.set(datasets[0])
                self.dataset_info_var.set(f"📊 {len(datasets)} COCO dataset(s) found")
            else:
                self.dataset_info_var.set("📊 No COCO datasets found")
                
        except Exception as e:
            print(f"Error refreshing preview datasets: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'dataset_info_var'):
                self.dataset_info_var.set("Error loading datasets")
    
    def log_rf_status(self, message):
        """Log status message to Roboflow log area"""
        try:
            if hasattr(self, 'rf_status_text'):
                self.rf_status_text.config(state=tk.NORMAL)
                self.rf_status_text.insert(tk.END, f"{message}\n")
                self.rf_status_text.see(tk.END)
                self.rf_status_text.config(state=tk.DISABLED)
                self.update()
        except Exception as e:
            print(f"Log error: {e}")
    
    def setup_roboflow_upload_tab(self):
        """Setup the Roboflow upload tab with step-by-step workflow"""
        # Main container with scrolling
        main_frame = tk.Frame(self.roboflow_frame, bg=APP_BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="🚀 Roboflow Project Management & Upload", 
                              font=("Arial", 18, "bold"), fg="white", bg=APP_BG_COLOR)
        title_label.pack(pady=(0, 20))
        
        # Step 1: API Key Configuration
        step1_frame = tk.LabelFrame(main_frame, text="Step 1: 🔑 API Key", 
                                   font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR,
                                   relief=tk.RAISED, bd=2)
        step1_frame.pack(fill="x", pady=(0, 15))
        
        api_container = tk.Frame(step1_frame, bg=APP_BG_COLOR)
        api_container.pack(fill="x", padx=10, pady=10)
        
        tk.Label(api_container, text="API Key:", font=("Arial", 10, "bold"), 
                fg="white", bg=APP_BG_COLOR, width=12, anchor="w").pack(side="left")
        self.rf_api_key_var = tk.StringVar()
        self.rf_api_entry = tk.Entry(api_container, textvariable=self.rf_api_key_var, 
                                    font=("Arial", 10), width=40, show="*")
        self.rf_api_entry.pack(side="left", padx=(10, 10))
        self.rf_api_entry.insert(0, "RKVbzglmD4K4cBfDFXRJ")  # Default API key
        
        self.rf_test_api_btn = tk.Button(api_container, text="🔍 Load Projects", 
                                        command=self.test_api_and_load_projects,
                                        bg="#4CAF50", fg="white", font=("Arial", 9, "bold"),
                                        relief=tk.RAISED, bd=2)
        self.rf_test_api_btn.pack(side="left", padx=(0, 10))
        
        # API Status
        self.rf_api_status_var = tk.StringVar(value="🔴 Enter API key and click 'Load Projects'")
        api_status_label = tk.Label(step1_frame, textvariable=self.rf_api_status_var, 
                                   font=("Arial", 9), fg="#ffcc00", bg=APP_BG_COLOR)
        api_status_label.pack(padx=10, pady=(0, 10))
        
        # Step 1.5: Create New Project (Optional)
        step1_5_frame = tk.LabelFrame(main_frame, text="Step 1.5: ✨ Create New Project (Optional)", 
                                     font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR,
                                     relief=tk.RAISED, bd=2)
        step1_5_frame.pack(fill="x", pady=(0, 15))
        
        # Project creation info
        create_info = tk.Label(step1_5_frame, text="Create a new project in your Roboflow workspace", 
                              font=("Arial", 9), fg="#cccccc", bg=APP_BG_COLOR)
        create_info.pack(padx=10, pady=(10, 2))
        
        # Project type info
        type_info = tk.Label(step1_5_frame, text="💡 Object Detection: For bounding boxes | Classification: For image labels", 
                            font=("Arial", 8), fg="#888888", bg=APP_BG_COLOR)
        type_info.pack(padx=10, pady=(0, 8))
        
        # Project name input
        name_container = tk.Frame(step1_5_frame, bg=APP_BG_COLOR)
        name_container.pack(fill="x", padx=10, pady=5)
        
        tk.Label(name_container, text="Project Name:", font=("Arial", 10, "bold"), 
                fg="white", bg=APP_BG_COLOR, width=15, anchor="w").pack(side="left")
        self.rf_new_project_name_var = tk.StringVar()
        self.rf_new_project_name_entry = tk.Entry(name_container, textvariable=self.rf_new_project_name_var, 
                                                 font=("Arial", 10), width=30)
        self.rf_new_project_name_entry.pack(side="left", padx=(5, 10))
        
        # Project type dropdown
        type_container = tk.Frame(step1_5_frame, bg=APP_BG_COLOR)
        type_container.pack(fill="x", padx=10, pady=5)
        
        tk.Label(type_container, text="Project Type:", font=("Arial", 10, "bold"), 
                fg="white", bg=APP_BG_COLOR, width=15, anchor="w").pack(side="left")
        self.rf_new_project_type_var = tk.StringVar(value="object-detection")
        self.rf_new_project_type_dropdown = ttk.Combobox(type_container, 
                                                        textvariable=self.rf_new_project_type_var,
                                                        values=["object-detection", "classification", "instance-segmentation", "semantic-segmentation"],
                                                        font=("Arial", 10), width=25, state="readonly")
        self.rf_new_project_type_dropdown.pack(side="left", padx=(5, 10))
        
        # License dropdown
        license_container = tk.Frame(step1_5_frame, bg=APP_BG_COLOR)
        license_container.pack(fill="x", padx=10, pady=5)
        
        tk.Label(license_container, text="License:", font=("Arial", 10, "bold"), 
                fg="white", bg=APP_BG_COLOR, width=15, anchor="w").pack(side="left")
        self.rf_new_project_license_var = tk.StringVar(value="private")
        self.rf_new_project_license_dropdown = ttk.Combobox(license_container, 
                                                           textvariable=self.rf_new_project_license_var,
                                                           values=["private", "MIT", "CC BY 4.0", "Public Domain"],
                                                           font=("Arial", 10), width=25, state="readonly")
        self.rf_new_project_license_dropdown.pack(side="left", padx=(5, 10))
        
        # Create project button
        create_button_container = tk.Frame(step1_5_frame, bg=APP_BG_COLOR)
        create_button_container.pack(pady=10)
        
        self.rf_create_project_btn = tk.Button(create_button_container, text="🆕 Create New Project", 
                                              command=self.create_roboflow_project,
                                              bg="#6f42c1", fg="white", font=("Arial", 10, "bold"),
                                              relief=tk.RAISED, bd=2, padx=15, pady=5, state="disabled")
        self.rf_create_project_btn.pack()
        
        # Project creation status
        self.rf_create_project_status_var = tk.StringVar(value="💡 Load projects first, then create new ones")
        create_status_label = tk.Label(step1_5_frame, textvariable=self.rf_create_project_status_var, 
                                      font=("Arial", 9), fg="#cccccc", bg=APP_BG_COLOR)
        create_status_label.pack(padx=10, pady=(0, 10))
        
        # Step 2: Project Selection
        step2_frame = tk.LabelFrame(main_frame, text="Step 2: 📂 Project Selection", 
                                   font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR,
                                   relief=tk.RAISED, bd=2)
        step2_frame.pack(fill="x", pady=(0, 15))
        
        project_container = tk.Frame(step2_frame, bg=APP_BG_COLOR)
        project_container.pack(fill="x", padx=10, pady=10)
        
        tk.Label(project_container, text="Project:", font=("Arial", 10, "bold"), 
                fg="white", bg=APP_BG_COLOR, width=12, anchor="w").pack(side="left")
        
        self.rf_project_var = tk.StringVar()
        self.rf_project_dropdown = ttk.Combobox(project_container, textvariable=self.rf_project_var,
                                               font=("Arial", 10), width=35, state="disabled")
        self.rf_project_dropdown.pack(side="left", padx=(10, 10))
        self.rf_project_dropdown.bind("<<ComboboxSelected>>", self.on_project_selected)
        
        # Project Status
        self.rf_project_status_var = tk.StringVar(value="⏳ Load projects first")
        project_status_label = tk.Label(step2_frame, textvariable=self.rf_project_status_var, 
                                       font=("Arial", 9), fg="#cccccc", bg=APP_BG_COLOR)
        project_status_label.pack(padx=10, pady=(0, 10))
        
        # Step 3: Dataset Selection
        step3_frame = tk.LabelFrame(main_frame, text="Step 3: 📊 Dataset Selection", 
                                   font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR,
                                   relief=tk.RAISED, bd=2)
        step3_frame.pack(fill="x", pady=(0, 15))
        
        dataset_container = tk.Frame(step3_frame, bg=APP_BG_COLOR)
        dataset_container.pack(fill="x", padx=10, pady=10)
        
        tk.Label(dataset_container, text="Dataset:", font=("Arial", 10, "bold"), 
                fg="white", bg=APP_BG_COLOR, width=12, anchor="w").pack(side="left")
        
        self.rf_dataset_var = tk.StringVar()
        self.rf_dataset_dropdown = ttk.Combobox(dataset_container, textvariable=self.rf_dataset_var,
                                               font=("Arial", 10), width=35, state="readonly")
        self.rf_dataset_dropdown.pack(side="left", padx=(10, 10))
        self.rf_dataset_dropdown.bind("<<ComboboxSelected>>", self.on_rf_dataset_selected)
        
        refresh_dataset_btn = tk.Button(dataset_container, text="🔄 Refresh", 
                                       command=self.refresh_all_datasets,
                                       bg="#17a2b8", fg="white", font=("Arial", 9, "bold"),
                                       relief=tk.RAISED, bd=2)
        refresh_dataset_btn.pack(side="left", padx=(5, 0))
        
        # Dataset Status
        self.rf_dataset_info_var = tk.StringVar(value="Select a dataset to upload")
        dataset_status_label = tk.Label(step3_frame, textvariable=self.rf_dataset_info_var, 
                                       font=("Arial", 9), fg="#cccccc", bg=APP_BG_COLOR)
        dataset_status_label.pack(padx=10, pady=(0, 10))
        
        # Step 4: Upload
        step4_frame = tk.LabelFrame(main_frame, text="Step 4: 🚀 Upload", 
                                   font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR,
                                   relief=tk.RAISED, bd=2)
        step4_frame.pack(fill="x", pady=(0, 15))
        
        upload_container = tk.Frame(step4_frame, bg=APP_BG_COLOR)
        upload_container.pack(pady=15)
        
        self.rf_upload_btn = tk.Button(upload_container, text="🚀 Upload Dataset to Roboflow", 
                                      font=("Arial", 14, "bold"), bg="#28a745", fg="white",
                                      command=self.upload_to_roboflow, relief=tk.RAISED, bd=3,
                                      padx=20, pady=10, state="disabled")
        self.rf_upload_btn.pack()
        
        # Upload info
        upload_info = tk.Label(step4_frame, text="Images will be uploaded as predictions to 'Unassigned' section", 
                              font=("Arial", 9), fg="#ffcc00", bg=APP_BG_COLOR)
        upload_info.pack(pady=(0, 10))
        
        # Status Log
        log_frame = tk.LabelFrame(main_frame, text="📝 Upload Log", 
                                 font=("Arial", 12, "bold"), fg="white", bg=APP_BG_COLOR,
                                 relief=tk.RAISED, bd=2)
        log_frame.pack(fill="both", expand=True)
        
        log_container = tk.Frame(log_frame, bg=APP_BG_COLOR)
        log_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.rf_status_text = tk.Text(log_container, height=8, width=70, 
                                     font=("Consolas", 9), bg="#2a2a2a", fg="white",
                                     wrap=tk.WORD, state=tk.DISABLED)
        
        log_scrollbar = tk.Scrollbar(log_container, orient="vertical", command=self.rf_status_text.yview)
        self.rf_status_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.rf_status_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")
        
        # Load datasets initially
        self.after(200, self.refresh_rf_datasets)
    
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
        
        # Extract dataset name from selection (remove status indicators)
        # Format: "dataset_name (COCO)"
        dataset_name = selected.replace(" (COCO)", "")
        
        try:
            # Get dataset info from database
            datasets = self.db_manager.get_datasets()
            selected_dataset = None
            for ds in datasets:
                if ds['name'] == dataset_name:
                    selected_dataset = ds
                    break
            
            if selected_dataset:
                dataset_path = selected_dataset['path']
                self.dataset_info_var.set(f"📁 Loading: {dataset_name} from {dataset_path}")
                self.load_dataset_images_from_folder(dataset_path)
            else:
                # Try default datasets folder
                default_path = os.path.join("datasets", dataset_name)
                if os.path.exists(default_path):
                    self.dataset_info_var.set(f"📁 Loading: {dataset_name} from {default_path}")
                    self.load_dataset_images_from_folder(default_path)
                else:
                    self.dataset_info_var.set("❌ Dataset not found")
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.dataset_info_var.set(f"❌ Error loading dataset: {str(e)}")
    
    def load_dataset_images_from_folder(self, folder_path):
        """Load and display images from a dataset folder using COCO JSON"""
        try:
            # Clear previous images and reset navigation
            self.dataset_images.clear()
            self.dataset_annotations = []
            self.dataset_categories = []
            self.current_preview_index = 0
            
            # Clear the preview canvas
            if hasattr(self, 'preview_canvas'):
                self.preview_canvas.delete("all")
            
            # Look for COCO JSON file
            coco_file = os.path.join(folder_path, "annotations.json")
            if not os.path.exists(coco_file):
                self.dataset_info_var.set(f"❌ No COCO annotations.json found in {os.path.basename(folder_path)}")
                return
            
            # Load COCO data
            import json
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            images_data = coco_data.get('images', [])
            annotations_data = coco_data.get('annotations', [])
            categories_data = coco_data.get('categories', [])
            categories_data = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
            
            if not images_data:
                self.dataset_info_var.set(f"📂 No images found in COCO JSON")
                return
            
            # Process images and store for single-image preview
            if not images_data:
                self.dataset_info_var.set(f"📂 No images found in COCO JSON")
                return
            
            # Store image data for single-image preview
            self.dataset_images = []
            for img_info in images_data:
                # Use full path if available, otherwise construct from images folder
                if 'file_path' in img_info and os.path.exists(img_info['file_path']):
                    image_path = img_info['file_path']
                else:
                    # Fallback: construct path from images folder
                    images_folder = os.path.join(folder_path, "images")
                    image_path = os.path.join(images_folder, img_info['file_name'])
                
                if os.path.exists(image_path):
                    self.dataset_images.append({
                        'file_path': image_path,
                        'file_name': img_info['file_name'],
                        'width': img_info.get('width', 0),
                        'height': img_info.get('height', 0),
                        'id': img_info['id']
                    })
            
            if not self.dataset_images:
                self.dataset_info_var.set(f"📂 No valid images found in dataset")
                return
            
            # Store annotations and categories
            self.dataset_annotations = annotations_data
            self.dataset_categories = categories_data
            
            # Reset navigation
            self.current_preview_index = 0
            
            # Update info
            labeled_count = len(set(ann['image_id'] for ann in annotations_data))
            total_annotations = len(annotations_data)
            total_images = len(self.dataset_images)
            
            self.dataset_info_var.set(f"✅ Loaded {total_images} images ({labeled_count} labeled, {total_annotations} annotations)")
            
            # Update navigation buttons
            self.update_navigation_buttons()
            
            # Show first image if available
            if self.dataset_images:
                self.show_single_image_with_annotations()
            
        except Exception as e:
            print(f"Error loading dataset images: {e}")
            self.dataset_info_var.set(f"❌ Error loading images: {str(e)}")
    
    def prev_preview_image(self):
        """Show previous image in preview"""
        if not self.dataset_images:
            return
        
        self.current_preview_index = (self.current_preview_index - 1) % len(self.dataset_images)
        self.show_single_image_with_annotations()
        self.update_navigation_buttons()
    
    def next_preview_image(self):
        """Show next image in preview"""
        if not self.dataset_images:
            return
        
        self.current_preview_index = (self.current_preview_index + 1) % len(self.dataset_images)
        self.show_single_image_with_annotations()
        self.update_navigation_buttons()
    
    def toggle_annotations_visibility(self):
        """Toggle the visibility of annotations on the current image"""
        if not self.dataset_images:
            return
        
        self.show_annotations = not self.show_annotations
        button_text = "🔲 Hide Annotations" if self.show_annotations else "🔳 Show Annotations"
        self.toggle_annotations_btn.config(text=button_text)
        
        # Refresh the current image
        self.show_single_image_with_annotations()
    
    def update_navigation_buttons(self):
        """Update the state of navigation buttons and counter"""
        if not self.dataset_images:
            self.prev_btn.config(state="disabled")
            self.next_btn.config(state="disabled")
            self.toggle_annotations_btn.config(state="disabled")
            self.image_counter_var.set("No images loaded")
            return
        
        # Enable buttons
        self.prev_btn.config(state="normal")
        self.next_btn.config(state="normal")
        self.toggle_annotations_btn.config(state="normal")
        
        # Update counter
        total_images = len(self.dataset_images)
        current_num = self.current_preview_index + 1
        self.image_counter_var.set(f"Image {current_num} of {total_images}")
        
        # Update button states at boundaries (optional visual feedback)
        if self.current_preview_index == 0:
            self.prev_btn.config(bg="#6c757d")  # Dimmer for first image
        else:
            self.prev_btn.config(bg="#007bff")  # Normal blue
            
        if self.current_preview_index == total_images - 1:
            self.next_btn.config(bg="#6c757d")  # Dimmer for last image
        else:
            self.next_btn.config(bg="#007bff")  # Normal blue
    
    def show_single_image_with_annotations(self):
        """Display the current image with annotations in the large preview canvas"""
        if not self.dataset_images or self.current_preview_index >= len(self.dataset_images):
            return
        
        try:
            # Get current image info
            image_info = self.dataset_images[self.current_preview_index]
            image_path = image_info.get('file_path', '')
            
            if not os.path.exists(image_path):
                self.current_image_info_var.set(f"❌ Image not found: {image_path}")
                return
            
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                self.current_image_info_var.set(f"❌ Could not load image: {image_path}")
                return
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Draw annotations if enabled
            if self.show_annotations and hasattr(self, 'dataset_annotations'):
                image_rgb = self.draw_annotations_on_image(image_rgb, image_info['id'])
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Calculate display size (fit to canvas while maintaining aspect ratio)
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not yet fully initialized, use default size
                canvas_width, canvas_height = 800, 600
            
            # Calculate scale to fit image in canvas
            img_width, img_height = pil_image.size
            scale_w = (canvas_width - 20) / img_width
            scale_h = (canvas_height - 20) / img_height
            scale = min(scale_w, scale_h, 1.0)  # Don't upscale
            
            # Resize image for display
            display_width = int(img_width * scale)
            display_height = int(img_height * scale)
            display_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.current_preview_photo = ImageTk.PhotoImage(display_image)
            
            # Clear canvas and display image
            self.preview_canvas.delete("all")
            
            # Center the image in the canvas
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            
            self.preview_canvas.create_image(canvas_center_x, canvas_center_y, 
                                           image=self.current_preview_photo, anchor=tk.CENTER)
            
            # Update scroll region for large images
            self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))
            
            # Update image info
            filename = os.path.basename(image_path)
            annotations_count = len([ann for ann in self.dataset_annotations 
                                   if ann.get('image_id') == image_info['id']])
            
            info_text = (f"📁 File: {filename} | "
                        f"📐 Size: {img_width}×{img_height} | "
                        f"🏷️ Annotations: {annotations_count} | "
                        f"📊 Display: {display_width}×{display_height} ({scale:.2%})")
            
            self.current_image_info_var.set(info_text)
            
        except Exception as e:
            self.current_image_info_var.set(f"❌ Error displaying image: {str(e)}")
            print(f"Error in show_single_image_with_annotations: {e}")
    
    def draw_annotations_on_image(self, image_rgb, image_id):
        """Draw COCO annotations on the image"""
        try:
            # Get annotations for this image
            image_annotations = [ann for ann in self.dataset_annotations 
                               if ann.get('image_id') == image_id]
            
            if not image_annotations:
                return image_rgb
            
            # Convert to PIL for drawing
            pil_image = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # Define colors for different categories
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', 
                     '#FFA500', '#800080', '#FFC0CB', '#008000', '#FFD700', '#4B0082']
            
            for i, annotation in enumerate(image_annotations):
                try:
                    # Get bounding box (COCO format: [x, y, width, height])
                    bbox = annotation.get('bbox', [])
                    if len(bbox) != 4:
                        continue
                    
                    x, y, width, height = bbox
                    x2, y2 = x + width, y + height
                    
                    # Get category info
                    category_id = annotation.get('category_id', 0)
                    color = colors[category_id % len(colors)]
                    
                    # Draw bounding box
                    draw.rectangle([x, y, x2, y2], outline=color, width=3)
                    
                    # Find category name
                    category_name = f"Category {category_id}"
                    if hasattr(self, 'dataset_categories'):
                        for cat in self.dataset_categories:
                            if cat.get('id') == category_id:
                                category_name = cat.get('name', category_name)
                                break
                    
                    # Draw label background
                    label_text = f"{category_name}"
                    bbox_label = draw.textbbox((0, 0), label_text)
                    label_width = bbox_label[2] - bbox_label[0]
                    label_height = bbox_label[3] - bbox_label[1]
                    
                    draw.rectangle([x, y - label_height - 4, x + label_width + 8, y], 
                                 fill=color, outline=color)
                    
                    # Draw label text
                    draw.text((x + 4, y - label_height - 2), label_text, fill='white')
                    
                except Exception as e:
                    print(f"Error drawing annotation: {e}")
                    continue
            
            return np.array(pil_image)
            
        except Exception as e:
            print(f"Error in draw_annotations_on_image: {e}")
            return image_rgb
    
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
        
        self.preview_info_var = tk.StringVar(value="")
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
        
        # Left side - Raw image (responsive size)
        raw_frame = tk.LabelFrame(self.split_container, text="🖼️ Original Image", 
                                 font=("Arial", 12, "bold"), fg="#17a2b8", bg=APP_BG_COLOR)
        raw_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5))
        
        # Calculate responsive canvas size based on available space
        available_width = int(self.winfo_screenwidth() * 0.25)  # ~25% of screen width
        available_height = int(self.winfo_screenheight() * 0.4)  # ~40% of screen height
        self.split_canvas_width = min(400, available_width)
        self.split_canvas_height = min(450, available_height)
        
        self.raw_image_canvas = tk.Canvas(raw_frame, width=self.split_canvas_width, height=self.split_canvas_height, 
                                         bg="#1a1a1a", highlightthickness=0)
        self.raw_image_canvas.pack(pady=10, expand=True, fill=tk.BOTH)
        
        # Right side - Labeled image (same size as raw)
        labeled_frame = tk.LabelFrame(self.split_container, text="🎯 Labeled Image", 
                                     font=("Arial", 12, "bold"), fg="#28a745", bg=APP_BG_COLOR)
        labeled_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10))
        
        self.labeled_image_canvas = tk.Canvas(labeled_frame, width=self.split_canvas_width, height=self.split_canvas_height, 
                                             bg="#1a1a1a", highlightthickness=0)
        self.labeled_image_canvas.pack(pady=10, expand=True, fill=tk.BOTH)
        
        # Single view container (initially hidden) - larger but still responsive
        self.single_container = tk.Frame(main_display_frame, bg=APP_BG_COLOR)
        
        self.single_canvas_width = min(800, int(self.winfo_screenwidth() * 0.5))
        self.single_canvas_height = min(500, int(self.winfo_screenheight() * 0.45))
        
        self.single_image_canvas = tk.Canvas(self.single_container, width=self.single_canvas_width, height=self.single_canvas_height, 
                                           bg="#1a1a1a", highlightthickness=0)
        self.single_image_canvas.pack(pady=10, expand=True, fill=tk.BOTH)
        
        # Image info label (hidden initially - no text shown in upload widget)
        self.main_image_info_var = tk.StringVar(value="")
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
        
        tk.Button(control_frame, text="� Create Dataset", 
                 font=("Arial", 12, "bold"), bg="#28a745", fg="white", width=15,
                 command=self.create_dataset_from_labels).pack(side=tk.LEFT, padx=5)
        
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
                    # Run YOLO inference automatically with safe fallback
                    results = self.safe_yolo_inference(self.current_selected_path)
                    
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
        """Show the current raw image in the appropriate canvas, WITHOUT annotations"""
        if not self.current_main_image:
            return
        
        try:
            img_width, img_height = self.current_main_image.size
            
            # Calculate display size based on preview mode
            if self.preview_mode.get() == "split":
                canvas_width, canvas_height = self.split_canvas_width, self.split_canvas_height
                canvas = self.raw_image_canvas
            else:
                canvas_width, canvas_height = self.single_canvas_width, self.single_canvas_height
                canvas = self.single_image_canvas
            
            # Create image for display (start with original - NO ANNOTATIONS)
            display_image = self.current_main_image.copy()
            
            # DO NOT draw annotations on raw image - this should always show the clean original
            # Removed: if self.current_selected_path in self.temp_annotations:
            #             display_image = self.draw_temp_annotations_on_image(display_image)
            
            # Resize while maintaining aspect ratio
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            display_image = display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_display_image = ImageTk.PhotoImage(display_image)
            
            # Display in canvas
            canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            canvas.create_image(x, y, anchor=tk.NW, image=self.current_display_image)
            
        except Exception as e:
            print(f"Error showing raw image: {e}")
    
    def draw_temp_annotations_on_image(self, image):
        """Draw temporary annotations on the image"""
        try:
            # Convert PIL image to OpenCV format for drawing
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Get annotations for this image
            temp_data = self.temp_annotations[self.current_selected_path]
            annotations = temp_data['annotations']
            
            # Draw each annotation
            for annotation in annotations:
                bbox = annotation['bbox']
                class_name = annotation['class_name']
                confidence = annotation['confidence']
                
                # Extract coordinates
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(cv_image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(cv_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Convert back to PIL format
            annotated_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            return annotated_image
            
        except Exception as e:
            print(f"Error drawing temporary annotations: {e}")
            return image
    
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
            
            # Run YOLO inference with safe fallback
            results = self.safe_yolo_inference(self.current_selected_path)
            
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
                canvas_width, canvas_height = self.split_canvas_width, self.split_canvas_height
                canvas = self.labeled_image_canvas
            else:
                canvas_width, canvas_height = self.single_canvas_width, self.single_canvas_height
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
                                   font=("Arial", 14, "bold"), 
                                     fg="white", bg=APP_BG_COLOR, 
                                     relief="raised", bd=2)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model selection section - vertical layout for better fit
        selection_frame = tk.Frame(model_frame, bg=APP_BG_COLOR)
        selection_frame.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(selection_frame, text="Select Model:", 
                font=("Arial", 11, "bold"), 
                fg="white", bg=APP_BG_COLOR).pack(anchor=tk.W, pady=(0, 5))
        
        # Model dropdown and load button
        model_controls = tk.Frame(selection_frame, bg=APP_BG_COLOR)
        model_controls.pack(fill=tk.X, pady=(0, 10))
        
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(model_controls, textvariable=self.model_var, 
                                          font=("Arial", 10), state="readonly")
        self.model_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.model_dropdown.bind('<<ComboboxSelected>>', self.on_model_selected)
        
        # Prominent Load Model button
        self.load_model_btn = tk.Button(model_controls, text="📥 Load Model", 
                                       font=("Arial", 11, "bold"), 
                                       bg="#007bff", fg="white", width=12,
                                       command=self.load_selected_model)
        self.load_model_btn.pack(side=tk.RIGHT)
        
        # Model status indicator
        self.model_status_var = tk.StringVar(value="No model loaded")
        model_status_label = tk.Label(selection_frame, textvariable=self.model_status_var, 
                                     font=("Arial", 10), fg="#ffc107", bg=APP_BG_COLOR,
                                     wraplength=320, justify=tk.LEFT)
        model_status_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Separator line
        separator = tk.Frame(selection_frame, height=1, bg="#555555")
        separator.pack(fill=tk.X, pady=(5, 10))
        
        # Model loading section - compact vertical layout
        tk.Label(selection_frame, text="Add New Model:", 
                font=("Arial", 11, "bold"), 
                fg="white", bg=APP_BG_COLOR).pack(anchor=tk.W, pady=(0, 5))
        
        # Browse and add controls
        loading_controls = tk.Frame(selection_frame, bg=APP_BG_COLOR)
        loading_controls.pack(fill=tk.X, pady=(0, 5))
        
        browse_model_btn = tk.Button(loading_controls, text="Browse", 
                                    font=("Arial", 10, "bold"), 
                                    bg="#28a745", fg="white", width=8,
                                    command=self.browse_model_file)
        browse_model_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.model_path_var = tk.StringVar()
        model_path_entry = tk.Entry(loading_controls, textvariable=self.model_path_var, 
                                   font=("Arial", 9), state="readonly")
        model_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        add_model_btn = tk.Button(loading_controls, text="Add", 
                                 font=("Arial", 10, "bold"), 
                                 bg="#ffc107", fg="black", width=6,
                                 command=self.add_model_to_database)
        add_model_btn.pack(side=tk.LEFT, padx=5)
    
    def create_dataset_section(self, parent):
        """Create dataset management section"""
        dataset_frame = tk.LabelFrame(parent, text="Dataset Management", 
                                     font=("Arial", 14, "bold"), 
                                     fg="white", bg=APP_BG_COLOR, 
                                     relief="raised", bd=2)
        dataset_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 5))
        
        # Dataset options row - stacked vertically for better fit
        options_frame = tk.Frame(dataset_frame, bg=APP_BG_COLOR)
        options_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Dataset type selection - stacked for better visibility
        self.dataset_type_var = tk.StringVar(value="new")
        new_dataset_rb = tk.Radiobutton(options_frame, text="Create New Dataset", 
                                       variable=self.dataset_type_var, value="new",
                                       font=("Arial", 11, "bold"), fg="white", bg=APP_BG_COLOR,
                                       selectcolor=APP_BG_COLOR, command=self.on_dataset_type_changed)
        new_dataset_rb.pack(anchor=tk.W, pady=(0, 5))
        
        existing_dataset_rb = tk.Radiobutton(options_frame, text="Add to Existing Dataset", 
                                            variable=self.dataset_type_var, value="existing",
                                            font=("Arial", 11, "bold"), fg="white", bg=APP_BG_COLOR,
                                            selectcolor=APP_BG_COLOR, command=self.on_dataset_type_changed)
        existing_dataset_rb.pack(anchor=tk.W)
        
        # New dataset row
        self.new_dataset_frame = tk.Frame(dataset_frame, bg=APP_BG_COLOR)
        self.new_dataset_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Dataset name row - vertical layout for better fit
        tk.Label(self.new_dataset_frame, text="Dataset Name:", 
                 font=("Arial", 11, "bold"), 
                 fg="white", bg=APP_BG_COLOR).pack(anchor=tk.W, pady=(0, 5))
        
        self.dataset_name_var = tk.StringVar()
        self.dataset_entry = tk.Entry(self.new_dataset_frame, textvariable=self.dataset_name_var, 
                                     font=("Arial", 10), width=35)
        self.dataset_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Dataset location row - vertical layout
        tk.Label(self.new_dataset_frame, text="Save Location:", 
                 font=("Arial", 11, "bold"), 
                 fg="white", bg=APP_BG_COLOR).pack(anchor=tk.W, pady=(0, 5))
        
        location_controls = tk.Frame(self.new_dataset_frame, bg=APP_BG_COLOR)
        location_controls.pack(fill=tk.X, pady=(0, 5))
        
        browse_location_btn = tk.Button(location_controls, text="Browse", 
                                       font=("Arial", 10, "bold"), 
                                       bg="#17a2b8", fg="white", width=8,
                                       command=self.browse_dataset_location)
        browse_location_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.location_var = tk.StringVar(value="datasets (default)")
        location_entry = tk.Entry(location_controls, textvariable=self.location_var, 
                                 font=("Arial", 9), state="readonly")
        location_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Existing dataset row - vertical layout
        self.existing_dataset_frame = tk.Frame(dataset_frame, bg=APP_BG_COLOR)
        self.existing_dataset_frame.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(self.existing_dataset_frame, text="Select Dataset:", 
                 font=("Arial", 11, "bold"), 
                 fg="white", bg=APP_BG_COLOR).pack(anchor=tk.W, pady=(0, 5))
        
        dataset_controls = tk.Frame(self.existing_dataset_frame, bg=APP_BG_COLOR)
        dataset_controls.pack(fill=tk.X)
        
        self.existing_dataset_var = tk.StringVar()
        self.dataset_dropdown = ttk.Combobox(dataset_controls, textvariable=self.existing_dataset_var, 
                                           font=("Arial", 10), state="readonly")
        self.dataset_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        refresh_datasets_btn = tk.Button(dataset_controls, text="Refresh", 
                                        font=("Arial", 10, "bold"), 
                                        bg="#17a2b8", fg="white", width=8,
                                        command=self.load_datasets)
        refresh_datasets_btn.pack(side=tk.LEFT)
        
        # Initially hide existing dataset frame
        self.existing_dataset_frame.pack_forget()
    
    def create_upload_section(self, parent):
        """Create image upload section"""
        upload_frame = tk.LabelFrame(parent, text="Image Upload", 
                                    font=("Arial", 14, "bold"), 
                                     fg="white", bg=APP_BG_COLOR, 
                                     relief="raised", bd=2)
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Upload options - vertical layout for better fit
        options_frame = tk.Frame(upload_frame, bg=APP_BG_COLOR)
        options_frame.pack(fill=tk.X, padx=15, pady=10)
        
        upload_files_btn = tk.Button(options_frame, text="Upload Individual Images", 
                                    font=("Arial", 11, "bold"), 
                                    bg="#28a745", fg="white",
                                    command=self.upload_images)
        upload_files_btn.pack(fill=tk.X, pady=(0, 5))
        
        upload_folder_btn = tk.Button(options_frame, text="Upload Image Folder", 
                                     font=("Arial", 11, "bold"), 
                                     bg="#007bff", fg="white",
                                     command=self.upload_image_folder)
        upload_folder_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Initialize the image_info_var for status tracking (no visible label)
        self.image_info_var = tk.StringVar(value="")
        
        # Upload info display (hidden but tracks upload count for status)
        self.upload_info_var = tk.StringVar(value="")
        # Note: No label widget created here to avoid showing text in upload section
    
    def create_action_section(self, parent):
        """Create action buttons section"""
        action_frame = tk.LabelFrame(parent, text="Actions", 
                                    font=("Arial", 14, "bold"), 
                                    fg="white", bg=APP_BG_COLOR, 
                                    relief="raised", bd=2)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        button_frame = tk.Frame(action_frame, bg=APP_BG_COLOR)
        button_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Arrange buttons in a 2x3 grid for better fit
        start_btn = tk.Button(button_frame, text="Start Labeling", 
                             font=("Arial", 11, "bold"), 
                             bg="#ffc107", fg="black",
                             command=self.start_labeling)
        start_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.create_dataset_btn = tk.Button(button_frame, text="📦 Create Dataset", 
                              font=("Arial", 11, "bold"), 
                              bg="#28a745", fg="white",
                              command=self.create_dataset_from_labels)
        self.create_dataset_btn.pack(fill=tk.X, pady=(0, 5))
        
        clear_btn = tk.Button(button_frame, text="Clear All", 
                             font=("Arial", 11, "bold"), 
                             bg="#dc3545", fg="white",
                             command=self.clear_all)
        clear_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Second row of buttons
        settings_row = tk.Frame(button_frame, bg=APP_BG_COLOR)
        settings_row.pack(fill=tk.X, pady=(0, 5))
        
        db_settings_btn = tk.Button(settings_row, text="DB Settings", 
                                   font=("Arial", 10, "bold"), 
                                   bg="#17a2b8", fg="white",
                                   command=self.show_db_settings)
        db_settings_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        settings_btn = tk.Button(settings_row, text="YOLO Settings", 
                                font=("Arial", 10, "bold"), 
                                bg="#6c757d", fg="white",
                                command=self.show_yolo_settings)
        settings_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
    
    def create_status_panel(self, parent):
        """Create enhanced status panel with device information"""
        status_frame = tk.LabelFrame(parent, text="Status", 
                                    font=("Arial", 14, "bold"), 
                                    fg="white", bg=APP_BG_COLOR, 
                                    relief="raised", bd=2)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Device status section
        self.device_status_var = tk.StringVar()
        self.update_device_status()
        device_status_label = tk.Label(status_frame, textvariable=self.device_status_var, 
                                     font=("Arial", 10, "bold"), fg="#00ff88", bg=APP_BG_COLOR,
                                     wraplength=280, justify=tk.LEFT)
        device_status_label.pack(anchor=tk.W, padx=15, pady=(10, 5))
        
        # Main status section
        self.status_var = tk.StringVar(value="🏁 Ready - Load a model and upload images to begin")
        status_label = tk.Label(status_frame, textvariable=self.status_var, 
                               font=("Arial", 10), fg="#cccccc", bg=APP_BG_COLOR,
                               wraplength=280, justify=tk.LEFT)
        status_label.pack(anchor=tk.W, padx=15, pady=(5, 5))
        
        # Dataset location info
        self.location_info_var = tk.StringVar(value="Dataset save location: datasets (default)")
        location_info_label = tk.Label(status_frame, textvariable=self.location_info_var, 
                                     font=("Arial", 9), fg="#888888", bg=APP_BG_COLOR,
                                     wraplength=280, justify=tk.LEFT)
        location_info_label.pack(anchor=tk.W, padx=15, pady=(0, 10))
    
    def update_device_status(self):
        """Update device status display with current device information"""
        if hasattr(self, 'device_status_var'):
            try:
                if self.optimal_device == 'cuda' and torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    try:
                        total_memory = torch.cuda.get_device_properties(0).total_memory
                        memory_gb = total_memory / (1024**3)
                        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
                        memory_info = f"{allocated_memory:.1f}/{memory_gb:.1f}GB"
                    except:
                        memory_info = "Memory info unavailable"
                    
                    self.device_status_var.set(f"🎮 GPU: {gpu_name} ({memory_info})")
                    
                elif self.optimal_device == 'mps':
                    self.device_status_var.set("🍎 Apple Silicon GPU (MPS)")
                    
                else:
                    self.device_status_var.set("💻 CPU Processing")
                    
            except Exception as e:
                self.device_status_var.set("⚠️ Device Status Unknown")
                print(f"Error updating device status: {e}")
    
    def schedule_device_status_update(self):
        """Schedule periodic device status updates"""
        try:
            # Update device status
            self.update_device_status()
            
            # Schedule next update in 10 seconds (only if window still exists)
            if hasattr(self, 'device_status_var'):
                self.after(10000, self.schedule_device_status_update)
        except Exception as e:
            print(f"Error in device status update scheduler: {e}")
    
    def on_dataset_type_changed(self):
        """Handle dataset type radio button change"""
        if self.dataset_type_var.get() == "new":
            self.existing_dataset_frame.pack_forget()
            self.new_dataset_frame.pack(fill=tk.X, padx=15, pady=10)
            # Update button text for new dataset
            self.create_dataset_btn.config(text="📦 Create Dataset")
        else:
            self.new_dataset_frame.pack_forget()
            self.existing_dataset_frame.pack(fill=tk.X, padx=15, pady=10)
            # Update button text for adding to existing dataset
            self.create_dataset_btn.config(text="➕ Add to Dataset")
    
    def load_models(self):
        """Load available models from database"""
        try:
            models = self.db_manager.get_models()
            model_names = [model['name'] for model in models]
        
            if model_names:
                self.model_dropdown['values'] = model_names
                self.model_dropdown.set(model_names[0])  # Select first model by default
                self.model_status_var.set(f"🔸 Selected: {model_names[0]}")
                self.load_model_btn.config(state='normal', bg="#007bff")
                self.status_var.set(f"Loaded {len(model_names)} models from database")
            else:
                self.model_dropdown['values'] = []
                self.model_status_var.set("No models available")
                self.load_model_btn.config(state='disabled', bg="#6c757d")
                self.status_var.set("No models found in database. Add models using 'Add New Model' section.")
        except Exception as e:
            self.model_dropdown['values'] = []
            self.model_status_var.set("Error loading models")
            self.load_model_btn.config(state='disabled', bg="#6c757d")
            raise e
    
    def load_datasets(self):
        """Load available datasets from database - only show existing folders"""
        try:
            datasets = self.db_manager.get_datasets()
            # Filter datasets to only include those with existing paths
            valid_datasets = []
            for dataset in datasets:
                if os.path.exists(dataset['path']):
                    valid_datasets.append(dataset)
            
            dataset_names = [f"{dataset['name']} ({dataset['image_count']} images)" for dataset in valid_datasets]
            
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
    
    def cleanup_deleted_datasets(self):
        """Remove datasets from database if their folders no longer exist"""
        try:
            datasets = self.db_manager.get_datasets()
            deleted_count = 0
            
            for dataset in datasets:
                if not os.path.exists(dataset['path']):
                    # Dataset folder no longer exists, remove from database
                    cursor = self.db_connection.cursor()
                    cursor.execute("DELETE FROM datasets WHERE dataset_name = %s", (dataset['name'],))
                    deleted_count += 1
                    print(f"Removed deleted dataset from database: {dataset['name']}")
            
            if deleted_count > 0:
                self.db_connection.commit()
                print(f"Cleaned up {deleted_count} deleted dataset(s) from database")
                
        except Exception as e:
            print(f"Error cleaning up deleted datasets: {e}")
    
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
        if selected_model:
            self.model_status_var.set(f"🔸 Selected: {selected_model}")
            self.load_model_btn.config(state='normal', bg="#007bff")
        else:
            self.model_status_var.set("No model selected")
            self.load_model_btn.config(state='disabled', bg="#6c757d")
    
    def load_selected_model(self):
        """Load the selected YOLO model with enhanced GPU/CPU handling"""
        selected_model_name = self.model_var.get()
        if not selected_model_name:
            messagebox.showerror("Error", "Please select a model first")
            return
        
        # Update button state during loading
        self.load_model_btn.config(state='disabled', text="⏳ Loading...", bg="#6c757d")
        self.model_status_var.set("🔄 Loading model...")
        self.update()
        
        # Get model details from database
        model_info = self.db_manager.get_model_by_name(selected_model_name)
        if not model_info:
            messagebox.showerror("Error", "Model not found in database")
            self.load_model_btn.config(state='normal', text="📥 Load Model", bg="#007bff")
            self.model_status_var.set(f"❌ Model not found: {selected_model_name}")
            return
        
        model_path = model_info['path']
        
        # Check if model file exists
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            self.load_model_btn.config(state='normal', text="📥 Load Model", bg="#007bff")
            self.model_status_var.set(f"❌ File not found: {selected_model_name}")
            return
        
        try:
            # Load YOLO model initially on CPU for safety
            self.status_var.set(f"🔄 Loading YOLO model: {selected_model_name}...")
            self.update()
            
            print(f"📥 Loading model from: {model_path}")
            self.yolo_model = YOLO(model_path)
            print("✅ Model loaded successfully on CPU")
            
            # Determine the best device and move model
            device_info = self.setup_model_device()
            
            # Update UI with success information
            self.current_model = model_info
            self.status_var.set(f"✅ Model loaded successfully on {device_info['name']}: {selected_model_name}")
            self.model_status_var.set(f"✅ {selected_model_name} - {device_info['name']}")
            self.load_model_btn.config(state='normal', text="✅ Loaded", bg="#28a745")
            
            # Update device status display
            self.update_device_status()
            
            # Show detailed success message
            success_message = (
                f"Model '{selected_model_name}' loaded successfully!\n\n"
                f"🖥️ Device: {device_info['name']}\n"
                f"📊 Classes: {len(self.yolo_model.names)} detection classes\n"
                f"📁 Path: {model_path}"
            )
            
            if device_info['memory']:
                success_message += f"\n💾 GPU Memory: {device_info['memory']}"
            
            messagebox.showinfo("Model Loaded Successfully", success_message)
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            print(f"❌ {error_msg}")
            
            self.load_model_btn.config(state='normal', text="📥 Load Model", bg="#007bff")
            self.model_status_var.set(f"❌ Load failed: {selected_model_name}")
            self.status_var.set(f"❌ Model loading failed: {str(e)}")
            messagebox.showerror("Model Loading Error", error_msg)
    
    def setup_model_device(self):
        """Setup the loaded model on the optimal device with comprehensive error handling"""
        device_info = {
            'device': 'cpu',
            'name': 'CPU',
            'memory': None,
            'success': False
        }
        
        try:
            print(f"🔧 Setting up model on device: {self.optimal_device}")
            
            if self.optimal_device == 'cuda':
                # Test CUDA compatibility before moving model
                print("🧪 Testing CUDA compatibility with loaded model...")
                test_result = self.test_cuda_inference()
                
                if test_result:
                    print("🎮 Moving model to CUDA GPU...")
                    self.yolo_model.to("cuda")
                    
                    # Get GPU info
                    gpu_name = torch.cuda.get_device_name(0)
                    try:
                        total_memory = torch.cuda.get_device_properties(0).total_memory
                        memory_gb = total_memory / (1024**3)
                        memory_info = f"{memory_gb:.1f}GB"
                    except:
                        memory_info = "Unknown"
                    
                    device_info.update({
                        'device': 'cuda',
                        'name': f"NVIDIA {gpu_name}",
                        'memory': memory_info,
                        'success': True
                    })
                    
                    print(f"✅ Model successfully loaded on CUDA: {gpu_name}")
                else:
                    print("❌ CUDA compatibility test failed, falling back to CPU")
                    self.optimal_device = 'cpu'
                    self.yolo_model.to("cpu")
                    device_info.update({
                        'device': 'cpu',
                        'name': 'CPU (CUDA incompatible)',
                        'success': True
                    })
                        
            elif self.optimal_device == 'mps':
                try:
                    print("🍎 Moving model to Apple Silicon GPU (MPS)...")
                    self.yolo_model.to("mps")
                    device_info.update({
                        'device': 'mps',
                        'name': 'Apple Silicon GPU (MPS)',
                        'success': True
                    })
                    print("✅ Model successfully loaded on MPS")
                    
                except Exception as mps_error:
                    print(f"❌ MPS loading failed: {mps_error}")
                    print("🔄 Falling back to CPU")
                    self.optimal_device = 'cpu'
                    self.yolo_model.to("cpu")
                    device_info.update({
                        'device': 'cpu',
                        'name': 'CPU (MPS fallback)',
                        'success': True
                    })
            else:
                # CPU mode
                print("💻 Using CPU for model inference")
                self.yolo_model.to("cpu")
                device_info.update({
                    'device': 'cpu',
                    'name': 'CPU',
                    'success': True
                })
            
            return device_info
            
        except Exception as device_error:
            print(f"❌ Device setup error: {device_error}")
            print("🔄 Forcing CPU fallback")
            
            try:
                self.yolo_model.to("cpu")
                self.optimal_device = 'cpu'
                device_info.update({
                    'device': 'cpu',
                    'name': 'CPU (forced fallback)',
                    'success': True
                })
                print("✅ Successfully fell back to CPU")
                
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {cpu_error}")
                device_info.update({
                    'device': 'cpu',
                    'name': 'CPU (error state)',
                    'success': False
                })
            
            return device_info
        
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Get the selected model name safely
            try:
                selected_name = self.model_var.get() or "Unknown Model"
            except:
                selected_name = "Unknown Model"
            
            self.load_model_btn.config(state='normal', text="📥 Load Model", bg="#007bff")
            self.model_status_var.set(f"❌ Load failed: {selected_name}")
            self.status_var.set(f"❌ Model loading failed: {str(e)}")
            messagebox.showerror("Model Loading Error", error_msg)
    
    def upload_images(self):
        """Upload individual images for labeling"""
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=FILE_CONFIG['supported_formats']
        )
        
        if file_paths:
            self.uploaded_images = list(file_paths)
            # Update internal tracking variables (no visible text)
            self.image_info_var.set(f"Loaded {len(self.uploaded_images)} images")
            self.upload_info_var.set(f"Uploaded {len(self.uploaded_images)} images")
            # Update status panel
            self.status_var.set(f"✅ Uploaded {len(self.uploaded_images)} images - Ready for labeling")
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
                # Update internal tracking variables (no visible text)
                self.image_info_var.set(f"Loaded {len(self.uploaded_images)} images from folder")
                self.upload_info_var.set(f"Uploaded {len(self.uploaded_images)} images from folder")
                # Update status panel
                self.status_var.set(f"✅ Uploaded {len(self.uploaded_images)} images from folder: {os.path.basename(folder_path)} - Ready for labeling")
                # Update image previews
                self.update_image_previews()
            else:
                messagebox.showwarning("No Images", "No supported image files found in the selected folder")
                self.status_var.set("❌ No images found in selected folder")
    
    def start_labeling(self):
        """Start the labeling process - store annotations temporarily"""
        if not self.yolo_model:
            messagebox.showerror("Error", "Please load a YOLO model first")
            self.status_var.set("❌ Cannot start labeling - No model loaded")
            return
        
        if not self.uploaded_images:
            messagebox.showerror("Error", "Please upload images first")
            self.status_var.set("❌ Cannot start labeling - No images uploaded")
            return
        
        # Initialize temporary annotation storage for this session
        self.temp_annotations = {}
        self.is_labeling_session_active = True
        
        # Get a temporary session name for tracking
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_session_name = f"session_{timestamp}"
        
        # Update status
        self.status_var.set(f"🔄 Starting labeling session for {len(self.uploaded_images)} images...")
        
        # Start labeling process in a separate thread
        threading.Thread(target=self.label_images_temporarily, daemon=True).start()
    
    def label_images_temporarily(self):
        """Label images and store annotations temporarily (not saved to disk yet)"""
        try:
            total_images = len(self.uploaded_images)
            labeled_count = 0
            
            for i, image_path in enumerate(self.uploaded_images):
                # Update status
                self.status_var.set(f"Labeling image {i+1}/{total_images}: {os.path.basename(image_path)}")
                self.update()
                
                # Run YOLO inference with safe fallback
                results = self.safe_yolo_inference(image_path)
                
                # Process results and store temporarily
                if self.store_temp_annotation(image_path, results[0]):
                    labeled_count += 1
            
            # Mark session as complete
            self.is_labeling_session_active = True  # Keep true until dataset creation
            
            # Complete
            rejected_count = total_images - labeled_count
            self.status_var.set(f"Labeling complete! {labeled_count}/{total_images} images annotated temporarily")
            
            # Show completion message
            self.after(0, lambda: messagebox.showinfo("Labeling Complete", 
                                                     f"Temporary labeling complete!\n\n"
                                                     f"✅ Images labeled: {labeled_count} (with detections)\n"
                                                     f"❌ Images skipped: {rejected_count} (no detections)\n"
                                                     f"📁 Total processed: {total_images}\n\n"
                                                     f"All detected classes are included in annotations.\n"
                                                     f"Click 'Create Dataset' to save them permanently."))
            
            # Update preview to show annotations
            self.after(0, self.update_image_previews)
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Labeling failed: {str(e)}"))
            self.status_var.set("Labeling failed")
    
    def store_temp_annotation(self, image_path, results):
        """Store annotation temporarily in memory"""
        try:
            # Store all detected classes, not just "roller"
            annotations = []
            
            if results.boxes is not None:
                for box in results.boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.yolo_model.names[class_id]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Store annotation info for all classes
                    annotations.append({
                        'class_name': class_name,
                        'class_id': class_id,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence
                    })
            
            # Store the image if it contains any detections
            if annotations:
                # Get image dimensions
                image = cv2.imread(image_path)
                if image is None:
                    return False
                
                height, width = image.shape[:2]
                
                # Store in temporary annotations
                self.temp_annotations[image_path] = {
                    'annotations': annotations,
                    'image_width': width,
                    'image_height': height,
                    'filename': os.path.basename(image_path)
                }
                
                class_names = [ann['class_name'] for ann in annotations]
                print(f"✅ Temporarily stored annotations for {os.path.basename(image_path)}: {len(annotations)} detections - Classes: {', '.join(set(class_names))}")
                return True
            else:
                print(f"Skipping image {os.path.basename(image_path)}: No detections found")
                return False
                
        except Exception as e:
            print(f"Error storing temporary annotation for {image_path}: {e}")
            return False
    
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
                # Note: No labels folder needed - using COCO format only
                
                # Add to database with full path
                self.db_manager.add_dataset(dataset_name, base_path, f"Dataset created on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                
                # Refresh all dataset dropdowns
                self.refresh_all_datasets()
                
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
                
                # Run YOLO inference with safe fallback
                results = self.safe_yolo_inference(image_path)
                
                # Process results and create COCO format annotations
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
        """Process YOLO results and save directly in COCO format - only if 'roller' class is detected"""
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
            
            # Update COCO annotations directly
            self.update_coco_annotations(image_filename, width, height, results, dest_image_path)
            
            print(f"✅ Added image {os.path.basename(image_path)}: Contains 'roller' class")
            return True
            
        except Exception as e:
            print(f"Error processing results for {image_path}: {e}")
            return False
    
    def update_coco_annotations(self, image_filename, width, height, results, image_path):
        """Update or create COCO annotations file with new image and annotations"""
        try:
            # Path to COCO annotations file
            coco_json_path = os.path.join(self.current_dataset_path, "annotations.json")
            
            # Load existing COCO data or create new structure
            if os.path.exists(coco_json_path):
                with open(coco_json_path, 'r') as f:
                    coco_data = json.load(f)
            else:
                # Create new COCO structure
                coco_data = {
                    "info": {
                        "description": "WelVision Dataset",
                        "version": "1.0",
                        "year": datetime.now().year,
                        "contributor": "WelVision Data Labeller",
                        "date_created": datetime.now().isoformat()
                    },
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": []
                }
                
                # Add categories from YOLO model
                if hasattr(self, 'yolo_model') and self.yolo_model:
                    for class_id, class_name in self.yolo_model.names.items():
                        coco_data["categories"].append({
                            "id": class_id + 1,  # COCO uses 1-based indexing
                            "name": class_name,
                            "supercategory": "object"
                        })
            
            # Get next image ID
            image_id = max([img["id"] for img in coco_data["images"]], default=0) + 1
            
            # Add image info
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height,
                "file_path": image_path  # Add full path for Roboflow compatibility
            })
            
            # Get next annotation ID
            annotation_id = max([ann["id"] for ann in coco_data["annotations"]], default=0) + 1
            
            # Add annotations
            if results.boxes is not None:
                for box in results.boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convert to COCO format (top-left corner + width/height)
                    x = float(x1)
                    y = float(y1)
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    
                    # Get class ID and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Add annotation in COCO format
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,  # COCO uses 1-based indexing
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "confidence": confidence
                    })
                    annotation_id += 1
            
            # Save updated COCO data
            with open(coco_json_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
                
        except Exception as e:
            print(f"Error updating COCO annotations: {e}")
    
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
    
    def create_dataset_from_labels(self):
        """Create a dataset in COCO format from temporary annotations"""
        try:
            # Check if we have temporary annotations
            if not self.temp_annotations:
                messagebox.showwarning("No Annotations", "No temporary annotations found. Please run 'Start Labeling' first.")
                self.status_var.set("❌ No annotations to save - Run 'Start Labeling' first")
                return
            
            # Update status
            self.status_var.set("🔄 Preparing to create dataset from temporary annotations...")
            
            # Get dataset information
            dataset_name = ""
            if self.dataset_type_var.get() == "new":
                dataset_name = self.dataset_name_var.get().strip()
                if not dataset_name:
                    messagebox.showerror("Error", "Please enter a dataset name")
                    self.status_var.set("❌ Dataset creation failed - No name provided")
                    return
            else:
                selected_dataset = self.existing_dataset_var.get()
                if not selected_dataset:
                    messagebox.showerror("Error", "Please select an existing dataset")
                    self.status_var.set("❌ Dataset creation failed - No dataset selected")
                    return
                # Extract dataset name from "name (count images)" format
                dataset_name = selected_dataset.split(" (")[0]
            
            # Create dataset directory structure
            self.status_var.set(f"🔄 Creating dataset structure: {dataset_name}...")
            if not self.create_dataset_structure(dataset_name):
                self.status_var.set("❌ Dataset creation failed - Could not create directory structure")
                return
            
            # Create progress dialog
            progress_window = tk.Toplevel(self)
            progress_window.title("Creating Dataset")
            progress_window.geometry("400x150")
            progress_window.resizable(False, False)
            progress_window.transient(self)
            progress_window.grab_set()
            
            # Center the window
            progress_window.geometry("+%d+%d" % (
                self.winfo_rootx() + 50,
                self.winfo_rooty() + 50
            ))
            
            tk.Label(progress_window, text="Creating COCO dataset from annotations...", font=('Arial', 12)).pack(pady=20)
            
            progress_bar = ttk.Progressbar(progress_window, mode='determinate', length=300)
            progress_bar.pack(pady=10)
            progress_bar['maximum'] = len(self.temp_annotations)
            
            status_label = tk.Label(progress_window, text="Initializing...")
            status_label.pack(pady=5)
            
            # Process temporary annotations and create dataset
            threading.Thread(target=self.save_temp_annotations_to_dataset, 
                           args=(progress_window, progress_bar, status_label),
                           daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create dataset: {str(e)}")
    
    def save_temp_annotations_to_dataset(self, progress_window, progress_bar, status_label):
        """Save temporary annotations to permanent dataset"""
        try:
            # Determine dataset info
            dataset_name = ""
            is_existing_dataset = self.dataset_type_var.get() == "existing"
            
            if is_existing_dataset:
                selected_dataset = self.existing_dataset_var.get()
                dataset_name = selected_dataset.split(" (")[0]
            else:
                dataset_name = self.dataset_name_var.get().strip()
            
            self.current_dataset_name = dataset_name
            
            # Get dataset path
            if self.dataset_save_location:
                self.current_dataset_path = os.path.join(self.dataset_save_location, dataset_name)
            else:
                self.current_dataset_path = os.path.join(FILE_CONFIG['dataset_base_path'], dataset_name)
            
            # Initialize COCO structure
            coco_data = {
                "info": {
                    "description": f"Dataset created from labeling session - {dataset_name}",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "contributor": "WelVision YOLO Data Labeller",
                    "date_created": datetime.now().isoformat()
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": []
            }
            
            # If adding to existing dataset, load existing COCO data
            existing_image_count = 0
            existing_annotation_count = 0
            coco_json_path = os.path.join(self.current_dataset_path, "annotations.json")
            
            if is_existing_dataset and os.path.exists(coco_json_path):
                try:
                    with open(coco_json_path, 'r') as f:
                        coco_data = json.load(f)
                    
                    # Get existing counts for ID continuation
                    existing_image_count = len(coco_data.get("images", []))
                    existing_annotation_count = len(coco_data.get("annotations", []))
                    
                    print(f"Loading existing dataset: {existing_image_count} images, {existing_annotation_count} annotations")
                    
                except Exception as e:
                    print(f"Error loading existing COCO data: {e}")
                    # Continue with empty structure if loading fails
            
            # Ensure categories exist for all classes found in temp annotations
            existing_categories = {cat['name']: cat['id'] for cat in coco_data.get("categories", [])}
            
            # Find all unique classes in temp annotations
            all_classes = set()
            for temp_data in self.temp_annotations.values():
                for annotation in temp_data['annotations']:
                    all_classes.add(annotation['class_name'])
            
            # Add any missing categories
            next_category_id = max([cat['id'] for cat in coco_data.get("categories", [])], default=0) + 1
            for class_name in all_classes:
                if class_name not in existing_categories:
                    coco_data["categories"].append({
                        "id": next_category_id,
                        "name": class_name,
                        "supercategory": "object"
                    })
                    existing_categories[class_name] = next_category_id
                    next_category_id += 1
            
            # Start IDs after existing ones
            image_id = existing_image_count + 1
            annotation_id = existing_annotation_count + 1
            saved_count = 0
            
            # Process each temporarily annotated image
            for i, (image_path, temp_data) in enumerate(self.temp_annotations.items()):
                # Update progress
                self.after(0, lambda: status_label.config(text=f"Processing {os.path.basename(image_path)}..."))
                self.after(0, lambda: progress_bar.config(value=i))
                
                try:
                    # Copy image to dataset
                    image_filename = temp_data['filename']
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
                    
                    # Add image info to COCO
                    coco_data["images"].append({
                        "id": image_id,
                        "width": temp_data['image_width'],
                        "height": temp_data['image_height'],
                        "file_name": image_filename,
                        "path": dest_image_path
                    })
                    
                    # Add annotations to COCO
                    for annotation in temp_data['annotations']:
                        x1, y1, x2, y2 = annotation['bbox']
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        # Get correct category ID for this class
                        category_id = existing_categories[annotation['class_name']]
                        
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [x1, y1, width, height],
                            "area": area,
                            "iscrowd": 0,
                            "confidence": annotation['confidence']
                        })
                        annotation_id += 1
                    
                    image_id += 1
                    saved_count += 1
                    
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue
            
            # Save COCO annotations
            coco_json_path = os.path.join(self.current_dataset_path, "annotations.json")
            with open(coco_json_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            # Update dataset image count in database
            total_images = existing_image_count + saved_count
            self.db_manager.update_dataset_image_count(self.current_dataset_name, total_images)
            
            # Clear temporary annotations
            self.temp_annotations = {}
            self.is_labeling_session_active = False
            
            # Prepare success message
            is_existing = self.dataset_type_var.get() == "existing"
            if is_existing:
                action_text = "Images added to existing dataset successfully!"
                detail_text = f"✅ New images added: {saved_count}\n📊 Total images in dataset: {total_images}"
            else:
                action_text = "Dataset created successfully!"
                detail_text = f"✅ Images saved: {saved_count}"
            
            # Update status and close progress
            self.after(0, lambda: progress_window.destroy())
            self.after(0, lambda: messagebox.showinfo("Dataset Operation Complete", 
                                                     f"{action_text}\n\n"
                                                     f"Dataset: {self.current_dataset_name}\n"
                                                     f"{detail_text}\n"
                                                     f"📁 Location: {self.current_dataset_path}\n\n"
                                                     f"Temporary annotations have been cleared."))
            
            # Refresh datasets list
            self.after(0, self.refresh_all_datasets)
            self.after(0, self.update_image_previews)  # Update preview to remove annotations
            
        except Exception as e:
            self.after(0, lambda: progress_window.destroy())
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to save dataset: {str(e)}"))
            print(f"Error saving temporary annotations: {e}")
            progress_bar['maximum'] = len(image_files)
            annotation_id = 1
            
            for idx, image_file in enumerate(image_files):
                image_path = os.path.join(images_dir, image_file)
                
                # Update progress
                progress_bar['value'] = idx + 1
                status_label.config(text=f"Processing: {image_file}")
                progress_window.update()
                
                try:
                    # Get image dimensions
                    with Image.open(image_path) as img:
                        width, height = img.size
                    
                    # Add image to COCO
                    image_id = idx + 1
                    coco_data["images"].append({
                        "id": image_id,
                        "width": width,
                        "height": height,
                        "file_name": image_file,
                        "file_path": image_path
                    })
                    
                    # Get annotations for this image
                    cursor.execute("""
                        SELECT class_name, x, y, width, height, confidence 
                        FROM annotations 
                        WHERE dataset_name = %s AND image_filename = %s
                    """, (self.current_dataset_name, image_file))
                    
                    annotations = cursor.fetchall()
                    
                    # Add annotations to COCO
                    for ann in annotations:
                        class_name, x, y, w, h, confidence = ann
                        
                        if class_name in class_to_id:
                            # Convert relative coordinates to absolute
                            abs_x = x * width
                            abs_y = y * height
                            abs_w = w * width
                            abs_h = h * height
                            
                            coco_data["annotations"].append({
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": class_to_id[class_name],
                                "bbox": [abs_x, abs_y, abs_w, abs_h],
                                "area": abs_w * abs_h,
                                "iscrowd": 0,
                                "confidence": confidence
                            })
                            annotation_id += 1
                    
                except Exception as e:
                    print(f"Error processing {image_file}: {str(e)}")
                    continue
            
            # Save COCO file
            annotations_file = os.path.join(dataset_path, 'annotations.json')
            
            try:
                with open(annotations_file, 'w') as f:
                    json.dump(coco_data, f, indent=2)
                
                progress_window.destroy()
                
                # Show success message
                total_images = len(coco_data["images"])
                total_annotations = len(coco_data["annotations"])
                total_categories = len(coco_data["categories"])
                
                messagebox.showinfo(
                    "Dataset Created Successfully!", 
                    f"COCO dataset created successfully!\n\n"
                    f"📊 Statistics:\n"
                    f"   • Images: {total_images}\n"
                    f"   • Annotations: {total_annotations}\n"
                    f"   • Categories: {total_categories}\n\n"
                    f"📁 Location: {annotations_file}\n\n"
                    f"✅ Dataset is ready for preview and Roboflow upload!"
                )
                
                # Refresh all dataset dropdowns
                self.refresh_all_datasets()
                
            except Exception as e:
                progress_window.destroy()
                messagebox.showerror("Save Error", f"Error saving COCO file: {str(e)}")
                return
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Error", f"Error creating dataset: {str(e)}")

    def export_to_coco(self):
        """Validate and display COCO JSON format (dataset is already in COCO format)"""
        if not self.current_dataset_name:
            messagebox.showerror("Error", "Please create a dataset first before exporting")
            return
        
        try:
            # Get dataset path
            if self.dataset_save_location:
                dataset_path = os.path.join(self.dataset_save_location, self.current_dataset_name)
            else:
                dataset_path = os.path.join(FILE_CONFIG['dataset_base_path'], self.current_dataset_name)
            
            if not os.path.exists(dataset_path):
                messagebox.showerror("Error", f"Dataset path not found: {dataset_path}")
                return
            
            # Check if COCO annotations file exists
            coco_json_path = os.path.join(dataset_path, "annotations.json")
            
            if not os.path.exists(coco_json_path):
                messagebox.showwarning("No Annotations", "No annotations found. Please add some images and annotations first.")
                return
            
            self.status_var.set("🔄 Validating COCO format...")
            self.update()
            
            # Load and validate COCO data
            with open(coco_json_path, 'r') as f:
                coco_data = json.load(f)
            
            # Show success message with statistics
            num_images = len(coco_data.get("images", []))
            num_annotations = len(coco_data.get("annotations", []))
            num_categories = len(coco_data.get("categories", []))
            
            categories = coco_data.get("categories", [])
            class_names = [cat.get('name', 'Unknown') for cat in categories]
            
            success_msg = f"""COCO JSON dataset is ready!
            
📊 Dataset Statistics:
• Images: {num_images}
• Annotations: {num_annotations}
• Categories: {num_categories}
• Classes: {', '.join(class_names) if class_names else 'None'}

📁 File location: {coco_json_path}

This dataset is ready for upload to Roboflow!"""
            
            messagebox.showinfo("COCO Dataset Ready", success_msg)
            self.status_var.set(f"✅ COCO dataset validated: {num_images} images, {num_annotations} annotations")
            
            # Refresh all dataset dropdowns
            self.refresh_all_datasets()
            
            # Ask if user wants to open the folder
            if messagebox.askyesno("Open Folder", "Would you like to open the dataset folder?"):
                import subprocess
                subprocess.Popen(f'explorer "{dataset_path}"')
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid COCO JSON format: {str(e)}"
            messagebox.showerror("Export Error", error_msg)
            self.status_var.set("❌ COCO validation failed")
        except Exception as e:
            error_msg = f"Error validating COCO format: {str(e)}"
            messagebox.showerror("Export Error", error_msg)
            self.status_var.set("❌ COCO validation failed")
    
    def clear_all(self):
        """Clear all uploaded images and reset interface"""
        # Clear main image list
        self.uploaded_images = []
        
        # Clear temporary annotations
        self.temp_annotations = {}
        self.is_labeling_session_active = False
        self.temp_session_name = ""
        
        # Reset current image state variables
        self.current_image_index = 0
        self.current_main_image = None
        self.current_labeled_image = None
        self.current_selected_path = None
        self.current_detections = None
        
        # Clear photo references
        if hasattr(self, 'current_preview_photo'):
            self.current_preview_photo = None
        if hasattr(self, 'current_display_image'):
            self.current_display_image = None
        
        # Reset info variables (hidden from user)
        if hasattr(self, 'image_info_var') and self.image_info_var:
            self.image_info_var.set("")
        if hasattr(self, 'upload_info_var') and self.upload_info_var:
            self.upload_info_var.set("")
        if hasattr(self, 'preview_info_var'):
            self.preview_info_var.set("")
        if hasattr(self, 'main_image_info_var'):
            self.main_image_info_var.set("")
        
        # Update status panel - this is the main visible status indicator
        self.status_var.set(f"🧹 Cleared all images and annotations - Ready for new upload")
        
        # Reset dataset variables
        self.dataset_name_var.set("")
        self.dataset_save_location = ""
        self.location_var.set("datasets (default)")
        self.location_info_var.set("Dataset save location: datasets (default)")
        if hasattr(self, 'existing_dataset_var'):
            self.existing_dataset_var.set("")
        
        # Clear all image canvases
        if hasattr(self, 'clear_canvases'):
            self.clear_canvases()
        
        # Update image previews to reflect cleared state
        self.update_image_previews()
    
    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.destroy()
    
    def test_api_and_load_projects(self):
        """Step 1: Test API key and load projects directly"""
        try:
            api_key = self.rf_api_key_var.get().strip()
            if not api_key:
                messagebox.showerror("Error", "Please enter your Roboflow API key!")
                return
            
            self.rf_api_status_var.set("🟡 Testing API key...")
            self.rf_test_api_btn.config(state="disabled", text="⏳ Testing...")
            self.update()
            
            # Test API connection and load projects directly
            try:
                from roboflow import Roboflow
                self.roboflow_instance = Roboflow(api_key=api_key)
                
                # Get the default workspace for this API key
                self.rf_api_status_var.set("🟡 Loading projects...")
                self.update()
                
                workspace = self.roboflow_instance.workspace()
                self.rf_workspace_name = getattr(workspace, 'url', getattr(workspace, 'id', 'default'))
                workspace_display_name = getattr(workspace, 'name', self.rf_workspace_name)
                
                self.log_rf_status(f"Connected to workspace: {workspace_display_name}")
                
                # Load projects directly
                projects = []
                try:
                    # Use the project_list property we discovered!
                    self.log_rf_status(f"🔍 Loading projects from workspace...")
                    project_list = workspace.project_list
                    
                    for project_info in project_list:
                        project_id = project_info.get('id', '')
                        project_name = project_info.get('name', '')
                        
                        # Extract just the project slug from the full ID
                        if '/' in project_id:
                            project_slug = project_id.split('/')[-1]
                        else:
                            project_slug = project_id
                        
                        if project_slug:
                            projects.append(project_slug)
                            self.log_rf_status(f"Found project: {project_name} (slug: {project_slug})")
                    
                    self.log_rf_status(f"Total projects discovered: {len(projects)}")
                    
                except Exception as proj_error:
                    self.log_rf_status(f"Project discovery failed: {proj_error}")
                    # Fallback to known project
                    projects = ["chatter-scratch-damage-2"]
                
                # Update project dropdown
                self.rf_project_dropdown['values'] = projects
                self.rf_project_dropdown.config(state="readonly")
                
                if projects:
                    self.rf_project_dropdown.set(projects[0])
                    self.rf_project_status_var.set(f"✅ Found {len(projects)} project(s). Select one to continue.")
                    self.log_rf_status(f"Projects loaded: {', '.join(projects)}")
                    
                    # Enable project creation
                    self.rf_create_project_btn.config(state="normal")
                    self.rf_create_project_status_var.set("✨ Ready to create new projects")
                else:
                    self.rf_project_status_var.set("⚠️ No projects found")
                    # Still enable project creation even if no projects found
                    self.rf_create_project_btn.config(state="normal")
                    self.rf_create_project_status_var.set("✨ Ready to create your first project")
                
                self.rf_api_status_var.set(f"🟢 API valid! {len(projects)} projects loaded.")
                self.rf_test_api_btn.config(state="normal", text="✅ Projects Loaded")
                
                # Auto-select first project
                if projects:
                    self.on_project_selected()
                
            except Exception as e:
                self.rf_api_status_var.set(f"🔴 API test failed: {str(e)}")
                self.rf_test_api_btn.config(state="normal", text="🔍 Load Projects")
                self.log_rf_status(f"API test failed: {str(e)}")
                messagebox.showerror("API Error", f"Failed to connect with API key:\n{str(e)}")
                
        except Exception as e:
            self.rf_api_status_var.set(f"🔴 Error: {str(e)}")
            self.rf_test_api_btn.config(state="normal", text="🔍 Load Projects")
            self.log_rf_status(f"Error: {str(e)}")
    
    def on_workspace_selected(self, event=None):
        """Step 2: Load projects for selected workspace (legacy function - now using default workspace)"""
        try:
            # Since we use the default workspace automatically, this function is not needed
            # but kept for compatibility
            if not hasattr(self, 'roboflow_instance'):
                return
            
            self.log_rf_status("Using default workspace (workspace selection not needed)")
            
            # Projects are already loaded in test_api_and_load_projects
            return
            
        except Exception as e:
            error_msg = f"Error in workspace selection: {str(e)}"
            self.log_rf_status(error_msg)
    
    def create_roboflow_project(self):
        """Create a new project in Roboflow workspace using REST API"""
        try:
            # Validate inputs
            project_name = self.rf_new_project_name_var.get().strip()
            if not project_name:
                messagebox.showerror("Error", "Please enter a project name!")
                return
            
            project_type = self.rf_new_project_type_var.get()
            license_type = self.rf_new_project_license_var.get()
            api_key = self.rf_api_key_var.get().strip()
            
            if not api_key:
                messagebox.showerror("Error", "Please enter your API key first!")
                return
            
            # Update UI to show creation in progress
            self.rf_create_project_btn.config(state="disabled", text="⏳ Creating...")
            self.rf_create_project_status_var.set("🟡 Creating new project...")
            self.update()
            
            self.log_rf_status(f"🆕 Creating new project: {project_name}")
            self.log_rf_status(f"   Type: {project_type}")
            self.log_rf_status(f"   License: {license_type}")
            self.log_rf_status(f"   Using Roboflow REST API...")
            
            # Get workspace info first
            try:
                import requests
                
                # Get workspace information
                self.log_rf_status("🔍 Getting workspace information...")
                workspace_response = requests.get(f"https://api.roboflow.com/?api_key={api_key}")
                
                if workspace_response.status_code != 200:
                    raise Exception(f"Failed to get workspace info: {workspace_response.status_code}")
                
                workspace_data = workspace_response.json()
                workspace_id = workspace_data.get('workspace', 'default')
                self.log_rf_status(f"✅ Workspace ID: {workspace_id}")
                
                # Create annotation name (alphanumeric only as per docs)
                import re
                annotation_name = re.sub(r'[^a-zA-Z0-9]', '', project_name.lower())[:20]  # Max 20 chars, alphanumeric only
                if not annotation_name:
                    annotation_name = "annotation1"
                
                self.log_rf_status(f"📝 Annotation group: {annotation_name}")
                
                # Prepare project creation payload according to documentation
                payload = {
                    "name": project_name,
                    "type": project_type,
                    "annotation": annotation_name
                }
                
                # Add license if not private (as per docs requirement)
                if license_type.lower() != "private":
                    payload["license"] = license_type
                
                self.log_rf_status(f"📦 Payload: {payload}")
                
                # Make POST request to create project
                create_url = f"https://api.roboflow.com/{workspace_id}/projects"
                headers = {
                    'Content-Type': 'application/json'
                }
                params = {
                    'api_key': api_key
                }
                
                self.log_rf_status(f"🌐 POST {create_url}")
                create_response = requests.post(
                    create_url,
                    json=payload,
                    headers=headers,
                    params=params
                )
                
                self.log_rf_status(f"📋 Response Status: {create_response.status_code}")
                self.log_rf_status(f"📋 Response Text: {create_response.text}")
                
                if create_response.status_code == 200 or create_response.status_code == 201:
                    # Success!
                    response_data = create_response.json()
                    self.log_rf_status(f"✅ Project created successfully!")
                    self.log_rf_status(f"📋 Response Data: {response_data}")
                    
                    # Extract project ID/slug
                    project_id = response_data.get('id', '')
                    if '/' in project_id:
                        project_slug = project_id.split('/')[-1]
                    else:
                        project_slug = project_id
                    
                    if not project_slug:
                        # Fallback: create slug from name
                        project_slug = project_name.lower().replace(' ', '-').replace('_', '-')
                    
                    self.log_rf_status(f"📁 Project slug: {project_slug}")
                    
                    # Update the project dropdown with the new project
                    current_projects = list(self.rf_project_dropdown['values'])
                    if project_slug not in current_projects:
                        current_projects.append(project_slug)
                        self.rf_project_dropdown['values'] = current_projects
                    
                    # Select the newly created project
                    self.rf_project_dropdown.set(project_slug)
                    self.rf_project_var.set(project_slug)
                    
                    # Update status
                    self.rf_create_project_status_var.set(f"✅ Project '{project_name}' created successfully!")
                    self.rf_project_status_var.set(f"✅ Using newly created project: {project_name}")
                    
                    # Clear the input fields
                    self.rf_new_project_name_var.set("")
                    
                    # Show success message
                    messagebox.showinfo("Success", 
                                      f"Project '{project_name}' created successfully!\n\n"
                                      f"Type: {project_type}\n"
                                      f"License: {license_type}\n"
                                      f"Project ID: {project_id}\n"
                                      f"Annotation Group: {annotation_name}\n\n"
                                      f"The project has been automatically selected for upload.")
                    
                    # Refresh project list and trigger selection
                    try:
                        self.test_api_and_load_projects()
                    except Exception as refresh_error:
                        self.log_rf_status(f"⚠️ Could not refresh project list: {refresh_error}")
                    
                    # Trigger project selection
                    self.on_project_selected()
                    
                else:
                    # Failed
                    try:
                        error_data = create_response.json()
                        error_message = error_data.get('error', {}).get('message', f"HTTP {create_response.status_code}")
                    except:
                        error_message = f"HTTP {create_response.status_code}: {create_response.text}"
                    
                    self.log_rf_status(f"❌ Project creation failed: {error_message}")
                    
                    # Provide helpful error messages
                    if create_response.status_code == 400:
                        suggestion = "Check project name (must be unique) and parameters."
                    elif create_response.status_code == 401:
                        suggestion = "Check your API key permissions."
                    elif create_response.status_code == 403:
                        suggestion = "You may not have permission to create projects in this workspace."
                    elif create_response.status_code == 409:
                        suggestion = "A project with this name already exists. Try a different name."
                    else:
                        suggestion = "Please check your internet connection and try again."
                    
                    messagebox.showerror("Project Creation Failed", 
                                       f"Failed to create project '{project_name}':\n\n"
                                       f"Error: {error_message}\n\n"
                                       f"Suggestion: {suggestion}")
                    
                    self.rf_create_project_status_var.set(f"❌ Creation failed: {error_message}")
                    
            except requests.exceptions.RequestException as req_error:
                error_msg = f"Network error: {str(req_error)}"
                self.log_rf_status(f"❌ {error_msg}")
                self.rf_create_project_status_var.set(f"❌ Network error")
                messagebox.showerror("Network Error", f"Network error during project creation:\n{error_msg}")
                
            except Exception as api_error:
                error_msg = f"API error: {str(api_error)}"
                self.log_rf_status(f"❌ {error_msg}")
                self.rf_create_project_status_var.set(f"❌ API error")
                messagebox.showerror("API Error", f"API error during project creation:\n{error_msg}")
                
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.log_rf_status(f"❌ {error_msg}")
            self.rf_create_project_status_var.set(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", error_msg)
            
        finally:
            # Re-enable the create button
            self.rf_create_project_btn.config(state="normal", text="🆕 Create New Project")
    
    def on_project_selected(self, event=None):
        """Step 2: Handle project selection"""
        try:
            project_name = self.rf_project_var.get()
            
            if not project_name or not hasattr(self, 'roboflow_instance'):
                return
            
            self.rf_project_status_var.set("🟡 Validating project...")
            self.update()
            
            self.log_rf_status(f"Selected project: {project_name}")
            
            # Validate project access
            try:
                workspace = self.roboflow_instance.workspace()
                
                # Try multiple methods to access the project
                project = None
                access_error = None
                
                # Method 1: Try direct project access
                try:
                    project = workspace.project(project_name)
                    self.log_rf_status(f"✅ Direct project access successful")
                except Exception as direct_error:
                    self.log_rf_status(f"⚠️ Direct access failed: {direct_error}")
                    access_error = direct_error
                
                # Method 2: If direct access fails, try to find the project in the list
                if not project:
                    try:
                        self.log_rf_status(f"🔍 Searching for project in workspace...")
                        projects_list = workspace.list_projects()
                        
                        for proj in projects_list:
                            proj_slug = getattr(proj, 'slug', getattr(proj, 'id', str(proj)))
                            proj_name = getattr(proj, 'name', proj_slug)
                            
                            self.log_rf_status(f"   Found: {proj_name} (slug: {proj_slug})")
                            
                            # Try to match by slug or name
                            if proj_slug == project_name or proj_name.lower() == project_name.lower():
                                project = proj
                                self.log_rf_status(f"✅ Found matching project: {proj_name}")
                                break
                                
                    except Exception as list_error:
                        self.log_rf_status(f"⚠️ Project list access failed: {list_error}")
                
                # Method 3: Try REST API to get project info
                if not project:
                    try:
                        import requests
                        api_key = self.rf_api_key_var.get().strip()
                        workspace_response = requests.get(f"https://api.roboflow.com/?api_key={api_key}")
                        workspace_data = workspace_response.json()
                        workspace_id = workspace_data.get('workspace', 'default')
                        
                        # Try to get project via REST API
                        project_url = f"https://api.roboflow.com/{workspace_id}/{project_name}"
                        project_response = requests.get(project_url, params={"api_key": api_key})
                        
                        if project_response.status_code == 200:
                            project_data = project_response.json()
                            
                            # Create a mock project object
                            class MockProjectInfo:
                                def __init__(self, data):
                                    self.name = data.get('project', {}).get('name', project_name)
                                    self.type = data.get('project', {}).get('type', 'object-detection')
                                    self.images = data.get('project', {}).get('images', 0)
                                    self.id = project_name
                            
                            project = MockProjectInfo(project_data)
                            self.log_rf_status(f"✅ REST API project access successful")
                        else:
                            self.log_rf_status(f"⚠️ REST API access failed: {project_response.status_code}")
                            
                    except Exception as rest_error:
                        self.log_rf_status(f"⚠️ REST API project validation failed: {rest_error}")
                
                if project:
                    # Get project details
                    project_display_name = getattr(project, 'name', project_name)
                    project_type = getattr(project, 'type', 'unknown')
                    project_images = getattr(project, 'images', 0)

                    self.log_rf_status(f"Project details: {project_display_name} ({project_type}, {project_images} images)")

                    self.rf_project_status_var.set(f"✅ Project '{project_display_name}' selected successfully")

                    # Check if dataset is also selected
                    dataset_name = self.rf_dataset_var.get()
                    if dataset_name and hasattr(self, 'current_rf_dataset_path'):
                        self.rf_upload_btn.config(state="normal")
                        self.log_rf_status(f"🎯 Ready to upload '{dataset_name}' to '{project_name}'")
                    else:
                        self.rf_upload_btn.config(state="disabled")
                        self.rf_dataset_info_var.set("Select a dataset to upload to this project")
                else:
                    raise Exception(f"Could not access project '{project_name}'. Original error: {access_error}")
                    
            except Exception as project_error:
                self.rf_project_status_var.set(f"🔴 Error accessing project: {project_error}")
                self.log_rf_status(f"Project access error: {project_error}")
                self.rf_upload_btn.config(state="disabled")
                
        except Exception as e:
            self.rf_project_status_var.set(f"🔴 Error: {str(e)}")
    
    def refresh_rf_datasets(self):
        """Refresh datasets for Roboflow upload - show only datasets with COCO format"""
        try:
            datasets = []
            
            # Get datasets from database
            db_datasets = self.db_manager.get_datasets()
            for ds in db_datasets:
                # First check if dataset folder actually exists
                if not os.path.exists(ds['path']):
                    continue  # Skip datasets with non-existent paths
                    
                coco_file = os.path.join(ds['path'], "annotations.json")
                if os.path.exists(coco_file):
                    try:
                        import json
                        with open(coco_file, 'r') as f:
                            coco_data = json.load(f)
                        # Only add if valid COCO file
                        datasets.append(f"{ds['name']} (COCO)")
                    except:
                        # Skip datasets with invalid COCO files
                        continue
                # Skip datasets without COCO format
            
            # Also scan default datasets folder
            default_datasets_path = "datasets"
            if os.path.exists(default_datasets_path):
                for folder_name in os.listdir(default_datasets_path):
                    folder_path = os.path.join(default_datasets_path, folder_name)
                    if os.path.isdir(folder_path):
                        # Skip if already added from database
                        if any(item.startswith(folder_name + " ") for item in datasets):
                            continue
                            
                        coco_file = os.path.join(folder_path, "annotations.json")
                        if os.path.exists(coco_file):
                            try:
                                import json
                                with open(coco_file, 'r') as f:
                                    coco_data = json.load(f)
                                # Only add if valid COCO file
                                datasets.append(f"{folder_name} (COCO)")
                            except:
                                # Skip datasets with invalid COCO files
                                continue
                        # Skip datasets without COCO format
            
            self.rf_dataset_dropdown['values'] = datasets
            if datasets:
                self.rf_dataset_dropdown.set(datasets[0])
                self.on_rf_dataset_selected()
                self.rf_dataset_info_var.set(f"✅ Found {len(datasets)} COCO dataset(s) ready for upload")
            else:
                self.rf_dataset_info_var.set("❌ No COCO datasets found")
            
        except Exception as e:
            self.rf_dataset_info_var.set(f"❌ Error loading datasets: {e}")
            print(f"Error refreshing RF datasets: {e}")
    
    def on_rf_dataset_selected(self, event=None):
        """Step 4: Handle dataset selection"""
        try:
            selected = self.rf_dataset_var.get()
            if not selected:
                return
            
            # Extract dataset name from selection (remove status indicators)
            # Format: "dataset_name (COCO)"
            dataset_name = selected.replace(" (COCO)", "")
            
            # Find dataset path
            dataset_path = None
            db_datasets = self.db_manager.get_datasets()
            for ds in db_datasets:
                if ds['name'] == dataset_name:
                    dataset_path = ds['path']
                    break
            
            if not dataset_path:
                # Check default datasets folder
                default_path = os.path.join("datasets", dataset_name)
                if os.path.exists(default_path):
                    dataset_path = default_path
            
            if dataset_path:
                coco_file = os.path.join(dataset_path, "annotations.json")
                try:
                    # Load COCO data to show info
                    import json
                    with open(coco_file, 'r') as f:
                        coco_data = json.load(f)
                        
                        image_count = len(coco_data.get('images', []))
                        ann_count = len(coco_data.get('annotations', []))
                        categories = [cat['name'] for cat in coco_data.get('categories', [])]
                        
                        info_text = f"✅ Dataset: {dataset_name}\n"
                        info_text += f"📊 {image_count} images, {ann_count} annotations\n"
                        info_text += f"🏷️ Classes: {', '.join(categories)}"
                        
                        self.rf_dataset_info_var.set(info_text)
                        self.current_rf_dataset_path = dataset_path
                        
                        # Check if all steps are complete
                        if (hasattr(self, 'roboflow_instance') and 
                            self.rf_project_var.get()):
                            self.rf_upload_btn.config(state="normal")
                    
                except Exception as coco_error:
                    self.rf_dataset_info_var.set(f"❌ Error reading COCO file: {coco_error}")
            else:
                self.rf_dataset_info_var.set("❌ Dataset path not found")
                
        except Exception as e:
            self.rf_dataset_info_var.set(f"❌ Error: {e}")
    
    def upload_to_roboflow(self):
        """Step 4: Upload dataset to Roboflow using workspace.upload_dataset method"""
        try:
            # Validate all steps are complete
            api_key = self.rf_api_key_var.get().strip()
            project_id = self.rf_project_var.get().strip()
            dataset_name = self.rf_dataset_var.get()
            
            if not all([api_key, project_id, dataset_name]):
                messagebox.showerror("Error", "Please complete all steps before uploading!")
                return
            
            if not hasattr(self, 'current_rf_dataset_path'):
                messagebox.showerror("Error", "Please select a dataset first!")
                return
            
            # Verify dataset structure
            dataset_path = self.current_rf_dataset_path
            coco_file = os.path.join(dataset_path, "annotations.json")
            images_dir = os.path.join(dataset_path, "images")
            
            if not os.path.exists(coco_file):
                messagebox.showerror("Error", f"COCO annotations file not found: {coco_file}")
                return
                
            if not os.path.exists(images_dir):
                messagebox.showerror("Error", f"Images directory not found: {images_dir}")
                return
            
            # Check if upload is already in progress
            if hasattr(self, 'upload_in_progress') and self.upload_in_progress:
                self.log_rf_status("⚠️ Upload already in progress. Please wait...")
                return
            
            # Clear status and start upload
            self.rf_status_text.config(state=tk.NORMAL)
            self.rf_status_text.delete(1.0, tk.END)
            self.rf_status_text.config(state=tk.DISABLED)
            
            self.log_rf_status("🚀 Starting Roboflow workspace upload...")
            self.log_rf_status(f"📁 Dataset: {dataset_name}")
            self.log_rf_status(f"📦 Project: {project_id}")
            self.log_rf_status(f"� Dataset Path: {dataset_path}")
            self.log_rf_status(f"📋 Upload Method: workspace.upload_dataset()")
            self.log_rf_status(f"⚡ Workers: 12 (high performance)")
            
            # Start workspace upload in thread
            self.start_workspace_upload(dataset_path, api_key, project_id)
            
        except Exception as e:
            self.log_rf_status(f"❌ Upload error: {e}")
            messagebox.showerror("Upload Error", f"Upload preparation failed: {e}")
    
    def start_workspace_upload(self, dataset_path, api_key, project_id):
        """Start the workspace upload process in a separate thread"""
        # Disable upload button and start progress
        self.rf_upload_btn.config(state='disabled', text="🔄 Uploading...")
        self.upload_in_progress = True
        
        # Start upload thread
        self.upload_thread = threading.Thread(
            target=self.workspace_upload_worker,
            args=(dataset_path, api_key, project_id),
            daemon=True
        )
        self.upload_thread.start()
        
        # Start progress monitor
        self.monitor_workspace_upload_progress()

    def workspace_upload_worker(self, dataset_path, api_key, project_id):
        """Worker function that runs workspace upload in separate thread"""
        try:
            # Call the actual workspace upload function
            success = self.upload_dataset_using_workspace(dataset_path, api_key, project_id)
            
            # Send result back to main thread
            if hasattr(self, 'upload_queue'):
                self.upload_queue.put(('complete', success))
            else:
                # Fallback if upload_queue doesn't exist
                self.after(0, lambda: self.handle_upload_complete(success))
            
        except Exception as e:
            if hasattr(self, 'upload_queue'):
                self.upload_queue.put(('error', str(e)))
            else:
                self.after(0, lambda: self.handle_upload_error(str(e)))

    def monitor_workspace_upload_progress(self):
        """Monitor workspace upload progress and update UI"""
        try:
            # Check for messages from upload thread
            while hasattr(self, 'upload_queue'):
                try:
                    message_type, data = self.upload_queue.get_nowait()
                    
                    if message_type == 'complete':
                        self.handle_upload_complete(data)
                        return
                        
                    elif message_type == 'error':
                        self.handle_upload_error(data)
                        return
                        
                    elif message_type == 'progress':
                        self.log_rf_status(data)
                        
                except queue.Empty:
                    break
                    
        except Exception as e:
            self.log_rf_status(f"⚠️ Progress monitor error: {e}")
        
        # Continue monitoring if upload is still in progress
        if hasattr(self, 'upload_in_progress') and self.upload_in_progress:
            self.after(1000, self.monitor_workspace_upload_progress)

    def handle_upload_complete(self, success):
        """Handle upload completion"""
        self.upload_in_progress = False
        self.rf_upload_btn.config(state='normal', text="🚀 Upload Dataset to Roboflow")
        
        if success:
            self.log_rf_status("🎉 Dataset upload completed successfully!")
            messagebox.showinfo("Upload Complete", 
                              "Dataset uploaded successfully to Roboflow!\n\n"
                              "You can now view and manage your dataset in the Roboflow dashboard.")
        else:
            self.log_rf_status("❌ Dataset upload failed or completed with errors")
            messagebox.showerror("Upload Failed", 
                                "Dataset upload failed or completed with errors.\n"
                                "Check the status log for details.")

    def handle_upload_error(self, error_msg):
        """Handle upload error"""
        self.upload_in_progress = False
        self.rf_upload_btn.config(state='normal', text="🚀 Upload Dataset to Roboflow")
        self.log_rf_status(f"❌ Upload error: {error_msg}")
        messagebox.showerror("Upload Error", f"Upload failed: {error_msg}")

    def upload_dataset_using_workspace(self, dataset_path, api_key, project_id):
        """Upload dataset using workspace.upload_dataset method (efficient bulk upload)"""
        try:
            # Import required modules
            try:
                import roboflow
                if hasattr(self, 'upload_queue'):
                    self.upload_queue.put(('progress', "✅ Roboflow library loaded"))
                else:
                    self.after(0, lambda: self.log_rf_status("✅ Roboflow library loaded"))
            except ImportError:
                error_msg = "❌ Roboflow library not found. Please install: pip install roboflow"
                if hasattr(self, 'upload_queue'):
                    self.upload_queue.put(('progress', error_msg))
                else:
                    self.after(0, lambda: self.log_rf_status(error_msg))
                return False
            
            # Initialize Roboflow
            progress_msg = "🔄 Initializing Roboflow connection..."
            if hasattr(self, 'upload_queue'):
                self.upload_queue.put(('progress', progress_msg))
            else:
                self.after(0, lambda: self.log_rf_status(progress_msg))
            
            rf = roboflow.Roboflow(api_key=api_key)
            
            # Get workspace
            workspace = rf.workspace()
            workspace_name = getattr(workspace, 'name', 'default')
            
            success_msg = f"✅ Connected to workspace: {workspace_name}"
            if hasattr(self, 'upload_queue'):
                self.upload_queue.put(('progress', success_msg))
            else:
                self.after(0, lambda: self.log_rf_status(success_msg))
            
            # Prepare upload parameters
            upload_msg = f"🚀 Starting workspace.upload_dataset()..."
            if hasattr(self, 'upload_queue'):
                self.upload_queue.put(('progress', upload_msg))
            else:
                self.after(0, lambda: self.log_rf_status(upload_msg))
            
            params_msg = f"📦 Project ID: {project_id}"
            params_msg2 = f"📂 Dataset Path: {dataset_path}"
            params_msg3 = f"⚡ Workers: 12 (high performance)"
            params_msg4 = f"📋 License: MIT"
            params_msg5 = f"🎯 Type: object-detection"
            
            if hasattr(self, 'upload_queue'):
                self.upload_queue.put(('progress', params_msg))
                self.upload_queue.put(('progress', params_msg2))
                self.upload_queue.put(('progress', params_msg3))
                self.upload_queue.put(('progress', params_msg4))
                self.upload_queue.put(('progress', params_msg5))
                self.upload_queue.put(('progress', "-" * 50))
            else:
                self.after(0, lambda: self.log_rf_status(params_msg))
                self.after(0, lambda: self.log_rf_status(params_msg2))
                self.after(0, lambda: self.log_rf_status(params_msg3))
                self.after(0, lambda: self.log_rf_status(params_msg4))
                self.after(0, lambda: self.log_rf_status(params_msg5))
                self.after(0, lambda: self.log_rf_status("-" * 50))
            
            # Upload dataset using workspace method (bulk upload - much faster!)
            result = workspace.upload_dataset(
                dataset_path,           # Dataset directory path
                project_id,             # Project ID (will create or use existing)
                num_workers=12,         # High performance with 12 workers
                project_license="MIT",  # License type
                project_type="object-detection",  # Project type
                batch_name=None,        # No specific batch name
                num_retries=0          # No retries for faster upload
            )
            
            # Log completion
            completion_msg = "🎉 Workspace upload completed!"
            result_msg = f"📊 Upload result: {result}"
            final_msg = f"📁 Dataset uploaded to project: {project_id}"
            dashboard_msg = f"🌐 Check your Roboflow dashboard to view the uploaded data"
            
            if hasattr(self, 'upload_queue'):
                self.upload_queue.put(('progress', "-" * 50))
                self.upload_queue.put(('progress', completion_msg))
                self.upload_queue.put(('progress', result_msg))
                self.upload_queue.put(('progress', final_msg))
                self.upload_queue.put(('progress', dashboard_msg))
            else:
                self.after(0, lambda: self.log_rf_status("-" * 50))
                self.after(0, lambda: self.log_rf_status(completion_msg))
                self.after(0, lambda: self.log_rf_status(result_msg))
                self.after(0, lambda: self.log_rf_status(final_msg))
                self.after(0, lambda: self.log_rf_status(dashboard_msg))
            
            return True
            
        except Exception as e:
            error_msg = f"❌ Workspace upload failed: {str(e)}"
            if hasattr(self, 'upload_queue'):
                self.upload_queue.put(('progress', error_msg))
            else:
                self.after(0, lambda: self.log_rf_status(error_msg))
            return False
        self.rf_upload_btn.config(state='disabled', text="🔄 Uploading...")
        self.upload_in_progress = True
        
        # Start upload thread
        self.upload_thread = threading.Thread(
            target=self.threaded_upload_worker,
            args=(coco_file, api_key, project_id),
            daemon=True
        )
        self.upload_thread.start()
        
        # Start progress monitor
        self.monitor_upload_progress()

    def threaded_upload_worker(self, coco_file, api_key, project_id):
        """Worker function that runs in separate thread"""
        try:
            # Call the actual upload function
            success = self.upload_dataset_to_roboflow_threaded(coco_file, api_key, project_id)
            
            # Send result back to main thread
            self.upload_queue.put(('complete', success))
            
        except Exception as e:
            self.upload_queue.put(('error', str(e)))

    def monitor_upload_progress(self):
        """Monitor upload progress and update UI"""
        try:
            # Check for messages from upload thread
            while True:
                message_type, data = self.upload_queue.get_nowait()
                
                if message_type == 'complete':
                    # Upload finished
                    self.upload_in_progress = False
                    self.rf_upload_btn.config(state='normal', text="🚀 Upload as Predictions to Roboflow")
                    
                    if data:  # Success
                        self.log_rf_status("🎉 Threaded upload completed successfully!")
                    else:  # Failed
                        self.log_rf_status("❌ Upload failed. Check logs above.")
                    return
                    
                elif message_type == 'error':
                    # Upload error
                    self.upload_in_progress = False
                    self.rf_upload_btn.config(state='normal', text="🚀 Upload as Predictions to Roboflow")
                    self.log_rf_status(f"❌ Upload error: {data}")
                    return
                    
                elif message_type == 'progress':
                    # Progress update
                    self.log_rf_status(data)
                    
        except queue.Empty:
            # No messages yet, check again in 100ms
            if self.upload_in_progress:
                self.after(100, self.monitor_upload_progress)

    def upload_dataset_to_roboflow_threaded(self, coco_file, api_key, project_id):
        """Optimized threaded upload with progress tracking"""
        try:
            # Import required modules
            try:
                from roboflow import Roboflow
                import tempfile
                self.upload_queue.put(('progress', "✅ Roboflow library loaded"))
            except ImportError:
                self.upload_queue.put(('progress', "❌ Roboflow library not found. Please install: pip install roboflow"))
                return False
            
            # Initialize Roboflow
            self.upload_queue.put(('progress', "🔄 Initializing Roboflow connection..."))
            rf = Roboflow(api_key=api_key)
            project = rf.workspace().project(project_id)
            self.upload_queue.put(('progress', "✅ Connected to Roboflow"))
            self.upload_queue.put(('progress', "📋 Upload Mode: Predictions (images will go to 'Unassigned')"))
            
            # Load COCO data
            self.upload_queue.put(('progress', "📄 Loading COCO annotations..."))
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            images_data = coco_data.get('images', [])
            annotations_data = coco_data.get('annotations', [])
            categories_data = coco_data.get('categories', [])
            info_data = coco_data.get('info', {})
            
            total_images = len(images_data)
            self.upload_queue.put(('progress', f"📂 Starting threaded upload of {total_images} images..."))
            self.upload_queue.put(('progress', f"📊 Total annotations: {len(annotations_data)}"))
            self.upload_queue.put(('progress', f"📈 Categories: {len(categories_data)}"))
            self.upload_queue.put(('progress', "-" * 50))
            
            if not images_data:
                self.upload_queue.put(('progress', "❌ No images found in COCO file"))
                return False
            
            # Upload images with optimized approach
            successful_uploads = 0
            failed_uploads = 0
            
            for i, image_info in enumerate(images_data, 1):
                try:
                    # Get image path
                    if 'file_path' in image_info and os.path.exists(image_info['file_path']):
                        image_path = image_info['file_path']
                    else:
                        # Fallback: construct from images folder
                        dataset_dir = os.path.dirname(coco_file)
                        images_dir = os.path.join(dataset_dir, "images")
                        image_path = os.path.join(images_dir, image_info['file_name'])
                    
                    filename = image_info['file_name']
                    
                    # Progress update every 5 images or at the end
                    if i % 5 == 0 or i == total_images:
                        progress_pct = (i / total_images) * 100
                        self.upload_queue.put(('progress', f"📈 Progress: {progress_pct:.1f}% ({i}/{total_images})"))
                    
                    if not os.path.exists(image_path):
                        self.upload_queue.put(('progress', f"❌ Missing: {filename}"))
                        failed_uploads += 1
                        continue
                    
                    # Get annotations for this specific image only
                    image_annotations = [
                        ann for ann in annotations_data 
                        if ann['image_id'] == image_info['id']
                    ]
                    
                    # Create minimal COCO file for this image only
                    mini_coco = {
                        "info": info_data,
                        "images": [image_info],
                        "annotations": image_annotations,
                        "categories": categories_data
                    }
                    
                    # Create temporary file with minimal COCO data
                    temp_coco_path = None
                    try:
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                            json.dump(mini_coco, temp_file)
                            temp_coco_path = temp_file.name
                        
                        # Upload with minimal COCO file (much faster!)
                        response = project.single_upload(
                            image_path=image_path,
                            annotation_path=temp_coco_path,  # Small file instead of huge one!
                            is_prediction=True,  # Always upload as predictions
                            num_retry_uploads=1  # Reduced retries for speed
                        )
                        
                        successful_uploads += 1
                        
                        # Log every 10th success to avoid spam
                        if i % 10 == 0 or len(image_annotations) > 0:
                            ann_count = len(image_annotations)
                            self.upload_queue.put(('progress', f"✅ Uploaded: {filename} ({ann_count} annotations)"))
                        
                    except Exception as upload_error:
                        failed_uploads += 1
                        self.upload_queue.put(('progress', f"❌ Failed: {filename} - {str(upload_error)[:50]}..."))
                    
                    finally:
                        # Clean up temporary file
                        if temp_coco_path and os.path.exists(temp_coco_path):
                            try:
                                os.unlink(temp_coco_path)
                            except:
                                pass
                    
                except Exception as e:
                    failed_uploads += 1
                    self.upload_queue.put(('progress', f"❌ Error processing {image_info.get('file_name', 'unknown')}: {str(e)[:50]}..."))
            
            # Final summary
            self.upload_queue.put(('progress', "-" * 50))
            self.upload_queue.put(('progress', "🎉 Threaded Upload Complete!"))
            self.upload_queue.put(('progress', f"   ✅ Successful: {successful_uploads}"))
            self.upload_queue.put(('progress', f"   ❌ Failed: {failed_uploads}"))
            self.upload_queue.put(('progress', f"   📊 Total: {total_images}"))
            self.upload_queue.put(('progress', f"   📁 Destination: Unassigned (Predictions for Review)"))
            self.upload_queue.put(('progress', f"⚡ Upload was optimized with threading!"))
            
            return successful_uploads > 0
            
        except Exception as e:
            self.upload_queue.put(('progress', f"❌ Upload failed: {str(e)}"))
            return False
    
    def perform_roboflow_upload(self, coco_file, api_key, workspace_id, project_id):
        """Optimized Roboflow upload with minimal COCO files per image"""
        try:
            # Import required modules
            try:
                from roboflow import Roboflow
                import tempfile
                self.log_rf_status("✅ Roboflow library loaded")
            except ImportError:
                self.log_rf_status("❌ Roboflow library not found. Please install: pip install roboflow")
                self.rf_upload_btn.config(state=tk.NORMAL, text="🚀 Upload as Predictions to Roboflow")
                return
            
            # Initialize Roboflow
            self.log_rf_status("🔄 Initializing Roboflow connection...")
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(workspace_id).project(project_id)
            self.log_rf_status("✅ Connected to Roboflow")
            self.log_rf_status("📋 Upload Mode: Predictions (images will go to 'Unassigned')")
            
            # Load COCO data
            self.log_rf_status("📄 Loading COCO annotations...")
            import json
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            images_data = coco_data.get('images', [])
            annotations_data = coco_data.get('annotations', [])
            categories_data = coco_data.get('categories', [])
            info_data = coco_data.get('info', {})
            
            total_images = len(images_data)
            self.log_rf_status(f"📂 Starting optimized upload of {total_images} images...")
            self.log_rf_status(f"📊 Total annotations: {len(annotations_data)}")
            self.log_rf_status(f"📈 Categories: {len(categories_data)}")
            self.log_rf_status("-" * 50)
            
            if not images_data:
                self.log_rf_status("❌ No images found in COCO file")
                self.rf_upload_btn.config(state=tk.NORMAL, text="🚀 Upload as Predictions to Roboflow")
                return
            
            # Upload images with optimized approach
            successful_uploads = 0
            failed_uploads = 0
            
            for i, image_info in enumerate(images_data, 1):
                try:
                    # Get image path
                    if 'file_path' in image_info and os.path.exists(image_info['file_path']):
                        image_path = image_info['file_path']
                    else:
                        # Fallback: construct from images folder
                        dataset_dir = os.path.dirname(coco_file)
                        images_dir = os.path.join(dataset_dir, "images")
                        image_path = os.path.join(images_dir, image_info['file_name'])
                    
                    filename = image_info['file_name']
                    
                    # Progress update every 10 images or at the end
                    if i % 10 == 0 or i == total_images:
                        progress_pct = (i / total_images) * 100
                        self.log_rf_status(f"📈 Progress: {progress_pct:.1f}% ({i}/{total_images})")
                    
                    if not os.path.exists(image_path):
                        self.log_rf_status(f"❌ Missing: {filename}")
                        failed_uploads += 1
                        continue
                    
                    # Get annotations for this specific image only
                    image_annotations = [
                        ann for ann in annotations_data 
                        if ann['image_id'] == image_info['id']
                    ]
                    
                    # Create minimal COCO file for this image only
                    mini_coco = {
                        "info": info_data,
                        "images": [image_info],
                        "annotations": image_annotations,
                        "categories": categories_data
                    }
                    
                    # Create temporary file with minimal COCO data
                    temp_coco_path = None
                    try:
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                            json.dump(mini_coco, temp_file)
                            temp_coco_path = temp_file.name
                        
                        # Upload with minimal COCO file (much faster!)
                        response = project.single_upload(
                            image_path=image_path,
                            annotation_path=temp_coco_path,  # Small file instead of huge one!
                            is_prediction=True,  # Always upload as predictions
                            num_retry_uploads=1  # Reduced retries for speed
                        )
                        
                        successful_uploads += 1
                        
                        # Log every 5th success to avoid spam
                        if i % 5 == 0 or len(image_annotations) > 0:
                            ann_count = len(image_annotations)
                            self.log_rf_status(f"✅ Uploaded: {filename} ({ann_count} annotations)")
                        
                    except Exception as upload_error:
                        failed_uploads += 1
                        self.log_rf_status(f"❌ Failed: {filename} - {str(upload_error)[:50]}...")
                    
                    finally:
                        # Clean up temporary file
                        if temp_coco_path and os.path.exists(temp_coco_path):
                            try:
                                os.unlink(temp_coco_path)
                            except:
                                pass
                    
                except Exception as e:
                    failed_uploads += 1
                    self.log_rf_status(f"❌ Error processing {image_info.get('file_name', 'unknown')}: {str(e)[:50]}...")
            
            # Final summary
            self.log_rf_status("-" * 50)
            self.log_rf_status("🎉 Upload Complete!")
            self.log_rf_status(f"   ✅ Successful: {successful_uploads}")
            self.log_rf_status(f"   ❌ Failed: {failed_uploads}")
            self.log_rf_status(f"   📊 Total: {total_images}")
            self.log_rf_status(f"   📁 Destination: Unassigned (Predictions for Review)")
            
            if successful_uploads > 0:
                self.log_rf_status("🎉 Optimized upload completed successfully!")
                self.log_rf_status("📋 Images uploaded as predictions to 'Unassigned' section")
                messagebox.showinfo("Upload Complete", 
                    f"Successfully uploaded {successful_uploads} images to Roboflow!\n"
                    f"Failed: {failed_uploads}\n\n"
                    f"📋 Images are in 'Unassigned' section for review\n"
                    f"⚡ Upload was optimized for speed!")
            else:
                self.log_rf_status("❌ Upload failed - no images were uploaded")
                messagebox.showerror("Upload Failed", "No images were successfully uploaded")
            
        except Exception as e:
            self.log_rf_status(f"❌ Upload failed: {e}")
            messagebox.showerror("Upload Error", f"Upload failed: {e}")
        
        finally:
            # Re-enable upload button
            self.rf_upload_btn.config(state=tk.NORMAL, text="🚀 Upload as Predictions to Roboflow")

# Import simpledialog for model name input
import tkinter.simpledialog

def main():
    """Main function to run the application"""
    app = YOLOLabelerApp()
    app.mainloop()

if __name__ == "__main__":
    main() 
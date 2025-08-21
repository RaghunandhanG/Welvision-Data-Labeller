"""
WelVision YOLO Data Labeller - Simple Working Version
A comprehensive image labeling application using YOLO v8 models
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import shutil
import mysql.connector
from ultralytics import YOLO
from PIL import Image, ImageTk
import torch

class YOLOLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title('WelVision Data Labeller')
        self.root.geometry('1200x800')
        
        # Initialize variables
        self.current_model = None
        self.yolo_model = None
        self.uploaded_images = []
        self.current_image_index = 0
        self.current_dataset_path = None
        
        # Database config
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '1469',
            'database': 'welvision_db'
        }
        
        # Status variable
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        # Detect device
        self.detect_device()
        
        # Create GUI
        self.create_widgets()
        
        # Load models
        try:
            self.load_models()
        except Exception as e:
            self.status_var.set(f"Database error: {str(e)}")
    
    def detect_device(self):
        """Detect the best available device"""
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f"Using device: {self.device}")
    
    def create_widgets(self):
        """Create the GUI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Model section
        model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding="10")
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_dropdown = ttk.Combobox(model_frame, state="readonly", width=50)
        self.model_dropdown.grid(row=0, column=1, padx=5)
        ttk.Button(model_frame, text="Load Model", command=self.load_selected_model).grid(row=0, column=2, padx=5)
        
        # Add new model
        ttk.Label(model_frame, text="Add Model Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.new_model_name = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.new_model_name, width=30).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Model Path:").grid(row=2, column=0, sticky=tk.W)
        self.new_model_path = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.new_model_path, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(model_frame, text="Browse", command=self.browse_model).grid(row=2, column=2, padx=5)
        ttk.Button(model_frame, text="Add Model", command=self.add_model).grid(row=3, column=1, pady=5)
        
        # Dataset section
        dataset_frame = ttk.LabelFrame(main_frame, text="Dataset Management", padding="10")
        dataset_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(dataset_frame, text="Dataset Name:").grid(row=0, column=0, sticky=tk.W)
        self.dataset_name = tk.StringVar()
        ttk.Entry(dataset_frame, textvariable=self.dataset_name, width=30).grid(row=0, column=1, padx=5)
        ttk.Button(dataset_frame, text="Create Dataset", command=self.create_dataset).grid(row=0, column=2, padx=5)
        
        # Upload section
        upload_frame = ttk.LabelFrame(main_frame, text="Image Upload", padding="10")
        upload_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(upload_frame, text="Upload Images", command=self.upload_images).grid(row=0, column=0, padx=5)
        ttk.Button(upload_frame, text="Upload Folder", command=self.upload_folder).grid(row=0, column=1, padx=5)
        self.image_count_label = ttk.Label(upload_frame, text="No images loaded")
        self.image_count_label.grid(row=0, column=2, padx=10)
        
        # Image display
        display_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        display_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.image_canvas = tk.Canvas(display_frame, width=600, height=400, bg='white')
        self.image_canvas.grid(row=0, column=0, columnspan=3, pady=5)
        
        # Navigation
        ttk.Button(display_frame, text="Previous", command=self.prev_image).grid(row=1, column=0, padx=5)
        self.image_info_label = ttk.Label(display_frame, text="No image")
        self.image_info_label.grid(row=1, column=1, padx=10)
        ttk.Button(display_frame, text="Next", command=self.next_image).grid(row=1, column=2, padx=5)
        
        # Action buttons
        action_frame = ttk.LabelFrame(main_frame, text="Actions", padding="10")
        action_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(action_frame, text="Label All Images", command=self.label_all_images).grid(row=0, column=0, padx=5)
        ttk.Button(action_frame, text="Label Current Image", command=self.label_current_image).grid(row=0, column=1, padx=5)
        ttk.Button(action_frame, text="Clear All", command=self.clear_all).grid(row=0, column=2, padx=5)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
    
    def get_db_connection(self):
        """Get database connection"""
        try:
            return mysql.connector.connect(**self.db_config)
        except mysql.connector.Error as e:
            messagebox.showerror("Database Error", f"Could not connect to database: {e}")
            return None
    
    def load_models(self):
        """Load models from database"""
        conn = self.get_db_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM models ORDER BY name")
            models = cursor.fetchall()
            model_names = [model[0] for model in models]
            self.model_dropdown['values'] = model_names
            if model_names:
                self.model_dropdown.set(model_names[0])
            cursor.close()
            conn.close()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {e}")
    
    def browse_model(self):
        """Browse for model file"""
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            self.new_model_path.set(file_path)
    
    def add_model(self):
        """Add new model to database"""
        name = self.new_model_name.get().strip()
        path = self.new_model_path.get().strip()
        
        if not name or not path:
            messagebox.showerror("Error", "Please provide model name and path")
            return
        
        if not os.path.exists(path):
            messagebox.showerror("Error", "Model file does not exist")
            return
        
        conn = self.get_db_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO models (name, path) VALUES (%s, %s) ON DUPLICATE KEY UPDATE path = VALUES(path)", (name, path))
            conn.commit()
            cursor.close()
            conn.close()
            
            messagebox.showinfo("Success", "Model added successfully")
            self.new_model_name.set("")
            self.new_model_path.set("")
            self.load_models()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add model: {e}")
    
    def load_selected_model(self):
        """Load the selected model"""
        model_name = self.model_dropdown.get()
        if not model_name:
            messagebox.showerror("Error", "Please select a model")
            return
        
        conn = self.get_db_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT path FROM models WHERE name = %s", (model_name,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not result:
                messagebox.showerror("Error", "Model not found")
                return
            
            model_path = result[0]
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return
            
            self.status_var.set("Loading model...")
            self.root.update()
            
            self.yolo_model = YOLO(model_path)
            self.yolo_model.to(self.device)
            self.current_model = {'name': model_name, 'path': model_path}
            
            self.status_var.set(f"Model loaded: {model_name} on {self.device}")
            messagebox.showinfo("Success", f"Model '{model_name}' loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.status_var.set("Failed to load model")
    
    def create_dataset(self):
        """Create a new dataset"""
        name = self.dataset_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Please provide dataset name")
            return
        
        dataset_path = os.path.join('datasets', name)
        images_path = os.path.join(dataset_path, 'images')
        labels_path = os.path.join(dataset_path, 'labels')
        
        try:
            os.makedirs(images_path, exist_ok=True)
            os.makedirs(labels_path, exist_ok=True)
            self.current_dataset_path = dataset_path
            self.status_var.set(f"Dataset created: {name}")
            messagebox.showinfo("Success", f"Dataset '{name}' created successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create dataset: {e}")
    
    def upload_images(self):
        """Upload images"""
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_paths:
            self.uploaded_images = list(file_paths)
            self.current_image_index = 0
            self.update_image_display()
            self.image_count_label.config(text=f"{len(self.uploaded_images)} images loaded")
    
    def upload_folder(self):
        """Upload folder of images"""
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        
        if folder_path:
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            image_files = []
            
            for file in os.listdir(folder_path):
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(folder_path, file))
            
            if image_files:
                self.uploaded_images = image_files
                self.current_image_index = 0
                self.update_image_display()
                self.image_count_label.config(text=f"{len(self.uploaded_images)} images loaded")
            else:
                messagebox.showinfo("No Images", "No image files found in selected folder")
    
    def update_image_display(self):
        """Update image display"""
        if not self.uploaded_images:
            self.image_info_label.config(text="No image")
            return
        
        image_path = self.uploaded_images[self.current_image_index]
        
        try:
            image = Image.open(image_path)
            image.thumbnail((600, 400))
            self.photo = ImageTk.PhotoImage(image)
            
            self.image_canvas.delete("all")
            self.image_canvas.create_image(300, 200, image=self.photo)
            
            self.image_info_label.config(text=f"Image {self.current_image_index + 1} of {len(self.uploaded_images)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {e}")
    
    def prev_image(self):
        """Show previous image"""
        if self.uploaded_images and self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image_display()
    
    def next_image(self):
        """Show next image"""
        if self.uploaded_images and self.current_image_index < len(self.uploaded_images) - 1:
            self.current_image_index += 1
            self.update_image_display()
    
    def label_current_image(self):
        """Label current image"""
        if not self.yolo_model:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if not self.uploaded_images:
            messagebox.showerror("Error", "Please upload images first")
            return
        
        if not self.current_dataset_path:
            messagebox.showerror("Error", "Please create a dataset first")
            return
        
        image_path = self.uploaded_images[self.current_image_index]
        self.process_image(image_path)
        messagebox.showinfo("Success", "Image labeled successfully!")
    
    def label_all_images(self):
        """Label all images"""
        if not self.yolo_model:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if not self.uploaded_images:
            messagebox.showerror("Error", "Please upload images first")
            return
        
        if not self.current_dataset_path:
            messagebox.showerror("Error", "Please create a dataset first")
            return
        
        total = len(self.uploaded_images)
        processed = 0
        
        for i, image_path in enumerate(self.uploaded_images):
            try:
                self.status_var.set(f"Processing {i+1}/{total}")
                self.root.update()
                
                self.process_image(image_path)
                processed += 1
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        self.status_var.set(f"Completed: {processed}/{total} images processed")
        messagebox.showinfo("Complete", f"Processed {processed} of {total} images")
    
    def process_image(self, image_path):
        """Process single image"""
        try:
            # Run YOLO
            results = self.yolo_model(image_path, conf=0.25)
            
            # Get image size
            image = Image.open(image_path)
            img_width, img_height = image.size
            
            # Prepare labels
            labels = []
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Convert to YOLO format
                        x_center = (x1 + x2) / 2 / img_width
                        y_center = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        labels.append([cls, x_center, y_center, width, height])
            
            # Save image and labels
            image_name = os.path.basename(image_path)
            images_dir = os.path.join(self.current_dataset_path, 'images')
            labels_dir = os.path.join(self.current_dataset_path, 'labels')
            
            # Copy image
            dest_image = os.path.join(images_dir, image_name)
            shutil.copy2(image_path, dest_image)
            
            # Save labels
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)
            
            with open(label_path, 'w') as f:
                for label in labels:
                    f.write(' '.join(map(str, label)) + '\n')
            
        except Exception as e:
            raise Exception(f"Error processing {image_path}: {e}")
    
    def clear_all(self):
        """Clear all data"""
        self.uploaded_images = []
        self.current_image_index = 0
        self.image_canvas.delete("all")
        self.image_count_label.config(text="No images loaded")
        self.image_info_label.config(text="No image")
        self.status_var.set("Cleared all data")

def main():
    root = tk.Tk()
    app = YOLOLabelerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

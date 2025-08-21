import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import logging
import os
from PIL import Image, ImageTk
import tempfile
import zipfile

from config import APP_TITLE, APP_GEOMETRY, SUPPORTED_FORMATS
from database_manager import DatabaseManager
from yolo_processor import YOLOProcessor
from dataset_manager import DatasetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_labeller.log'),
        logging.StreamHandler()
    ]
)

class YOLOLabellerApp:
    """Main application class for YOLO Image Labeller."""
    
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(APP_GEOMETRY)
        
        # Initialize managers
        self.db_manager = DatabaseManager()
        self.yolo_processor = YOLOProcessor()
        self.dataset_manager = DatasetManager()
        
        # Application state
        self.selected_images = []
        self.current_dataset = None
        self.processing = False
        
        # Create GUI
        self.create_widgets()
        self.load_models()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """Create and layout all GUI widgets."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Model Selection Section
        model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding="5")
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly", width=40)
        self.model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)
        
        ttk.Button(model_frame, text="Refresh Models", command=self.load_models).grid(row=0, column=2, padx=(5, 0))
        ttk.Button(model_frame, text="Add Model", command=self.add_model_dialog).grid(row=0, column=3, padx=(5, 0))
        
        model_frame.columnconfigure(1, weight=1)
        
        # Dataset Section
        dataset_frame = ttk.LabelFrame(main_frame, text="Dataset Management", padding="5")
        dataset_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # New Dataset
        ttk.Label(dataset_frame, text="New Dataset Name:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.dataset_name_var = tk.StringVar()
        self.dataset_entry = ttk.Entry(dataset_frame, textvariable=self.dataset_name_var, width=30)
        self.dataset_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(dataset_frame, text="Create Dataset", command=self.create_dataset).grid(row=0, column=2, padx=(5, 0))
        
        # Existing Dataset
        ttk.Label(dataset_frame, text="Existing Dataset:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.existing_dataset_var = tk.StringVar()
        self.existing_dataset_combo = ttk.Combobox(dataset_frame, textvariable=self.existing_dataset_var, state="readonly", width=30)
        self.existing_dataset_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))
        
        ttk.Button(dataset_frame, text="Load Dataset", command=self.load_existing_dataset).grid(row=1, column=2, padx=(5, 0), pady=(5, 0))
        ttk.Button(dataset_frame, text="Refresh", command=self.refresh_datasets).grid(row=1, column=3, padx=(5, 0), pady=(5, 0))
        
        dataset_frame.columnconfigure(1, weight=1)
        
        # Image Upload Section
        upload_frame = ttk.LabelFrame(main_frame, text="Image Upload", padding="5")
        upload_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(upload_frame, text="Select Images", command=self.select_images).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(upload_frame, text="Select Folder", command=self.select_folder).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(upload_frame, text="Upload ZIP", command=self.upload_zip).grid(row=0, column=2, padx=(0, 5))
        
        self.images_label = ttk.Label(upload_frame, text="No images selected")
        self.images_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Processing Options
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="5")
        options_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(options_frame, text="Confidence:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.confidence_var = tk.DoubleVar(value=0.25)
        confidence_scale = ttk.Scale(options_frame, from_=0.1, to=0.9, variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        self.confidence_label = ttk.Label(options_frame, text="0.25")
        self.confidence_label.grid(row=0, column=2, padx=(5, 0))
        self.confidence_var.trace('w', self.update_confidence_label)
        
        ttk.Label(options_frame, text="IoU:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.iou_var = tk.DoubleVar(value=0.45)
        iou_scale = ttk.Scale(options_frame, from_=0.1, to=0.9, variable=self.iou_var, orient=tk.HORIZONTAL)
        iou_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        self.iou_label = ttk.Label(options_frame, text="0.45")
        self.iou_label.grid(row=1, column=2, padx=(5, 0))
        self.iou_var.trace('w', self.update_iou_label)
        
        options_frame.columnconfigure(1, weight=1)
        
        # Process Button
        self.process_button = ttk.Button(main_frame, text="Process Images", command=self.process_images, state="disabled")
        self.process_button.grid(row=4, column=0, columnspan=2, pady=(0, 10))
        
        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Status Label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.grid(row=6, column=0, columnspan=2, sticky=tk.W)
        
        # Log Text Area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)
        
        # Redirect logging to text widget
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging to display in the text widget."""
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
            
            def emit(self, record):
                msg = self.format(record)
                def append():
                    self.text_widget.configure(state='normal')
                    self.text_widget.insert(tk.END, msg + '\n')
                    self.text_widget.configure(state='disabled')
                    self.text_widget.see(tk.END)
                self.text_widget.after(0, append)
        
        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(text_handler)
    
    def load_models(self):
        """Load models from database and populate dropdown."""
        try:
            models = self.db_manager.get_all_models()
            model_names = [model['name'] for model in models]
            self.model_combo['values'] = model_names
            
            if model_names:
                self.model_combo.set(model_names[0])
                self.on_model_selected()
            
            logging.info(f"Loaded {len(models)} models from database")
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            messagebox.showerror("Error", f"Failed to load models: {e}")
    
    def on_model_selected(self, event=None):
        """Handle model selection."""
        model_name = self.model_var.get()
        if not model_name:
            return
        
        try:
            model_info = self.db_manager.get_model_by_name(model_name)
            if model_info:
                success = self.yolo_processor.load_model(model_info['path'], model_info['name'])
                if success:
                    # Update model usage statistics
                    self.db_manager.update_model_usage(model_info['id'])
                    self.status_var.set(f"Model loaded: {model_name}")
                    self.update_process_button_state()
                    logging.info(f"Successfully loaded model: {model_name}")
                else:
                    self.status_var.set("Failed to load model")
                    messagebox.showerror("Error", f"Failed to load model: {model_name}")
        except Exception as e:
            logging.error(f"Error selecting model: {e}")
            messagebox.showerror("Error", f"Error selecting model: {e}")
    
    def add_model_dialog(self):
        """Show dialog to add a new model."""
        dialog = ModelDialog(self.root, self.db_manager)
        if dialog.result:
            self.load_models()
    
    def create_dataset(self):
        """Create a new dataset."""
        dataset_name = self.dataset_name_var.get().strip()
        if not dataset_name:
            messagebox.showerror("Error", "Please enter a dataset name")
            return
        
        if not self.yolo_processor.is_model_loaded():
            messagebox.showerror("Error", "Please select and load a model first")
            return
        
        try:
            dataset_path = self.dataset_manager.create_dataset(
                dataset_name, 
                self.yolo_processor.model_name
            )
            
            if dataset_path:
                self.current_dataset = dataset_name
                self.status_var.set(f"Created dataset: {dataset_name}")
                self.update_process_button_state()
                self.refresh_datasets()
                logging.info(f"Created dataset: {dataset_name}")
            else:
                messagebox.showerror("Error", "Failed to create dataset")
        except Exception as e:
            logging.error(f"Error creating dataset: {e}")
            messagebox.showerror("Error", f"Error creating dataset: {e}")
    
    def refresh_datasets(self):
        """Refresh the existing datasets dropdown."""
        try:
            datasets = self.dataset_manager.get_existing_datasets()
            dataset_names = [dataset['name'] for dataset in datasets]
            self.existing_dataset_combo['values'] = dataset_names
            logging.info(f"Found {len(datasets)} existing datasets")
        except Exception as e:
            logging.error(f"Error refreshing datasets: {e}")
    
    def load_existing_dataset(self):
        """Load an existing dataset."""
        dataset_name = self.existing_dataset_var.get()
        if not dataset_name:
            messagebox.showerror("Error", "Please select a dataset")
            return
        
        self.current_dataset = dataset_name
        self.status_var.set(f"Loaded dataset: {dataset_name}")
        self.update_process_button_state()
        logging.info(f"Loaded existing dataset: {dataset_name}")
    
    def select_images(self):
        """Select individual images."""
        filetypes = [
            ("Image files", " ".join(f"*{ext}" for ext in SUPPORTED_FORMATS)),
            ("All files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=filetypes
        )
        
        if files:
            self.selected_images = list(files)
            self.images_label.config(text=f"{len(files)} images selected")
            self.update_process_button_state()
            logging.info(f"Selected {len(files)} images")
    
    def select_folder(self):
        """Select a folder containing images."""
        folder = filedialog.askdirectory(title="Select Folder with Images")
        if folder:
            image_files = []
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if YOLOProcessor.is_supported_image(file):
                        image_files.append(os.path.join(root, file))
            
            if image_files:
                self.selected_images = image_files
                self.images_label.config(text=f"{len(image_files)} images found in folder")
                self.update_process_button_state()
                logging.info(f"Found {len(image_files)} images in folder: {folder}")
            else:
                messagebox.showinfo("Info", "No supported images found in the selected folder")
    
    def upload_zip(self):
        """Upload and extract images from a ZIP file."""
        zip_file = filedialog.askopenfilename(
            title="Select ZIP file",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
        )
        
        if zip_file:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    image_files = []
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if YOLOProcessor.is_supported_image(file):
                                # Copy to a permanent location
                                src_path = os.path.join(root, file)
                                dest_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", "yolo_extracted")
                                os.makedirs(dest_dir, exist_ok=True)
                                dest_path = os.path.join(dest_dir, file)
                                
                                import shutil
                                shutil.copy2(src_path, dest_path)
                                image_files.append(dest_path)
                    
                    if image_files:
                        self.selected_images = image_files
                        self.images_label.config(text=f"{len(image_files)} images extracted from ZIP")
                        self.update_process_button_state()
                        logging.info(f"Extracted {len(image_files)} images from ZIP: {zip_file}")
                    else:
                        messagebox.showinfo("Info", "No supported images found in the ZIP file")
            except Exception as e:
                logging.error(f"Error extracting ZIP: {e}")
                messagebox.showerror("Error", f"Error extracting ZIP file: {e}")
    
    def update_confidence_label(self, *args):
        """Update confidence label."""
        self.confidence_label.config(text=f"{self.confidence_var.get():.2f}")
    
    def update_iou_label(self, *args):
        """Update IoU label."""
        self.iou_label.config(text=f"{self.iou_var.get():.2f}")
    
    def update_process_button_state(self):
        """Update the state of the process button."""
        if (self.yolo_processor.is_model_loaded() and 
            self.selected_images and 
            self.current_dataset and 
            not self.processing):
            self.process_button.config(state="normal")
        else:
            self.process_button.config(state="disabled")
    
    def process_images(self):
        """Process selected images with the loaded model."""
        if self.processing:
            return
        
        self.processing = True
        self.process_button.config(state="disabled")
        self.progress_var.set(0)
        
        # Run processing in a separate thread
        threading.Thread(target=self._process_images_thread, daemon=True).start()
    
    def _process_images_thread(self):
        """Process images in a separate thread."""
        try:
            # Create temporary directory for labels
            temp_dir = tempfile.mkdtemp()
            
            def progress_callback(current, total, successful):
                progress = (current / total) * 100
                self.root.after(0, lambda: self.progress_var.set(progress))
                self.root.after(0, lambda: self.status_var.set(f"Processing: {current}/{total} ({successful} successful)"))
            
            # Process images
            success = self.yolo_processor.process_images_batch(
                self.selected_images,
                temp_dir,
                progress_callback,
                self.confidence_var.get(),
                self.iou_var.get()
            )
            
            if success:
                # Copy images and labels to dataset
                label_files = []
                for image_path in self.selected_images:
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    label_path = os.path.join(temp_dir, f"{image_name}.txt")
                    if os.path.exists(label_path):
                        label_files.append(label_path)
                    else:
                        label_files.append(None)
                
                # Add to dataset
                dataset_success = self.dataset_manager.add_images_to_dataset(
                    self.current_dataset,
                    self.selected_images,
                    label_files
                )
                
                if dataset_success:
                    # Update class names in dataset
                    class_names = self.yolo_processor.get_class_names()
                    dataset_path = os.path.join(self.dataset_manager.base_path, self.current_dataset)
                    self.dataset_manager.update_class_names(dataset_path, class_names)
                    
                    self.root.after(0, lambda: self.status_var.set("Processing completed successfully"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Images processed and added to dataset successfully!"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to add images to dataset"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Image processing failed"))
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            logging.error(f"Error in processing thread: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {e}"))
        finally:
            self.processing = False
            self.root.after(0, lambda: self.process_button.config(state="normal"))
            self.root.after(0, self.update_process_button_state)
            self.root.after(0, lambda: self.progress_var.set(0))
    
    def on_closing(self):
        """Handle application closing."""
        if self.processing:
            if messagebox.askokcancel("Quit", "Processing is in progress. Do you want to quit anyway?"):
                self.db_manager.disconnect()
                self.root.destroy()
        else:
            self.db_manager.disconnect()
            self.root.destroy()


class ModelDialog:
    """Dialog for adding new models to the database."""
    
    def __init__(self, parent, db_manager):
        self.db_manager = db_manager
        self.result = False
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Add New Model")
        self.dialog.geometry("600x350")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model name
        ttk.Label(main_frame, text="Model Name:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(main_frame, textvariable=self.name_var, width=50)
        name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Model path
        ttk.Label(main_frame, text="Model Path:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(main_frame, textvariable=self.path_var, width=40)
        path_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(main_frame, text="Browse", command=self.browse_model).grid(row=1, column=2, padx=(5, 0), pady=(0, 5))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=(20, 0))
        
        ttk.Button(button_frame, text="Add Model", command=self.add_model).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.LEFT)
        
        main_frame.columnconfigure(1, weight=1)
        
        # Focus on name entry
        name_entry.focus()
    
    def browse_model(self):
        """Browse for model file."""
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model File",
            filetypes=[("PyTorch models", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            self.path_var.set(file_path)
    
    def add_model(self):
        """Add the model to database."""
        name = self.name_var.get().strip()
        path = self.path_var.get().strip()
        
        if not name:
            messagebox.showerror("Error", "Please enter a model name")
            return
        
        if not path:
            messagebox.showerror("Error", "Please select a model file")
            return
        
        if not os.path.exists(path):
            messagebox.showerror("Error", "Model file does not exist")
            return
        
        try:
            success = self.db_manager.add_model(name, path)
            if success:
                self.result = True
                self.dialog.destroy()
                messagebox.showinfo("Success", f"Model '{name}' added successfully")
            else:
                messagebox.showerror("Error", "Failed to add model to database")
        except Exception as e:
            messagebox.showerror("Error", f"Error adding model: {e}")
    
    def cancel(self):
        """Cancel dialog."""
        self.dialog.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = YOLOLabellerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main() 
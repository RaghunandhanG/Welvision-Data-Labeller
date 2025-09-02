"""
GUI Components Module
Reusable GUI components and widgets for the YOLO Labeler application
"""

import tkinter as tk
from tkinter import ttk


class StatusBar:
    """Status bar component for showing application status"""
    
    def __init__(self, parent):
        self.frame = tk.Frame(parent, relief=tk.SUNKEN, bd=1)
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        self.status_label = tk.Label(
            self.frame, 
            textvariable=self.status_var, 
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.frame,
            variable=self.progress_var,
            mode='determinate',
            length=200
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)
    
    def set_status(self, text):
        """Update status text"""
        self.status_var.set(text)
    
    def set_progress(self, value, maximum=100):
        """Update progress bar"""
        self.progress_bar.configure(maximum=maximum)
        self.progress_var.set(value)
    
    def pack(self, **kwargs):
        """Pack the status bar frame"""
        self.frame.pack(**kwargs)


class ModelSelectionFrame:
    """Model selection frame with dropdown and load button"""
    
    def __init__(self, parent, load_callback=None):
        self.frame = tk.LabelFrame(parent, text="🤖 Model Selection", padx=10, pady=10)
        self.load_callback = load_callback
        
        # Model selection
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            self.frame, 
            textvariable=self.model_var, 
            state="readonly",
            width=30
        )
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Load button
        self.load_btn = tk.Button(
            self.frame,
            text="📥 Load Model",
            command=self._on_load_clicked
        )
        self.load_btn.pack(side=tk.LEFT)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("No model loaded")
        self.status_label = tk.Label(
            self.frame,
            textvariable=self.status_var,
            fg="gray"
        )
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
    
    def _on_load_clicked(self):
        """Handle load button click"""
        if self.load_callback:
            self.load_callback()
    
    def update_models(self, models):
        """Update available models"""
        model_names = [model['name'] for model in models]
        self.model_combo['values'] = model_names
        
        if model_names and not self.model_var.get():
            self.model_var.set(model_names[0])
    
    def get_selected_model(self):
        """Get currently selected model"""
        return self.model_var.get()
    
    def set_status(self, text, color="black"):
        """Update status text"""
        self.status_var.set(text)
        self.status_label.configure(fg=color)
    
    def set_load_button_state(self, state, text=None, bg=None):
        """Update load button state"""
        self.load_btn.configure(state=state)
        if text:
            self.load_btn.configure(text=text)
        if bg:
            self.load_btn.configure(bg=bg)
    
    def pack(self, **kwargs):
        """Pack the frame"""
        self.frame.pack(**kwargs)


class DatasetSelectionFrame:
    """Dataset selection frame with dropdown and buttons"""
    
    def __init__(self, parent, select_callback=None, create_callback=None):
        self.frame = tk.LabelFrame(parent, text="📁 Dataset Selection", padx=10, pady=10)
        self.select_callback = select_callback
        self.create_callback = create_callback
        
        # Dataset selection
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(
            self.frame,
            textvariable=self.dataset_var,
            state="readonly",
            width=25
        )
        self.dataset_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Select button
        self.select_btn = tk.Button(
            self.frame,
            text="📂 Select",
            command=self._on_select_clicked
        )
        self.select_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Create button
        self.create_btn = tk.Button(
            self.frame,
            text="➕ Create New",
            command=self._on_create_clicked
        )
        self.create_btn.pack(side=tk.LEFT)
        
        # Info label
        self.info_var = tk.StringVar()
        self.info_var.set("No dataset selected")
        self.info_label = tk.Label(
            self.frame,
            textvariable=self.info_var,
            fg="gray"
        )
        self.info_label.pack(side=tk.LEFT, padx=(10, 0))
    
    def _on_select_clicked(self):
        """Handle select button click"""
        if self.select_callback:
            self.select_callback()
    
    def _on_create_clicked(self):
        """Handle create button click"""
        if self.create_callback:
            self.create_callback()
    
    def update_datasets(self, datasets):
        """Update available datasets"""
        dataset_names = [f"{ds['name']} ({ds['images_count']} images)" for ds in datasets]
        self.dataset_combo['values'] = dataset_names
        
        if dataset_names and not self.dataset_var.get():
            self.dataset_var.set(dataset_names[0])
    
    def get_selected_dataset(self):
        """Get currently selected dataset name"""
        selected = self.dataset_var.get()
        if selected and '(' in selected:
            return selected.split(' (')[0]  # Extract name before image count
        return selected
    
    def set_info(self, text, color="black"):
        """Update info text"""
        self.info_var.set(text)
        self.info_label.configure(fg=color)
    
    def pack(self, **kwargs):
        """Pack the frame"""
        self.frame.pack(**kwargs)


class ImageCanvas:
    """Canvas component for displaying and annotating images"""
    
    def __init__(self, parent, width=800, height=600):
        self.frame = tk.Frame(parent)
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(
            self.frame,
            bg="white",
            width=width,
            height=height,
            scrollregion=(0, 0, width, height)
        )
        
        # Scrollbars
        self.h_scrollbar = tk.Scrollbar(self.frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set
        )
        
        # Grid layout
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure grid weights
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        
        # Image variables
        self.current_image = None
        self.image_id = None
        self.scale_factor = 1.0
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
    
    def load_image(self, image_path):
        """Load and display image"""
        try:
            from PIL import Image, ImageTk
            
            # Load image
            self.current_image = Image.open(image_path)
            
            # Create PhotoImage
            self.photo_image = ImageTk.PhotoImage(self.current_image)
            
            # Clear canvas and add image
            self.canvas.delete("all")
            self.image_id = self.canvas.create_image(
                0, 0, 
                anchor=tk.NW, 
                image=self.photo_image
            )
            
            # Update scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def add_bounding_box(self, x1, y1, x2, y2, label="", color="red", width=2):
        """Add bounding box to canvas"""
        bbox_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline=color,
            width=width,
            tags="bbox"
        )
        
        if label:
            label_id = self.canvas.create_text(
                x1, y1 - 10,
                text=label,
                anchor=tk.SW,
                fill=color,
                font=("Arial", 10, "bold"),
                tags="label"
            )
            return bbox_id, label_id
        
        return bbox_id
    
    def clear_annotations(self):
        """Clear all annotations from canvas"""
        self.canvas.delete("bbox")
        self.canvas.delete("label")
    
    def _on_click(self, event):
        """Handle mouse click"""
        # Convert canvas coordinates to image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        # Implement annotation logic here
    
    def _on_drag(self, event):
        """Handle mouse drag"""
        # Implement drag logic here
        pass
    
    def _on_release(self, event):
        """Handle mouse release"""
        # Implement release logic here
        pass
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel for zooming"""
        # Implement zoom logic here
        pass
    
    def pack(self, **kwargs):
        """Pack the frame"""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame"""
        self.frame.grid(**kwargs)


class ProgressDialog:
    """Progress dialog for long-running operations"""
    
    def __init__(self, parent, title="Progress", message="Processing..."):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.resizable(False, False)
        self.dialog.grab_set()  # Make dialog modal
        
        # Center the dialog
        self.dialog.geometry("400x150")
        self.dialog.transient(parent)
        
        # Message label
        self.message_var = tk.StringVar()
        self.message_var.set(message)
        self.message_label = tk.Label(
            self.dialog,
            textvariable=self.message_var,
            wraplength=350
        )
        self.message_label.pack(pady=20)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.dialog,
            variable=self.progress_var,
            mode='determinate',
            length=350
        )
        self.progress_bar.pack(pady=10)
        
        # Cancel button
        self.cancelled = False
        self.cancel_btn = tk.Button(
            self.dialog,
            text="Cancel",
            command=self._on_cancel
        )
        self.cancel_btn.pack(pady=10)
    
    def update_progress(self, value, maximum=100, message=None):
        """Update progress"""
        self.progress_bar.configure(maximum=maximum)
        self.progress_var.set(value)
        
        if message:
            self.message_var.set(message)
        
        self.dialog.update()
    
    def _on_cancel(self):
        """Handle cancel button"""
        self.cancelled = True
        self.dialog.destroy()
    
    def is_cancelled(self):
        """Check if operation was cancelled"""
        return self.cancelled
    
    def close(self):
        """Close the dialog"""
        self.dialog.destroy()

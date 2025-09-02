"""
YOLO Model Manager Module
Handles YOLO model operations, device management, and inference
"""

import torch
import os
from ultralytics import YOLO
from config import YOLO_CONFIG


class YOLOModelManager:
    """Handles YOLO model operations and device management"""
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.device = None
        self.optimal_device = self.detect_optimal_device()
    
    def detect_optimal_device(self):
        """Detect the best available device for YOLO inference with enhanced GPU detection"""
        print("🔍 Detecting optimal device for YOLO inference...")
        
        try:
            # Check for CUDA GPU first
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)  # GB
                
                print(f"🔥 NVIDIA GPU detected: {gpu_name}")
                print(f"   📊 GPU Memory: {gpu_memory}GB")
                print(f"   🔢 GPU Count: {gpu_count}")
                
                # Test if CUDA is actually working
                try:
                    test_tensor = torch.randn(1, 3, 640, 640).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    print("✅ CUDA test successful - GPU ready for inference")
                    return {
                        'device': 'cuda',
                        'name': f'NVIDIA {gpu_name}',
                        'memory': gpu_memory,
                        'count': gpu_count
                    }
                except Exception as cuda_error:
                    print(f"⚠️ CUDA test failed: {cuda_error}")
                    print("🔄 Falling back to CPU processing")
            
            # Check for Apple Silicon MPS
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("🍎 Apple Silicon GPU (MPS) detected")
                return {
                    'device': 'mps',
                    'name': 'Apple Silicon GPU',
                    'memory': 'Unknown',
                    'count': 1
                }
            
            # Fall back to CPU
            else:
                import psutil
                cpu_count = psutil.cpu_count()
                memory_gb = psutil.virtual_memory().total // (1024**3)
                
                print("💻 No GPU detected - Using CPU processing")
                print(f"   🔢 CPU Cores: {cpu_count}")
                print(f"   📊 System Memory: {memory_gb}GB")
                
                return {
                    'device': 'cpu',
                    'name': f'CPU ({cpu_count} cores)',
                    'memory': memory_gb,
                    'count': cpu_count
                }
                
        except Exception as e:
            print(f"⚠️ Device detection error: {e}")
            print("💻 Falling back to CPU processing")
            return {
                'device': 'cpu',
                'name': 'CPU (fallback)',
                'memory': 'Unknown',
                'count': 1
            }
    
    def load_model(self, model_path):
        """Load YOLO model with enhanced GPU/CPU handling"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            print(f"📥 Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            self.model_path = model_path
            
            # Move model to optimal device
            device_info = self.optimal_device
            self.model.to(device_info['device'])
            self.device = device_info['device']
            
            print(f"✅ Model loaded successfully on {device_info['name']}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def test_cuda_inference(self):
        """Test CUDA inference capability"""
        if not self.model:
            return False
        
        try:
            # Create a dummy image path for testing (use any existing image)
            test_image_path = "yolov8n.pt"  # Fallback to model file if no image available
            
            if not hasattr(self, 'model') or self.model is None:
                return False
                
            # Try CUDA inference
            if torch.cuda.is_available():
                print("🧪 Testing CUDA inference...")
                _ = self.model(test_image_path, device='cuda', verbose=False)
                print("✅ CUDA inference test successful")
                return True
        except Exception as e:
            print(f"❌ CUDA inference test failed: {e}")
            return False
        
        return False
    
    def safe_yolo_inference(self, image_path, conf=None, max_det=None):
        """Perform YOLO inference with enhanced GPU/CPU fallback and device monitoring"""
        if conf is None:
            conf = YOLO_CONFIG['confidence_threshold']
        if max_det is None:
            max_det = YOLO_CONFIG['max_detections']
        
        if not self.model:
            raise ValueError("No model loaded")
        
        try:
            # Perform inference with device-specific optimizations
            results = self.model(
                source=image_path,
                conf=conf,
                max_det=max_det,
                device=self.device,
                verbose=False,
                save=False,
                show=False
            )
            
            return results
            
        except torch.cuda.OutOfMemoryError:
            print("⚠️ GPU out of memory, switching to CPU...")
            torch.cuda.empty_cache()
            self.device = 'cpu'
            self.model.to('cpu')
            
            # Retry with CPU
            results = self.model(
                source=image_path,
                conf=conf,
                max_det=max_det,
                device='cpu',
                verbose=False,
                save=False,
                show=False
            )
            return results
            
        except Exception as e:
            print(f"❌ Inference error: {e}")
            raise e
    
    def get_device_info(self):
        """Get current device information"""
        return self.optimal_device
    
    def get_model_classes(self):
        """Get model class names"""
        if self.model:
            return self.model.names
        return None
    
    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model is not None

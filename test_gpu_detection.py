#!/usr/bin/env python3
"""
GPU/CPU Detection Test Script for WelVision YOLO Data Labeller
This script tests the GPU detection and fallback functionality.
"""

import torch
import sys
import os

def test_gpu_detection():
    """Test GPU detection similar to the main app"""
    print("🔍 Testing GPU Detection for YOLO Data Labeller")
    print("=" * 50)
    
    # Test PyTorch installation
    print(f"PyTorch version: {torch.__version__}")
    
    # Test CUDA availability
    if torch.cuda.is_available():
        print(f"🎮 CUDA Available: YES")
        print(f"🔢 CUDA Devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        # Test basic CUDA operations
        try:
            print("🧪 Testing basic CUDA operations...")
            test_tensor = torch.randn(100, 100).cuda()
            result = test_tensor.matmul(test_tensor)
            _ = result.cpu()
            print("✅ Basic CUDA operations: PASSED")
            
            # Test torchvision operations
            try:
                import torchvision
                print("🧪 Testing torchvision CUDA operations...")
                test_boxes = torch.tensor([[10, 10, 50, 50]], device='cuda', dtype=torch.float32)
                test_scores = torch.tensor([0.9], device='cuda', dtype=torch.float32)
                _ = torchvision.ops.nms(test_boxes, test_scores, iou_threshold=0.5)
                print("✅ Torchvision CUDA operations: PASSED")
                recommended_device = 'cuda'
                
            except Exception as tv_error:
                print(f"❌ Torchvision CUDA operations: FAILED - {tv_error}")
                print("⚠️ Recommendation: Use CPU mode due to torchvision incompatibility")
                recommended_device = 'cpu'
                
        except Exception as cuda_error:
            print(f"❌ Basic CUDA operations: FAILED - {cuda_error}")
            recommended_device = 'cpu'
            
    else:
        print("🎮 CUDA Available: NO")
        
        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 Apple Silicon MPS Available: YES")
            
            try:
                print("🧪 Testing MPS operations...")
                test_tensor = torch.randn(100, 100).to('mps')
                result = test_tensor.matmul(test_tensor)
                _ = result.cpu()
                print("✅ MPS operations: PASSED")
                recommended_device = 'mps'
                
            except Exception as mps_error:
                print(f"❌ MPS operations: FAILED - {mps_error}")
                recommended_device = 'cpu'
        else:
            print("🍎 Apple Silicon MPS Available: NO")
            recommended_device = 'cpu'
    
    # Final recommendation
    print("\n" + "=" * 50)
    print("📋 DEVICE RECOMMENDATION:")
    
    if recommended_device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ Use GPU: {gpu_name} ({total_memory:.1f}GB)")
        print("🚀 Expected performance: FAST inference with GPU acceleration")
        
    elif recommended_device == 'mps':
        print("✅ Use Apple Silicon GPU (MPS)")
        print("🚀 Expected performance: Accelerated inference on Apple Silicon")
        
    else:
        print("✅ Use CPU")
        print("🐌 Expected performance: Slower inference but stable")
        
    print("\n💡 The YOLO Data Labeller will automatically detect and use the optimal device.")
    print("💡 If GPU fails during inference, it will automatically fallback to CPU.")
    
    return recommended_device

def test_yolo_compatibility():
    """Test YOLO model compatibility if available"""
    print("\n" + "=" * 50)
    print("🤖 Testing YOLO Model Compatibility")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO: Available")
        
        # Try to load a basic model (this will download if not present)
        print("📥 Testing YOLO model loading...")
        try:
            model = YOLO('yolov8n.pt')  # Nano model for testing
            print("✅ YOLO model loading: SUCCESS")
            
            # Test prediction with dummy image
            import numpy as np
            import tempfile
            
            # Create a dummy image
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            temp_path = tempfile.mktemp(suffix='.jpg')
            
            try:
                import cv2
                cv2.imwrite(temp_path, dummy_image)
                
                print("🧪 Testing YOLO inference...")
                results = model(temp_path, verbose=False)
                print("✅ YOLO inference: SUCCESS")
                
                os.unlink(temp_path)
                
            except Exception as inference_error:
                print(f"❌ YOLO inference: FAILED - {inference_error}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as model_error:
            print(f"❌ YOLO model loading: FAILED - {model_error}")
            
    except ImportError:
        print("❌ Ultralytics YOLO: NOT AVAILABLE")
        print("💡 Install with: pip install ultralytics")

if __name__ == "__main__":
    print("WelVision YOLO Data Labeller - GPU/CPU Detection Test")
    print("This script tests your system's compatibility with GPU acceleration.")
    print()
    
    recommended_device = test_gpu_detection()
    test_yolo_compatibility()
    
    print("\n" + "=" * 50)
    print("🏁 Test Complete!")
    print(f"🎯 Recommended device for YOLO Data Labeller: {recommended_device.upper()}")
    print("=" * 50)

"""
Device Detection Utility for WelVision YOLO Data Labeller
Run this script to check your system's PyTorch and CUDA capabilities
"""

import torch
import sys

def check_pytorch_installation():
    """Check PyTorch installation and capabilities"""
    print("🔍 WelVision YOLO Data Labeller - Device Check")
    print("=" * 50)
    
    # PyTorch version
    print(f"📦 PyTorch Version: {torch.__version__}")
    print(f"🐍 Python Version: {sys.version}")
    
    # CUDA availability
    print(f"\n🖥️  CUDA Support:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
    print(f"   CUDA Device Count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"\n🚀 GPU Devices:")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"   GPU {i}: {device_name} ({device_memory:.1f} GB)")
    
    # MPS availability (for Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        print(f"\n🍎 MPS (Apple Silicon) Available: {torch.backends.mps.is_available()}")
    
    print(f"\n⚙️  Recommendations:")
    
    if torch.cuda.is_available():
        print("   ✅ CUDA is available! You can use:")
        print("      device: 'auto' or 'cuda' in config.py for GPU acceleration")
        print("      device: 'cpu' for CPU-only processing")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("   ✅ MPS is available! You can use:")
        print("      device: 'mps' in config.py for Apple Silicon acceleration")
        print("      device: 'cpu' for CPU-only processing")
    else:
        print("   💡 No GPU acceleration available. Use:")
        print("      device: 'cpu' in config.py")
        print("      Consider installing CUDA-compatible PyTorch for GPU support")
    
    print(f"\n📝 Current config.py recommendation:")
    if torch.cuda.is_available():
        print("   YOLO_CONFIG = {'device': 'cuda'}  # For GPU acceleration")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("   YOLO_CONFIG = {'device': 'mps'}   # For Apple Silicon")
    else:
        print("   YOLO_CONFIG = {'device': 'cpu'}   # For CPU-only")
    
    print(f"\n🔧 Installation commands:")
    if not torch.cuda.is_available():
        print("   For CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("   For CPU-only:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")

if __name__ == "__main__":
    try:
        check_pytorch_installation()
    except Exception as e:
        print(f"❌ Error checking device capabilities: {e}")
        print("Please ensure PyTorch is installed: pip install torch")

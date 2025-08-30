# GPU/CPU Acceleration Guide
## WelVision YOLO Data Labeller

### Overview
The WelVision YOLO Data Labeller automatically detects and uses the best available hardware for image annotation:
- **🎮 NVIDIA GPU (CUDA)** - Fastest performance
- **🍎 Apple Silicon (MPS)** - Optimized for Mac M1/M2/M3
- **💻 CPU** - Universal compatibility, slower but stable

### Automatic Device Detection

The application automatically:
1. **Detects available hardware** on startup
2. **Tests compatibility** with comprehensive checks
3. **Falls back gracefully** if GPU fails
4. **Monitors performance** during inference
5. **Switches to CPU** if GPU errors occur

### GPU Requirements

#### NVIDIA GPU (Recommended)
- **CUDA-compatible GPU** (GTX 1060 or newer recommended)
- **4GB+ VRAM** for typical models
- **Updated drivers** (latest NVIDIA drivers)
- **CUDA toolkit** (installed with PyTorch)

#### Apple Silicon (Mac)
- **M1, M2, or M3 processor**
- **8GB+ RAM** recommended
- **macOS 12.3+** for MPS support

### Performance Comparison

| Device Type | Speed | Memory Usage | Compatibility |
|-------------|-------|--------------|---------------|
| NVIDIA GPU  | 🚀🚀🚀 | 2-6GB VRAM | High with modern GPUs |
| Apple Silicon | 🚀🚀 | Shared RAM | Excellent on M1/M2/M3 |
| CPU | 🚀 | 1-4GB RAM | Universal |

### Status Indicators

The application shows real-time device status:

#### GPU Active
```
🎮 GPU: NVIDIA GeForce RTX 3060 (2.1/12.0GB)
✅ ModelName - NVIDIA GeForce RTX 3060
```

#### Apple Silicon Active  
```
🍎 Apple Silicon GPU (MPS)
✅ ModelName - Apple Silicon GPU (MPS)
```

#### CPU Mode
```
💻 CPU Processing
✅ ModelName - CPU
```

#### Auto-Fallback
```
💻 CPU Processing
✅ ModelName - CPU (Auto-fallback)
```

### Troubleshooting

#### Common GPU Issues

**Problem**: GPU detected but incompatible
```
⚠️ GPU detected but incompatible - Using CPU
```
**Solutions**:
- Update GPU drivers
- Reinstall PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Check CUDA version compatibility

**Problem**: Out of memory errors
```
CUDA out of memory
```
**Solutions**:
- Close other GPU applications
- Use smaller batch sizes
- Restart the application
- The app will auto-fallback to CPU

**Problem**: Model loading fails
```
❌ Model loading failed
```
**Solutions**:
- Check model file exists
- Verify model format (.pt, .onnx)
- Try loading on CPU first

#### Testing Your Setup

Run the GPU detection test:
```bash
# Windows
test_gpu.bat

# Linux/Mac
python test_gpu_detection.py
```

### Manual Device Override

You can force a specific device in `config.py`:

```python
YOLO_CONFIG = {
    'device': 'cpu',        # Force CPU
    'device': 'cuda',       # Force NVIDIA GPU
    'device': 'cuda:0',     # Force specific GPU
    'device': 'mps',        # Force Apple Silicon
    'device_fallback': True  # Enable auto-fallback
}
```

### Performance Tips

#### For NVIDIA GPU Users
- **Keep drivers updated** for best performance
- **Close other GPU applications** (games, mining, etc.)
- **Monitor GPU temperature** during long sessions
- **Use models appropriate for your VRAM**

#### For Apple Silicon Users
- **Use unified memory efficiently** 
- **Close unnecessary applications**
- **MPS is generally stable** but CPU fallback available

#### For CPU Users
- **Use lighter models** (YOLOv8n instead of YOLOv8x)
- **Process smaller batches** of images
- **Consider upgrading hardware** for better performance

### Error Messages Explained

| Message | Meaning | Action |
|---------|---------|---------|
| `🎮 GPU Ready` | GPU detected and working | Continue normally |
| `⚠️ GPU detected but incompatible` | GPU exists but can't be used | App uses CPU automatically |
| `🔄 Automatically falling back to CPU` | GPU failed during inference | App switched to CPU |
| `💻 CPU processing mode` | No GPU available | Normal CPU operation |
| `⚠️ Device detection failed` | Hardware detection error | App defaults to CPU |

### Advanced Configuration

#### Memory Management
- GPU memory is automatically managed
- Failed allocations trigger CPU fallback
- Memory is cleared between operations

#### Real-time Monitoring
- Device status updates every 10 seconds
- GPU memory usage is displayed
- Automatic failover is logged

#### Logging
All device operations are logged to console:
```
🔍 Detecting optimal device for YOLO inference...
🎮 NVIDIA GPU detected: GeForce RTX 3060
✅ CUDA compatibility test passed
📥 Loading model from: model.pt
✅ Model loaded successfully on CPU
🎮 Moving model to CUDA GPU...
✅ Model successfully loaded on CUDA: GeForce RTX 3060
```

### Getting Help

If you experience issues:
1. **Run the GPU test** (`test_gpu.bat` or `test_gpu_detection.py`)
2. **Check the console output** for detailed error messages
3. **Try manual device override** in config.py
4. **Update your drivers** and PyTorch installation
5. **Use CPU mode** as a reliable fallback

The application is designed to work on any system, automatically adapting to your hardware capabilities.

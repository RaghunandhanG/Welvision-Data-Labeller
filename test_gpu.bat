@echo off
echo WelVision YOLO Data Labeller - GPU Detection Test
echo ================================================
echo.
echo This script will test your system's GPU compatibility
echo for the YOLO Data Labeller application.
echo.
pause
echo.
echo Running GPU detection test...
echo.

python test_gpu_detection.py

echo.
echo ================================================
echo Test completed! Check the results above.
echo.
echo If you see GPU compatibility issues:
echo 1. Try updating your GPU drivers
echo 2. Reinstall PyTorch with CUDA support
echo 3. The app will automatically use CPU as fallback
echo.
pause

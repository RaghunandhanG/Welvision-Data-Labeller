@echo off
echo WelVision YOLO Data Labeller - Roboflow Project Creation Test
echo =============================================================
echo.
echo This script will test the new Roboflow project creation feature.
echo.
echo Before running this test:
echo 1. Make sure you have a Roboflow account
echo 2. Get your API key from https://app.roboflow.com/settings/api
echo 3. Have it ready to test the project creation
echo.
pause
echo.
echo Testing Roboflow SDK availability...
python -c "import roboflow; print('✅ Roboflow SDK is available')" 2>nul || (
    echo ❌ Roboflow SDK not found
    echo.
    echo Installing Roboflow SDK...
    pip install roboflow
)
echo.
echo Running Roboflow project creation test...
python test_roboflow_creation.py
echo.
echo ===============================================
echo Test completed!
echo.
echo To test the full functionality:
echo 1. Run the main application: python yolo_labeler_app.py
echo 2. Go to the "Roboflow Projects" tab
echo 3. Enter your API key and click "Load Projects"
echo 4. Try creating a new project using the form
echo.
pause

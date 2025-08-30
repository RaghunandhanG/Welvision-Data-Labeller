@echo off
echo Starting comprehensive Roboflow project creation test...
echo.

REM Set up Python environment
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Install required packages if not present
python -c "import roboflow" 2>nul || (
    echo Installing Roboflow package...
    pip install roboflow
)

python -c "import requests" 2>nul || (
    echo Installing requests package...
    pip install requests
)

REM Run the comprehensive test
echo.
echo Running comprehensive Roboflow test...
python test_roboflow_comprehensive.py

echo.
echo Test completed. Press any key to exit...
pause >nul

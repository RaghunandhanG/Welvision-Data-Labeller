@echo off
echo.
echo ===================================================
echo   WelVision YOLO Data Labeller
echo ===================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if requirements are installed
python -c "import ultralytics, mysql.connector, cv2" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
)

REM Check if database is configured
python -c "from config import DATABASE_CONFIG; assert DATABASE_CONFIG['password'] != '', 'Password not set'" >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Database password not configured!
    echo Please edit config.py and set your MySQL password
    echo.
    pause
)

echo Starting WelVision YOLO Data Labeller...
echo.
python yolo_labeler_app.py

if errorlevel 1 (
    echo.
    echo Application exited with an error
    pause
) 
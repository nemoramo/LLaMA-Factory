@echo off
chcp 65001 >nul
echo ========================================
echo   Endpointing WebUI - Installation
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/2] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/2] Installing dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo   Installation complete!
echo   Run 'run.bat' to start the WebUI.
echo ========================================
pause

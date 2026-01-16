@echo off
chcp 65001 >nul
echo ========================================
echo   Endpointing WebUI - Starting...
echo ========================================
echo.

python app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start. Please run install.bat first.
)
pause

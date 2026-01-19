@echo off
setlocal

rem Assumes this script is placed in the root of text-generation-webui
cd /D "%~dp0"

if not exist "start_windows.bat" (
  echo [ERROR] start_windows.bat not found. Please copy this file into the webui root.
  pause
  exit /b 1
)

call start_windows.bat --extensions endpointing_tool

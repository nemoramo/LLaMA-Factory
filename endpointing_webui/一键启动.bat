@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

title 语音端点检测标注工具 - 一键启动

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║        语音端点检测标注工具 - Endpointing WebUI              ║
echo ║                                                              ║
echo ║                     一键启动程序                             ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

cd /D "%~dp0"

REM ========================================
REM 第一步：检查 Python 是否已安装
REM ========================================
echo [1/4] 检查 Python 环境...

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ╔══════════════════════════════════════════════════════════════╗
    echo ║  [错误] 未检测到 Python！                                    ║
    echo ║                                                              ║
    echo ║  请按以下步骤安装 Python：                                   ║
    echo ║                                                              ║
    echo ║  1. 访问 https://www.python.org/downloads/                   ║
    echo ║  2. 下载 Python 3.10 或更高版本                              ║
    echo ║  3. 安装时务必勾选 "Add Python to PATH"                      ║
    echo ║  4. 安装完成后重新运行本程序                                 ║
    echo ╚══════════════════════════════════════════════════════════════╝
    echo.
    echo 按任意键打开 Python 下载页面...
    pause >nul
    start https://www.python.org/downloads/
    exit /b 1
)

REM 检查 Python 版本
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo     Python 版本: %PYTHON_VERSION%

REM ========================================
REM 第二步：创建/激活虚拟环境
REM ========================================
echo.
echo [2/4] 配置虚拟环境...

if not exist "venv" (
    echo     首次运行，正在创建虚拟环境...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo     [错误] 创建虚拟环境失败！
        pause
        exit /b 1
    )
    echo     虚拟环境创建成功！
) else (
    echo     虚拟环境已存在
)

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM ========================================
REM 第三步：安装依赖
REM ========================================
echo.
echo [3/4] 检查并安装依赖...

REM 检查是否需要安装依赖（通过检查 gradio 是否存在）
python -c "import gradio" >nul 2>&1
if %errorlevel% neq 0 (
    echo     正在安装依赖包，请稍候...
    echo     （首次安装可能需要几分钟）
    echo.
    python -m pip install --upgrade pip -q
    pip install -r requirements.txt -q
    if %errorlevel% neq 0 (
        echo.
        echo     [错误] 依赖安装失败！
        echo     请检查网络连接后重试。
        pause
        exit /b 1
    )
    echo     依赖安装完成！
) else (
    echo     依赖已安装
)

REM ========================================
REM 第四步：启动程序
REM ========================================
echo.
echo [4/4] 启动 WebUI...
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║   程序启动中，浏览器将自动打开...                            ║
echo ║                                                              ║
echo ║   如果浏览器没有自动打开，请手动访问：                       ║
echo ║   http://127.0.0.1:7860                                      ║
echo ║                                                              ║
echo ║   关闭此窗口将停止程序                                       ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM 启动浏览器（延迟3秒后）
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:7860"

REM 启动应用
python app.py

REM 如果程序退出
echo.
echo 程序已停止运行。
pause

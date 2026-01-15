@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo         LaCoGSEA One-Click Launcher
echo ==================================================

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher.
    pause
    exit /b
)

:: 2. Check if installed, if not install
pip show lacogsea >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] First-time setup: Installing LaCoGSEA and dependencies...
    pip install -e .
    if !errorlevel! neq 0 (
        echo [ERROR] Installation failed.
        pause
        exit /b
    )
    echo [SUCCESS] Dependencies installed.
)

:: 3. Launch GUI (Environment/Java check happens automatically inside GUI)
echo [INFO] Launching Graphical Interface...
lacogsea-gui

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start GUI. 
    echo Please try running 'install_windows.bat' manually to debug.
    pause
)

endlocal

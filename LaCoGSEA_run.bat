@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo         LaCoGSEA One-Click Launcher
echo ==================================================

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. 
    echo.
    echo --------------------------------------------------
    echo To use LaCoGSEA, please install Python 3.8-3.12 first:
    echo 1. Download from: https://www.python.org/downloads/
    echo 2. IMPORTANT: Check "Add Python to PATH" during installation.
    echo --------------------------------------------------
    echo.
    pause
    exit /b
)

:: 2. Check if installed, if not install
pip show lacogsea >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] First-time setup: Installing LaCoGSEA and dependencies...
    pip install -e .
    if !errorlevel! neq 0 (
        echo [ERROR] Installation failed. Try running: pip install -e . --user
        pause
        exit /b
    )
    echo [SUCCESS] Dependencies installed.
)

:: 3. Launch GUI
echo [INFO] Launching Graphical Interface...
lacogsea-gui

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start GUI. 
    echo Try running 'python -m lacogsea.gui' manually.
    pause
)

endlocal

@echo off
echo ==================================================
echo   LaCoGSEA Windows Installer
echo ==================================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher.
    pause
    exit /b
)

echo [1/3] Installing dependencies...
pip install -e .

echo [2/3] Setting up environment...
lacogsea setup --yes

echo [3/3] Finalizing...
echo.
echo ==================================================
echo   Installation Complete!
echo   You can now launch the GUI using 'run_gui.bat'
echo ==================================================
pause

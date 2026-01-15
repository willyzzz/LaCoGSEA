@echo off
setlocal

echo ==================================================
echo         LaCoGSEA One-Click Launcher
echo ==================================================

:: 1. Check Python
python --version >nul 2>&1
if errorlevel 1 goto NOPYTHON

:: 2. Create/Activate venv
if exist .venv goto ACTIVATE
echo [INFO] Creating isolated environment...
python -m venv .venv
if errorlevel 1 goto VENV_FAIL

:ACTIVATE
call .venv\Scripts\activate.bat

:: 3. Installation
if exist .venv\lacogsea_installed goto LAUNCH

echo [INFO] Installing core components (One-time setup)...
echo [NOTE] Showing installation logs for transparency.

:: Step 1: Install CPU Torch first with specific index
echo [1/2] Installing PyTorch (CPU version)...
pip install torch --index-url https://download.pytorch.org/whl/cpu --prefer-binary

:: Step 2: Install others from requirements
echo [2/2] Installing remaining dependencies...
pip install -r requirements.txt --prefer-binary

if errorlevel 1 goto INSTALL_FAIL
echo. > .venv\lacogsea_installed
echo [SUCCESS] Environment ready.

:LAUNCH
echo [INFO] Launching LaCoGSEA...
python -m lacogsea.gui
if errorlevel 1 (
    echo [ERROR] Application crashed. Checking dependencies...
    :: If it crashes, maybe some deps are missing, allow one retry of install
    del .venv\lacogsea_installed >nul 2>&1
    pause
)
goto END

:NOPYTHON
echo [ERROR] Python not found. Please install Python 3.8-3.12.
pause
exit /b

:VENV_FAIL
echo [ERROR] Failed to create environment.
pause
exit /b

:INSTALL_FAIL
echo [ERROR] Installation failed. This might be due to incompatible Python version or network issues.
pause
exit /b

:LAUNCH_FAIL
echo [ERROR] Failed to start. 
pause
exit /b

:END
endlocal

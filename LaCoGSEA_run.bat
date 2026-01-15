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
echo [INFO] Creating isolated environment (.venv)...
python -m venv .venv
if errorlevel 1 goto VENV_FAIL

:ACTIVATE
:: Force the script to use the local venv's python and pip
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "PIP_EXE=%~dp0.venv\Scripts\pip.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment seems corrupted. Deleting .venv...
    rmdir /s /q .venv
    goto :END
)

:: 3. Verification (How to test)
echo [DEBUG] Using Python from: %PYTHON_EXE%

:: 4. Installation
if exist .venv\lacogsea_installed goto LAUNCH

echo [INFO] Installing core components (One-time setup)...

:: Use python -m pip to be 100%% sure we are using the venv
"%PYTHON_EXE%" -m pip install --upgrade pip
echo [1/2] Installing PyTorch (CPU version)...
"%PYTHON_EXE%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --prefer-binary

echo [2/2] Installing remaining dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt --prefer-binary

if errorlevel 1 goto INSTALL_FAIL
echo. > .venv\lacogsea_installed
echo [SUCCESS] Environment ready.

:LAUNCH
echo [INFO] Launching LaCoGSEA...
"%PYTHON_EXE%" -m lacogsea.gui
if errorlevel 1 (
    echo [ERROR] Application crashed.
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
echo [ERROR] Installation failed.
pause
exit /b

:END
endlocal

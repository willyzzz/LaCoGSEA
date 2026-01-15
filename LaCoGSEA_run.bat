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

:: 3. Fast-track Installation
if exist .venv\lacogsea_installed goto LAUNCH
echo [1/2] Installing core components (One-time setup)...

:: Combined install for maximum speed + CPU Torch prioritization
pip install --quiet --no-warn-script-location --prefer-binary ^
    --extra-index-url https://download.pytorch.org/whl/cpu ^
    -r requirements.txt

if errorlevel 1 goto INSTALL_FAIL
echo. > .venv\lacogsea_installed

:LAUNCH
echo [2/2] Launching LaCoGSEA...
:: Run module directly to skip heavy package registration entries
python -m lacogsea.gui
if errorlevel 1 goto LAUNCH_FAIL
goto END

:NOPYTHON
echo [ERROR] Python not found.
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

:LAUNCH_FAIL
echo [ERROR] Failed to start. 
pause
exit /b

:END
endlocal

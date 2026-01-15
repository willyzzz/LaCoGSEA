@echo off
setlocal

echo ==================================================
echo         LaCoGSEA One-Click Launcher
echo ==================================================

:: 1. Check Python
python --version >nul 2>&1
if errorlevel 1 goto NOPYTHON

:: 2. Create/Activate Virtual Environment
if exist .venv goto ACTIVATE
echo [INFO] Creating isolated virtual environment (.venv)...
python -m venv .venv
if errorlevel 1 goto VENV_FAIL

:ACTIVATE
call .venv\Scripts\activate.bat

:: 3. Check/Install dependencies
if exist .venv\lacogsea_installed goto LAUNCH
echo [INFO] Installing light-weight dependencies...
python -m pip install --upgrade pip

:: Optimization: Install CPU-only Torch to save gigabytes of space and time
echo [INFO] Installing CPU-version of Torch (Optimized for speed)...
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo [INFO] Installing other requirements...
pip install -r requirements.txt

echo [INFO] Finalizing package...
pip install -e . --no-deps

if errorlevel 1 goto INSTALL_FAIL

echo. > .venv\lacogsea_installed
echo [SUCCESS] Environment ready.

:LAUNCH
echo [INFO] Launching Graphical Interface...
lacogsea-gui
if errorlevel 1 goto LAUNCH_FAIL
goto END

:NOPYTHON
echo [ERROR] Python not found.
pause
exit /b

:VENV_FAIL
echo [ERROR] Failed to create venv.
pause
exit /b

:INSTALL_FAIL
echo [ERROR] Installation failed.
pause
exit /b

:LAUNCH_FAIL
echo [ERROR] Failed to start GUI.
pause
exit /b

:END
endlocal

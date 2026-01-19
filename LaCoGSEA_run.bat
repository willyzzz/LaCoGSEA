@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo         LaCoGSEA Universal Launcher
echo    (Local Runtime: Python 3.8.8 x64 Verified)
echo ==================================================

set "RUNTIME_DIR=%~dp0.python_runtime"
set "PYTHON_EXE=%RUNTIME_DIR%\python.exe"
set "PIP_EXE=%RUNTIME_DIR%\Scripts\pip.exe"

:: 1. Check for Local Runtime
if exist "%PYTHON_EXE%" goto RUNTIME_READY

echo [INFO] No local runtime detected. Preparing Python 3.8.8...
echo [INFO] This is a one-time setup (approx. 25MB for Python + dependencies).

:: 2. Download Python 3.8.8 Embeddable Zip
echo [1/4] Downloading Python 3.8.8 executable...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $url='https://www.python.org/ftp/python/3.8.8/python-3.8.8-embed-amd64.zip'; $out='python_dist.zip'; Invoke-WebRequest -Uri $url -OutFile $out"
if errorlevel 1 goto FAIL

:: 3. Extract and Configure
echo [2/4] Extracting runtime...
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"
powershell -Command "Expand-Archive -Path 'python_dist.zip' -DestinationPath '%RUNTIME_DIR%' -Force"
del python_dist.zip

:: Enable site-packages in embeddable python (Crucial step)
echo [3/4] Tuning runtime configuration...
pushd "%RUNTIME_DIR%"
powershell -Command "(Get-Content python38._pth) -replace '#import site', 'import site' | Set-Content python38._pth"
popd

:: 4. Install Pip
echo [4/4] Activating package manager (pip)...
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/pip/3.8/get-pip.py' -OutFile 'get-pip.py'"
"%PYTHON_EXE%" get-pip.py --no-warn-script-location
del get-pip.py

:RUNTIME_READY
:: 5. Install/Verify Dependencies
if exist "%RUNTIME_DIR%\.installed_mark" goto LAUNCH

echo [INFO] Installing locked dependencies from requirements.txt...
"%PYTHON_EXE%" -m pip install --upgrade pip
echo [Installing] PyTorch (CPU 2.4.1)...
"%PYTHON_EXE%" -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu --prefer-binary
echo [Installing] Other core packages...
"%PYTHON_EXE%" -m pip install -r requirements.txt --prefer-binary
"%PYTHON_EXE%" -m pip install -e . --no-deps

if errorlevel 1 goto FAIL
echo. > "%RUNTIME_DIR%\.installed_mark"
echo [SUCCESS] Environment is fully synchronized and locked.

:LAUNCH
echo [INFO] Launching LaCoGSEA on Internal Python 3.8.8...
"%PYTHON_EXE%" -m lacogsea.gui
if errorlevel 1 (
    echo [ERROR] Application crashed.
    pause
)
goto END

:FAIL
echo ==================================================
echo [ERROR] Setup failed. Please check your internet.
echo ==================================================
pause

:END
endlocal

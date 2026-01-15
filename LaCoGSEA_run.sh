#!/bin/bash

echo "=================================================="
echo "        LaCoGSEA One-Click Launcher (Linux/macOS)"
echo "=================================================="

# 1. Check Python
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python3 not found. Please install Python 3.8 or higher."
    echo "Common command: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# 2. Create/Activate Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating isolated virtual environment (.venv)..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        exit 1
    fi
fi

# Activate
source .venv/bin/activate

# 3. Check/Install dependencies inside venv using requirements.txt
if [ ! -f ".venv/lacogsea_installed" ]; then
    echo "[INFO] Installing dependencies from requirements.txt..."
    echo "[NOTE] This may take a few minutes for the first time."
    python3 -m pip install --upgrade pip
    
    # Install core dependencies from requirements.txt
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies from requirements.txt."
        exit 1
    fi
    
    # Install the package itself in editable mode without reinstalling dependencies
    pip install -e . --no-deps
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install LaCoGSEA package."
        exit 1
    fi
    
    touch .venv/lacogsea_installed
    echo "[SUCCESS] Environment ready."
fi

# 4. Launch GUI
echo "[INFO] Launching Graphical Interface..."
lacogsea-gui

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to start GUI."
    echo "Try running 'python3 -m lacogsea.gui' manually."
fi

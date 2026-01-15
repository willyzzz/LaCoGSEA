#!/bin/bash

echo "=================================================="
echo "        LaCoGSEA One-Click Launcher (Linux/macOS)"
echo "=================================================="

# 1. Check Python
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python3 not found. Please install Python 3.8 or higher."
    echo "Common command: sudo apt install python3 python3-pip"
    exit 1
fi

# 2. Check if installed, if not install in user mode
if ! python3 -m pip show lacogsea &> /dev/null
then
    echo "[INFO] First-time setup: Installing LaCoGSEA and dependencies..."
    python3 -m pip install -e .
    if [ $? -ne 0 ]; then
        echo "[ERROR] Installation failed."
        exit 1
    fi
    echo "[SUCCESS] Dependencies installed."
fi

# 3. Launch GUI
echo "[INFO] Launching Graphical Interface..."
# Ensure local bin is in PATH for entry points
export PATH=$PATH:~/.local/bin
lacogsea-gui

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to start GUI."
    echo "Try running 'python3 -m lacogsea.gui' manually."
fi

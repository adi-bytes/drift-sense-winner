#!/bin/bash

echo "=============================================="
echo "  Drift-Sense Local Setup and Runner (Mac/Linux)"
echo "=============================================="

# Use python3 if available, else python
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
fi

if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Python is not installed. Please install Python 3.9-3.12."
    exit 1
fi

# Recreate virtual environment if it is broken or copied from another system
if [ -d "venv" ]; then
    venv/bin/python -c "import sys; sys.exit(0)" >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Existing virtual environment is broken or from another system. Recreating..."
        rm -rf venv
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate and install requirements
echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

# Run the app
echo "Starting Drift-Sense UI..."
$PYTHON_CMD -m streamlit run ui.py

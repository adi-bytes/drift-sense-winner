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

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate and install requirements
echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run the app
echo "Starting Drift-Sense UI..."
streamlit run ui.py

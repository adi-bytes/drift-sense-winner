@echo off
echo ==============================================
echo    Drift-Sense Local Setup and Runner (Windows)
echo ==============================================

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not added to PATH. Please install Python 3.9-3.12.
    pause
    exit /b
)

:: Recreate virtual environment if it is broken or copied from another system
if exist "venv" (
    venv\Scripts\python.exe -c "import sys; sys.exit(0)" >nul 2>&1
    if errorlevel 1 (
        echo Existing virtual environment is broken or from another system. Recreating...
        rmdir /s /q venv
    )
)

:: If venv exists but activate.bat doesn't (partially created), remove it too
if exist "venv" (
    if not exist "venv\Scripts\activate.bat" (
        echo Virtual environment is incomplete. Recreating...
        rmdir /s /q venv
    )
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment... Please wait, this may take a minute.
    python -m venv venv
)

:: Activate and install requirements
echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: Run the app
echo Starting Drift-Sense UI...
python -m streamlit run ui.py
pause

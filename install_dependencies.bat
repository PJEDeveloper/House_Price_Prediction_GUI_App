@echo off

:: Navigate to project directory
cd /d "%~dp0"

:: Install dependencies
pip install -r requirements.txt

echo Dependencies installed successfully.

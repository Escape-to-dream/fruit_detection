@echo off
cd /d "%~dp0\.."
set PYTHONPATH=%CD%\src
.\.venv\Scripts\python.exe -m fruit_detection.compare_models
.\.venv\Scripts\python.exe -m fruit_detection.export_summary

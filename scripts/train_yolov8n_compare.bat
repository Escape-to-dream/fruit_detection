@echo off
cd /d "%~dp0\.."
set PYTHONPATH=%CD%\src
.\.venv\Scripts\python.exe -m fruit_detection.train_yolo --name yolov8n_fruits --weights yolov8n.pt --epochs 20 --imgsz 640 --batch 16

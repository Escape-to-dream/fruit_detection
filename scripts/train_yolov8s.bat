@echo off
cd /d "%~dp0\.."
set PYTHONPATH=%CD%\src
.\.venv\Scripts\python.exe -m fruit_detection.train_yolo --name yolov8s_fruits --weights yolov8s.pt --epochs 20 --imgsz 640 --batch 16

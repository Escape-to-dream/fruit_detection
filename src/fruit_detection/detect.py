from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from fruit_detection.paths import MODELS_DIR, REPORTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the trained fruit detector.")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--weights", type=Path, default=MODELS_DIR / "yolov8s_fruits_best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--name", type=str, default="manual_predict")
    args = parser.parse_args()
    model = YOLO(str(args.weights))
    model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, save=True, project=str(REPORTS_DIR / "predictions"), name=args.name, exist_ok=True)


if __name__ == "__main__":
    main()

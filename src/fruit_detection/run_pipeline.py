from __future__ import annotations

import argparse
import subprocess
import sys

from fruit_detection.paths import root_dir


def run(module_args: list[str]) -> None:
    print("\n>>> python -m", " ".join(module_args))
    subprocess.run([sys.executable, "-m", *module_args], cwd=root_dir(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full fruit detection workflow.")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    run(["fruit_detection.dataset_report"])
    if not args.skip_train:
        run(["fruit_detection.train_yolo", "--name", "yolov8s_fruits", "--weights", "yolov8s.pt", "--epochs", "50", "--imgsz", "640", "--batch", "16"])
        run(["fruit_detection.train_yolo", "--name", "yolov8n_fruits", "--weights", "yolov8n.pt", "--epochs", "50", "--imgsz", "640", "--batch", "16"])
        run(["fruit_detection.train_yolo", "--name", "yolov8n-cbam_fruits", "--weights", "yolov8n-cbam", "--epochs", "50", "--imgsz", "640", "--batch", "16"])
    run(["fruit_detection.compare_models"])
    run(["fruit_detection.export_summary"])


if __name__ == "__main__":
    main()
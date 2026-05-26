from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from fruit_detection.paths import DATASET_DIR, MODELS_DIR, REPORTS_DIR, root_dir, ensure_dir


def select_device(device: str) -> str:
    if device == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device


def benchmark(model: YOLO, image_dir: Path, imgsz: int, device: str, max_images: int) -> float:
    images = sorted(image_dir.glob("*.jpg"))[:max_images]
    if not images:
        return 0.0
    for path in images[:8]:
        model.predict(str(path), imgsz=imgsz, device=device, verbose=False)
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for path in images:
        model.predict(str(path), imgsz=imgsz, device=device, verbose=False)
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return len(images) / elapsed if elapsed else 0.0


def save_metrics(model_name: str, metrics_data: dict) -> None:
    metrics_dir = ensure_dir(REPORTS_DIR / "metrics")
    json_path = metrics_dir / f"{model_name}_metrics.json"
    csv_path = metrics_dir / f"{model_name}_metrics.csv"
    json_path.write_text(json.dumps(metrics_data, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics_data.items():
            if isinstance(value, (str, int, float)) or value is None:
                writer.writerow([key, value])


def copy_report_assets(run_dir: Path, model_name: str) -> None:
    figures = ensure_dir(REPORTS_DIR / "figures")
    for source_name, target_name in [
        ("results.png", f"{model_name}_training_results.png"),
        ("confusion_matrix.png", f"{model_name}_confusion_matrix.png"),
        ("confusion_matrix_normalized.png", f"{model_name}_confusion_matrix_normalized.png"),
        ("BoxPR_curve.png", f"{model_name}_pr_curve.png"),
        ("val_batch0_pred.jpg", f"{model_name}_validation_predictions.jpg"),
    ]:
        src = run_dir / source_name
        if src.exists():
            shutil.copy2(src, figures / target_name)


def train(args: argparse.Namespace) -> dict:
    device = select_device(args.device)
    data_yaml = args.dataset_dir / "data.yaml"
    model = YOLO(args.weights)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        pretrained=True,
        device=device,
        workers=args.workers,
        project=str(root_dir() / "runs" / args.name),
        name="train",
        exist_ok=True,
        plots=True,
        seed=args.seed,
    )
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    models_dir = ensure_dir(MODELS_DIR)
    shutil.copy2(best, models_dir / f"{args.name}_best.pt")
    shutil.copy2(last, models_dir / f"{args.name}_last.pt")

    trained = YOLO(str(best))
    val = trained.val(data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch, device=device, workers=args.workers, plots=True)
    fps = benchmark(trained, args.dataset_dir / "test" / "images", args.imgsz, device, args.speed_samples)
    pred_dir = REPORTS_DIR / "predictions"
    trained.predict(
        source=str(args.dataset_dir / "test" / "images"),
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        save=True,
        project=str(pred_dir),
        name=f"{args.name}_test_predictions",
        exist_ok=True,
        max_det=50,
    )
    metrics_data = {
        "model": args.name,
        "weights": args.weights,
        "best_weight": str(models_dir / f"{args.name}_best.pt"),
        "run_dir": str(run_dir),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "map50": float(val.box.map50),
        "map50_95": float(val.box.map),
        "precision": float(val.box.mp),
        "recall": float(val.box.mr),
        "fps": float(fps),
    }
    save_metrics(args.name, metrics_data)
    copy_report_assets(run_dir, args.name)
    return metrics_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 fruit detector.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--name", type=str, default="yolov8s_fruits")
    parser.add_argument("--weights", type=str, default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed-samples", type=int, default=160)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    metrics = train(args)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

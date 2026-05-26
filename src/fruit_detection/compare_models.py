from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from fruit_detection.paths import REPORTS_DIR, ensure_dir


def load_metric(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def table_image(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.2), dpi=180)
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2d4059")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f7f7f7" if row % 2 else "white")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def bar(df: pd.DataFrame, col: str, output: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(7, 4), dpi=160)
    bars = plt.bar(df["model"], df[col], color=["#2878b5", "#c85200", "#4c956c"][: len(df)])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    for item in bars:
        v = item.get_height()
        plt.text(item.get_x() + item.get_width() / 2, v, f"{v:.3f}" if v <= 1.5 else f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def prediction_sheet(reports_dir: Path, output: Path) -> None:
    dirs = [
        ("YOLOv8s", reports_dir / "predictions" / "yolov8s_fruits_test_predictions"),
        ("YOLOv8n", reports_dir / "predictions" / "yolov8n_fruits_test_predictions"),
        ("YOLOv8n-CBAM", reports_dir / "predictions" / "yolov8n-cbam_fruits_test_predictions"),
    ]
    font = ImageFont.load_default()
    rows = []
    for title, folder in dirs:
        images = sorted(folder.glob("*.jpg"))[:6]
        if not images:
            continue
        thumb_size = (240, 240)
        row = Image.new("RGB", (len(images) * thumb_size[0], thumb_size[1] + 28), (246, 246, 246))
        draw = ImageDraw.Draw(row)
        draw.text((8, 8), title, fill=(30, 30, 30), font=font)
        for idx, path in enumerate(images):
            img = Image.open(path).convert("RGB").resize(thumb_size, Image.Resampling.LANCZOS)
            row.paste(img, (idx * thumb_size[0], 28))
        rows.append(row)
    if not rows:
        return
    sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows) + 12), (235, 235, 235))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + 12
    sheet.save(output, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metric tables and model comparison screenshots.")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--models", nargs="+", default=["yolov8s_fruits", "yolov8n_fruits", "yolov8n-cbam_fruits"])
    args = parser.parse_args()
    metrics_dir = ensure_dir(args.reports_dir / "metrics")
    figures = ensure_dir(args.reports_dir / "figures")
    rows = []
    for model in args.models:
        path = metrics_dir / f"{model}_metrics.json"
        if not path.exists():
            continue
        data = load_metric(path)
        rows.append(
            {
                "model": data["model"],
                "mAP@0.5": round(data["map50"], 4),
                "mAP@0.5:0.95": round(data["map50_95"], 4),
                "precision": round(data["precision"], 4),
                "recall": round(data["recall"], 4),
                "FPS": round(data["fps"], 2),
                "epochs": data["epochs"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(metrics_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    if not df.empty:
        table_image(df, figures / "model_comparison_table.png")
        bar(df, "mAP@0.5", figures / "model_comparison_map50.png", "YOLO model comparison: mAP@0.5", "mAP@0.5")
        bar(df, "FPS", figures / "model_comparison_fps.png", "YOLO model comparison: inference speed", "FPS")
        prediction_sheet(args.reports_dir, figures / "model_prediction_comparison.jpg")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
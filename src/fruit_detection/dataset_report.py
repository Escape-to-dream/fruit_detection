from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw, ImageFont

from fruit_detection.paths import DATASET_DIR, REPORTS_DIR, ensure_dir


def load_data_yaml(dataset_dir: Path) -> dict:
    return yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))


def read_label(label_path: Path, width: int, height: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    records = []
    if not label_path.exists():
        return records
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:])
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        records.append((cls_id, (x1, y1, x2, y2)))
    return records


def font(size: int = 16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_boxes(image_path: Path, label_path: Path, names: list[str]) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = ["#e63946", "#f4a261", "#7cb518", "#fb8500", "#6a994e", "#2a9d8f"]
    text_font = font(16)
    for cls_id, box in read_label(label_path, img.width, img.height):
        color = colors[cls_id % len(colors)]
        name = names[cls_id]
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = name
        tw = draw.textlength(label, font=text_font) + 8
        draw.rectangle((x1, max(0, y1 - 24), x1 + tw, y1), fill=color)
        draw.text((x1 + 4, max(0, y1 - 22)), label, fill="white", font=text_font)
    return img


def count_dataset(dataset_dir: Path, names: list[str]) -> dict:
    summary: dict[str, dict] = {}
    total_counter = Counter()
    for split in ["train", "valid", "test"]:
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        images = sorted(image_dir.glob("*.jpg"))
        labels = sorted(label_dir.glob("*.txt"))
        counter = Counter()
        objects = 0
        for label_path in labels:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    cls_id = int(float(line.split()[0]))
                    counter[names[cls_id]] += 1
                    total_counter[names[cls_id]] += 1
                    objects += 1
        summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": objects,
            "class_counts": dict(counter),
        }
    summary["total"] = {
        "images": sum(summary[s]["images"] for s in ["train", "valid", "test"]),
        "labels": sum(summary[s]["labels"] for s in ["train", "valid", "test"]),
        "objects": sum(summary[s]["objects"] for s in ["train", "valid", "test"]),
        "class_counts": dict(total_counter),
    }
    return summary


def save_distribution(summary: dict, names: list[str], output: Path) -> None:
    counts = [summary["total"]["class_counts"].get(name, 0) for name in names]
    plt.figure(figsize=(9, 5), dpi=160)
    bars = plt.bar(names, counts, color=["#e63946", "#f4a261", "#7cb518", "#fb8500", "#6a994e", "#2a9d8f"])
    plt.title("Fruit dataset class distribution")
    plt.ylabel("Object count")
    plt.xticks(rotation=25, ha="right")
    for bar, value in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def save_samples(dataset_dir: Path, names: list[str], output: Path, sample_count: int, seed: int) -> None:
    rng = random.Random(seed)
    image_dir = dataset_dir / "valid" / "images"
    image_paths = sorted(image_dir.glob("*.jpg"))
    selected = rng.sample(image_paths, min(sample_count, len(image_paths)))
    thumbs = []
    for image_path in selected:
        label_path = dataset_dir / "valid" / "labels" / f"{image_path.stem}.txt"
        img = draw_boxes(image_path, label_path, names).resize((256, 256), Image.Resampling.LANCZOS)
        thumbs.append(img)
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * 256 + (cols + 1) * 12, rows * 256 + (rows + 1) * 12), (245, 245, 245))
    for idx, img in enumerate(thumbs):
        r, c = divmod(idx, cols)
        canvas.paste(img, (12 + c * 268, 12 + r * 268))
    canvas.save(output, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dataset statistics and visualization screenshots.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_data_yaml(args.dataset_dir)
    names = data["names"]
    figures = ensure_dir(args.reports_dir / "figures")
    metrics = ensure_dir(args.reports_dir / "metrics")
    summary = count_dataset(args.dataset_dir, names)
    (metrics / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_distribution(summary, names, figures / "dataset_class_distribution.png")
    save_samples(args.dataset_dir, names, figures / "dataset_label_samples.jpg", args.samples, args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

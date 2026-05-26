from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fruit_detection.paths import REPORTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a concise Chinese experiment summary.")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    dataset = json.loads((args.reports_dir / "metrics" / "dataset_summary.json").read_text(encoding="utf-8"))
    comparison_path = args.reports_dir / "metrics" / "model_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    lines = [
        "# 实验结果摘要",
        "",
        "## 数据集",
        "",
        "- 数据来源：`lightly-ai/dataset_fruits_detection`，其 README 标注来源为 Kaggle Fruit Detection 数据集。",
        "- 数据格式：YOLOv8 目标检测格式。",
        "- 类别：Apple、Banana、Grape、Orange、Pineapple、Watermelon。",
        f"- 图片数量：训练集 {dataset['train']['images']} 张，验证集 {dataset['valid']['images']} 张，测试集 {dataset['test']['images']} 张，总计 {dataset['total']['images']} 张。",
        f"- 标注目标数：总计 {dataset['total']['objects']} 个目标。",
        "",
        "## 参考项目实现路线",
        "",
        "- 参考仓库：`https://github.com/zolppy/object-detection-with-yolo`。",
        "- 主模型：YOLOv8s 预训练模型 `yolov8s.pt`。",
        "- 主训练参数：epochs=20，imgsz=640，batch=16，pretrained=True。",
        "- 对比模型：YOLOv8n，使用同一数据集和同一训练流程。",
        "",
        "## 模型对比",
        "",
    ]
    if not comparison.empty:
        header = "| " + " | ".join(comparison.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(comparison.columns)) + " |"
        lines.extend([header, sep])
        for _, row in comparison.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in comparison.columns) + " |")
        lines.append("")
    lines.extend(
        [
            "## 主要文件",
            "",
            "- 数据集截图：`reports/figures/dataset_label_samples.jpg`",
            "- 类别分布：`reports/figures/dataset_class_distribution.png`",
            "- YOLOv8s 训练曲线：`reports/figures/yolov8s_fruits_training_results.png`",
            "- YOLOv8s 混淆矩阵：`reports/figures/yolov8s_fruits_confusion_matrix.png`",
            "- 模型对比表截图：`reports/figures/model_comparison_table.png`",
            "- 测试集预测对比：`reports/figures/model_prediction_comparison.jpg`",
        ]
    )
    (args.reports_dir / "实验结果摘要.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.reports_dir / "实验结果摘要.md")


if __name__ == "__main__":
    main()

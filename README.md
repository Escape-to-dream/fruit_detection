# 基于 YOLOv8 的多种水果检测项目

本项目参考 `https://github.com/zolppy/object-detection-with-yolo` 的说明和代码流程完成：使用 Ultralytics YOLOv8，加载预训练 YOLOv8s，在公开水果检测数据集上训练、验证、推理，并输出训练曲线、模型权重、指标表和检测截图。

## 数据集

数据集放在 `data/dataset_fruits_detection/`，克隆自 `lightly-ai/dataset_fruits_detection`。该数据集 README 说明来源为 Kaggle Fruit Detection 数据集，包含 8479 张 640x640 图片，6 类水果：

- Apple
- Banana
- Grape
- Orange
- Pineapple
- Watermelon

数据划分：

- 训练集：7108 张
- 验证集：914 张
- 测试集：457 张

## 训练方案

主模型与参考项目保持一致：

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    pretrained=True
)
```

本地脚本中同时训练 `YOLOv8n` 做轻量模型对比。

## 目录说明

- `references/object-detection-with-yolo/`：参考 GitHub 项目代码。
- `data/dataset_fruits_detection/`：真实水果检测数据集。
- `src/fruit_detection/`：训练、统计、对比和推理脚本。
- `models/`：训练完成的模型权重。
- `runs/`：Ultralytics 原始训练输出。
- `reports/metrics/`：指标 JSON/CSV。
- `reports/figures/`：训练曲线、混淆矩阵、数据集样例、对比图。
- `reports/predictions/`：测试集检测结果图。
- `scripts/`：Windows 批处理入口。

## Windows 环境迁移

复制整个项目到其他 Windows 后运行：

```bat
scripts\setup_windows.bat
```

如果有 NVIDIA GPU，脚本默认安装 CUDA 12.8 对应的 PyTorch wheel。没有 GPU 时，可把 `scripts/setup_windows.ps1` 中 PyTorch 安装命令替换为 CPU 版。

## 运行

生成数据集截图与统计：

```bat
scripts\dataset_report.bat
```

训练参考主模型：

```bat
scripts\train_yolov8s.bat
```

训练对比模型：

```bat
scripts\train_yolov8n_compare.bat
```

生成对比表和摘要：

```bat
scripts\compare_models.bat
```

完整流程：

```bat
scripts\run_full_pipeline.bat
```

## 交付物

1. 完整数据集：`data/dataset_fruits_detection/`
2. 完成训练模型：`models/yolov8s_fruits_best.pt`、`models/yolov8n_fruits_best.pt`
3. 训练过程数据及截图：`runs/`、`reports/figures/*training_results.png`
4. 与其他模型的对比数据和截图：`reports/metrics/model_comparison.csv`、`reports/figures/model_comparison_*.png`
5. 可迁移 Windows Python 虚拟环境：`requirements.txt`、`scripts/setup_windows.ps1`、`scripts/setup_windows.bat`

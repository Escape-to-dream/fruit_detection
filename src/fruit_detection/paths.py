from __future__ import annotations

from pathlib import Path


def root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


DATASET_DIR = root_dir() / "data" / "dataset_fruits_detection"
REPORTS_DIR = root_dir() / "reports"
MODELS_DIR = root_dir() / "models"

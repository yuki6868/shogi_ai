# backend/ai/path_config.py

from __future__ import annotations

from pathlib import Path


AI_DIR = Path(__file__).resolve().parent
BACKEND_DIR = AI_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent

DEFAULT_DATASET_DIR = WORKSPACE_DIR / "dataset"

DEFAULT_WDOOR_ARCHIVE = WORKSPACE_DIR / "wdoor2025.7z"


def find_wdoor_archive() -> Path:
    candidates = [
        WORKSPACE_DIR / "wdoor2025.7z",
        WORKSPACE_DIR / "wdoor2025.7r",
        PROJECT_DIR / "wdoor2025.7z",
        PROJECT_DIR / "wdoor2025.7r",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "wdoor2025.7z が見つかりません。"
        f"探した場所: {[str(p) for p in candidates]}"
    )


def ensure_dataset_dir() -> Path:
    DEFAULT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DATASET_DIR
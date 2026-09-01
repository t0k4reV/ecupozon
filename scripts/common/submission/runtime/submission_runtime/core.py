"""Shared configuration, validation, and filesystem helpers for submission runtime."""

from __future__ import annotations

import html
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SUPPLEMENTS_CATEGORY = "БАД"
FLAMMABLE_CATEGORY = "Легковоспламеняющиеся"
SUPPORTED_CATEGORIES = (SUPPLEMENTS_CATEGORY, FLAMMABLE_CATEGORY)
ADAPTER_NAME_BY_CATEGORY = {SUPPLEMENTS_CATEGORY: "supplements", FLAMMABLE_CATEGORY: "flammable"}
REQUIRED_TEST_DATA_COLUMNS = {"id", "name", "description", "category"}
SUBMISSION_RESULT_PATTERN = re.compile(r"^<комментарий>(.*)<вердикт>(бан|не бан)$", re.DOTALL)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

CLASSIFICATION_RULES = {
    SUPPLEMENTS_CATEGORY: (
        "1: есть прямое указание БАД или dietary supplement. "
        "0: спортивное питание, явное отрицание принадлежности к БАД или "
        "отсутствие маркировки БАД."
    ),
    FLAMMABLE_CATEGORY: (
        "1: самостоятельный источник воспламенения, горючее вещество/газ "
        "или такой товар входит в комплект. 0: пустое устройство для работы "
        "с огнём, встроенный поджиг, горючий компонент другого изделия или "
        "легковоспламеняющийся предмет не входит в комплект."
    ),
}


@dataclass(frozen=True)
class CommentEvidence:
    source: str
    reason: str
    photos: tuple[int, ...]
    confidence: str
    fact: str = ""


def log_event(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def get_submission_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def max_images_for_category(submission_config: Mapping[str, Any], category: str) -> int:
    configured_counts = submission_config.get("max_images", {})
    default_count = 1 if category == SUPPLEMENTS_CATEGORY else 3
    image_count = (
        int(configured_counts.get(category, default_count))
        if isinstance(configured_counts, Mapping)
        else default_count
    )
    if not 1 <= image_count <= 5:
        raise ValueError(f"Invalid image count for {category}: {image_count}")
    return image_count


def load_submission_config(submission_directory: Path) -> dict[str, Any]:
    config_path = submission_directory / "submission_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Submission config is absent: {config_path}")
    submission_config = json.loads(config_path.read_text(encoding="utf-8"))
    if set(submission_config.get("adapters", {})) != set(SUPPORTED_CATEGORIES):
        raise ValueError("Config must contain adapters for both categories")
    if set(submission_config.get("thresholds", {})) != set(SUPPORTED_CATEGORIES):
        raise ValueError("Config must contain thresholds for both categories")
    for category, threshold in submission_config["thresholds"].items():
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"Invalid threshold for {category}: {threshold}")
        max_images_for_category(submission_config, category)
    if not isinstance(submission_config.get("ocr"), Mapping):
        raise ValueError("Config must contain EasyOCR settings")
    classification_categories = submission_config["ocr"].get("classification_categories", [])
    if not isinstance(classification_categories, list) or not set(
        classification_categories
    ).issubset(SUPPORTED_CATEGORIES):
        raise ValueError("Invalid OCR classification categories")
    return submission_config


def load_test_data(csv_path: Path) -> pd.DataFrame:
    products = pd.read_csv(csv_path, dtype={"id": "string"})
    missing_columns = REQUIRED_TEST_DATA_COLUMNS - set(products.columns)
    if missing_columns:
        raise ValueError(f"Missing input columns: {sorted(missing_columns)}")
    if products["id"].isna().any() or products["id"].duplicated().any():
        raise ValueError("Input ids must be non-empty and unique")
    categories = set(products["category"].dropna().astype(str))
    unsupported_categories = categories - set(SUPPORTED_CATEGORIES)
    if products["category"].isna().any() or unsupported_categories:
        raise ValueError(f"Unsupported categories: {sorted(unsupported_categories)}")
    products = products.copy()
    # Preserve the exact classifier prompt used by the validated best runtime.
    # Comment rules sanitize their own copy through normalize_text().
    products["name"] = products["name"].fillna("").astype(str)
    products["description"] = products["description"].fillna("").astype(str)
    products["category"] = products["category"].astype(str)
    return products


def resolve_shared_model_path(model_id: str) -> Path:
    shared_models_directory = Path(os.environ.get("SHARED_MODELS_PATH", "/shared_models"))
    model_directory = shared_models_directory / model_id
    if not model_directory.is_dir():
        raise FileNotFoundError(
            f"Offline base model is unavailable: {model_directory}. "
            "Set SHARED_MODELS_PATH to the directory containing google/."
        )
    return model_directory


def find_product_image_paths(
    images_directory: Path, product_id: str, limit: int
) -> list[tuple[int, Path]]:
    product_image_directory = images_directory / product_id
    if not product_image_directory.is_dir():
        return []
    image_paths = sorted(
        image_path
        for image_path in product_image_directory.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return list(enumerate(image_paths[:limit], start=1))

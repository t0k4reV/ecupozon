"""Create the runtime configuration shared by ZIP building and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUPPLEMENTS_CATEGORY = "БАД"
FLAMMABLE_CATEGORY = "Легковоспламеняющиеся"
SUPPORTED_CATEGORIES = (SUPPLEMENTS_CATEGORY, FLAMMABLE_CATEGORY)


def build_submission_config(
    *,
    solution: str,
    model_id: str,
    adapters: Mapping[str, str],
    inference_by_category: Mapping[str, Mapping[str, object]],
    ocr_affects_classification: bool,
) -> dict[str, Any]:
    """Build and validate the exact configuration consumed by the runtime."""
    expected_categories = set(SUPPORTED_CATEGORIES)
    if set(adapters) != expected_categories or set(inference_by_category) != expected_categories:
        raise ValueError("Runtime configuration requires both supported categories")

    for category, inference in inference_by_category.items():
        threshold = inference.get("threshold")
        max_images = inference.get("max_images")
        aggregation = inference.get("aggregation")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise TypeError(f"Invalid threshold for {category}")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"Invalid threshold for {category}: {threshold}")
        if not isinstance(max_images, int) or isinstance(max_images, bool) or max_images < 1:
            raise TypeError(f"Invalid max_images for {category}")
        if aggregation not in {"single", "max"}:
            raise ValueError(f"Invalid aggregation for {category}: {aggregation}")

    return {
        "solution": solution,
        "model_id": model_id,
        "adapters": dict(adapters),
        "thresholds": {
            category: float(inference_by_category[category]["threshold"])
            for category in SUPPORTED_CATEGORIES
        },
        "batch_size": 2,
        "seed": 2026,
        "max_description_chars": 8000,
        "max_image_side": 1024,
        "max_images": {
            category: int(inference_by_category[category]["max_images"])
            for category in SUPPORTED_CATEGORIES
        },
        "ocr": {
            "model_directory": "artifacts/easyocr",
            "languages": ["ru", "en"],
            "classification_categories": (
                [FLAMMABLE_CATEGORY] if ocr_affects_classification else []
            ),
            "supplements_images": 0 if ocr_affects_classification else 1,
            "flammable_images": int(inference_by_category[FLAMMABLE_CATEGORY]["max_images"]),
            "recognition_batch_size": 128,
            "workers": 4,
            "canvas_size": 2560,
            "min_confidence": 0.25,
        },
    }

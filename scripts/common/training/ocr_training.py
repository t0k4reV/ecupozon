"""EasyOCR cache helpers shared by OCR-v1 preparation and training."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.common.training.lora_training import MAX_DESCRIPTION_CHARS, TrainingProfile

MAX_OCR_CHARS = 2000


def normalize_ocr_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def get_ocr_cache_key(product_id: str, image_path: Path) -> str:
    return f"{product_id}/{image_path.name}"


def load_ocr_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.is_file():
        raise FileNotFoundError(f"OCR cache not found: {cache_path}")
    cache: dict[str, str] = {}
    with cache_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid OCR JSONL at line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"OCR cache line {line_number} is not an object")
            product_id = str(record.get("id", "")).strip()
            image_name = Path(str(record.get("image_path", ""))).name
            if not product_id or not image_name:
                raise ValueError(f"OCR cache line {line_number} has no id or image path")
            key = f"{product_id}/{image_name}"
            text = normalize_ocr_text(record.get("ocr"))[:MAX_OCR_CHARS]
            if key in cache and cache[key] != text:
                if not cache[key] and text:
                    cache[key] = text
                    continue
                if cache[key] and not text:
                    continue
                raise ValueError(f"Conflicting duplicate OCR record: {key}")
            cache[key] = text
    if not cache:
        raise ValueError(f"OCR cache contains no records: {cache_path}")
    return cache


def get_photo_number(product: dict[str, Any], selected_image_path: Path) -> int:
    image_paths = product.get("image_paths")
    if isinstance(image_paths, list):
        selected_name = selected_image_path.name
        for index, relative_path in enumerate(image_paths[:3], start=1):
            if Path(str(relative_path)).name == selected_name:
                return index
    return 1


def build_ocr_training_content(
    product: dict[str, Any],
    profile: TrainingProfile,
    selected_image_paths: list[Path],
    ocr_cache: dict[str, str],
) -> list[dict[str, str]]:
    content: list[dict[str, str]] = []
    if selected_image_paths:
        image_path = selected_image_paths[0]
        photo_number = get_photo_number(product, image_path)
        ocr_text = ocr_cache.get(get_ocr_cache_key(str(product["id"]), image_path), "")
        content.append(
            {
                "type": "text",
                "text": f"OCR изображения {photo_number}: {ocr_text or 'текст не распознан'}",
            }
        )
    description = str(product.get("description", ""))[:MAX_DESCRIPTION_CHARS]
    prompt = (
        f"Категория: {profile.category}\n"
        f"Правила: {profile.classification_rule}\n"
        f"Название: {product.get('name', '')}\n"
        f"Описание: {description}\n"
        "OCR каждой фотографии может содержать ошибки. "
        "Определи соответствие правилам. Ответь только цифрой 0 или 1."
    )
    content.append({"type": "text", "text": prompt})
    return content

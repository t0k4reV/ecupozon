"""Offline EasyOCR extraction for comment evidence."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .core import (
    SUPPLEMENTS_CATEGORY,
    find_product_image_paths,
    log_event,
    max_images_for_category,
    normalize_text,
)
from .rules import detect_comment_evidence, expected_label_for_reason


def calculate_file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_precomputed_ocr_texts(
    *,
    test_products: pd.DataFrame,
    images_directory: Path,
    submission_config: Mapping[str, Any],
    cache_path: Path,
    expected_sha256: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load a validated JSONL OCR cache using live-runtime absolute path keys."""
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    cache_sha256 = calculate_file_sha256(cache_path)
    if expected_sha256 is not None and cache_sha256 != expected_sha256:
        raise ValueError(
            f"OCR cache SHA-256 mismatch: expected {expected_sha256}, found {cache_sha256}"
        )

    needed_path_by_key: dict[tuple[str, str], Path] = {}
    for product in test_products.itertuples(index=False):
        product_id = str(product.id)
        image_limit = max_images_for_category(submission_config, str(product.category))
        for _, image_path in find_product_image_paths(
            images_directory,
            product_id,
            image_limit,
        ):
            key = (product_id, image_path.name)
            if key in needed_path_by_key:
                raise ValueError(f"Duplicate product image key: {product_id}/{image_path.name}")
            needed_path_by_key[key] = image_path.resolve()
    if not needed_path_by_key:
        raise ValueError("No product images are available for OCR cache validation")

    cached_record_by_key: dict[tuple[str, str], tuple[str, str]] = {}
    parsed_records = 0
    relevant_records = 0
    duplicate_records = 0
    recovered_errors = 0
    with cache_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            parsed_records += 1
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
            key = (product_id, image_name)
            if key not in needed_path_by_key:
                continue
            relevant_records += 1
            text = str(record.get("ocr", "") or "").strip()
            error_message = str(record.get("error", "") or "").strip()
            previous = cached_record_by_key.get(key)
            if previous is None:
                cached_record_by_key[key] = (text, error_message)
                continue
            duplicate_records += 1
            previous_text, previous_error = previous
            if previous_text == text:
                if previous_error and not error_message:
                    cached_record_by_key[key] = (text, error_message)
                continue
            if not previous_text and text and not error_message:
                cached_record_by_key[key] = (text, error_message)
                if previous_error:
                    recovered_errors += 1
                continue
            if previous_text and not text and error_message:
                continue
            raise ValueError(f"Conflicting OCR cache records for {product_id}/{image_name}")

    missing_keys = sorted(set(needed_path_by_key) - set(cached_record_by_key))
    if missing_keys:
        preview = [f"{product_id}/{image_name}" for product_id, image_name in missing_keys[:10]]
        raise ValueError(
            f"OCR cache misses {len(missing_keys)} required images; examples: {preview}"
        )
    ocr_text_by_path = {
        str(needed_path_by_key[key]): cached_record_by_key[key][0] for key in needed_path_by_key
    }
    metadata = {
        "path": str(cache_path),
        "sha256": cache_sha256,
        "parsed_records": parsed_records,
        "relevant_records": relevant_records,
        "required_images": len(needed_path_by_key),
        "non_empty_images": sum(bool(text) for text in ocr_text_by_path.values()),
        "duplicate_records": duplicate_records,
        "recovered_errors": recovered_errors,
    }
    return ocr_text_by_path, metadata


def _normalize_ocr_output(ocr_detections: Sequence[Any], min_confidence: float) -> str:
    recognized_lines: list[tuple[float, float, str]] = []
    for detection in ocr_detections:
        if not isinstance(detection, (list, tuple)) or len(detection) < 3:
            continue
        bounding_box, text, confidence = (
            detection[0],
            normalize_text(detection[1]),
            float(detection[2]),
        )
        if not text or confidence < min_confidence:
            continue
        try:
            left = min(float(point[0]) for point in bounding_box)
            top = min(float(point[1]) for point in bounding_box)
        except (TypeError, ValueError, IndexError):
            left = top = 0.0
        recognized_lines.append((top, left, text))
    recognized_lines.sort(key=lambda detection: (round(detection[0] / 20), detection[1]))
    return " ".join(detection[2] for detection in recognized_lines)


def extract_ocr_texts(
    *,
    test_products: pd.DataFrame,
    images_directory: Path,
    submission_directory: Path,
    submission_config: Mapping[str, Any],
    strict: bool = False,
) -> dict[str, str]:
    ocr_settings = submission_config["ocr"]
    classification_categories = {
        str(category) for category in ocr_settings.get("classification_categories", [])
    }

    image_paths_to_process: list[Path] = []
    seen_image_paths: set[str] = set()
    for product in test_products.itertuples(index=False):
        category = str(product.category)
        image_limit = max_images_for_category(submission_config, category)
        if category in classification_categories:
            for _, image_path in find_product_image_paths(
                images_directory, str(product.id), image_limit
            ):
                image_key = str(image_path.resolve())
                if image_key not in seen_image_paths:
                    seen_image_paths.add(image_key)
                    image_paths_to_process.append(image_path)
            continue
        text_evidence = detect_comment_evidence(
            category=category,
            name=str(product.name),
            description=str(product.description),
            product_id=str(product.id),
            images_directory=images_directory,
            image_limit=0,
            ocr_text_by_path={},
        )
        if category == SUPPLEMENTS_CATEGORY:
            if text_evidence is not None:
                continue
            image_limit = min(image_limit, int(ocr_settings.get("supplements_images", 1)))
        else:
            if text_evidence is None or expected_label_for_reason(text_evidence.reason) != 1:
                continue
            image_limit = min(image_limit, int(ocr_settings.get("flammable_images", 3)))
        for _, image_path in find_product_image_paths(
            images_directory, str(product.id), image_limit
        ):
            image_key = str(image_path.resolve())
            if image_key not in seen_image_paths:
                seen_image_paths.add(image_key)
                image_paths_to_process.append(image_path)
    if not image_paths_to_process:
        return {}
    log_event("ocr_planned", images=len(image_paths_to_process), rows=len(test_products))

    try:
        import easyocr
    except ImportError as error:
        if strict:
            raise RuntimeError("EasyOCR is required for comment auditing") from error
        log_event("ocr_unavailable", error=f"{type(error).__name__}: {error}")
        return {}

    model_directory = Path(str(ocr_settings.get("model_directory", "artifacts/easyocr")))
    if not model_directory.is_absolute():
        model_directory = submission_directory / model_directory
    ocr_languages = list(ocr_settings.get("languages", ["ru", "en"]))
    started_at = time.perf_counter()
    ocr_text_by_path: dict[str, str] = {}
    try:
        ocr_reader = easyocr.Reader(
            ocr_languages,
            gpu="cuda",
            verbose=False,
            model_storage_directory=str(model_directory),
            download_enabled=False,
            cudnn_benchmark=False,
        )
    except Exception as error:
        if strict:
            raise RuntimeError("EasyOCR initialization failed") from error
        log_event("ocr_initialization_failed", error=f"{type(error).__name__}: {error}")
        return {}

    batch_size = int(ocr_settings.get("recognition_batch_size", 128))
    workers = int(ocr_settings.get("workers", 4))
    canvas_size = int(ocr_settings.get("canvas_size", 2560))
    min_confidence = float(ocr_settings.get("min_confidence", 0.25))
    for processed_count, image_path in enumerate(image_paths_to_process, start=1):
        image_key = str(image_path.resolve())
        try:
            ocr_result = ocr_reader.readtext(
                str(image_path),
                detail=1,
                paragraph=False,
                decoder="greedy",
                batch_size=batch_size,
                workers=workers,
                canvas_size=canvas_size,
                mag_ratio=1.0,
            )
            ocr_text_by_path[image_key] = _normalize_ocr_output(ocr_result, min_confidence)
        except Exception as error:
            ocr_text_by_path[image_key] = ""
            log_event(
                "ocr_image_failed",
                image_path=image_path.name,
                error=f"{type(error).__name__}: {error}",
            )
        if processed_count % 100 == 0 or processed_count == len(image_paths_to_process):
            log_event("ocr_progress", done=processed_count, total=len(image_paths_to_process))

    del ocr_reader
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    log_event(
        "ocr_complete",
        images=len(image_paths_to_process),
        non_empty=sum(bool(value) for value in ocr_text_by_path.values()),
        seconds=round(time.perf_counter() - started_at, 3),
    )
    return ocr_text_by_path

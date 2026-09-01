"""Run production inference and audit deterministic comments on labeled rows."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.common.submission.configuration import (
    FLAMMABLE_CATEGORY,
    SUPPLEMENTS_CATEGORY,
    SUPPORTED_CATEGORIES,
)
from scripts.common.submission.download_easyocr import EASYOCR_MODELS
from scripts.common.submission.runtime.submission_runtime.comments import (
    build_submission_comment_records,
)
from scripts.common.submission.runtime.submission_runtime.core import (
    ADAPTER_NAME_BY_CATEGORY,
    find_product_image_paths,
    max_images_for_category,
)
from scripts.common.submission.runtime.submission_runtime.inference import (
    get_label_token_ids,
    predict_category_probabilities,
)
from scripts.common.submission.runtime.submission_runtime.pipeline import (
    format_submission_result,
    validate_submission_output,
)
from scripts.common.submission.runtime.submission_runtime.rules import expected_label_for_reason
from scripts.common.training.lora_training import calculate_sha256

COMMENT_FORBIDDEN_CHARACTERS = ("\n", "\r", "<", ">")


def verify_easyocr_weights(project_root: Path) -> None:
    models_directory = project_root / "artifacts" / "easyocr"
    for filename, metadata in EASYOCR_MODELS.items():
        model_path = models_directory / filename
        if not model_path.is_file() or calculate_sha256(model_path) != metadata["sha256"]:
            raise ValueError(f"EasyOCR weight is absent or invalid: {model_path}")


def load_model_with_adapters(
    project_root: Path,
    adapter_directory_by_category: dict[str, Path],
) -> tuple[Any, Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for post-training evaluation")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark = False

    model_directory = project_root / "models" / "google" / "gemma-4-E4B-it"
    if not model_directory.is_dir():
        raise FileNotFoundError(model_directory)
    processor = AutoProcessor.from_pretrained(model_directory, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForMultimodalLM.from_pretrained(
        model_directory,
        dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    base_model.config.use_cache = True
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_directory_by_category[SUPPLEMENTS_CATEGORY]),
        adapter_name=ADAPTER_NAME_BY_CATEGORY[SUPPLEMENTS_CATEGORY],
        local_files_only=True,
    )
    model.load_adapter(
        str(adapter_directory_by_category[FLAMMABLE_CATEGORY]),
        adapter_name=ADAPTER_NAME_BY_CATEGORY[FLAMMABLE_CATEGORY],
        local_files_only=True,
    )
    model.eval()
    return processor, base_model, model


def score_products(
    products: pd.DataFrame,
    *,
    model: Any,
    processor: Any,
    images_directory: Path,
    submission_config: dict[str, Any],
    ocr_text_by_path: dict[str, str],
    batch_size: int,
) -> pd.DataFrame:
    label_token_ids = get_label_token_ids(processor)
    rows: list[dict[str, Any]] = []
    for category in SUPPORTED_CATEGORIES:
        category_products = products.loc[products["category"].eq(category)]
        if category_products.empty:
            continue
        max_images = max_images_for_category(submission_config, category)
        results = predict_category_probabilities(
            category_products=category_products,
            category=category,
            adapter_name=ADAPTER_NAME_BY_CATEGORY[category],
            model=model,
            processor=processor,
            label_token_ids=label_token_ids,
            images_directory=images_directory,
            batch_size=batch_size,
            submission_config=submission_config,
            max_images=max_images,
            ocr_text_by_path=ocr_text_by_path,
        )
        for (source_index, product), result in zip(
            category_products.iterrows(), results, strict=True
        ):
            row: dict[str, Any] = {
                "source_index": int(source_index),
                "id": str(product["id"]),
                "category": category,
                "label": int(product["label"]),
                "probability": float(result["probability"]),
                "p_max": float(result["probability"]),
                "threshold": float(submission_config["thresholds"][category]),
                "selected_photo": result["selected_photo"],
                "p_no_image": math.nan,
            }
            row.update({f"p_img{index}": math.nan for index in range(1, max_images + 1)})
            for photo_number, probability in result["image_probabilities"]:
                column = "p_no_image" if photo_number is None else f"p_img{photo_number}"
                row[column] = float(probability)
            row["prediction"] = int(row["probability"] >= row["threshold"])
            row["correct"] = row["prediction"] == row["label"]
            rows.append(row)
    predictions = pd.DataFrame(rows).sort_values("source_index").reset_index(drop=True)
    if len(predictions) != len(products):
        raise RuntimeError("Prediction count differs from evaluation row count")
    return predictions


def build_comments_review(
    products: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    images_directory: Path,
    ocr_text_by_path: dict[str, str],
    submission_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_by_index = predictions.set_index("source_index")
    labels_by_index = {
        int(index): int(row.prediction) for index, row in prediction_by_index.iterrows()
    }
    comment_records, comment_statistics = build_submission_comment_records(
        test_products=products,
        predicted_labels_by_index=labels_by_index,
        images_directory=images_directory,
        ocr_text_by_path=ocr_text_by_path,
        submission_config=submission_config,
    )

    review_rows: list[dict[str, Any]] = []
    invalid_ids: list[str] = []
    for source_index, product in products.iterrows():
        prediction = prediction_by_index.loc[int(source_index)]
        comment_record = comment_records[int(source_index)]
        comment = str(comment_record["comment"])
        result = format_submission_result(int(prediction["prediction"]), comment)
        image_limit = max_images_for_category(submission_config, str(product["category"]))
        existing_photos = {
            number
            for number, _ in find_product_image_paths(
                images_directory,
                str(product["id"]),
                image_limit,
            )
        }
        referenced_photos = {
            int(value) for value in re.findall(r"\bфото\s+(\d+)\b", comment, re.IGNORECASE)
        }
        evidence_reason = str(comment_record["evidence_reason"])
        evidence_consistent = not evidence_reason or (
            expected_label_for_reason(evidence_reason) == int(prediction["prediction"])
        )
        comment_valid = (
            50 <= len(comment) <= 300
            and not any(character in comment for character in COMMENT_FORBIDDEN_CHARACTERS)
            and referenced_photos.issubset(existing_photos)
            and evidence_consistent
        )
        if not comment_valid:
            invalid_ids.append(str(product["id"]))
        review_rows.append(
            {
                "id": str(product["id"]),
                "category": str(product["category"]),
                "label": int(product["label"]),
                "prediction": int(prediction["prediction"]),
                "correct": bool(prediction["correct"]),
                "probability": float(prediction["probability"]),
                "threshold": float(prediction["threshold"]),
                "selected_photo": prediction["selected_photo"],
                "name": str(product["name"]),
                "description": str(product["description"]),
                "comment": comment,
                "comment_length": len(comment),
                "comment_valid": comment_valid,
                "comment_source": comment_record["comment_source"],
                "evidence_source": comment_record["evidence_source"],
                "evidence_reason": evidence_reason,
                "evidence_confidence": comment_record["evidence_confidence"],
                "evidence_photos": ",".join(
                    str(number) for number in comment_record["evidence_photos"]
                ),
                "conflicting_reason": comment_record["conflicting_reason"],
                "ocr_photo": comment_record["ocr_photo"],
                "ocr_excerpt": comment_record["ocr_excerpt"],
                "result": result,
            }
        )

    review = pd.DataFrame(review_rows)
    validate_submission_output(products, review[["id", "result"]].copy())
    audit = {
        "passed": not invalid_ids,
        "rows": len(review),
        "valid_comments": int(review["comment_valid"].sum()),
        "invalid_ids": invalid_ids,
        "classification_errors_for_review": int((~review["correct"]).sum()),
        "comment_sources": dict(Counter(review["comment_source"])),
        "evidence_reasons": dict(Counter(reason for reason in review["evidence_reason"] if reason)),
        "runtime_statistics": comment_statistics,
        "ocr_images_processed": len(ocr_text_by_path),
        "ocr_images_with_text": sum(bool(text) for text in ocr_text_by_path.values()),
    }
    return review, audit

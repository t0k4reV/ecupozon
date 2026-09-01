"""Rebuild and compare deterministic comments without running model inference."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.common.evaluation.data import load_labeled_products
from scripts.common.evaluation.runtime import (
    build_comments_review,
    verify_easyocr_weights,
)
from scripts.common.solution_profile import load_solution_profile
from scripts.common.submission.build_archive import load_and_validate_training_manifest
from scripts.common.submission.configuration import (
    SUPPORTED_CATEGORIES,
    build_submission_config,
)
from scripts.common.submission.runtime.submission_runtime.ocr import (
    extract_ocr_texts,
    load_precomputed_ocr_texts,
)
from scripts.common.training.lora_training import MODEL_ID, calculate_sha256, write_json_atomic

HISTORICAL_COMMENTS_SHA256 = "d6639e42d16be79e5e85d2eb78439afb9f6e67febe7901572c4fff54f20673ed"
HISTORICAL_OCR_CACHE_SHA256 = "0999fcf59309d5d411c6fe35540ae2eab7b244567ce00baa2c75383aafaf3bf0"
REQUIRED_PREDICTION_COLUMNS = {
    "id",
    "category",
    "label",
    "prediction",
    "probability",
    "threshold",
    "selected_photo",
}
REQUIRED_GOLDEN_COLUMNS = {
    "id",
    "category",
    "gemma_v2_prediction",
    "gemma_v2_probability_1",
    "threshold",
    "comment",
    "result",
}
MISMATCH_COLUMNS = (
    "id",
    "category",
    "name",
    "mismatch_kind",
    "golden_prediction",
    "current_prediction",
    "golden_probability",
    "current_probability",
    "golden_threshold",
    "current_threshold",
    "golden_comment",
    "current_comment",
    "golden_result",
    "current_result",
)


def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_comments_audit_arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ocr-cache", type=Path)
    parser.add_argument("--golden-comments", type=Path)
    parser.add_argument("--data-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _normalize_comparison_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def _load_predictions(predictions_path: Path, products: pd.DataFrame) -> pd.DataFrame:
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    predictions = pd.read_csv(predictions_path, dtype={"id": "string"})
    missing_columns = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"Prediction columns are absent: {sorted(missing_columns)}")
    if predictions["id"].isna().any() or predictions["id"].duplicated().any():
        raise ValueError("Prediction IDs must be non-empty and unique")

    product_ids = products["id"].astype("string")
    if set(predictions["id"]) != set(product_ids):
        missing_ids = set(product_ids) - set(predictions["id"])
        extra_ids = set(predictions["id"]) - set(product_ids)
        raise ValueError(
            f"Prediction/data ID mismatch: missing={len(missing_ids)}, extra={len(extra_ids)}"
        )
    ordered = predictions.set_index("id", drop=False).loc[product_ids].reset_index(drop=True)
    if not ordered["category"].astype(str).equals(products["category"].reset_index(drop=True)):
        raise ValueError("Prediction categories differ from evaluation data")
    labels = pd.to_numeric(ordered["label"], errors="raise").astype(int)
    if not labels.equals(products["label"].reset_index(drop=True)):
        raise ValueError("Prediction labels differ from evaluation data")
    predicted_labels = pd.to_numeric(ordered["prediction"], errors="raise").astype(int)
    if not predicted_labels.isin({0, 1}).all():
        raise ValueError("Predictions must contain only 0 and 1")
    probabilities = pd.to_numeric(ordered["probability"], errors="raise").astype(float)
    thresholds = pd.to_numeric(ordered["threshold"], errors="raise").astype(float)
    if not probabilities.between(0.0, 1.0).all() or not thresholds.between(0.0, 1.0).all():
        raise ValueError("Probabilities and thresholds must be between 0 and 1")

    ordered = ordered.copy()
    ordered.insert(0, "source_index", products.index.astype(int))
    ordered["label"] = labels
    ordered["prediction"] = predicted_labels
    ordered["probability"] = probabilities
    ordered["threshold"] = thresholds
    ordered["correct"] = ordered["prediction"].eq(ordered["label"])
    return ordered


def _load_golden_comments(golden_path: Path, expected_ids: set[str]) -> pd.DataFrame:
    if not golden_path.is_file():
        raise FileNotFoundError(golden_path)
    actual_sha256 = calculate_sha256(golden_path)
    if actual_sha256 != HISTORICAL_COMMENTS_SHA256:
        raise ValueError(
            "Historical comments SHA-256 mismatch: "
            f"expected {HISTORICAL_COMMENTS_SHA256}, found {actual_sha256}"
        )
    golden = pd.read_csv(golden_path, dtype={"id": "string"})
    missing_columns = REQUIRED_GOLDEN_COLUMNS - set(golden.columns)
    if missing_columns:
        raise ValueError(f"Golden comment columns are absent: {sorted(missing_columns)}")
    if golden["id"].isna().any() or golden["id"].duplicated().any():
        raise ValueError("Golden comment IDs must be non-empty and unique")
    if set(golden["id"]) != expected_ids:
        raise ValueError("Golden comments do not cover the same product IDs")
    return golden


def compare_comments_with_golden(
    current: pd.DataFrame,
    golden: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    golden_columns = [
        "id",
        "category",
        "gemma_v2_prediction",
        "gemma_v2_probability_1",
        "threshold",
        "comment",
        "result",
    ]
    comparison = current.merge(
        golden[golden_columns],
        on="id",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_current", "_golden"),
    )
    if not comparison["_merge"].eq("both").all():
        raise ValueError("Current and golden comment IDs do not match")
    if not comparison["category_current"].eq(comparison["category_golden"]).all():
        raise ValueError("Current and golden categories do not match")

    comparison["verdict_same"] = comparison["prediction"].eq(comparison["gemma_v2_prediction"])
    comparison["comment_exact"] = comparison["comment_current"].eq(comparison["comment_golden"])
    comparison["comment_normalized_exact"] = (
        comparison["comment_current"]
        .map(_normalize_comparison_text)
        .eq(comparison["comment_golden"].map(_normalize_comparison_text))
    )
    similarities = [
        SequenceMatcher(
            None,
            _normalize_comparison_text(current_comment),
            _normalize_comparison_text(golden_comment),
            autojunk=False,
        ).ratio()
        for current_comment, golden_comment in zip(
            comparison["comment_current"],
            comparison["comment_golden"],
            strict=True,
        )
    ]
    comparison["similarity"] = similarities
    same_verdict = comparison["verdict_same"]
    same_verdict_rows = int(same_verdict.sum())
    exact_same_verdict_rows = int((same_verdict & comparison["comment_exact"]).sum())
    normalized_same_verdict_rows = int(
        (same_verdict & comparison["comment_normalized_exact"]).sum()
    )
    valid_comments = int(current["comment_valid"].sum())
    reproduction_passed = (
        valid_comments == len(current)
        and same_verdict_rows > 0
        and exact_same_verdict_rows == same_verdict_rows
    )
    report = {
        "schema": "e_cup.comments_v4_reproduction",
        "schema_version": 1,
        "comments_reproduction_passed": reproduction_passed,
        "rows": len(comparison),
        "valid_comments": valid_comments,
        "verdict_same": same_verdict_rows,
        "verdict_same_rate": same_verdict_rows / len(comparison),
        "comment_exact": int(comparison["comment_exact"].sum()),
        "comment_exact_rate": float(comparison["comment_exact"].mean()),
        "comment_normalized_exact": int(comparison["comment_normalized_exact"].sum()),
        "comment_normalized_exact_rate": float(comparison["comment_normalized_exact"].mean()),
        "exact_when_same_verdict": exact_same_verdict_rows,
        "exact_when_same_verdict_rate": exact_same_verdict_rows / same_verdict_rows,
        "normalized_exact_when_same_verdict": normalized_same_verdict_rows,
        "normalized_exact_when_same_verdict_rate": (
            normalized_same_verdict_rows / same_verdict_rows
        ),
        "mean_similarity": float(comparison["similarity"].mean()),
        "median_similarity": float(comparison["similarity"].median()),
        "minimum_similarity": float(comparison["similarity"].min()),
        "mismatches": int((~comparison["comment_exact"]).sum()),
        "same_verdict_text_mismatches": int((same_verdict & ~comparison["comment_exact"]).sum()),
    }

    mismatches = comparison.loc[~comparison["comment_exact"]].copy()
    mismatch_rows = pd.DataFrame(
        {
            "id": mismatches["id"],
            "category": mismatches["category_current"],
            "name": mismatches["name"],
            "mismatch_kind": mismatches["verdict_same"].map(
                {True: "comment_changed", False: "verdict_changed"}
            ),
            "golden_prediction": mismatches["gemma_v2_prediction"],
            "current_prediction": mismatches["prediction"],
            "golden_probability": mismatches["gemma_v2_probability_1"],
            "current_probability": mismatches["probability"],
            "golden_threshold": mismatches["threshold_golden"],
            "current_threshold": mismatches["threshold_current"],
            "golden_comment": mismatches["comment_golden"],
            "current_comment": mismatches["comment_current"],
            "golden_result": mismatches["result_golden"],
            "current_result": mismatches["result_current"],
        },
        columns=MISMATCH_COLUMNS,
    )
    return report, mismatch_rows


def _write_outputs(
    output_directory: Path,
    *,
    comments_review: pd.DataFrame,
    comments_audit: dict[str, Any],
    reproduction_report: dict[str, Any],
    mismatches: pd.DataFrame,
) -> None:
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite comments audit: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_directory.parent,
        prefix=f".{output_directory.name}.",
    ) as temporary_name:
        temporary_directory = Path(temporary_name)
        comments_review.to_csv(temporary_directory / "comments_review.csv", index=False)
        mismatches.to_csv(temporary_directory / "comment_mismatches.csv", index=False)
        write_json_atomic(temporary_directory / "comments_audit.json", comments_audit)
        write_json_atomic(
            temporary_directory / "comments_reproduction.json",
            reproduction_report,
        )
        temporary_directory.replace(output_directory)


def run_comments_audit(args: argparse.Namespace, *, profile_path: Path) -> Path:
    project_root = args.project_root.expanduser().resolve()
    images_directory = args.images_dir.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    data_csv = (
        args.data_csv.expanduser().resolve()
        if args.data_csv is not None
        else project_root / "data_no_duplicates.csv"
    )
    solution_profile = load_solution_profile(profile_path)
    solution = str(solution_profile["solution"])
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "artifacts" / "gemma_lora" / solution / "comments_v4_audit"
    )
    if not images_directory.is_dir():
        raise FileNotFoundError(images_directory)
    if not data_csv.is_file():
        raise FileNotFoundError(data_csv)
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite comments audit: {output_directory}")

    products = load_labeled_products(data_csv)
    predictions = _load_predictions(predictions_path, products)
    _, adapter_directories, inference_by_category = load_and_validate_training_manifest(
        project_root,
        solution,
        solution_profile["training"]["flammable_inference"],
    )
    submission_config = build_submission_config(
        solution=solution,
        model_id=MODEL_ID,
        adapters={
            category: str(directory.relative_to(project_root))
            for category, directory in adapter_directories.items()
        },
        inference_by_category=inference_by_category,
        ocr_affects_classification=bool(
            solution_profile["submission"]["ocr_affects_classification"]
        ),
    )
    expected_thresholds = {
        category: float(submission_config["thresholds"][category])
        for category in SUPPORTED_CATEGORIES
    }
    actual_thresholds = {
        category: sorted(
            predictions.loc[predictions["category"].eq(category), "threshold"].unique()
        )
        for category in SUPPORTED_CATEGORIES
    }
    if any(
        actual_thresholds[category] != [expected]
        for category, expected in expected_thresholds.items()
    ):
        raise ValueError(
            f"Prediction thresholds differ from training manifest: {actual_thresholds}"
        )

    ocr_cache_metadata: dict[str, Any] | None = None
    if args.ocr_cache is not None:
        cache_path = args.ocr_cache.expanduser().resolve()
        expected_cache_sha256 = (
            HISTORICAL_OCR_CACHE_SHA256 if args.golden_comments is not None else None
        )
        ocr_text_by_path, ocr_cache_metadata = load_precomputed_ocr_texts(
            test_products=products,
            images_directory=images_directory,
            submission_config=submission_config,
            cache_path=cache_path,
            expected_sha256=expected_cache_sha256,
        )
        ocr_mode = "precomputed_full_cache"
    else:
        verify_easyocr_weights(project_root)
        ocr_text_by_path = extract_ocr_texts(
            test_products=products,
            images_directory=images_directory,
            submission_directory=project_root,
            submission_config=submission_config,
            strict=True,
        )
        ocr_mode = "live_candidate_ocr"

    comments_review, comments_audit = build_comments_review(
        products,
        predictions,
        images_directory=images_directory,
        ocr_text_by_path=ocr_text_by_path,
        submission_config=submission_config,
    )
    comments_audit["ocr_mode"] = ocr_mode
    if ocr_cache_metadata is not None:
        comments_audit["ocr_cache"] = ocr_cache_metadata

    mismatches = pd.DataFrame(columns=MISMATCH_COLUMNS)
    if args.golden_comments is not None:
        golden_path = args.golden_comments.expanduser().resolve()
        golden = _load_golden_comments(golden_path, set(products["id"]))
        reproduction_report, mismatches = compare_comments_with_golden(
            comments_review,
            golden,
        )
        reproduction_report["golden_comments_sha256"] = HISTORICAL_COMMENTS_SHA256
    else:
        reproduction_report = {
            "schema": "e_cup.comments_v4_reproduction",
            "schema_version": 1,
            "comments_reproduction_passed": None,
            "status": "golden_comparison_not_requested",
            "rows": len(comments_review),
            "valid_comments": int(comments_review["comment_valid"].sum()),
        }
    reproduction_report["ocr_mode"] = ocr_mode
    reproduction_report["provenance"] = {
        "data_csv": str(data_csv),
        "data_csv_sha256": calculate_sha256(data_csv),
        "predictions": str(predictions_path),
        "predictions_sha256": calculate_sha256(predictions_path),
        "ocr_cache_sha256": (
            ocr_cache_metadata["sha256"] if ocr_cache_metadata is not None else None
        ),
    }

    _write_outputs(
        output_directory,
        comments_review=comments_review,
        comments_audit=comments_audit,
        reproduction_report=reproduction_report,
        mismatches=mismatches,
    )
    print(json.dumps(reproduction_report, ensure_ascii=False, indent=2))
    print(f"Comments audit: {output_directory}")
    return output_directory

"""Solution-agnostic post-training evaluation pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path
from typing import Any

from scripts.common.evaluation.data import (
    calculate_holdout_metrics,
    compare_with_baseline,
    load_labeled_products,
    load_validation_indices,
)
from scripts.common.evaluation.runtime import (
    build_comments_review,
    load_model_with_adapters,
    score_products,
    verify_easyocr_weights,
)
from scripts.common.solution_profile import load_solution_profile
from scripts.common.submission.build_archive import load_and_validate_training_manifest
from scripts.common.submission.configuration import (
    SUPPORTED_CATEGORIES,
    build_submission_config,
)
from scripts.common.submission.runtime.submission_runtime.ocr import extract_ocr_texts
from scripts.common.training.lora_training import MODEL_ID, calculate_sha256, write_json_atomic

BASELINE_SCHEMA = "e_cup.gemma_lora_reproduction_baseline"


def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_evaluation_arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument(
        "--data-csv",
        type=Path,
        help="Labeled CSV to evaluate; defaults to <project-root>/data_no_duplicates.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory for reports; existing paths are never overwritten.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "all"),
        default="validation",
        help="Evaluate holdouts only, or score all rows while keeping holdout metrics separate.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def load_reproduction_baseline(baseline_path: Path, solution: str) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (
        not isinstance(baseline, dict)
        or baseline.get("schema") != BASELINE_SCHEMA
        or baseline.get("schema_version") != 1
        or baseline.get("solution") != solution
    ):
        raise ValueError(f"Unsupported reproduction baseline: {baseline_path}")
    if set(baseline.get("categories", {})) != set(SUPPORTED_CATEGORIES):
        raise ValueError(f"Baseline must describe both categories: {baseline_path}")
    completeness = baseline.get("completeness")
    if completeness not in {"complete", "partial"}:
        raise ValueError(f"Invalid baseline completeness: {baseline_path}")
    if not all(isinstance(value, dict) for value in baseline["categories"].values()):
        raise TypeError(f"Baseline category entries must be objects: {baseline_path}")
    missing_metrics = [
        category
        for category, category_baseline in baseline["categories"].items()
        if not isinstance(category_baseline.get("metrics"), dict)
    ]
    if completeness == "complete" and missing_metrics:
        raise ValueError(f"Complete baseline lacks metrics for: {missing_metrics}")
    return baseline


def _reproduction_status(comparison: dict[str, Any], comments_passed: bool) -> str:
    prefix = "complete" if comparison["fully_verified"] else "partial"
    suffix = "pass" if comparison["checks_passed"] and comments_passed else "fail"
    return f"{prefix}_{suffix}"


def save_reports(
    output_directory: Path,
    *,
    solution: str,
    baseline_path: Path,
    predictions: Any,
    comments_review: Any,
    metrics_by_category: dict[str, Any],
    comments_audit: dict[str, Any],
    baseline_comparison: dict[str, Any],
    training_manifest: dict[str, Any],
    training_manifest_path: Path,
    data_csv: Path,
    scope: str,
    holdout_rows: int,
) -> dict[str, Any]:
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_directory.parent,
        prefix=f".{output_directory.name}.",
    ) as temporary_name:
        temporary_directory = Path(temporary_name)
        predictions.drop(columns=["source_index"]).to_csv(
            temporary_directory / "predictions.csv",
            index=False,
        )
        comments_review.to_csv(temporary_directory / "comments_review.csv", index=False)
        write_json_atomic(temporary_directory / "metrics.json", metrics_by_category)
        write_json_atomic(temporary_directory / "comments_audit.json", comments_audit)
        evaluation_passed = baseline_comparison["checks_passed"] and comments_audit["passed"]
        status = _reproduction_status(baseline_comparison, comments_audit["passed"])
        historical_reproduction_passed = (
            evaluation_passed if baseline_comparison["fully_verified"] else None
        )
        report = {
            "schema": "e_cup.gemma_lora_post_training_evaluation",
            "schema_version": 1,
            "solution": solution,
            "scope": scope,
            "rows_scored": len(predictions),
            "holdout_rows": holdout_rows,
            "evaluation_passed": evaluation_passed,
            "historical_reproduction_status": status,
            "historical_reproduction_passed": historical_reproduction_passed,
            "reproduction_passed": historical_reproduction_passed,
            "baseline_comparison": baseline_comparison,
            "comments_audit": comments_audit,
            "provenance": {
                "base_model_revision": training_manifest["base_model_revision"],
                "training_manifest_sha256": calculate_sha256(training_manifest_path),
                "data_csv": data_csv.name,
                "data_csv_sha256": calculate_sha256(data_csv),
                "baseline_sha256": calculate_sha256(baseline_path),
            },
            "artifacts": {
                "predictions": "predictions.csv",
                "comments_review": "comments_review.csv",
                "metrics": "metrics.json",
                "comments_audit": "comments_audit.json",
            },
        }
        write_json_atomic(temporary_directory / "reproduction_report.json", report)
        temporary_directory.replace(output_directory)
    return report


def run_solution_evaluation(
    args: argparse.Namespace,
    *,
    profile_path: Path,
) -> Path:
    project_root = args.project_root.expanduser().resolve()
    images_directory = args.images_dir.expanduser().resolve()
    data_csv = (
        args.data_csv.expanduser().resolve()
        if args.data_csv is not None
        else project_root / "data_no_duplicates.csv"
    )
    solution_profile = load_solution_profile(profile_path)
    solution = str(solution_profile["solution"])
    reproduction_profile = solution_profile["training"]["reproduction"]
    baseline_path = project_root / str(reproduction_profile["baseline"])
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "artifacts" / "gemma_lora" / solution / "evaluation"
    )
    training_manifest_path = (
        project_root / "artifacts" / "gemma_lora" / solution / "training_manifest.json"
    )
    if not images_directory.is_dir():
        raise FileNotFoundError(images_directory)
    if not data_csv.is_file():
        raise FileNotFoundError(data_csv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output_directory}")

    baseline = load_reproduction_baseline(baseline_path, solution)
    training_manifest, adapter_directories, inference_by_category = (
        load_and_validate_training_manifest(
            project_root,
            solution,
            solution_profile["training"]["flammable_inference"],
        )
    )
    products = load_labeled_products(data_csv)
    validation_indices = load_validation_indices(project_root, products, training_manifest)
    selected_indices = (
        set(products.index) if args.scope == "all" else set().union(*validation_indices.values())
    )
    evaluation_products = products.loc[products.index.isin(selected_indices)].copy()
    verify_easyocr_weights(project_root)

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
    ocr_text_by_path = extract_ocr_texts(
        test_products=evaluation_products,
        images_directory=images_directory,
        submission_directory=project_root,
        submission_config=submission_config,
        strict=True,
    )

    processor = base_model = model = None
    try:
        processor, base_model, model = load_model_with_adapters(
            project_root,
            adapter_directories,
        )
        predictions = score_products(
            evaluation_products,
            model=model,
            processor=processor,
            images_directory=images_directory,
            submission_config=submission_config,
            ocr_text_by_path=ocr_text_by_path,
            batch_size=args.batch_size,
        )
    finally:
        del model, base_model, processor
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    metrics_by_category = calculate_holdout_metrics(predictions, validation_indices)
    baseline_comparison = compare_with_baseline(metrics_by_category, baseline)
    comments_review, comments_audit = build_comments_review(
        evaluation_products,
        predictions,
        images_directory=images_directory,
        ocr_text_by_path=ocr_text_by_path,
        submission_config=submission_config,
    )
    report = save_reports(
        output_directory,
        solution=solution,
        baseline_path=baseline_path,
        predictions=predictions,
        comments_review=comments_review,
        metrics_by_category=metrics_by_category,
        comments_audit=comments_audit,
        baseline_comparison=baseline_comparison,
        training_manifest=training_manifest,
        training_manifest_path=training_manifest_path,
        data_csv=data_csv,
        scope=args.scope,
        holdout_rows=sum(len(indices) for indices in validation_indices.values()),
    )
    print(f"Post-training evaluation: {output_directory}")
    print(f"Evaluation passed: {report['evaluation_passed']}")
    print(f"Historical reproduction: {report['historical_reproduction_status']}")
    return output_directory

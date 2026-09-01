"""Load labeled holdouts and compare their metrics with a recorded baseline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.common.submission.configuration import SUPPORTED_CATEGORIES
from scripts.common.submission.runtime.submission_runtime.core import load_test_data
from scripts.common.training.evaluation import calculate_binary_metrics


def load_labeled_products(csv_path: Path) -> pd.DataFrame:
    products = load_test_data(csv_path)
    try:
        label_rows = pd.read_csv(csv_path, usecols=["label"])
    except ValueError as error:
        raise ValueError(f"Evaluation CSV has no label column: {csv_path}") from error
    labels = pd.to_numeric(label_rows["label"], errors="raise")
    if len(labels) != len(products) or not labels.isin({0, 1}).all():
        raise ValueError("Evaluation labels must contain only 0 and 1")
    products["label"] = labels.astype(int).to_numpy()
    return products


def load_validation_indices(
    project_root: Path,
    products: pd.DataFrame,
    training_manifest: dict[str, Any],
) -> dict[str, set[int]]:
    indices_by_category: dict[str, set[int]] = {}
    for category in SUPPORTED_CATEGORIES:
        category_result = training_manifest["categories"][category]
        adapter_directory = project_root / Path(str(category_result["adapter"]))
        validation_path = adapter_directory.parent / "validation_ids.csv"
        if not validation_path.is_file():
            raise FileNotFoundError(validation_path)
        validation_ids = pd.read_csv(validation_path, dtype={"id": "string"})
        if validation_ids.columns.tolist() != ["id", "label"]:
            raise ValueError(f"Invalid validation IDs file: {validation_path}")
        if validation_ids.empty or validation_ids["id"].isna().any():
            raise ValueError(f"Validation IDs are empty or invalid: {validation_path}")
        if validation_ids["id"].duplicated().any():
            raise ValueError(f"Duplicate validation IDs: {validation_path}")

        category_rows = products.loc[products["category"].eq(category)]
        row_index_by_id = dict(zip(category_rows["id"], category_rows.index, strict=True))
        missing_ids = set(validation_ids["id"]) - set(row_index_by_id)
        if missing_ids:
            raise ValueError(
                f"Evaluation data lacks {len(missing_ids)} validation rows for {category}"
            )
        indices = {int(row_index_by_id[product_id]) for product_id in validation_ids["id"]}
        actual_labels = products.loc[sorted(indices), ["id", "label"]].set_index("id")["label"]
        expected_labels = validation_ids.set_index("id")["label"].astype(int)
        if not actual_labels.sort_index().equals(expected_labels.sort_index()):
            raise ValueError(f"Validation labels differ from training split for {category}")
        indices_by_category[category] = indices
    return indices_by_category


def calculate_holdout_metrics(
    predictions: pd.DataFrame,
    validation_indices_by_category: dict[str, set[int]],
) -> dict[str, dict[str, Any]]:
    metrics_by_category: dict[str, dict[str, Any]] = {}
    for category in SUPPORTED_CATEGORIES:
        category_predictions = predictions.loc[
            predictions["source_index"].isin(validation_indices_by_category[category])
        ]
        if len(category_predictions) != len(validation_indices_by_category[category]):
            raise RuntimeError(f"Missing holdout predictions for {category}")
        metrics = calculate_binary_metrics(
            category_predictions["label"],
            category_predictions["probability"],
            float(category_predictions["threshold"].iloc[0]),
        )
        metrics["labels"] = {
            str(label): int(count)
            for label, count in category_predictions["label"].value_counts().sort_index().items()
        }
        metrics_by_category[category] = metrics
    return metrics_by_category


def _compare_category(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    metric_tolerance: float,
    count_tolerance: int,
) -> dict[str, Any]:
    expected_metrics = expected.get("metrics")
    metrics_available = isinstance(expected_metrics, dict)
    metric_deltas = (
        {
            name: abs(float(actual[name]) - float(expected_metrics[name]))
            for name in ("f1", "precision", "recall")
        }
        if metrics_available
        else None
    )
    count_deltas = (
        {
            name: abs(int(actual[name]) - int(expected_metrics[name]))
            for name in ("tn", "fp", "fn", "tp")
        }
        if metrics_available
        else None
    )
    checks: dict[str, bool | None] = {
        "samples_match": (
            int(actual["samples"]) == int(expected["samples"])
            if expected.get("samples") is not None
            else None
        ),
        "labels_match": (
            actual["labels"] == expected["labels"] if expected.get("labels") is not None else None
        ),
        "threshold_matches": math.isclose(
            float(actual["threshold"]),
            float(expected["inference"]["threshold"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "metrics_within_tolerance": (
            all(delta <= metric_tolerance for delta in metric_deltas.values())
            if metric_deltas is not None
            else None
        ),
        "confusion_within_tolerance": (
            all(delta <= count_tolerance for delta in count_deltas.values())
            if count_deltas is not None
            else None
        ),
    }
    available_checks = [value for value in checks.values() if value is not None]
    fully_verified = all(value is not None for value in checks.values())
    return {
        "passed": all(available_checks),
        "fully_verified": fully_verified,
        "checks": checks,
        "metric_deltas": metric_deltas,
        "confusion_deltas": count_deltas,
        "expected": expected,
        "actual": actual,
    }


def compare_with_baseline(
    metrics_by_category: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    metric_tolerance = float(baseline["metric_absolute_tolerance"])
    count_tolerance = int(baseline["confusion_count_tolerance"])
    comparisons = {
        category: _compare_category(
            metrics_by_category[category],
            baseline["categories"][category],
            metric_tolerance=metric_tolerance,
            count_tolerance=count_tolerance,
        )
        for category in SUPPORTED_CATEGORIES
    }
    return {
        "checks_passed": all(result["passed"] for result in comparisons.values()),
        "fully_verified": all(result["fully_verified"] for result in comparisons.values()),
        "metric_absolute_tolerance": metric_tolerance,
        "confusion_count_tolerance": count_tolerance,
        "categories": comparisons,
    }

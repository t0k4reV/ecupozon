"""Audit saved binary-classification probabilities without loading a model."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.common.training.evaluation import (
    calculate_binary_metrics,
    calculate_threshold_curve,
    select_best_threshold,
)


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_saved_threshold(
    predictions: pd.DataFrame,
    selection_path: Path | None,
    saved_threshold: float | None,
) -> tuple[float, str, str | None]:
    if saved_threshold is not None:
        return float(saved_threshold), "command_line", None

    if "threshold" in predictions:
        thresholds = pd.to_numeric(predictions["threshold"], errors="raise").unique()
        if len(thresholds) == 1:
            return float(thresholds[0]), "predictions", None

    if selection_path is None:
        raise ValueError("Saved threshold is unavailable; provide --saved-threshold or --selection")
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    threshold = selection.get("threshold") if isinstance(selection, dict) else None
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError(f"Selection has no numeric threshold: {selection_path}")
    return float(threshold), "selection", calculate_sha256(selection_path)


def _validate_threshold(value: float, name: str) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1: {value}")
    return threshold


def _probability_ranges(labels: pd.Series, probabilities: pd.Series) -> dict[str, Any]:
    ranges: dict[str, dict[str, float]] = {}
    for label, name in ((0, "negative"), (1, "positive")):
        values = probabilities.loc[labels.eq(label)]
        if values.empty:
            raise ValueError(f"Threshold audit requires label {label}")
        ranges[name] = {"min": float(values.min()), "max": float(values.max())}
    return {
        **ranges,
        "separation_margin": ranges["positive"]["min"] - ranges["negative"]["max"],
    }


def run_threshold_audit(
    *,
    predictions_path: Path,
    output_directory: Path,
    solution: str,
    category: str,
    historical_threshold: float,
    saved_threshold: float | None = None,
    selection_path: Path | None = None,
) -> dict[str, Any]:
    """Compare historical, saved and exact-optimal thresholds."""
    predictions_path = predictions_path.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    selection_path = selection_path.expanduser().resolve() if selection_path else None
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite threshold audit: {output_directory}")
    if selection_path is None:
        default_selection_path = predictions_path.with_name("selection.json")
        if default_selection_path.is_file():
            selection_path = default_selection_path

    predictions = pd.read_csv(predictions_path)
    required_columns = {"label", "probability"}
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"Predictions are missing columns: {sorted(missing_columns)}")
    if "category" in predictions:
        predictions = predictions.loc[predictions["category"].eq(category)].copy()
    if predictions.empty:
        raise ValueError(f"Predictions contain no rows for {category}")

    labels = pd.to_numeric(predictions["label"], errors="raise")
    probabilities = pd.to_numeric(predictions["probability"], errors="raise").astype(float)
    if not labels.isin({0, 1}).all():
        raise ValueError("Prediction labels must be binary 0/1")
    labels = labels.astype(int)

    selected_threshold, selected_source, selection_sha256 = _load_saved_threshold(
        predictions,
        selection_path,
        saved_threshold,
    )
    historical_threshold = _validate_threshold(historical_threshold, "Historical threshold")
    selected_threshold = _validate_threshold(selected_threshold, "Saved threshold")
    optimal_metrics = select_best_threshold(labels, probabilities)
    curve = calculate_threshold_curve(labels, probabilities)
    label_counts = labels.value_counts().sort_index()

    report = {
        "schema": "e_cup.threshold_audit",
        "schema_version": 1,
        "solution": solution,
        "category": category,
        "rows": len(predictions),
        "labels": {str(label): int(count) for label, count in label_counts.items()},
        "probability_ranges": _probability_ranges(labels, probabilities),
        "thresholds": {
            "historical_threshold_on_current_predictions": {
                "source": "reproduction_baseline",
                "metrics": calculate_binary_metrics(
                    labels,
                    probabilities,
                    historical_threshold,
                ),
            },
            "saved_adapter": {
                "source": selected_source,
                "metrics": calculate_binary_metrics(labels, probabilities, selected_threshold),
            },
            "validation_optimal": {
                "source": "exact_validation_probabilities",
                "selection_policy": "f1_then_precision_then_larger_threshold",
                "metrics": optimal_metrics,
            },
        },
        "provenance": {
            "predictions_file": predictions_path.name,
            "predictions_sha256": calculate_sha256(predictions_path),
            "selection_file": selection_path.name if selection_path else None,
            "selection_sha256": selection_sha256,
        },
        "artifacts": {
            "threshold_curve": "threshold_curve.csv",
        },
    }

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_directory.parent,
        prefix=f".{output_directory.name}.",
    ) as temporary_name:
        temporary_directory = Path(temporary_name)
        pd.DataFrame(curve).to_csv(temporary_directory / "threshold_curve.csv", index=False)
        (temporary_directory / "threshold_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_directory.replace(output_directory)
    return report

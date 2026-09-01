"""Metrics shared by category evaluators and post-training validation."""

from __future__ import annotations

import math
from typing import Any


def calculate_binary_metrics(
    labels: Any,
    probabilities: Any,
    threshold: float,
) -> dict[str, Any]:
    """Calculate deterministic binary metrics for a probability threshold."""
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1: {threshold}")

    labels_array = labels.astype(int).to_numpy()
    probabilities_array = probabilities.astype(float).to_numpy()
    if len(labels_array) == 0 or len(labels_array) != len(probabilities_array):
        raise ValueError("Labels and probabilities must be non-empty and have equal length")
    if set(labels_array.tolist()) - {0, 1}:
        raise ValueError("Labels must contain only 0 and 1")
    if not all(math.isfinite(float(value)) for value in probabilities_array):
        raise ValueError("Probabilities must be finite")
    if any(float(value) < 0.0 or float(value) > 1.0 for value in probabilities_array):
        raise ValueError("Probabilities must be between 0 and 1")

    predictions = (probabilities_array >= threshold).astype(int)
    precision = float(precision_score(labels_array, predictions, zero_division=0))
    recall = float(recall_score(labels_array, predictions, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(labels_array, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "samples": int(len(labels_array)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "precision": precision,
        "recall": recall,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def get_threshold_candidates(probabilities: Any) -> tuple[float, ...]:
    """Return every threshold that can produce a distinct prediction set."""
    probabilities_array = probabilities.astype(float).to_numpy()
    if len(probabilities_array) == 0:
        raise ValueError("Probabilities must be non-empty")
    values = [float(value) for value in probabilities_array]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Probabilities must be finite")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Probabilities must be between 0 and 1")
    return tuple(sorted({0.0, 1.0, *values}))


def calculate_threshold_curve(labels: Any, probabilities: Any) -> list[dict[str, Any]]:
    """Calculate metrics for the exact set of meaningful thresholds."""
    return [
        calculate_binary_metrics(labels, probabilities, threshold)
        for threshold in get_threshold_candidates(probabilities)
    ]


def select_best_threshold(labels: Any, probabilities: Any) -> dict[str, Any]:
    """Select by F1, then precision, then the larger exact threshold."""
    candidates = calculate_threshold_curve(labels, probabilities)
    return max(
        candidates,
        key=lambda metrics: (
            metrics["f1"],
            metrics["precision"],
            metrics["threshold"],
        ),
    )


def add_prediction_columns(
    predictions: Any,
    probability_column: str,
    threshold: float,
) -> Any:
    """Return a copy with the probability used in production and its verdict."""
    if probability_column not in predictions:
        raise ValueError(f"Missing probability column: {probability_column}")
    enriched = predictions.copy()
    enriched["probability"] = enriched[probability_column].astype(float)
    enriched["prediction"] = (enriched["probability"] >= threshold).astype(int)
    enriched["correct"] = enriched["prediction"].eq(enriched["label"].astype(int))
    return enriched

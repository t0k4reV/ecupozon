from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.common.evaluation.data import compare_with_baseline
from scripts.common.evaluation.threshold_audit import run_threshold_audit
from scripts.common.submission.configuration import SUPPORTED_CATEGORIES
from scripts.common.training.evaluation import select_best_threshold

FLAMMABLE_CATEGORY = "Легковоспламеняющиеся"


class ThresholdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = pd.Series([0, 0, 1, 1])
        self.probabilities = pd.Series([0.0001, 0.0002305067, 0.0011695101857185364, 0.99])

    def test_exact_selector_finds_threshold_below_old_grid(self) -> None:
        metrics = select_best_threshold(self.labels, self.probabilities)

        self.assertEqual(metrics["threshold"], 0.0011695101857185364)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (2, 0, 0, 2))

    def test_audit_keeps_historical_and_saved_thresholds_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            predictions_path = root / "validation_predictions.csv"
            output_directory = root / "audit"
            pd.DataFrame(
                {
                    "id": ["1", "2", "3", "4"],
                    "category": [FLAMMABLE_CATEGORY] * 4,
                    "label": self.labels,
                    "probability": self.probabilities,
                    "threshold": [0.98] * 4,
                }
            ).to_csv(predictions_path, index=False)

            report = run_threshold_audit(
                predictions_path=predictions_path,
                output_directory=output_directory,
                solution="v1_ocr",
                category=FLAMMABLE_CATEGORY,
                historical_threshold=0.005,
            )

            historical = report["thresholds"]["historical_threshold_on_current_predictions"][
                "metrics"
            ]
            saved = report["thresholds"]["saved_adapter"]["metrics"]
            optimal = report["thresholds"]["validation_optimal"]["metrics"]
            self.assertEqual(historical["threshold"], 0.005)
            self.assertEqual(saved["threshold"], 0.98)
            self.assertAlmostEqual(optimal["threshold"], 0.0011695101857185364)
            self.assertTrue((output_directory / "threshold_curve.csv").is_file())
            written_report = json.loads(
                (output_directory / "threshold_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(written_report, report)

    def test_partial_baseline_remains_distinct_with_reconstructed_metrics(self) -> None:
        metrics = {
            "threshold": 0.5,
            "samples": 4,
            "f1": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "tn": 2,
            "fp": 0,
            "fn": 0,
            "tp": 2,
            "labels": {"0": 2, "1": 2},
        }
        baseline = {
            "completeness": "partial",
            "metric_absolute_tolerance": 0.005,
            "confusion_count_tolerance": 3,
            "categories": {
                category: {
                    "samples": 4,
                    "labels": {"0": 2, "1": 2},
                    "inference": {"threshold": 0.5},
                    "metrics": metrics,
                }
                for category in SUPPORTED_CATEGORIES
            },
        }

        comparison = compare_with_baseline(
            {category: metrics for category in SUPPORTED_CATEGORIES},
            baseline,
        )

        self.assertTrue(comparison["checks_passed"])
        self.assertFalse(comparison["fully_verified"])
        self.assertEqual(comparison["baseline_completeness"], "partial")


if __name__ == "__main__":
    unittest.main()

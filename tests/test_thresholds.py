from __future__ import annotations

import unittest

import pandas as pd

from scripts.common.evaluation.data import compare_with_baseline
from scripts.common.submission.configuration import (
    FLAMMABLE_CATEGORY,
    SUPPLEMENTS_CATEGORY,
    SUPPORTED_CATEGORIES,
    build_submission_config,
)
from scripts.common.training.build_training_manifest import validate_inference_contract
from scripts.common.training.evaluation import select_best_threshold


class ThresholdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = pd.Series([0, 0, 1, 1])
        self.probabilities = pd.Series([0.0001, 0.0002305067, 0.0011695101857185364, 0.99])

    def test_exact_selector_finds_threshold_below_old_grid(self) -> None:
        metrics = select_best_threshold(self.labels, self.probabilities)

        self.assertEqual(metrics["threshold"], 0.0011695101857185364)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (2, 0, 0, 2))

    def test_reference_comparison_is_secondary_for_partial_baseline(self) -> None:
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

    def test_selected_thresholds_flow_into_submission_config(self) -> None:
        supplements_inference = {
            "max_images": 1,
            "aggregation": "single",
            "threshold": 0.8,
        }
        flammable_shape = {"max_images": 3, "aggregation": "max"}
        flammable_inference = {**flammable_shape, "threshold": 0.85}
        validate_inference_contract(
            SUPPLEMENTS_CATEGORY,
            supplements_inference,
            flammable_shape,
        )
        validate_inference_contract(
            FLAMMABLE_CATEGORY,
            flammable_inference,
            flammable_shape,
        )

        config = build_submission_config(
            solution="test",
            model_id="model",
            adapters={SUPPLEMENTS_CATEGORY: "supplements", FLAMMABLE_CATEGORY: "flammable"},
            inference_by_category={
                SUPPLEMENTS_CATEGORY: supplements_inference,
                FLAMMABLE_CATEGORY: flammable_inference,
            },
            ocr_affects_classification=False,
        )

        self.assertEqual(config["thresholds"][SUPPLEMENTS_CATEGORY], 0.8)
        self.assertEqual(config["thresholds"][FLAMMABLE_CATEGORY], 0.85)


if __name__ == "__main__":
    unittest.main()

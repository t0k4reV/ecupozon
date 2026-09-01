"""Audit v1-OCR flammable thresholds from saved validation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.common.evaluation.threshold_audit import run_threshold_audit
from scripts.common.submission.configuration import FLAMMABLE_CATEGORY

SOLUTION_DIRECTORY = Path(__file__).resolve().parent
BASELINE_PATH = SOLUTION_DIRECTORY / "reproduction_baseline.json"


def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--saved-threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    historical_threshold = baseline["categories"][FLAMMABLE_CATEGORY]["inference"]["threshold"]
    report = run_threshold_audit(
        predictions_path=args.predictions,
        output_directory=args.output_dir,
        solution="v1_ocr",
        category=FLAMMABLE_CATEGORY,
        historical_threshold=float(historical_threshold),
        saved_threshold=args.saved_threshold,
        selection_path=args.selection,
    )
    historical = report["thresholds"]["historical_threshold_on_current_predictions"]["metrics"]
    optimal = report["thresholds"]["validation_optimal"]["metrics"]
    print(
        f"Historical threshold {historical['threshold']}: F1={historical['f1']:.6f}; "
        f"validation threshold {optimal['threshold']}: F1={optimal['f1']:.6f}"
    )
    print(f"Threshold audit: {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()

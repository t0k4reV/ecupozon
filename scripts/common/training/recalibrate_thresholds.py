"""Recalculate production thresholds from saved validation predictions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.common.solution_profile import load_solution_profile
from scripts.common.training.build_training_manifest import load_result
from scripts.common.training.evaluation import add_prediction_columns, select_best_threshold
from scripts.common.training.lora_training import get_default_project_root

CALIBRATION_POLICY = "validation_f1_then_precision_then_threshold"
REPORT_PATH = Path("artifacts/gemma_lora/recalibration_report.json")
MANIFEST_PATHS = {
    "v1_ocr": Path("artifacts/gemma_lora/v1_ocr/training_manifest.json"),
    "v2_img3": Path("artifacts/gemma_lora/v2_img3/training_manifest.json"),
}
PROFILE_PATHS = {
    solution: Path(f"scripts/solutions/{solution}/profile.json") for solution in MANIFEST_PATHS
}


@dataclass(frozen=True)
class CalibrationTarget:
    name: str
    category: str
    experiment: str
    result_path: Path
    calibration_predictions_path: Path
    probability_column: str
    selected_mode: str
    solutions: tuple[str, ...]
    diagnostic_probability_column: str | None = None


TARGETS = (
    CalibrationTarget(
        name="supplements",
        category="БАД",
        experiment="shared",
        result_path=Path(
            "artifacts/gemma_lora/shared/adapters/gemma_e4b_supplements/training_result.json"
        ),
        calibration_predictions_path=Path(
            "artifacts/gemma_lora/v2_img3/evaluation/predictions.csv"
        ),
        probability_column="p_img1",
        selected_mode="single_first_image",
        solutions=("v1_ocr", "v2_img3"),
    ),
    CalibrationTarget(
        name="v1_ocr_flammable",
        category="Легковоспламеняющиеся",
        experiment="v1_ocr",
        result_path=Path(
            "artifacts/gemma_lora/v1_ocr/adapters/gemma_e4b_flammable_ocr/training_result.json"
        ),
        calibration_predictions_path=Path("artifacts/gemma_lora/v1_ocr/evaluation/predictions.csv"),
        probability_column="p_img1",
        selected_mode="single_first_image_with_ocr",
        solutions=("v1_ocr",),
    ),
    CalibrationTarget(
        name="v2_img3_flammable",
        category="Легковоспламеняющиеся",
        experiment="v2_img3",
        result_path=Path(
            "artifacts/gemma_lora/v2_img3/adapters/gemma_e4b_flammable/training_result.json"
        ),
        calibration_predictions_path=Path(
            "artifacts/gemma_lora/v2_img3/evaluation/predictions.csv"
        ),
        probability_column="p_max",
        selected_mode="max_first_three",
        solutions=("v2_img3",),
        diagnostic_probability_column="p_img1",
    ),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically update threshold-derived artifacts; default is read-only dry-run.",
    )
    return parser.parse_args()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()


def _csv_bytes(dataframe: pd.DataFrame) -> bytes:
    stream = io.StringIO()
    dataframe.to_csv(stream, index=False)
    return stream.getvalue().encode()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _build_selection(
    target: CalibrationTarget,
    predictions: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required_columns = {"id", "label", target.probability_column}
    if target.diagnostic_probability_column:
        required_columns.add(target.diagnostic_probability_column)
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"{target.name} predictions are missing columns: {sorted(missing_columns)}"
        )
    if predictions.empty or predictions["id"].isna().any():
        raise ValueError(f"{target.name} predictions are empty or contain invalid ids")
    if predictions["id"].astype(str).duplicated().any():
        raise ValueError(f"{target.name} predictions contain duplicate ids")

    labels = pd.to_numeric(predictions["label"], errors="raise")
    if not labels.isin({0, 1}).all() or set(labels.astype(int)) != {0, 1}:
        raise ValueError(f"{target.name} predictions require both binary labels")
    selected_metrics = select_best_threshold(labels, predictions[target.probability_column])
    selection: dict[str, Any] = {
        "selected_mode": target.selected_mode,
        "threshold_policy": CALIBRATION_POLICY,
    }
    if target.diagnostic_probability_column:
        selection["first_image"] = select_best_threshold(
            labels,
            predictions[target.diagnostic_probability_column],
        )
        selection["max_first_three"] = selected_metrics
    else:
        selection.update(selected_metrics)
    return selection, selected_metrics


def _load_calibration_predictions(
    project_root: Path,
    target: CalibrationTarget,
    saved_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, Path, bytes]:
    """Load final-model holdout scores and align them with the saved split."""
    source_path = project_root / target.calibration_predictions_path
    source_content = source_path.read_bytes()
    source_predictions = pd.read_csv(source_path, dtype={"id": "string"})
    if "category" not in source_predictions:
        raise ValueError(f"Calibration predictions lack category: {source_path}")
    source_predictions = source_predictions.loc[
        source_predictions["category"].eq(target.category)
    ].copy()
    saved_predictions = saved_predictions.copy()
    saved_predictions["id"] = saved_predictions["id"].astype("string")

    saved_by_id = saved_predictions.set_index("id")
    source_by_id = source_predictions.set_index("id")
    if not saved_by_id.index.is_unique or not source_by_id.index.is_unique:
        raise ValueError(f"Calibration predictions contain duplicate ids for {target.name}")
    if set(saved_by_id.index) != set(source_by_id.index):
        raise ValueError(f"Calibration holdout differs from the saved split for {target.name}")
    source_by_id = source_by_id.reindex(saved_by_id.index)
    saved_labels = pd.to_numeric(saved_by_id["label"], errors="raise").astype(int)
    source_labels = pd.to_numeric(source_by_id["label"], errors="raise").astype(int)
    if not saved_labels.equals(source_labels):
        raise ValueError(f"Calibration labels differ from the saved split for {target.name}")

    probability_columns = [column for column in source_by_id if column.startswith("p_")]
    if target.probability_column not in probability_columns:
        raise ValueError(f"Calibration predictions lack {target.probability_column}: {source_path}")
    for column in probability_columns:
        if column in saved_by_id:
            saved_by_id[column] = source_by_id[column].to_numpy()
    return saved_by_id.reset_index(), source_path, source_content


def _prepare_target(
    project_root: Path,
    target: CalibrationTarget,
) -> tuple[dict[Path, bytes], dict[str, Any], dict[str, Any]]:
    result_path = project_root / target.result_path
    result_payload = load_result(
        project_root,
        target.category,
        target.experiment,
        target.result_path,
    )
    output_directory = result_path.parent
    predictions_path = output_directory / "validation_predictions.csv"
    selection_path = output_directory / "selection.json"
    saved_predictions = pd.read_csv(predictions_path, dtype={"id": "string"})
    expected_rows = int(result_payload["result"]["split"]["validation"])
    if len(saved_predictions) != expected_rows:
        raise ValueError(
            f"{target.name} validation rows differ from the saved split: "
            f"{len(saved_predictions)} != {expected_rows}"
        )
    predictions, source_path, source_content = _load_calibration_predictions(
        project_root,
        target,
        saved_predictions,
    )

    selection, selected_metrics = _build_selection(target, predictions)
    calibrated_predictions = add_prediction_columns(
        predictions,
        target.probability_column,
        float(selected_metrics["threshold"]),
    )
    calibrated_predictions["threshold"] = float(selected_metrics["threshold"])
    predictions_content = _csv_bytes(calibrated_predictions)
    selection_content = _json_bytes(selection)

    calibrated_result = copy.deepcopy(result_payload)
    result = calibrated_result["result"]
    old_threshold = float(result["inference"]["threshold"])
    result["inference"]["threshold"] = float(selected_metrics["threshold"])
    result["validation"] = selection
    result["calibration"] = {
        "policy": CALIBRATION_POLICY,
        "predictions": str(target.calibration_predictions_path),
        "predictions_sha256": _sha256_bytes(source_content),
    }
    result["validation_artifacts"] = {
        "selection.json": _sha256_bytes(selection_content),
        "validation_predictions.csv": _sha256_bytes(predictions_content),
    }
    result_content = _json_bytes(calibrated_result)
    prepared = {
        predictions_path: predictions_content,
        selection_path: selection_content,
        result_path: result_content,
        source_path: source_content,
    }
    report = {
        "category": target.category,
        "probability_column": target.probability_column,
        "old_threshold": old_threshold,
        "selected_threshold": float(selected_metrics["threshold"]),
        "metrics": selected_metrics,
        "calibration_predictions": str(target.calibration_predictions_path),
        "calibration_predictions_sha256": _sha256_bytes(source_content),
        "adapter_files": result["files"],
        "validation_predictions_sha256_before": hashlib.sha256(
            predictions_path.read_bytes()
        ).hexdigest(),
        "validation_predictions_sha256_after": _sha256_bytes(predictions_content),
    }
    return prepared, calibrated_result, report


def build_recalibration_plan(
    project_root: Path,
) -> tuple[dict[Path, bytes], dict[str, Any], dict[Path, str]]:
    project_root = project_root.expanduser().resolve()
    prepared: dict[Path, bytes] = {}
    calibrated_results: dict[str, dict[str, Any]] = {}
    target_reports: dict[str, Any] = {}
    original_hashes: dict[Path, str] = {}

    for target in TARGETS:
        target_prepared, calibrated_result, target_report = _prepare_target(
            project_root,
            target,
        )
        for path, content in target_prepared.items():
            prepared[path] = content
            original_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        calibrated_results[target.name] = calibrated_result
        target_reports[target.name] = target_report

    manifest_reports: dict[str, Any] = {}
    for solution, relative_manifest_path in MANIFEST_PATHS.items():
        manifest_path = project_root / relative_manifest_path
        manifest = _read_json(manifest_path)
        if (
            manifest.get("schema") != "e_cup.gemma_lora_training"
            or manifest.get("schema_version") != 1
            or manifest.get("solution") != solution
        ):
            raise ValueError(f"Unsupported training manifest: {manifest_path}")
        calibrated_manifest = copy.deepcopy(manifest)
        for target in TARGETS:
            if solution not in target.solutions:
                continue
            original_result = _read_json(project_root / target.result_path)["result"]
            if manifest["categories"].get(target.category) != original_result:
                raise ValueError(
                    f"{solution} manifest is stale for {target.category}; refusing recalibration"
                )
            calibrated_manifest["categories"][target.category] = calibrated_results[target.name][
                "result"
            ]
        profile = load_solution_profile(project_root / PROFILE_PATHS[solution])
        calibrated_manifest["reproduction"] = profile["training"]["reproduction"]
        manifest_content = _json_bytes(calibrated_manifest)
        prepared[manifest_path] = manifest_content
        original_hashes[manifest_path] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        manifest_reports[solution] = {
            "sha256_before": original_hashes[manifest_path],
            "sha256_after": _sha256_bytes(manifest_content),
            "thresholds": {
                category: float(category_result["inference"]["threshold"])
                for category, category_result in calibrated_manifest["categories"].items()
            },
        }

    report = {
        "schema": "e_cup.threshold_recalibration",
        "schema_version": 1,
        "policy": CALIBRATION_POLICY,
        "targets": target_reports,
        "manifests": manifest_reports,
    }
    return prepared, report, original_hashes


def _write_atomic(path: Path, content: bytes) -> None:
    temporary_path = path.with_suffix(path.suffix + ".recalibration.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def apply_recalibration_plan(
    project_root: Path,
    prepared: dict[Path, bytes],
    report: dict[str, Any],
    original_hashes: dict[Path, str],
) -> list[Path]:
    for path, expected_hash in original_hashes.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Calibration input changed after validation: {path}")

    changed_paths = [path for path, content in prepared.items() if path.read_bytes() != content]
    report_path = project_root / REPORT_PATH
    report_content = _json_bytes(report)
    if not changed_paths:
        if not report_path.is_file():
            report_path.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(report_path, report_content)
        return []

    def write_order(path: Path) -> tuple[int, str]:
        if path.name in {"selection.json", "validation_predictions.csv"}:
            return (0, str(path))
        if path.name == "training_result.json":
            return (1, str(path))
        return (2, str(path))

    for path in sorted(changed_paths, key=write_order):
        _write_atomic(path, prepared[path])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(report_path, report_content)
    return changed_paths


def main() -> None:
    args = parse_arguments()
    project_root = args.project_root.expanduser().resolve()
    prepared, report, original_hashes = build_recalibration_plan(project_root)
    for name, target in report["targets"].items():
        print(
            f"{name}: {target['old_threshold']:.16g} -> "
            f"{target['selected_threshold']:.16g}; F1={target['metrics']['f1']:.6f}"
        )
    changed = [path for path, content in prepared.items() if path.read_bytes() != content]
    if not args.apply:
        print(f"Dry-run: {len(changed)} files would change; pass --apply to update them")
        return
    updated = apply_recalibration_plan(
        project_root,
        prepared,
        report,
        original_hashes,
    )
    if updated:
        print(f"Updated {len(updated)} files; report: {project_root / REPORT_PATH}")
    else:
        print(f"Threshold artifacts are already calibrated; report: {project_root / REPORT_PATH}")


if __name__ == "__main__":
    main()

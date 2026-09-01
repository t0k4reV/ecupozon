"""Validate both trained adapters and build the combined training manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.common.solution_profile import load_solution_profile
from scripts.common.training.lora_training import (
    MODEL_ID,
    calculate_sha256,
    get_default_project_root,
    write_json_atomic,
)

SHARED_SUPPLEMENTS_RESULT = Path(
    "artifacts/gemma_lora/shared/adapters/gemma_e4b_supplements/training_result.json"
)
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
REQUIRED_VALIDATION_FILES = ("selection.json", "validation_predictions.csv")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    parser.add_argument("--profile", type=Path, required=True)
    return parser.parse_args()


def load_result(
    project_root: Path,
    category: str,
    expected_experiment: str,
    relative_path: Path,
) -> dict[str, Any]:
    result_path = project_root / relative_path
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "e_cup.gemma_lora_category_training"
        or payload.get("schema_version") != 1
        or payload.get("model_id") != MODEL_ID
    ):
        raise ValueError(f"Invalid category training result: {result_path}")
    result = payload.get("result")
    if (
        not isinstance(result, dict)
        or result.get("category") != category
        or result.get("experiment") != expected_experiment
    ):
        raise ValueError(f"Category mismatch in {result_path}")
    adapter_path = Path(str(result.get("adapter", "")))
    if adapter_path.is_absolute() or ".." in adapter_path.parts:
        raise ValueError(f"Unsafe adapter path in {result_path}")
    adapter_directory = (project_root / adapter_path).resolve()
    if project_root != adapter_directory and project_root not in adapter_directory.parents:
        raise ValueError(f"Adapter escapes project root: {adapter_directory}")
    expected_files = result.get("files")
    if not isinstance(expected_files, dict):
        raise ValueError(f"Adapter hashes are missing in {result_path}")
    for filename in REQUIRED_ADAPTER_FILES:
        adapter_file = adapter_directory / filename
        expected_hash = expected_files.get(filename)
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not adapter_file.is_file()
            or calculate_sha256(adapter_file) != expected_hash
        ):
            raise ValueError(f"Adapter file is missing or stale: {adapter_file}")
    validation = result.get("validation")
    validation_artifacts = result.get("validation_artifacts")
    if not isinstance(validation, dict) or not isinstance(validation_artifacts, dict):
        raise ValueError(f"Validation results are missing in {result_path}")
    output_directory = result_path.parent
    for filename in REQUIRED_VALIDATION_FILES:
        validation_file = output_directory / filename
        expected_hash = validation_artifacts.get(filename)
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not validation_file.is_file()
            or calculate_sha256(validation_file) != expected_hash
        ):
            raise ValueError(f"Validation file is missing or stale: {validation_file}")
    return payload


def validate_inference_contract(
    category: str,
    inference: object,
    flammable_inference: dict[str, object],
) -> None:
    if not isinstance(inference, dict):
        raise TypeError(f"Inference contract must be an object for {category}")
    threshold = inference.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError(f"Invalid inference threshold for {category}")
    if category == "БАД":
        if inference != {"max_images": 1, "aggregation": "single", "threshold": 0.7}:
            raise ValueError("Supplements inference contract does not match production")
    else:
        if (
            inference.get("max_images") != flammable_inference["max_images"]
            or inference.get("aggregation") != flammable_inference["aggregation"]
            or not 0 < float(threshold) < 1
        ):
            raise ValueError("Flammable inference contract does not match production")


def main() -> None:
    args = parse_arguments()
    project_root = args.project_root.expanduser().resolve()
    solution_profile = load_solution_profile(args.profile)
    solution = solution_profile["solution"]
    training_profile = solution_profile["training"]
    flammable_inference = training_profile["flammable_inference"]
    results = {
        "БАД": load_result(
            project_root,
            "БАД",
            "shared",
            SHARED_SUPPLEMENTS_RESULT,
        ),
        "Легковоспламеняющиеся": load_result(
            project_root,
            "Легковоспламеняющиеся",
            solution,
            Path(training_profile["flammable_result"]),
        ),
    }
    first = results["БАД"]
    for category, payload in results.items():
        for field in ("base_model_revision", "data", "seed", "parameters"):
            if payload.get(field) != first.get(field):
                raise ValueError(f"Training provenance differs for {category}: {field}")
        validate_inference_contract(
            category,
            payload["result"].get("inference"),
            flammable_inference,
        )

    manifest = {
        "schema": "e_cup.gemma_lora_training",
        "schema_version": 1,
        "solution": solution,
        "model_id": MODEL_ID,
        "base_model_revision": first["base_model_revision"],
        "data": first["data"],
        "seed": first["seed"],
        "parameters": first["parameters"],
        "categories": {category: payload["result"] for category, payload in results.items()},
    }
    reproduction = training_profile.get("reproduction")
    if reproduction is not None:
        manifest["reproduction"] = reproduction
    output_path = project_root / "artifacts" / "gemma_lora" / solution / "training_manifest.json"
    if output_path.exists():
        current = json.loads(output_path.read_text(encoding="utf-8"))
        if current == manifest:
            print(f"Verified existing training manifest: {output_path}")
            return
        raise FileExistsError(f"Refusing to overwrite different manifest: {output_path}")
    write_json_atomic(output_path, manifest)
    print(f"Training manifest: {output_path}")


if __name__ == "__main__":
    main()

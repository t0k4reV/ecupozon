"""Build the offline Gemma LoRA + EasyOCR submission ZIP from local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.common.solution_profile import load_solution_profile
from scripts.common.submission.configuration import (
    FLAMMABLE_CATEGORY,
    SUPPLEMENTS_CATEGORY,
    build_submission_config,
)

MODEL_ID = "google/gemma-4-E4B-it"
RUNTIME_FILE_PATHS = (
    "run.py",
    "submission_runtime/__init__.py",
    "submission_runtime/pipeline.py",
    "submission_runtime/comments.py",
    "submission_runtime/core.py",
    "submission_runtime/inference.py",
    "submission_runtime/ocr.py",
    "submission_runtime/rules.py",
)
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
EASYOCR_MODEL_HASHES = {
    "craft_mlt_25k.pth": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
    "cyrillic_g2.pth": "48d0f3b58f28aa64651ab1032cc2d498c4de25135829668e87c14e7a07529f29",
}
VENDORED_DISTRIBUTION_VERSIONS = {
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "safetensors": "0.8.0",
    "easyocr": "1.7.2",
    "opencv-python-headless": "5.0.0.93",
    "scikit-image": "0.26.0",
    "python-bidi": "0.6.11",
    "pyclipper": "1.4.0",
    "Shapely": "2.1.2",
    "ImageIO": "2.37.4",
    "tifffile": "2026.8.23",
    "lazy-loader": "0.5",
    "PyYAML": "6.0.2",
    "networkx": "3.6.1",
}


def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(json_path: Path, payload: object) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_safe_relative_path(relative_path: str) -> Path:
    posix_path = PurePosixPath(relative_path)
    if not relative_path or posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"Unsafe relative path: {relative_path!r}")
    return Path(*posix_path.parts)


def copy_verified_file(source_path: Path, destination_path: Path) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        if not destination_path.is_file():
            raise ValueError(f"Vendored path collision: {destination_path}")
        if calculate_sha256(source_path) != calculate_sha256(destination_path):
            raise ValueError(f"Vendored file collision: {destination_path}")
        return
    shutil.copy2(source_path, destination_path)


def copy_runtime_files(
    runtime_source_directory: Path,
    staging_directory: Path,
) -> dict[str, str]:
    actual_files = {
        path.relative_to(runtime_source_directory).as_posix()
        for path in runtime_source_directory.rglob("*.py")
        if path.is_file()
    }
    expected_files = set(RUNTIME_FILE_PATHS)
    if actual_files != expected_files:
        raise ValueError(
            "Runtime files do not match the reviewed allowlist: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )

    file_hashes = {}
    for relative_path in RUNTIME_FILE_PATHS:
        source_path = runtime_source_directory / parse_safe_relative_path(relative_path)
        destination_path = staging_directory / parse_safe_relative_path(relative_path)
        copy_verified_file(source_path, destination_path)
        file_hashes[relative_path] = calculate_sha256(source_path)
    return file_hashes


def vendor_distribution(
    staging_directory: Path,
    distribution_name: str,
    expected_version: str,
) -> dict[str, object]:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"Required distribution is not installed: {distribution_name}"
        ) from error
    if distribution.version != expected_version:
        raise RuntimeError(
            f"{distribution_name}=={expected_version} is required, found {distribution.version}"
        )

    copied = 0
    copied_bytes = 0
    for entry in distribution.files or ():
        relative = PurePosixPath(str(entry))
        if relative.is_absolute() or ".." in relative.parts:
            continue
        if "__pycache__" in relative.parts or relative.suffix == ".pyc":
            continue
        source = Path(distribution.locate_file(entry)).resolve()
        if not source.is_file():
            continue
        destination = staging_directory / Path(*relative.parts)
        copy_verified_file(source, destination)
        copied += 1
        copied_bytes += source.stat().st_size
    if copied == 0:
        raise RuntimeError(f"Distribution has no package files: {distribution_name}")
    return {"version": distribution.version, "files": copied, "bytes": copied_bytes}


def build_submission_readme(solution_profile: dict[str, Any]) -> str:
    submission_profile = solution_profile["submission"]
    return f"""# E-CUP Gemma {solution_profile["display_name"]} + comments v4

The archive contains two Gemma-4 E4B LoRA adapters and an offline EasyOCR
runtime. {submission_profile["classifier_description"]} Deterministic comment rules never change the
Gemma verdict.

Runtime contract:

```bash
python -u run.py --test_data_path /data/test.csv --output_path /output/submission.csv
```

The base model must be available at
`$SHARED_MODELS_PATH/google/gemma-4-E4B-it`. Images are read from
`<test csv directory>/images/<id>/`. Network access is not used during inference.
"""


def is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_lora_adapter(
    adapter_directory: Path,
    expected_files: dict[str, object],
) -> None:
    from safetensors import safe_open

    for filename in REQUIRED_ADAPTER_FILES:
        adapter_file = adapter_directory / filename
        expected_sha256 = expected_files.get(filename)
        if (
            not is_sha256_digest(expected_sha256)
            or not adapter_file.is_file()
            or calculate_sha256(adapter_file) != expected_sha256
        ):
            raise ValueError(f"Adapter hash mismatch: {adapter_file}")

    adapter_config = json.loads(
        (adapter_directory / "adapter_config.json").read_text(encoding="utf-8")
    )
    if not isinstance(adapter_config, dict) or adapter_config.get("peft_type") != "LORA":
        raise ValueError(f"Invalid LoRA adapter config: {adapter_directory}")
    with safe_open(
        adapter_directory / "adapter_model.safetensors",
        framework="pt",
        device="cpu",
    ) as adapter_weights:
        if not list(adapter_weights.keys()):
            raise ValueError(f"LoRA adapter contains no tensors: {adapter_directory}")


def load_and_validate_training_manifest(
    project_root: Path,
    solution: str,
    flammable_inference: dict[str, object],
) -> tuple[dict[str, object], dict[str, Path], dict[str, dict[str, object]]]:
    manifest_path = project_root / "artifacts" / "gemma_lora" / solution / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    training_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(training_manifest, dict):
        raise TypeError("Training manifest root must be a JSON object")
    if training_manifest.get("schema") != "e_cup.gemma_lora_training":
        raise ValueError("Unsupported training manifest schema")
    if training_manifest.get("schema_version") != 1:
        raise ValueError("Unsupported training manifest version")
    if training_manifest.get("model_id") != MODEL_ID:
        raise ValueError("Training manifest uses another base model")
    if training_manifest.get("solution") != solution:
        raise ValueError("Training manifest belongs to another solution")
    base_model_revision = training_manifest.get("base_model_revision")
    if (
        not isinstance(base_model_revision, str)
        or len(base_model_revision) != 40
        or not all(character in "0123456789abcdef" for character in base_model_revision)
    ):
        raise ValueError("Training manifest has no valid base model revision")
    category_manifests = training_manifest.get("categories")
    if not isinstance(category_manifests, dict) or set(category_manifests) != {
        SUPPLEMENTS_CATEGORY,
        FLAMMABLE_CATEGORY,
    }:
        raise ValueError("Training manifest must contain both category adapters")

    adapter_directory_by_category = {}
    inference_config_by_category: dict[str, dict[str, object]] = {}
    for category, category_manifest in category_manifests.items():
        if not isinstance(category_manifest, dict):
            raise TypeError(f"Invalid category manifest: {category}")
        inference = category_manifest.get("inference")
        if not isinstance(inference, dict):
            raise TypeError(f"Invalid inference contract for {category}")
        expected_shape = (
            {"max_images": 1, "aggregation": "single"}
            if category == SUPPLEMENTS_CATEGORY
            else flammable_inference
        )
        if any(inference.get(key) != value for key, value in expected_shape.items()):
            raise ValueError(f"Inference contract mismatch for {category}")
        threshold = inference.get("threshold")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise TypeError(f"Invalid inference threshold for {category}")
        if category == SUPPLEMENTS_CATEGORY and float(threshold) != 0.7:
            raise ValueError("Supplements threshold must be 0.7")
        if category == FLAMMABLE_CATEGORY and not 0 < float(threshold) < 1:
            raise ValueError("Flammable threshold must be between 0 and 1")
        relative_adapter_path = parse_safe_relative_path(str(category_manifest.get("adapter", "")))
        adapter_directory = (project_root / relative_adapter_path).resolve()
        if project_root != adapter_directory and project_root not in adapter_directory.parents:
            raise ValueError(f"Adapter escapes project root: {adapter_directory}")
        expected_files = category_manifest.get("files")
        if not isinstance(expected_files, dict):
            raise TypeError(f"Missing adapter hashes for {category}")
        validate_lora_adapter(adapter_directory, expected_files)
        adapter_directory_by_category[category] = adapter_directory
        inference_config_by_category[category] = dict(inference)
    return training_manifest, adapter_directory_by_category, inference_config_by_category


def build_submission_staging_directory(
    project_root: Path,
    staging_directory: Path,
    solution_profile: dict[str, Any],
) -> dict[str, object]:
    solution = str(solution_profile["solution"])
    training_profile = solution_profile["training"]
    submission_profile = solution_profile["submission"]
    flammable_inference = training_profile["flammable_inference"]
    (
        training_manifest,
        adapter_directory_by_category,
        inference_config_by_category,
    ) = load_and_validate_training_manifest(
        project_root,
        solution,
        flammable_inference,
    )
    runtime_hashes = copy_runtime_files(
        Path(__file__).resolve().parent / "runtime", staging_directory
    )

    bundled_adapter_directory_by_category = {
        SUPPLEMENTS_CATEGORY: staging_directory / "artifacts" / "adapters" / "supplements",
        FLAMMABLE_CATEGORY: staging_directory / "artifacts" / "adapters" / "flammable",
    }
    adapter_artifacts = {}
    for category, source_directory in adapter_directory_by_category.items():
        bundled_directory = bundled_adapter_directory_by_category[category]
        adapter_file_hashes = {}
        for filename in REQUIRED_ADAPTER_FILES:
            bundled_path = bundled_directory / filename
            copy_verified_file(source_directory / filename, bundled_path)
            adapter_file_hashes[filename] = calculate_sha256(bundled_path)
        adapter_artifacts[category] = {
            "source": str(source_directory.relative_to(project_root)),
            "bundle_path": str(bundled_directory.relative_to(staging_directory)),
            "files": adapter_file_hashes,
        }

    easyocr_models_directory = project_root / "artifacts" / "easyocr"
    easyocr_model_hashes = {}
    for filename, expected_sha256 in EASYOCR_MODEL_HASHES.items():
        source_path = easyocr_models_directory / filename
        if not source_path.is_file() or calculate_sha256(source_path) != expected_sha256:
            raise ValueError(f"EasyOCR weight is missing or invalid: {source_path}")
        bundled_path = staging_directory / "artifacts" / "easyocr" / filename
        copy_verified_file(source_path, bundled_path)
        easyocr_model_hashes[filename] = expected_sha256
    vendored_distributions = {
        distribution_name: vendor_distribution(
            staging_directory,
            distribution_name,
            version,
        )
        for distribution_name, version in VENDORED_DISTRIBUTION_VERSIONS.items()
    }
    submission_config = build_submission_config(
        solution=solution,
        model_id=MODEL_ID,
        adapters={
            SUPPLEMENTS_CATEGORY: "artifacts/adapters/supplements",
            FLAMMABLE_CATEGORY: "artifacts/adapters/flammable",
        },
        inference_by_category=inference_config_by_category,
        ocr_affects_classification=bool(submission_profile["ocr_affects_classification"]),
    )
    write_json(staging_directory / "submission_config.json", submission_config)
    write_json(
        staging_directory / "metadata.json",
        {
            "image": "odsai/ecup26-quality-baseline:1.0",
            "entry_point": "python -u run.py",
        },
    )
    (staging_directory / "README.md").write_text(
        build_submission_readme(solution_profile), encoding="utf-8"
    )

    production_manifest = {
        "schema": submission_profile["manifest_schema"],
        "schema_version": 4,
        "model_id": MODEL_ID,
        "base_model_revision": training_manifest["base_model_revision"],
        "artifacts": adapter_artifacts,
        "runtime_files": runtime_hashes,
        "inference": {
            "max_images": submission_config["max_images"],
            "aggregation": {
                category: profile["aggregation"]
                for category, profile in inference_config_by_category.items()
            },
            "thresholds": submission_config["thresholds"],
        },
        "decision": {
            "classification": submission_profile["classifier_description"],
            "comments_version": "v4",
            "comments": "deterministic card-specific text and OCR explanations",
            "generative_comment_calls": 0,
            "ocr": "bundled EasyOCR ru+en with offline weights",
            "ocr_effect_on_label": (
                "flammable_classifier_input"
                if submission_profile["ocr_affects_classification"]
                else "none"
            ),
        },
        "ocr_models": easyocr_model_hashes,
        "vendored_distributions": vendored_distributions,
        "training_manifest_sha256": calculate_sha256(
            project_root / "artifacts" / "gemma_lora" / solution / "training_manifest.json"
        ),
    }
    write_json(staging_directory / "production_manifest.json", production_manifest)
    report = {
        "status": "built_from_training_artifacts",
        "solution": solution,
        "built_at_utc": datetime.now(UTC).isoformat(),
        "runtime_files": runtime_hashes,
        "classification_changes_from_training_manifest": "none",
        "comments_version": "v4",
        "ocr_bundled": True,
        "ocr_models": easyocr_model_hashes,
        "vendored_distributions": vendored_distributions,
        "training_manifest_schema_version": training_manifest["schema_version"],
        "training_manifest_sha256": production_manifest["training_manifest_sha256"],
    }
    write_json(staging_directory / "BUILD_REPORT.json", report)
    return report


def calculate_zip_member_sha256(archive: zipfile.ZipFile, member_name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member_name) as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_verified_submission_archive(
    staging_directory: Path,
    output_path: Path,
    runtime_hashes: dict[str, str],
) -> None:
    if output_path.suffix.lower() != ".zip":
        raise ValueError(f"Submission output must use the .zip extension: {output_path}")
    temporary_archive_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.exists() or temporary_archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path} / {temporary_archive_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            temporary_archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for source_path in sorted(staging_directory.rglob("*")):
                if not source_path.is_file():
                    continue
                member_name = source_path.relative_to(staging_directory).as_posix()
                parse_safe_relative_path(member_name)
                archive.write(source_path, member_name)
        with zipfile.ZipFile(temporary_archive_path) as archive:
            member_names = archive.namelist()
            if len(member_names) != len(set(member_names)):
                raise ValueError("Submission ZIP contains duplicate members")
            if archive.testzip() is not None:
                raise ValueError("Submission ZIP failed CRC validation")
            required_members = set(RUNTIME_FILE_PATHS) | {
                "README.md",
                "metadata.json",
                "submission_config.json",
                "production_manifest.json",
                "BUILD_REPORT.json",
                "artifacts/adapters/supplements/adapter_config.json",
                "artifacts/adapters/supplements/adapter_model.safetensors",
                "artifacts/adapters/flammable/adapter_config.json",
                "artifacts/adapters/flammable/adapter_model.safetensors",
                "artifacts/easyocr/craft_mlt_25k.pth",
                "artifacts/easyocr/cyrillic_g2.pth",
            }
            missing_members = required_members - set(member_names)
            if missing_members:
                raise ValueError(f"Submission ZIP is incomplete: {sorted(missing_members)}")
            for member_name, expected_sha256 in runtime_hashes.items():
                if calculate_zip_member_sha256(archive, member_name) != expected_sha256:
                    raise ValueError(
                        f"Submission ZIP contains an invalid runtime file: {member_name}"
                    )
            for filename, expected_sha256 in EASYOCR_MODEL_HASHES.items():
                member_name = f"artifacts/easyocr/{filename}"
                if calculate_zip_member_sha256(archive, member_name) != expected_sha256:
                    raise ValueError(
                        f"Submission ZIP contains invalid EasyOCR weights: {member_name}"
                    )
        temporary_archive_path.replace(output_path)
    except Exception:
        temporary_archive_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_arguments()
    project_root = args.project_root.expanduser().resolve()
    solution_profile = load_solution_profile(args.profile)
    submission_profile = solution_profile["submission"]
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else project_root / "artifacts" / "submissions" / str(submission_profile["archive_name"])
    )
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Build the submission with Python 3.12")
    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("Build the submission on Linux x86_64 to vendor compatible binaries")

    with tempfile.TemporaryDirectory(prefix="ecup-submission-") as temporary_name:
        staging_directory = Path(temporary_name)
        build_report = build_submission_staging_directory(
            project_root,
            staging_directory,
            solution_profile,
        )
        create_verified_submission_archive(
            staging_directory,
            output_path,
            build_report["runtime_files"],
        )
    print(
        json.dumps(
            {
                "status": "built",
                "output": str(output_path),
                "bytes": output_path.stat().st_size,
                "sha256": calculate_sha256(output_path),
                "runtime_files": build_report["runtime_files"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

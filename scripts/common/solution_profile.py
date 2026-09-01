"""Load and validate the explicit profile owned by one solution."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

PROFILE_SCHEMA = "e_cup.solution_profile"


def _require_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Solution profile field {field!r} must be an object")
    return value


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Solution profile field {field!r} must be a non-empty string")
    return value


def _validate_relative_path(value: object, field: str) -> str:
    path = PurePosixPath(_require_text(value, field))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Solution profile field {field!r} must be a safe relative path")
    return path.as_posix()


def load_solution_profile(profile_path: Path) -> dict[str, Any]:
    """Return a validated solution profile without adding hidden defaults."""
    resolved_path = profile_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    try:
        profile = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid solution profile JSON: {resolved_path}") from error
    profile = _require_object(profile, "root")
    if profile.get("schema") != PROFILE_SCHEMA or profile.get("schema_version") != 1:
        raise ValueError(f"Unsupported solution profile schema: {resolved_path}")

    _require_text(profile.get("solution"), "solution")
    _require_text(profile.get("display_name"), "display_name")
    training = _require_object(profile.get("training"), "training")
    submission = _require_object(profile.get("submission"), "submission")

    _validate_relative_path(training.get("flammable_result"), "training.flammable_result")
    if training.get("ocr_cache") is not None:
        _validate_relative_path(training["ocr_cache"], "training.ocr_cache")
    inference = _require_object(
        training.get("flammable_inference"),
        "training.flammable_inference",
    )
    if inference.get("aggregation") not in {"single", "max"}:
        raise ValueError("training.flammable_inference.aggregation must be 'single' or 'max'")
    max_images = inference.get("max_images")
    if not isinstance(max_images, int) or isinstance(max_images, bool) or max_images < 1:
        raise TypeError("training.flammable_inference.max_images must be a positive integer")
    if inference["aggregation"] == "single" and max_images != 1:
        raise ValueError("single aggregation requires exactly one image")
    reproduction = _require_object(training.get("reproduction"), "training.reproduction")
    _validate_relative_path(
        reproduction.get("baseline"),
        "training.reproduction.baseline",
    )
    _require_text(
        reproduction.get("post_training_module"),
        "training.reproduction.post_training_module",
    )

    archive_name = _require_text(submission.get("archive_name"), "submission.archive_name")
    if PurePosixPath(archive_name).name != archive_name or not archive_name.endswith(".zip"):
        raise ValueError("submission.archive_name must be a ZIP filename")
    _require_text(submission.get("manifest_schema"), "submission.manifest_schema")
    _require_text(
        submission.get("classifier_description"),
        "submission.classifier_description",
    )
    if not isinstance(submission.get("ocr_affects_classification"), bool):
        raise TypeError("submission.ocr_affects_classification must be boolean")
    return profile

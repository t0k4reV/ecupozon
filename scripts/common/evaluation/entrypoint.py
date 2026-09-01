"""Shared command-line orchestration for solution post-training pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common.pipeline import run_stage


def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_post_training_arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--data-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scope", choices=("validation", "all"), default="validation")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def run_post_training_pipeline(description: str, evaluation_module: str) -> None:
    args = parse_post_training_arguments(description)
    project_root = args.project_root.expanduser().resolve()
    run_stage(
        "[1/2] Downloading and verifying EasyOCR weights",
        "scripts.common.submission.download_easyocr",
        "--project-root",
        str(project_root),
        cwd=project_root,
    )
    evaluation_arguments = [
        "--project-root",
        str(project_root),
        "--images-dir",
        str(args.images_dir.expanduser().resolve()),
        "--scope",
        args.scope,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.data_csv is not None:
        evaluation_arguments.extend(["--data-csv", str(args.data_csv.expanduser().resolve())])
    if args.output_dir is not None:
        evaluation_arguments.extend(["--output-dir", str(args.output_dir.expanduser().resolve())])
    run_stage(
        "[2/2] Evaluating adapters and auditing comments",
        evaluation_module,
        *evaluation_arguments,
        cwd=project_root,
    )

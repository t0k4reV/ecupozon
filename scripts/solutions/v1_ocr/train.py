"""Run the complete reproducible training pipeline for the v1-OCR solution."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common.pipeline import run_stage
from scripts.common.solution_profile import load_solution_profile
from scripts.common.training.build_training_manifest import load_result

PROFILE_PATH = Path(__file__).resolve().with_name("profile.json")
SHARED_SUPPLEMENTS_RESULT = Path(
    "artifacts/gemma_lora/shared/adapters/gemma_e4b_supplements/training_result.json"
)


def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument(
        "--ocr-cache",
        type=Path,
        help="Reuse an existing EasyOCR JSONL instead of generating a new cache.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run one forward pass without training or writing adapters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    project_root = args.project_root.expanduser().resolve()
    images_directory = args.images_dir.expanduser().resolve()
    if not images_directory.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_directory}")

    common_arguments = ("--project-root", str(project_root))
    solution_profile = load_solution_profile(PROFILE_PATH)
    default_cache_path = project_root / solution_profile["training"]["ocr_cache"]
    cache_path = (
        args.ocr_cache.expanduser().resolve() if args.ocr_cache is not None else default_cache_path
    )

    run_stage(
        "[1/6] Preparing Gemma",
        "scripts.common.training.download_gemma",
        *common_arguments,
        cwd=project_root,
    )
    run_stage(
        "[2/6] Preparing training data",
        "scripts.common.training.prepare_dataset",
        *common_arguments,
        "--images-dir",
        str(images_directory),
        cwd=project_root,
    )
    if args.ocr_cache is None and not cache_path.is_file():
        if args.check_only:
            raise FileNotFoundError(
                "--check-only requires an existing --ocr-cache; cache generation is skipped"
            )
        run_stage(
            "[3/6] Downloading EasyOCR models",
            "scripts.common.submission.download_easyocr",
            *common_arguments,
            cwd=project_root,
        )
        run_stage(
            "[4/6] Building OCR training cache",
            "scripts.solutions.v1_ocr.build_ocr_cache",
            *common_arguments,
            "--images-dir",
            str(images_directory),
            "--output",
            str(cache_path),
            cwd=project_root,
        )
    else:
        print(f"[3-4/6] Reusing OCR cache: {cache_path}", flush=True)

    training_arguments = [*common_arguments, "--images-dir", str(images_directory)]
    if args.check_only:
        training_arguments.append("--check-only")
    shared_result = project_root / SHARED_SUPPLEMENTS_RESULT
    if args.check_only or not shared_result.is_file():
        run_stage(
            "[5/6] Training supplements adapter",
            "scripts.common.training.train_supplements",
            *training_arguments,
            cwd=project_root,
        )
    else:
        load_result(
            project_root,
            "БАД",
            "shared",
            SHARED_SUPPLEMENTS_RESULT,
        )
        print(f"[5/6] Reusing supplements adapter: {shared_result}", flush=True)
    run_stage(
        "[6/6] Training OCR flammable-goods adapter",
        "scripts.solutions.v1_ocr.train_flammable",
        *training_arguments,
        "--ocr-cache",
        str(cache_path),
        cwd=project_root,
    )
    if not args.check_only:
        run_stage(
            "[manifest] Building v1-OCR training manifest",
            "scripts.common.training.build_training_manifest",
            *common_arguments,
            "--profile",
            str(PROFILE_PATH),
            cwd=project_root,
        )


if __name__ == "__main__":
    main()

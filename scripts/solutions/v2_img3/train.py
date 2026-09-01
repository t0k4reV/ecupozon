"""Run the complete reproducible training pipeline for the v2-img3 solution."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common.pipeline import run_stage
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
        "--check-only",
        action="store_true",
        help="Run one forward pass per adapter without training or writing adapters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    project_root = args.project_root.expanduser().resolve()
    images_directory = args.images_dir.expanduser().resolve()
    if not images_directory.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_directory}")
    common_arguments = ("--project-root", str(project_root))

    total_stages = 4 if args.check_only else 5
    run_stage(
        f"[1/{total_stages}] Preparing Gemma",
        "scripts.common.training.download_gemma",
        *common_arguments,
        cwd=project_root,
    )
    run_stage(
        f"[2/{total_stages}] Preparing training data",
        "scripts.common.training.prepare_dataset",
        *common_arguments,
        "--images-dir",
        str(images_directory),
        cwd=project_root,
    )
    training_arguments = [*common_arguments, "--images-dir", str(images_directory)]
    if args.check_only:
        training_arguments.append("--check-only")
    shared_result = project_root / SHARED_SUPPLEMENTS_RESULT
    if args.check_only or not shared_result.is_file():
        run_stage(
            f"[3/{total_stages}] Training supplements adapter",
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
        print(f"[3/{total_stages}] Reusing supplements adapter: {shared_result}", flush=True)
    run_stage(
        f"[4/{total_stages}] Training flammable-goods adapter",
        "scripts.solutions.v2_img3.train_flammable",
        *training_arguments,
        cwd=project_root,
    )
    if not args.check_only:
        run_stage(
            "[5/5] Building training manifest",
            "scripts.common.training.build_training_manifest",
            *common_arguments,
            "--profile",
            str(PROFILE_PATH),
            cwd=project_root,
        )


if __name__ == "__main__":
    main()

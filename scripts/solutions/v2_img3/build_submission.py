"""Build the offline submission archive for the v2-img3 solution."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common.pipeline import run_stage

PROFILE_PATH = Path(__file__).resolve().with_name("profile.json")


def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    project_root = args.project_root.expanduser().resolve()
    common_arguments = ("--project-root", str(project_root))
    run_stage(
        "[1/2] Downloading and verifying EasyOCR weights",
        "scripts.common.submission.download_easyocr",
        *common_arguments,
        cwd=project_root,
    )
    build_arguments = [*common_arguments, "--profile", str(PROFILE_PATH)]
    if args.output is not None:
        build_arguments.extend(["--output", str(args.output.expanduser().resolve())])
    run_stage(
        "[2/2] Building and validating v2-img3 submission ZIP",
        "scripts.common.submission.build_archive",
        *build_arguments,
        cwd=project_root,
    )


if __name__ == "__main__":
    main()

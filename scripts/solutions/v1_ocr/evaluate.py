"""Evaluate trained v1 OCR adapters and audit production comments on labeled data."""

from pathlib import Path

from scripts.common.evaluation.pipeline import (
    parse_evaluation_arguments,
    run_solution_evaluation,
)

SOLUTION_DIRECTORY = Path(__file__).resolve().parent


def main() -> None:
    run_solution_evaluation(
        parse_evaluation_arguments(__doc__ or "Evaluate v1-OCR"),
        profile_path=SOLUTION_DIRECTORY / "profile.json",
    )


if __name__ == "__main__":
    main()

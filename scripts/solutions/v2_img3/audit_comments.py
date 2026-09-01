"""Rebuild v2 comments without Gemma inference and compare them with comments v4."""

from pathlib import Path

from scripts.common.evaluation.comments_audit import (
    parse_comments_audit_arguments,
    run_comments_audit,
)

SOLUTION_DIRECTORY = Path(__file__).resolve().parent


def main() -> None:
    run_comments_audit(
        parse_comments_audit_arguments(__doc__ or "Audit v2 comments"),
        profile_path=SOLUTION_DIRECTORY / "profile.json",
    )


if __name__ == "__main__":
    main()

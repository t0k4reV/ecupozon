"""Small helpers for composing pipeline stages from Python modules."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_stage(
    stage_name: str,
    module: str,
    *arguments: str,
    cwd: Path | None = None,
) -> None:
    """Run one pipeline stage and stop immediately if it fails."""
    print(f"{stage_name}: {module}", flush=True)
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        check=True,
        cwd=cwd,
    )

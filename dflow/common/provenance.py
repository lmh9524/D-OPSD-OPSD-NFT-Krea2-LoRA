"""Record what code produced a run.

This is what replaces long-lived experiment branches. A branch kept alive to "remember" a
training run rots: it stops receiving fixes from ``main`` and nobody can tell what it diverged on.
A commit SHA plus the uncommitted diff pins the code exactly, costs nothing, and leaves ``main``
as the only long-lived branch.

Written once per run into the run directory, so months later "which code was this?" has an answer
that does not depend on anyone's memory of branch names.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def collect() -> dict[str, Any]:
    """Git state plus enough environment to explain a numerical difference."""
    import torch

    diff = _git("diff", "HEAD")
    record: dict[str, Any] = {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(diff),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import diffusers

        record["diffusers"] = diffusers.__version__
    except ImportError:
        record["diffusers"] = None

    # Kept separate: a diff can be large, and the summary above should stay readable.
    record["git_diff"] = diff or None
    return record


def write(directory: Path | str) -> dict[str, Any]:
    """Write ``provenance.json`` into the run directory and return the record."""
    record = collect()
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "provenance.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )
    return record


def describe(record: dict[str, Any]) -> str:
    sha = (record.get("git_sha") or "unknown")[:9]
    dirty = "+dirty" if record.get("git_dirty") else ""
    return (
        f"code {sha}{dirty} on {record.get('git_branch')} | "
        f"torch {record.get('torch')} | diffusers {record.get('diffusers')} | "
        f"gpu {record.get('gpu')}"
    )


__all__ = ["collect", "describe", "write"]

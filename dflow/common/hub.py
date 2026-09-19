"""Resolve a Hub repo id to a local directory.

Needed because the components disagree about what a path means: diffusers' ``load_config`` accepts
repo ids, ``safetensors.load_file`` needs a file, and transformers' ``subfolder=`` lookup fails
offline against a partially-downloaded snapshot — which is the normal state here, since a useful
local copy of klein deliberately omits the preview images and the 18 GB single-file duplicate.

Resolving per subfolder rather than per repo is what keeps that partial copy usable: asking for the
whole repo raises ``IncompleteSnapshotError``, asking for ``transformer/`` succeeds.
"""

from __future__ import annotations

import pathlib
import re

#: ``namespace/name``. Deliberately strict: a mistyped local path must not be mistaken for a repo
#: id, or the error becomes a 404 from the Hub instead of "no such directory".
_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def looks_like_repo_id(path: str | pathlib.Path) -> bool:
    text = str(path)
    if text.startswith(("/", "./", "../", "~")):
        return False
    return bool(_REPO_ID.match(text))


def localize(path: str | pathlib.Path, *, subfolder: str = "") -> pathlib.Path:
    """Return a local directory for ``path``, downloading only ``subfolder`` if needed."""
    candidate = pathlib.Path(path)
    if candidate.is_dir():
        return candidate / subfolder if subfolder else candidate

    if not looks_like_repo_id(path):
        raise FileNotFoundError(
            f"{candidate} is not a directory, and does not look like a Hub repo id "
            f"(expected 'namespace/name')."
        )

    from huggingface_hub import snapshot_download

    patterns = [f"{subfolder}/*", "*.json"] if subfolder else None
    root = pathlib.Path(snapshot_download(str(path), allow_patterns=patterns))
    resolved = root / subfolder if subfolder else root
    if not resolved.is_dir():
        raise FileNotFoundError(f"{subfolder!r} not found in {path}")
    return resolved


__all__ = ["localize", "looks_like_repo_id"]

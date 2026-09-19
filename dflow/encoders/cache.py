"""Precomputed prompt embeddings.

The point is to stop loading the text encoder at all. On one 80 GB card that is the largest single
saving available: klein's Qwen3 is ~16 GB in bf16, against ~24 GB for the transformer itself.

**The disk cost is real and worth stating plainly.** FLUX.2 stacks three encoder layers, so one
prompt is ``512 x 12288`` in bf16 — 12.6 MB. A thousand unique prompts is 12.6 GB. That is a good
trade for a few-hundred-image LoRA set and a bad one for a million-caption corpus, so this is opt-in
rather than the default.

Padding cannot be trimmed to save space. The encoder pads to ``max_length`` *and attends over the
padding*, so positions past the prompt hold real values rather than zeros; storing a short sequence
and re-padding with zeros at load time would change the conditioning.

Layout is a directory of shards plus an index, not one file per prompt: a hundred thousand 12 MB
files is a filesystem problem, and ``safe_open`` reads one tensor out of a shard without loading the
rest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import torch

INDEX_NAME = "index.json"
SHARD_PREFIX = "embeds"
#: Tensors per shard. Roughly 3 GB per shard at klein's 12.6 MB per prompt — large enough that file
#: count stays sane, small enough that a partial write loses little.
SHARD_SIZE = 256


def prompt_key(prompt: str, *, out_layers: Iterable[int], max_length: int) -> str:
    """Content hash of everything that changes the embedding.

    ``out_layers`` and ``max_length`` are part of the key because changing either produces different
    values for the same text. Leaving them out is the failure mode this guards against: a cache
    built with ``(9, 18, 27)`` silently reused for ``(10, 20, 30)`` trains on the wrong conditioning
    with no error anywhere.
    """
    digest = hashlib.sha256()
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(",".join(str(layer) for layer in out_layers).encode("utf-8"))
    digest.update(f"|{max_length}".encode())
    return digest.hexdigest()[:32]


class TextEmbedCache:
    """Read/write access to a directory of precomputed embeddings."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self._index: dict[str, str] = {}
        self._handles: dict[str, object] = {}
        index_path = self.directory / INDEX_NAME
        if index_path.is_file():
            self._index = json.loads(index_path.read_text(encoding="utf-8"))

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def __len__(self) -> int:
        return len(self._index)

    @property
    def exists(self) -> bool:
        return bool(self._index)

    # ------------------------------------------------------------------------- writing

    def write(self, entries: dict[str, torch.Tensor]) -> None:
        """Append entries, sharded. Existing keys are kept; re-running only adds what is missing."""
        from safetensors.torch import save_file

        self.directory.mkdir(parents=True, exist_ok=True)
        pending = {key: value for key, value in entries.items() if key not in self._index}
        if not pending:
            return

        existing_shards = sorted(self.directory.glob(f"{SHARD_PREFIX}-*.safetensors"))
        next_shard = len(existing_shards)
        items = list(pending.items())
        for start in range(0, len(items), SHARD_SIZE):
            chunk = dict(items[start : start + SHARD_SIZE])
            name = f"{SHARD_PREFIX}-{next_shard:05d}.safetensors"
            # clone(), not just contiguous(): the caller usually passes slices of one batch tensor,
            # which share storage, and safetensors refuses to save aliased tensors rather than
            # silently duplicating them.
            save_file(
                {key: value.detach().cpu().clone() for key, value in chunk.items()},
                str(self.directory / name),
            )
            for key in chunk:
                self._index[key] = name
            next_shard += 1

        # Index last, so a crash mid-write leaves unreferenced shards rather than an index that
        # promises tensors nobody wrote.
        (self.directory / INDEX_NAME).write_text(
            json.dumps(self._index, indent=0, sort_keys=True), encoding="utf-8"
        )

    # ------------------------------------------------------------------------- reading

    def get(self, key: str, *, device: torch.device | None = None) -> torch.Tensor | None:
        from safetensors import safe_open

        shard = self._index.get(key)
        if shard is None:
            return None
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.directory / shard), framework="pt")
            self._handles[shard] = handle
        tensor = handle.get_tensor(key)  # type: ignore[attr-defined]
        return tensor.to(device) if device is not None else tensor

    def require(self, keys: Iterable[str]) -> None:
        """Fail before training starts if the cache is incomplete.

        Checked up front because the alternative is discovering a miss thousands of steps in, when
        the encoder has already been left unloaded.
        """
        missing = [key for key in keys if key not in self._index]
        if missing:
            raise KeyError(
                f"{len(missing)} prompts are not in the cache at {self.directory} "
                f"(e.g. {missing[:2]}). Re-run tools/data/precompute_text_embeds.py, or enable the text "
                f"encoder so they can be computed online."
            )


__all__ = ["INDEX_NAME", "SHARD_SIZE", "TextEmbedCache", "prompt_key"]

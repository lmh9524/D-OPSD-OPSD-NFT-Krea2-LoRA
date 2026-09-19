"""Prompts and reward metadata, read from a JSONL manifest.

    {"prompt": "a sign reading HELLO", "text": "HELLO"}

Every field other than ``prompt`` becomes reward metadata, unchanged. That is deliberately loose:
which extras a run needs is decided by which rewards are enabled, and a schema that enumerated them
would have to be edited for every new scorer. ``schema.validate`` is where a run asserts the fields
its rewards actually require.

There is no collate function either: ``data/collate.py`` keeps non-tensor values as a list, so a
per-sample metadata dict already arrives as the list of dicts the schema describes. A second
implementation here would be duplicated infrastructure, which is the bug the layering prevents.

There is no retry wrapper and no bucketing here, because there are no files to fail to load and no
shapes to bucket — a prompt is a string. The rollout's target extent comes from the config, not
from the data, which is also why an RL run does not vary in token count the way ref2img does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class PromptSample:
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


def read_manifest(path: Path) -> list[PromptSample]:
    """Parse the JSONL manifest, naming the offending line rather than raising a bare KeyError."""
    samples: list[PromptSample] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "prompt" not in record:
                    raise KeyError("prompt")
                extras = {k: v for k, v in record.items() if k != "prompt"}
                samples.append(PromptSample(prompt=str(record["prompt"]), metadata=extras))
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    if not samples:
        raise ValueError(f"{path} contains no samples")
    return samples


class PromptDataset(Dataset):
    """Prompts for an RL run. One sample is a prompt plus whatever the rewards need."""

    def __init__(self, root: str | Path, *, manifest: str = "train.jsonl") -> None:
        self.root = Path(root)
        path = self.root / manifest
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. An RL run reads prompts from a JSONL manifest — one object per "
                f'line, at least {{"prompt": "..."}}.'
            )
        self.samples = read_manifest(path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {"prompt": sample.prompt, "metadata": dict(sample.metadata)}


__all__ = ["PromptDataset", "PromptSample", "read_manifest"]

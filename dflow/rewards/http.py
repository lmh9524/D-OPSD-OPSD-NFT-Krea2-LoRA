"""Scoring in another process, for models that cannot share the card.

The memory arithmetic forces this to exist from the first commit rather than arriving later. A
klein-9B LoRA step is 50.9-53.6 GiB, and ``trainer/loop.py`` empties the allocator cache before a
preview because one run reached 78.7 of 79.18 GiB. A VLM judge — HPSv3 is a Qwen2-VL-7B, verl-omni's
``genrm_ocr.py`` a Qwen3-VL — is ~15 GB. There is no room. So the only way to use one is out of
process, on another card or another host.

## Failures are visible, never scored as zero

A flaky endpoint must not end a multi-hour run, and it must not quietly become a reward of zero
either: zero is a real score, indistinguishable from a bad image, and a run that trains against a
half-broken scorer looks like a training problem. So ``retries`` bounds the attempts and exhaustion
**raises**. Deciding to continue with a degraded reward is the caller's call to make explicitly,
not a default buried in a helper.

## Concurrency is a real constraint, not a detail

Under data parallelism every rank scores its own trajectories, so the endpoint sees ``dp_size``
concurrent clients at the same moment in every step — the reward phase is synchronised by
construction. An endpoint sized for one client will time out for the others.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from dflow.rewards.base import validate_pixels


class HTTPReward:
    """Posts images to a scorer and reads back one number per image.

    The wire format is deliberately plain — base64 PNGs, prompts, metadata, and a ``scores`` array
    back — so a scorer can be forty lines of FastAPI wrapping whatever model, rather than something
    that has to know about this repo.
    """

    def __init__(
        self,
        url: str,
        *,
        name: str = "http",
        timeout: float = 120.0,
        retries: int = 3,
        backoff: float = 2.0,
        batch_size: int = 8,
    ) -> None:
        if retries < 1:
            raise ValueError(f"retries must be >= 1, got {retries}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.url = url
        self.name = name
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.batch_size = batch_size

    # ---------------------------------------------------------------------------- transport

    @staticmethod
    def _encode(images: torch.Tensor) -> list[str]:
        from PIL import Image

        encoded: list[str] = []
        for frame in images:
            buffer = io.BytesIO()
            Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()).save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        return encoded

    def _post(self, payload: dict[str, Any]) -> list[float]:
        body = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.retries):
            request = urllib.request.Request(
                self.url, data=body, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    scores = json.loads(response.read())["scores"]
                if len(scores) != len(payload["images"]):
                    raise ValueError(
                        f"scorer returned {len(scores)} scores for "
                        f"{len(payload['images'])} images"
                    )
                return [float(score) for score in scores]
            except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
                last = error
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff * (attempt + 1))
        raise RuntimeError(
            f"scoring at {self.url} failed after {self.retries} attempts: {last}. Raising rather "
            f"than scoring zero — a zero reward is a real score, so a degraded scorer would look "
            f"like a training problem instead of an outage."
        ) from last

    # ------------------------------------------------------------------------------ scoring

    def score(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        metadata: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        validate_pixels(images, len(prompts))
        scores: list[float] = []
        for start in range(0, images.shape[0], self.batch_size):
            stop = start + self.batch_size
            scores.extend(
                self._post(
                    {
                        "images": self._encode(images[start:stop]),
                        "prompts": list(prompts[start:stop]),
                        "metadata": [dict(entry) for entry in metadata[start:stop]],
                    }
                )
            )
        return torch.tensor(scores, dtype=torch.float32, device=images.device)


__all__ = ["HTTPReward"]

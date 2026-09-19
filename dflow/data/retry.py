"""Recovery from unreadable samples.

Real datasets contain truncated JPEGs, dead symlinks and files that decode on one machine and not
another. Without this, one bad file ends a multi-day run.

Shared rather than per-dataset for one specific reason: **resampling must stay inside the sample's
own bucket**. vflow's pattern was

    except Exception:
        index = random.randrange(len(self))

which is fine without bucketing and wrong with it — the replacement can land in a different
resolution bucket, and then the batch fails to collate. Every dataset re-implementing retry is one
chance per dataset to get that wrong, so it lives here and takes the bucket assignment as an input.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from typing import Any

from torch.utils.data import Dataset

from dflow.config import RetryConfig

logger = logging.getLogger(__name__)


class RetryDataset(Dataset):
    """Wrap a dataset so unreadable samples are replaced instead of fatal.

    Args:
        dataset: the dataset to wrap.
        config: attempt limit and whether to stay inside the bucket.
        bucket_of: maps an index to a bucket key. Without it every sample shares one bucket, which is
            correct only while there is no bucketing.
    """

    def __init__(
        self,
        dataset: Dataset,
        config: RetryConfig,
        *,
        bucket_of: Callable[[int], Any] | None = None,
    ) -> None:
        if config.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.dataset = dataset
        self.config = config
        self.bucket_of = bucket_of

        self._members: dict[Any, list[int]] = {}
        if bucket_of is not None and config.same_bucket:
            for index in range(len(dataset)):  # type: ignore[arg-type]
                self._members.setdefault(bucket_of(index), []).append(index)

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def _candidates(self, index: int) -> list[int]:
        if not self._members:
            return list(range(len(self)))
        return self._members[self.bucket_of(index)]  # type: ignore[misc]

    def __getitem__(self, index: int) -> Any:
        # Seeded from the index so a retry is reproducible: a run that crashes and resumes makes the
        # same substitution rather than silently training on different data.
        rng = random.Random(index)
        candidates = self._candidates(index)
        attempted: list[int] = []

        for attempt in range(self.config.max_attempts):
            try:
                return self.dataset[index]
            except Exception as error:  # noqa: BLE001 - any decode failure is recoverable
                attempted.append(index)
                logger.warning(
                    "sample %s failed (attempt %d/%d): %s",
                    index,
                    attempt + 1,
                    self.config.max_attempts,
                    error,
                )
                remaining = [i for i in candidates if i not in attempted]
                if not remaining:
                    break
                index = rng.choice(remaining)

        raise RuntimeError(
            f"gave up after {self.config.max_attempts} attempts; tried indices {attempted}. "
            f"Either the data is broadly unreadable or the failure is in the dataset code, not the "
            f"files — retrying further would hide it."
        )


__all__ = ["RetryDataset"]

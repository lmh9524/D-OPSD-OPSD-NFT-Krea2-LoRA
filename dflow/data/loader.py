"""DataLoader construction.

One function, so the pieces are wired the same way every time: retry wraps the dataset, the sampler is
flat or bucketed depending on whether the task supplies an assignment, and both resume from a
consumed-batch count.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch.utils.data import DataLoader, Dataset

from dflow.config import DataConfig
from dflow.data.bucket import BucketedBatchSampler
from dflow.data.collate import collate
from dflow.data.retry import RetryDataset
from dflow.data.sampler import ResumableDistributedBatchSampler


def build_dataloader(
    dataset: Dataset,
    config: DataConfig,
    *,
    dp_rank: int,
    dp_size: int,
    seed: int,
    bucket_assignments: Sequence[int] | None = None,
    num_buckets: int | None = None,
) -> DataLoader:
    """Wire dataset, retry, sampler and loader.

    ``dp_rank``/``dp_size`` — not global rank and world size. Context-parallel ranks work on different
    slices of the *same* sample and must therefore read the *same* batch; only data-parallel ranks get
    different data.

    Pass ``bucket_assignments`` (one bucket index per sample) to batch within buckets. Without it the
    sampler is flat, which is correct only when every sample already shares a shape. Retry becomes
    bucket-aware automatically in that case, since resampling across buckets would break collation.
    """
    loader = config.loader
    bucketed = bucket_assignments is not None and config.bucket.enabled

    if bucketed:
        if num_buckets is None:
            raise ValueError("bucket_assignments requires num_buckets")
        assignments = list(bucket_assignments)  # type: ignore[arg-type]
        wrapped: Dataset = RetryDataset(
            dataset, config.retry, bucket_of=lambda index: assignments[index]
        )
        sampler: Any = BucketedBatchSampler(
            assignments=assignments,
            num_buckets=num_buckets,
            batch_size=loader.batch_size,
            config=config.bucket,
            num_replicas=dp_size,
            rank=dp_rank,
            seed=seed,
            shuffle=loader.shuffle,
        )
    else:
        wrapped = RetryDataset(dataset, config.retry)
        sampler = ResumableDistributedBatchSampler(
            dataset_size=len(dataset),  # type: ignore[arg-type]
            batch_size=loader.batch_size,
            num_replicas=dp_size,
            rank=dp_rank,
            seed=seed,
            shuffle=loader.shuffle,
            drop_last=loader.drop_last,
        )

    kwargs: dict[str, Any] = {
        "dataset": wrapped,
        "batch_sampler": sampler,
        "collate_fn": collate,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
    }
    if loader.num_workers > 0:
        kwargs["prefetch_factor"] = loader.prefetch_factor
        kwargs["persistent_workers"] = loader.persistent_workers
    return DataLoader(**kwargs)


__all__ = ["build_dataloader"]

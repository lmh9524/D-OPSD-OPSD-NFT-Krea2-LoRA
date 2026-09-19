"""Data-pipeline configuration: mechanism only.

Task-specific dataset settings (which columns, how many references, what resolutions)
live with the task, in ``dflow/tasks/<name>/``. This module configures the machinery
that is identical across tasks.

Pure declaration. This module must not import anything else from ``dflow``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(kw_only=True, slots=True)
class DataLoaderConfig:
    batch_size: int = 1
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    shuffle: bool = True
    drop_last: bool = True


@dataclass(kw_only=True, slots=True)
class BucketConfig:
    """Resolution bucketing.

    The bucket *table* comes from the task; this configures how the sampler uses it.
    """

    enabled: bool = True
    # Sequence length must be a multiple of this. 1 while cp == 1; set to the CP degree
    # once context parallelism is on, because diffusers asserts the sharded dimension
    # is divisible by the CP size.
    seq_multiple: int = 1
    # Drop buckets that cannot fill one global batch, instead of padding them.
    drop_underfull: bool = True


@dataclass(kw_only=True, slots=True)
class RetryConfig:
    """Recovery from unreadable samples.

    Resampling must stay inside the sample's own bucket. Drawing uniformly from the
    whole dataset (vflow's pattern) can land in a different bucket, which then fails to
    collate with the rest of the batch.
    """

    max_attempts: int = 8
    same_bucket: bool = True


@dataclass(kw_only=True, slots=True)
class DataConfig:
    loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    bucket: BucketConfig = field(default_factory=BucketConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)


__all__ = ["BucketConfig", "DataConfig", "DataLoaderConfig", "RetryConfig"]

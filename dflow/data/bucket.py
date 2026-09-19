"""Resolution bucketing, resumable and distributed.

The last of the six shared-infrastructure items, and the one that is genuinely hard: bucketing,
exact resume and rank distribution all interact.

Why bucket at all: fixed square crops throw away composition. A 16:9 photo centre-cropped to a
square loses a third of the frame, and training on that teaches the model to compose for squares.
Buckets keep native aspect ratios by grouping samples that share a shape.

Three properties, each of which fails silently if dropped:

* **Every sample in a batch shares a bucket.** Otherwise collation fails — loudly, at least.
* **Every rank in a step draws from the *same* bucket.** Functionally optional under plain data
  parallelism, but it keeps step time balanced, and it becomes mandatory with context parallelism,
  where ranks split one sample's sequence and must agree on its length.
* **The plan is a pure function of (seed, epoch).** Resume replays it from a batch count instead of
  storing a permutation, exactly as the flat sampler does.

Sequence length must also be divisible by the context-parallel degree
(``diffusers/hooks/context_parallel.py:273``), which is what ``seq_multiple`` is for. At ``cp == 1``
it is 1 and constrains nothing.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Sampler

from dflow.config import BucketConfig


@dataclass(frozen=True, slots=True)
class Bucket:
    """One target shape."""

    height: int
    width: int

    @property
    def aspect_ratio(self) -> float:
        return self.height / self.width

    def tokens(self, compression: int) -> int:
        return (self.height // compression) * (self.width // compression)

    def __str__(self) -> str:
        return f"{self.height}x{self.width}"


def build_bucket_table(
    *,
    target_pixels: int,
    aspect_ratios: Sequence[float],
    multiple_of: int = 16,
) -> tuple[Bucket, ...]:
    """Buckets of roughly constant area across a set of aspect ratios.

    Constant area rather than constant side length, so every bucket costs about the same to train:
    token count is proportional to area, and a batch that changes cost by 4x between buckets makes
    step time and memory unpredictable.

    ``multiple_of`` is the VAE's effective compression — a side that is not a multiple of it cannot
    be patchified evenly.
    """
    if target_pixels <= 0:
        raise ValueError("target_pixels must be positive")
    if not aspect_ratios:
        raise ValueError("need at least one aspect ratio")

    buckets: list[Bucket] = []
    for ratio in aspect_ratios:
        if ratio <= 0:
            raise ValueError(f"aspect ratio must be positive, got {ratio}")
        height = math.sqrt(target_pixels * ratio)
        width = height / ratio
        rounded = Bucket(
            height=max(multiple_of, round(height / multiple_of) * multiple_of),
            width=max(multiple_of, round(width / multiple_of) * multiple_of),
        )
        if rounded not in buckets:
            buckets.append(rounded)
    return tuple(buckets)


def assign_bucket(height: int, width: int, buckets: Sequence[Bucket]) -> int:
    """Index of the bucket whose aspect ratio is closest to this image's.

    Closest ratio rather than closest area: the whole point is to preserve composition, and area is
    already near-constant across the table.
    """
    if not buckets:
        raise ValueError("bucket table is empty")
    ratio = height / width
    return min(range(len(buckets)), key=lambda i: abs(buckets[i].aspect_ratio - ratio))


class BucketedBatchSampler(Sampler[list[int]]):
    """Batches drawn from one bucket at a time, resumable from a consumed-batch count.

    ``assignments[i]`` is the bucket index of dataset sample ``i``.
    """

    def __init__(
        self,
        *,
        assignments: Sequence[int],
        num_buckets: int,
        batch_size: int,
        config: BucketConfig,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} outside [0, {num_replicas})")
        if not assignments:
            raise ValueError("no samples to bucket")
        if max(assignments) >= num_buckets:
            raise ValueError(
                f"assignment {max(assignments)} exceeds num_buckets={num_buckets}"
            )

        self.assignments = list(assignments)
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.config = config
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.global_batch = batch_size * num_replicas

        self.members: list[list[int]] = [[] for _ in range(num_buckets)]
        for index, bucket in enumerate(self.assignments):
            self.members[bucket].append(index)

        self.batches_per_epoch = sum(
            len(group) // self.global_batch for group in self.members
        )
        if self.batches_per_epoch == 0:
            largest = max(len(group) for group in self.members)
            raise ValueError(
                f"no bucket can fill a global batch of {batch_size} x {num_replicas} "
                f"(largest holds {largest} samples). Reduce batch_size, widen the buckets, or "
                f"disable bucketing."
            )
        self.step = 0

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("step must be non-negative")
        self.step = step

    def dropped_per_epoch(self) -> int:
        """Samples lost to partial global batches. Reported, never silent."""
        return sum(len(group) % self.global_batch for group in self.members)

    def _epoch_plan(self, epoch: int) -> list[tuple[int, list[int]]]:
        """``(bucket, global_batch_indices)`` for one epoch, mixed across buckets.

        Pure function of ``(seed, epoch)``, so resume replays it rather than storing it.
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)

        plan: list[tuple[int, list[int]]] = []
        for bucket, group in enumerate(self.members):
            if len(group) < self.global_batch:
                continue
            order = (
                [group[i] for i in torch.randperm(len(group), generator=generator).tolist()]
                if self.shuffle
                else list(group)
            )
            usable = (len(order) // self.global_batch) * self.global_batch
            for start in range(0, usable, self.global_batch):
                plan.append((bucket, order[start : start + self.global_batch]))

        if self.shuffle:
            # Mix bucket order so consecutive steps do not all share one shape, which would make the
            # gradient a tour of aspect ratios rather than a sample of them.
            plan = [plan[i] for i in torch.randperm(len(plan), generator=generator).tolist()]
        return plan

    def __iter__(self) -> Iterator[list[int]]:
        step = self.step
        while True:
            epoch, offset = divmod(step, self.batches_per_epoch)
            plan = self._epoch_plan(epoch)
            for index in range(offset, len(plan)):
                _, global_indices = plan[index]
                start = self.rank * self.batch_size
                step += 1
                self.step = step
                # Every rank slices the same global batch, so all ranks share the bucket.
                yield global_indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return self.batches_per_epoch


__all__ = [
    "Bucket",
    "BucketedBatchSampler",
    "assign_bucket",
    "build_bucket_table",
]

"""Resumable distributed batch sampler.

Two properties, both of which fail silently when absent.

**Exactly resumable from a consumed-batch count.** ``set_step(n)`` positions the sampler as if
``n`` batches had already been yielded, deriving epoch and offset from ``n`` rather than replaying.
The trainer supplies its own count — not the sampler's internal cursor — because with
``num_workers > 0`` the loader prefetches, so the sampler is always ahead of what the model
consumed. Restoring the cursor would skip the prefetched-but-unconsumed batches.

**Infinite.** Training is measured in steps, not epochs, so the iterator never ends and the loop
never has to distinguish "epoch boundary" from "resume boundary". Epochs still exist internally, as
the unit that reshuffles.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class ResumableDistributedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} outside [0, {num_replicas})")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last

        if drop_last:
            global_batch = batch_size * num_replicas
            self.batches_per_epoch = dataset_size // global_batch
            self.samples_per_rank = self.batches_per_epoch * batch_size
        else:
            self.samples_per_rank = math.ceil(dataset_size / num_replicas)
            self.batches_per_epoch = math.ceil(self.samples_per_rank / batch_size)

        if self.batches_per_epoch == 0:
            raise ValueError(
                f"{dataset_size} samples cannot fill one global batch of "
                f"{batch_size} x {num_replicas}. Reduce batch_size or set drop_last=False."
            )
        self.total_samples = self.samples_per_rank * num_replicas
        self.step = 0

    def set_step(self, step: int) -> None:
        """Position the sampler as if ``step`` batches had been yielded on this rank."""
        if step < 0:
            raise ValueError("step must be non-negative")
        self.step = step

    def _epoch_indices(self, epoch: int) -> list[int]:
        if self.shuffle:
            # Seeded by epoch, so the order is a pure function of (seed, epoch) and a resume
            # reproduces it without storing the permutation.
            generator = torch.Generator()
            generator.manual_seed(self.seed + epoch)
            indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            indices = list(range(self.dataset_size))

        if self.drop_last:
            indices = indices[: self.total_samples]
        else:
            padding = self.total_samples - len(indices)
            if padding > 0:
                repeats = math.ceil(padding / len(indices))
                indices += (indices * repeats)[:padding]

        indices = indices[self.rank : self.total_samples : self.num_replicas]
        if len(indices) != self.samples_per_rank:
            raise RuntimeError(
                f"rank {self.rank} got {len(indices)} samples, expected {self.samples_per_rank}"
            )
        return indices

    def __iter__(self) -> Iterator[list[int]]:
        step = self.step
        while True:
            epoch, offset = divmod(step, self.batches_per_epoch)
            indices = self._epoch_indices(epoch)
            for index in range(offset, self.batches_per_epoch):
                start = index * self.batch_size
                step += 1
                self.step = step
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return self.batches_per_epoch


__all__ = ["ResumableDistributedBatchSampler"]

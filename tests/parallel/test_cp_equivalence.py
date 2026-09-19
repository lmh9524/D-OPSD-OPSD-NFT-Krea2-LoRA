"""Context parallelism: the contract it must satisfy, recorded before it exists.

Skipped. CP is not implemented -- ``reduce_cp_gradients`` is a no-op at ``cp == 1`` and raises
otherwise. This file exists so the acceptance criteria are written down while the reasoning is fresh,
rather than reconstructed later from the code that happens to have been written.

Running these needs at least two GPUs, so they will not run on this host regardless.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="context parallelism is not implemented; see dflow/runtime/cp.py")


def test_loss_matches_the_single_rank_run():
    """Same seed, same data, cp=1 vs cp=2: the loss must agree to numerical tolerance.

    The whole point of CP is that it changes only *where* the work happens. Any loss difference means
    the sequence split is dropping or duplicating tokens.
    """


def test_gradients_match_the_single_rank_run():
    """The one that catches the reduction bug.

    diffusers' CP gather does not rescale gradients -- its semantics are "sum across the CP dim" --
    while FSDP's reduce-scatter averages over dp_shard * cp. Without the correction in
    reduce_cp_gradients, gradients come out a factor of cp_size too small: training still runs, the
    loss still falls, and the effective learning rate is quietly wrong.
    """


def test_clip_threshold_is_computed_on_corrected_gradients():
    """reduce_cp_gradients must run before clip_grad_norm_.

    Clipping first measures the norm of gradients that are still cp_size off, so the threshold means
    nothing. Nothing raises either way; only the resulting norm differs.
    """


def test_cp_ranks_receive_the_same_batch_and_noise():
    """dp_rank excludes the CP dimension, so a CP group shares data and RNG.

    Ranks splitting one sample's sequence must agree on which sample it is. Seeding from dp_rank gives
    that for free -- but only as long as dp_rank keeps excluding cp.
    """


def test_sequence_length_is_divisible_by_the_cp_degree():
    """EquipartitionSharder asserts it (context_parallel.py:273), which is what seq_multiple is for.

    Multi-reference makes this live: the concatenated length varies with reference count and
    resolution, so bucket tables must be built with seq_multiple=cp_size.
    """


def test_native_cp_plan_composes_with_fsdp():
    """enable_parallelism accepts a custom mesh, and the mesh already folds cp into the FSDP axis.

    If this turns out not to hold, the fallback is a hand-written Ulysses processor -- but the plan is
    to use theirs, so this is the assumption to check first.
    """

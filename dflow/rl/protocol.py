"""What the rollout needs from a token sequence and from the policy — and nothing more.

The consumer owns the interface. That is the same rule that puts ``ParallelSpec`` in ``runtime/``
(L2) while ``models/`` (L3) produces it, and it is what lets ``rl/`` sit at L4 without importing
``tasks/`` or ``models/``: a rollout walks *a* sequence and calls *something* that returns a
velocity, and it must not know that the sequence came from a reference-conditioning builder or that
the velocity came from a 9B transformer under FSDP with classifier-free guidance around it.

``ReferenceSequence`` (``dflow/tasks/ref2img/conditioning.py``) satisfies :class:`SequenceLike`
structurally — no base class, no import in either direction. ``tests/rl/test_rollout.py`` asserts
that, because a structural coupling is invisible until it breaks.

Both protocols are deliberately narrower than what the task offers. ``SequenceLike`` does not
require ``ids`` or ``target_offset``, not because a task lacks them but because the rollout never
touches them: position ids belong to the ``velocity_fn`` closure, and the target span is put back
by ``replace_target`` rather than sliced by offset. A protocol that listed everything available
would claim a coupling that does not exist.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch

#: ``(tokens, sigma) -> velocity``, over the **target span only**.
#:
#: * ``tokens``: ``(B, seq, C)`` — the full concatenated sequence, target span already swapped in.
#: * ``sigma``:  ``(B,)`` in [0, 1] — the normalised timestep. FLUX.2's forward multiplies by 1000
#:   itself, which is why ``flow_matching.py`` returns sigmas on this scale.
#: * returns:    ``(B, target_len, C)`` — the predicted velocity for the target span.
#:
#: Everything else is the closure's business, and deliberately so: which family calling convention
#: to use, where the text embeddings come from, whether to run two forwards for classifier-free
#: guidance and how to combine them, the cast to the model's dtype (the rollout hands over fp32,
#: since ``sde_step`` computes there), and slicing the output to the target span — which is
#: family-specific, since FLUX.2 puts the target first and Krea 2 puts it last.
#:
#: It is also where to micro-batch. A group of eight trajectories is one ``(8, seq, C)`` forward by
#: default; if that does not fit, the closure can split and concatenate without the rollout knowing
#: anything changed.
VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@runtime_checkable
class SequenceLike(Protocol):
    """A token sequence whose target span can be swapped out.

    The rollout replaces the target span once per denoising step and leaves everything else — clean
    reference latents, their position ids — untouched. That is the whole contract.
    """

    #: ``(B, seq, C)``. The rollout reads only its shape, device and dtype; the leading dimension
    #: is the group, so the caller expands it before handing the sequence over.
    tokens: torch.Tensor

    #: How many of those tokens are the target. The rollout draws noise of this length.
    target_len: int

    def replace_target(self, target_tokens: torch.Tensor) -> SequenceLike:
        """Return a sequence with a different target span, references and ids unchanged."""
        ...


__all__ = ["SequenceLike", "VelocityFn"]

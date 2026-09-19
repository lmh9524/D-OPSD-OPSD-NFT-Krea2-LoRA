"""Exponential moving average over trainable parameters.

DTensor-aware: under FSDP2 the parameters are DTensors, and ``lerp_`` between two DTensors with
matching placements works elementwise on the local shards, so no gathering is needed. Shadow
tensors are created by cloning the parameters, which inherits their sharding for free.

Only trainable parameters are tracked. For LoRA that is the adapters, which is also exactly what
gets published — so when EMA is on, the EMA weights *are* the deliverable and losing them to a
crash loses the run's output. Hence its place in the resumable manifest.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    """Shadow copy of the trainable parameters, updated as a running average."""

    def __init__(self, model: nn.Module, *, decay: float = 0.9999, update_every: int = 1) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if update_every < 1:
            raise ValueError("update_every must be >= 1")

        self.decay = decay
        self.update_every = update_every
        self.shadow: dict[str, torch.Tensor] = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self.shadow:
            raise ValueError("EMA found no trainable parameters to track")
        self._stashed: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def step(self, model: nn.Module, *, global_step: int) -> None:
        """Update the shadow weights. Skipped unless ``global_step`` lands on the interval."""
        if global_step % self.update_every:
            return
        for name, parameter in model.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is not None:
                shadow.lerp_(parameter.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: nn.Module) -> None:
        """Move shadow weights into the model, stashing the live ones.

        Used around validation sampling, so the images reflect what will be published.
        """
        if self._stashed is not None:
            raise RuntimeError("swap_in called twice without swap_out")
        self._stashed = {}
        for name, parameter in model.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is not None:
                self._stashed[name] = parameter.detach().clone()
                parameter.copy_(shadow)

    @torch.no_grad()
    def swap_out(self, model: nn.Module) -> None:
        if self._stashed is None:
            return
        for name, parameter in model.named_parameters():
            stashed = self._stashed.get(name)
            if stashed is not None:
                parameter.copy_(stashed)
        self._stashed = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.shadow)

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing = set(self.shadow) - set(state)
        if missing:
            raise ValueError(
                f"EMA state is missing {len(missing)} tracked parameters, e.g. "
                f"{sorted(missing)[:3]}. Resuming would mix averaged and fresh weights."
            )
        for name, shadow in self.shadow.items():
            shadow.copy_(state[name])


class LoRATeacherEMA:
    """EMA teacher for on-policy self-distillation, held as a second (frozen) LoRA adapter.

    Unlike :class:`EMA`, which keeps an independent shadow of the trainable parameters, the teacher
    weights here *are* live parameters of the model — a second adapter (see
    ``models.adapter.apply_dual_lora``). ``step`` updates them in place as an EMA of the student
    adapter, ``teacher = decay*teacher + (1-decay)*student``. The naive alternative, using the
    student's own live weights as the teacher, collapses training (D-OPSD paper); the EMA at
    momentum 0.9999 is what stabilises the alignment target while still tracking the student.

    It exposes the same ``step`` / ``state_dict`` / ``load_state_dict`` surface as :class:`EMA`, so
    it drops into ``fit(ema=...)`` and ``CheckpointManager`` with no loop change: ``fit`` calls
    ``step`` after each optimizer step, and the manager round-trips ``state_dict`` through the
    checkpoint's ``ema`` slot. That persistence is load-bearing — the teacher adapter is frozen, so
    a LoRA checkpoint (``ignore_frozen_params=True``) does **not** save it in the model shards; this
    object is what carries it across a resume.

    DTensor-aware for the same reason :class:`EMA` is: ``lerp_``/``copy_`` between two adapters with
    matching sharding operate on local shards, so no gather is needed.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        src_adapter: str = "default",
        dst_adapter: str = "teacher",
        decay: float = 0.9999,
        update_every: int = 1,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if update_every < 1:
            raise ValueError("update_every must be >= 1")
        self._model = model
        self.decay = decay
        self.update_every = update_every
        src_marker, dst_marker = f".{src_adapter}.", f".{dst_adapter}."
        names = {name for name, _ in model.named_parameters()}
        self.pairs: list[tuple[str, str]] = [
            (name, name.replace(src_marker, dst_marker))
            for name in names
            if "lora_" in name and src_marker in name
            and name.replace(src_marker, dst_marker) in names
        ]
        if not self.pairs:
            raise ValueError(
                f"LoRATeacherEMA found no {src_adapter!r} -> {dst_adapter!r} adapter pairs; "
                "was apply_dual_lora() called with these adapter names?"
            )

    @torch.no_grad()
    def step(self, model: nn.Module, *, global_step: int) -> None:
        """EMA the student adapter into the teacher. Skipped off the update interval."""
        if global_step % self.update_every:
            return
        params = dict(model.named_parameters())
        for src, dst in self.pairs:
            params[dst].data.lerp_(params[src].detach(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        params = dict(self._model.named_parameters())
        return {dst: params[dst].detach().clone() for _, dst in self.pairs}

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        params = dict(self._model.named_parameters())
        missing = {dst for _, dst in self.pairs} - set(state)
        if missing:
            raise ValueError(
                f"LoRATeacherEMA state is missing {len(missing)} teacher parameters, e.g. "
                f"{sorted(missing)[:3]}. Resuming would leave the teacher un-restored."
            )
        for _, dst in self.pairs:
            params[dst].data.copy_(state[dst])


__all__ = ["EMA", "LoRATeacherEMA"]

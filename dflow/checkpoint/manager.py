"""Checkpoint save and resume: the six resumable components, atomically.

``dflow.trainer.state.RESUMABLE`` lists what an exact restart needs; this module is what handles
all of it. A test asserts the two agree, so adding state to the manifest without handling it here
fails loudly.

Two properties worth their cost:

**Atomic.** Writes go to ``.step_XXXXXXX.tmp`` and are renamed only once every rank has finished.
Crashing mid-write then leaves the previous checkpoint intact instead of a half-written directory
that loads without error and resumes from corrupted weights.

**Two formats, on purpose.** Training state goes through
``torch.distributed.checkpoint``, which stores each rank's shard directly — no gather, so a 9B
model never has to fit in one process. The publishable artefact is written separately in diffusers
format, so the result is loadable by a stock pipeline without a conversion step.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from dflow.common.ema import EMA
from dflow.config import CheckpointConfig
from dflow.runtime.context import MeshBundle
from dflow.runtime.seed import gather_rng_states, restore_rng_state
from dflow.trainer.state import TrainState

_STEP = re.compile(r"step_(\d+)")
_META = "meta.pt"
_MODEL = "model"
_OPTIMIZER = "optimizer"
_EXPORT = "lora"


class CheckpointManager:
    def __init__(
        self,
        config: CheckpointConfig,
        *,
        mesh: MeshBundle,
        conditioning: dict[str, object] | None = None,
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.root = Path(config.directory)
        #: Recorded beside every exported LoRA so a sampler can reproduce this run's conditioning
        #: rather than depend on a flag being remembered. See ``lora_io.save_lora``.
        self.conditioning = conditioning

    # ----------------------------------------------------------------------- discovery

    def existing(self) -> list[tuple[int, Path]]:
        if not self.root.is_dir():
            return []
        found = [
            (int(match.group(1)), path)
            for path in self.root.glob("step_*")
            if path.is_dir() and (match := _STEP.fullmatch(path.name))
        ]
        return sorted(found)

    def resolve_resume(self) -> Path | None:
        if self.config.resume is None:
            return None
        if self.config.resume == "latest":
            found = self.existing()
            if not found:
                return None
            return found[-1][1]
        path = Path(self.config.resume)
        if not path.is_dir():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        return path

    # ---------------------------------------------------------------------------- save

    def _barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    def save(
        self,
        *,
        state: TrainState,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        ema: EMA | None,
        generator: torch.Generator,
        lora: bool,
    ) -> Path:
        path = self.root / f"step_{state.global_step:07d}"
        temporary = self.root / f".step_{state.global_step:07d}.tmp"
        if path.exists():
            raise FileExistsError(f"checkpoint already exists: {path}")

        if self.mesh.is_master:
            self.root.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
        self._barrier()

        # LoRA trains a small fraction of the parameters, so saving only those keeps a
        # checkpoint megabytes rather than tens of gigabytes.
        options = StateDictOptions(ignore_frozen_params=lora)
        dcp.save(get_model_state_dict(model, options=options), checkpoint_id=temporary / _MODEL)
        dcp.save(
            get_optimizer_state_dict(model, optimizer, options=options),
            checkpoint_id=temporary / _OPTIMIZER,
        )

        rng_states = gather_rng_states(generator, mesh=self.mesh)
        if self.mesh.is_master:
            torch.save(
                {
                    "train_state": state.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "rng_states": rng_states,
                    "ema": ema.state_dict() if ema is not None else None,
                    "world_size": self.mesh.world_size,
                },
                temporary / _META,
            )
            if self.config.export_diffusers and lora:
                from dflow.checkpoint.lora_io import save_lora

                save_lora(model, temporary / _EXPORT, conditioning=self.conditioning)

        self._barrier()
        if self.mesh.is_master:
            temporary.rename(path)
            self._prune()
        self._barrier()
        return path

    def _prune(self) -> None:
        keep = self.config.keep_latest
        if keep <= 0:
            return
        for _, path in self.existing()[:-keep]:
            shutil.rmtree(path, ignore_errors=True)

    # -------------------------------------------------------------------------- resume

    def load(
        self,
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        ema: EMA | None,
        generator: torch.Generator,
        lora: bool,
    ) -> TrainState:
        options = StateDictOptions(ignore_frozen_params=lora)

        model_state = get_model_state_dict(model, options=options)
        dcp.load(model_state, checkpoint_id=path / _MODEL)
        # strict=False is required, not a shortcut: a LoRA checkpoint deliberately holds only the
        # adapter tensors, so every frozen base weight is "missing" from it — they were already
        # loaded from the base checkpoint. Key correctness is still enforced, one layer up: dcp.load
        # validates the keys it fills against the checkpoint's own metadata and raises on a mismatch.
        set_model_state_dict(
            model,
            model_state,
            options=StateDictOptions(ignore_frozen_params=lora, strict=not lora),
        )

        optimizer_state = get_optimizer_state_dict(model, optimizer, options=options)
        dcp.load(optimizer_state, checkpoint_id=path / _OPTIMIZER)
        set_optimizer_state_dict(model, optimizer, optimizer_state, options=options)

        meta = torch.load(path / _META, map_location="cpu", weights_only=False)
        state = TrainState.from_state_dict(meta["train_state"])
        lr_scheduler.load_state_dict(meta["lr_scheduler"])
        restore_rng_state(generator, meta["rng_states"], mesh=self.mesh)

        if ema is not None:
            if meta.get("ema") is None:
                raise ValueError(
                    "EMA is enabled but the checkpoint holds no EMA state. Resuming would "
                    "restart the average from the current weights and publish the wrong ones."
                )
            ema.load_state_dict(meta["ema"])

        return state


__all__ = ["CheckpointManager"]

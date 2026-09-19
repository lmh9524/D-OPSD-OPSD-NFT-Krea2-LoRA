"""LoRA, via ``PeftAdapterMixin.add_adapter``.

Not ``peft.get_peft_model``. The distinction is structural, not stylistic:

``add_adapter`` calls ``inject_adapter_in_model``, which inserts ``lora_A``/``lora_B``
in place. Every existing module keeps its path, so the parallel spec, the FSDP wrap units
and the checkpoint keys all stay valid.

``get_peft_model`` wraps the model in a ``PeftModel``, rewriting every path to
``base_model.model.<...>.base_layer``. That is what forces the string surgery in vflow's
checkpoint code:

    key.replace("base_model.model.", "").replace("base_layer.", "")

and it means block paths derived from ``_no_split_modules`` no longer resolve.

Order matters when combining LoRA with meta-device init:

    build on meta -> add_adapter -> apply parallelism -> to_empty -> load base weights
                  -> reset_lora_parameters()

Adapter tensors created on meta are uninitialised after ``to_empty``, and loading the base
checkpoint does not touch them, so they must be re-initialised explicitly at the end.
Skipping that trains from garbage rather than from the intended zero-output initialisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from dflow.config import LoRAConfig


@dataclass(frozen=True, slots=True)
class ParameterSummary:
    trainable: int
    total: int

    @property
    def ratio(self) -> float:
        return 100.0 * self.trainable / self.total if self.total else 0.0

    def describe(self) -> str:
        return f"{self.trainable:,} / {self.total:,} parameters trainable ({self.ratio:.4f}%)"


def apply_lora(model: nn.Module, config: LoRAConfig, *, targets: tuple[str, ...]) -> nn.Module:
    """Inject LoRA adapters in place and freeze everything else.

    ``targets`` comes from the family (``default_lora_targets``) unless the config
    overrides it. For FLUX.2 it must include ``attn.to_qkv_mlp_proj``, or the 48
    single-stream blocks are silently left untouched.
    """
    if not config.enabled:
        raise ValueError("apply_lora called with LoRAConfig.enabled=False")
    if not targets:
        raise ValueError("LoRA needs at least one target module")

    from peft import LoraConfig

    adapter_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(targets),
        init_lora_weights=config.init_weights,
        bias="none",
    )
    model.add_adapter(adapter_config, adapter_name=config.adapter_name)
    freeze_base(model)
    return model


def freeze_base(model: nn.Module) -> ParameterSummary:
    """Train adapter tensors only.

    ``inject_adapter_in_model`` does not own the base model's ``requires_grad`` the way
    ``PeftModel`` does, so this is set explicitly rather than assumed.
    """
    trainable = 0
    total = 0
    for name, parameter in model.named_parameters():
        is_adapter = "lora_" in name
        parameter.requires_grad_(is_adapter)
        count = parameter.numel()
        total += count
        if is_adapter:
            trainable += count
    if trainable == 0:
        raise ValueError(
            "no adapter parameters found after injection — check that target_modules "
            "match real module names for this architecture"
        )
    return ParameterSummary(trainable=trainable, total=total)


def reset_lora_parameters(model: nn.Module, *, adapter_name: str = "default") -> int:
    """Re-initialise adapter tensors after ``to_empty()``.

    Returns the number of layers reset, so callers can assert it is non-zero rather than
    silently training from uninitialised memory.
    """
    from peft.tuners.lora import LoraLayer

    count = 0
    for module in model.modules():
        if isinstance(module, LoraLayer) and adapter_name in module.lora_A:
            module.reset_lora_parameters(adapter_name, init_lora_weights=True)
            count += 1
    return count


def parameter_summary(model: nn.Module) -> ParameterSummary:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return ParameterSummary(trainable=trainable, total=total)


def apply_dual_lora(
    model: nn.Module,
    config: LoRAConfig,
    *,
    targets: tuple[str, ...],
    teacher_name: str = "teacher",
) -> nn.Module:
    """Inject *two* LoRA adapters for on-policy self-distillation (D-OPSD).

    The trainable, deployable adapter keeps ``config.adapter_name`` (normally ``"default"``, so the
    exported LoRA is the student). A second, frozen adapter ``teacher_name`` holds the EMA copy of
    the student. Both are injected in place via ``add_adapter`` — the same path-preserving contract
    as :func:`apply_lora`, so the parallel spec, FSDP wrap units and checkpoint keys stay valid for
    both adapters.

    Only the student adapter's tensors are left trainable; the base model and the teacher adapter
    are frozen. Order is the same as single-adapter LoRA: call on the meta model, before
    parallelism; then after ``to_empty`` reset **both** adapters (``reset_lora_parameters`` once per
    name) and :func:`copy_adapter` student -> teacher so the teacher starts equal to the student.
    """
    if not config.enabled:
        raise ValueError("apply_dual_lora called with LoRAConfig.enabled=False")
    if not targets:
        raise ValueError("LoRA needs at least one target module")
    if teacher_name == config.adapter_name:
        raise ValueError(
            f"teacher adapter name {teacher_name!r} must differ from the student adapter "
            f"{config.adapter_name!r}"
        )

    from peft import LoraConfig

    adapter_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(targets),
        init_lora_weights=config.init_weights,
        bias="none",
    )
    model.add_adapter(adapter_config, adapter_name=config.adapter_name)
    model.add_adapter(adapter_config, adapter_name=teacher_name)

    # Train the student adapter only. ``add_adapter`` does not own requires_grad the way a
    # PeftModel would, so set it explicitly — base frozen, teacher frozen, student trainable.
    student_marker = f".{config.adapter_name}."
    trainable = 0
    for name, parameter in model.named_parameters():
        is_student = "lora_" in name and student_marker in name
        parameter.requires_grad_(is_student)
        if is_student:
            trainable += 1
    if trainable == 0:
        raise ValueError(
            "no trainable student-adapter parameters after dual injection — check target_modules "
            "match real module names for this architecture, and that adapter names are distinct"
        )
    # Student is the active adapter by default; the D-OPSD step toggles per forward.
    if hasattr(model, "set_adapter"):
        model.set_adapter(config.adapter_name)
    return model


def copy_adapter(model: nn.Module, *, src: str, dst: str) -> int:
    """Copy one LoRA adapter's tensors into another, matching by parameter name.

    Replaces ``.src.`` with ``.dst.`` in each ``lora_*`` parameter name. Used at init so the EMA
    teacher starts identical to the student, and by any warm-start that fills one adapter first.
    Returns the number of tensors copied (assert it is non-zero).
    """
    params = dict(model.named_parameters())
    src_marker, dst_marker = f".{src}.", f".{dst}."
    count = 0
    with torch.no_grad():
        for name, parameter in params.items():
            if "lora_" not in name or src_marker not in name:
                continue
            target = params.get(name.replace(src_marker, dst_marker))
            if target is not None:
                target.data.copy_(parameter.data)
                count += 1
    return count


__all__ = [
    "ParameterSummary",
    "apply_dual_lora",
    "apply_lora",
    "copy_adapter",
    "freeze_base",
    "parameter_summary",
    "reset_lora_parameters",
]

"""Export LoRA in the format the stock pipelines load.

This is the deliverable of a LoRA run, so the format matters more than the mechanism: the whole
point of injecting adapters with ``add_adapter`` — module paths unchanged — is that the result can
be handed to ``Flux2Pipeline.load_lora_weights`` with no conversion step.

Export therefore delegates to ``Flux2LoraLoaderMixin.save_lora_weights``. Writing the safetensors
by hand would mean reproducing its key naming, and a mismatch there does not raise: the loader
simply finds no matching keys and applies nothing, so inference looks like a LoRA that had no
effect.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"


def adapter_state_dict(model: nn.Module, *, adapter_name: str = "default") -> dict[str, torch.Tensor]:
    """The adapter tensors, in peft's naming.

    Works on a model with adapters *injected* (``add_adapter``) rather than wrapped, which is why
    the keys need no prefix surgery.
    """
    from peft.utils import get_peft_model_state_dict

    state = get_peft_model_state_dict(model, adapter_name=adapter_name)
    if not state:
        raise ValueError(
            f"no adapter tensors found for {adapter_name!r}; was apply_lora() called?"
        )
    return state


#: Written beside the weights, read back by the sampler. See ``save_lora``.
CONDITIONING_NAME = "conditioning.json"


def save_lora(
    model: nn.Module,
    directory: Path | str,
    *,
    adapter_name: str = "default",
    conditioning: dict[str, object] | None = None,
) -> Path:
    """Write a LoRA that ``load_lora_weights`` accepts, and how it was conditioned.

    Returns the directory. ``save_lora_weights`` names the weight file itself.

    ``conditioning`` records the settings a sampler has to reproduce: reference geometry, how many
    references reached the text encoder, whether the patch embedding ran as a convolution or a
    linear layer. It is written to ``conditioning.json`` so the sampler can read it back instead of
    relying on someone remembering a flag.

    That reliance has failed repeatedly and expensively here. A LoRA trained with image-grounded
    conditioning was evaluated text-only, and three rounds of results measured the mismatch. Then a
    run trained with nine grounded references was evaluated with the sampler's default of one, which
    voided its numbers and its images. Both times nothing raised: sampling under different
    conditioning produces plausible pictures, just not ones that say anything about the checkpoint.
    A file the sampler reads is the only version of this that cannot be forgotten.
    """
    import json

    from diffusers.loaders.lora_pipeline import Flux2LoraLoaderMixin

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    Flux2LoraLoaderMixin.save_lora_weights(
        save_directory=str(path),
        transformer_lora_layers=adapter_state_dict(model, adapter_name=adapter_name),
        safe_serialization=True,
    )
    if conditioning is not None:
        (path / CONDITIONING_NAME).write_text(
            json.dumps(conditioning, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return path


def load_conditioning(directory: Path | str) -> dict[str, object]:
    """Read back what ``save_lora`` recorded, or ``{}`` for a checkpoint written before it did."""
    import json

    path = Path(directory) / CONDITIONING_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_lora_state(directory: Path | str) -> dict[str, torch.Tensor]:
    """Read an exported LoRA back, via the same loader inference uses.

    Round-tripping through ``lora_state_dict`` rather than ``safetensors.load_file`` is what makes
    a test meaningful: it proves the pipeline's loader recognises our keys.
    """
    from diffusers.loaders.lora_pipeline import Flux2LoraLoaderMixin

    return Flux2LoraLoaderMixin.lora_state_dict(str(directory))


__all__ = [
    "CONDITIONING_NAME",
    "LORA_WEIGHT_NAME",
    "adapter_state_dict",
    "load_conditioning",
    "load_lora_state",
    "save_lora",
]

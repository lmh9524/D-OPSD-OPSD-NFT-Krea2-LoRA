"""Model registry: a name maps to a family and a default checkpoint.

Variants of one architecture share a family. The differences between FLUX.2 klein-4B,
klein-9B and dev-32B are values in ``config.json`` plus which text encoder they pair
with — not different code.
"""

from __future__ import annotations

from dataclasses import dataclass

from dflow.models.family.base import Family
from dflow.models.family.flux2 import Flux2Family
from dflow.models.family.krea2 import Krea2Family
from dflow.models.family.minimax_h3 import MiniMaxH3Family


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    family: Family
    #: Default HF repo id. Overridden by ``ModelConfig.path`` when set.
    default_repo: str
    #: Which text-encoder layers to stack. FLUX.2 feeds the concatenation of several
    #: intermediate layers to the transformer, which is why the transformer's
    #: ``joint_attention_dim`` is a multiple of the encoder's hidden size.
    text_out_layers: tuple[int, ...]
    notes: str = ""


_FLUX2 = Flux2Family()
_MINIMAX_H3 = MiniMaxH3Family()

#: Krea 2 taps twelve Qwen3-VL-4B decoder layers, not FLUX.2's three, and stacks them on a separate
#: axis rather than flattening them. Read from `Krea2Pipeline.__init__`'s
#: `text_encoder_select_layers` default; indices are into `hidden_states`, where 0 is the embedding
#: output. Both Krea 2 checkpoints use the same taps.
_KREA2_TEXT_LAYERS: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

REGISTRY: dict[str, RegistryEntry] = {
    "flux2-klein-base-9b": RegistryEntry(
        family=_FLUX2,
        default_repo="black-forest-labs/FLUX.2-klein-base-9B",
        text_out_layers=(9, 18, 27),
        notes="Undistilled; the variant BFL recommends for fine-tuning.",
    ),
    "flux2-klein-9b": RegistryEntry(
        family=_FLUX2,
        default_repo="black-forest-labs/FLUX.2-klein-9B",
        text_out_layers=(9, 18, 27),
        notes="Distilled.",
    ),
    "flux2-klein-4b": RegistryEntry(
        family=_FLUX2,
        default_repo="black-forest-labs/FLUX.2-klein-4B",
        text_out_layers=(9, 18, 27),
        notes="Smallest klein; a rank-16 LoRA reportedly fits in 24 GB.",
    ),
    "flux2-dev": RegistryEntry(
        family=_FLUX2,
        default_repo="black-forest-labs/FLUX.2-dev",
        text_out_layers=(10, 20, 30),
        notes="32B, Mistral Small 3.1 text encoder.",
    ),
    "krea2-raw": RegistryEntry(
        family=Krea2Family(distilled=False),
        default_repo="krea/Krea-2-Raw",
        text_out_layers=_KREA2_TEXT_LAYERS,
        notes="Undistilled midtrain checkpoint; mu follows resolution. The variant to fine-tune.",
    ),
    "krea2-turbo": RegistryEntry(
        family=Krea2Family(distilled=True),
        default_repo="krea/Krea-2-Turbo",
        text_out_layers=_KREA2_TEXT_LAYERS,
        notes=(
            "Few-step distilled (TDM). Pins mu=1.15 at every resolution. Fine-tuning a distilled "
            "checkpoint adapts a collapsed trajectory, so prefer krea2-raw unless the few-step "
            "behaviour is itself the thing being preserved."
        ),
    ),
    "minimax-h3": RegistryEntry(
        family=_MINIMAX_H3,
        default_repo="MiniMaxAI/MiniMax-H3",
        #: H3's text conditioning is one stream from its own encoder, not a stack of tapped decoder
        #: layers, so there is no layer selection to record here.
        text_out_layers=(),
        notes=(
            "Joint video + audio, and the first non-image backbone here. The transformer is "
            "vendored ahead of the diffusers pin (it does not exist in 0.39.0). Training needs a "
            "video task to build the packed-sequence layout; prepare_inputs raises until then."
        ),
    ),
}


def resolve(name: str) -> RegistryEntry:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}. Known: {sorted(REGISTRY)}. "
            f"Add an entry in dflow/models/registry.py."
        ) from None


def get_family(name: str) -> Family:
    return resolve(name).family


__all__ = ["REGISTRY", "RegistryEntry", "get_family", "resolve"]

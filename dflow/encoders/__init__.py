"""L3: frozen auxiliary models.

These wrappers own **lifecycle** — whether to load at all, dtype, placement, tiling — and delegate
every bit of encoding maths to diffusers. That split is deliberate: the dominant risk in this layer is
training diverging from inference, and the only way to guarantee it cannot is to run the same code
inference runs.
"""

from dflow.encoders.cache import TextEmbedCache, prompt_key
from dflow.encoders.text import TextConditioning, TextEncoder
from dflow.encoders.text_krea2 import Krea2TextConditioning, Krea2TextEncoder
from dflow.encoders.vae import VAEEncoder
from dflow.encoders.vae_krea2 import Krea2VAEEncoder

__all__ = [
    "Krea2TextConditioning",
    "Krea2TextEncoder",
    "Krea2VAEEncoder",
    "TextConditioning",
    "TextEmbedCache",
    "TextEncoder",
    "VAEEncoder",
    "prompt_key",
]

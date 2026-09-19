"""One loaded Krea 2 sampler, reusable across many renders.

Both `sample_krea2.py` and `eval_krea2_refs.py` go through this. They used to be a CLI and a
subprocess loop over that CLI, which meant a twenty-sample evaluation loaded a 13B transformer forty
times — around 25 s of the 30 s each render took. At the scale a sweep needs (eight checkpoints,
fifty samples, two arms) that is five hours of loading weights.

Keeping it one class rather than two code paths is deliberate. The last time this repository had two
implementations of the same denoise loop, one of them silently drifted to an older token layout and
three rounds of evaluation measured the sampler instead of the model.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from dflow.config import TextEncoderConfig, VAEConfig  # noqa: E402
from dflow.encoders.vae_krea2 import Krea2VAEEncoder  # noqa: E402
from dflow.models.family.krea2 import Krea2Family  # noqa: E402
from dflow.tasks.ref2img.conditioning import (  # noqa: E402
    build_krea2_sequence,
    build_krea2_text_ids,
)
from dflow.tasks.ref2img.sampling import denoise  # noqa: E402

#: Used only when a checkpoint predates `conditioning.json` and the caller passed nothing. These are
#: the values the earliest runs used, so an old checkpoint keeps behaving as it did.
FALLBACK = {
    "reference_registration": "center",
    "reference_t_scale": 1,
    "max_grounded_references": 1,
    "grounding_max_px": 384,
    "max_length": 1024,
}


@dataclass
class Settings:
    """Everything about conditioning that has to match how the checkpoint was trained."""

    reference_registration: str = "center"
    reference_t_scale: int = 1
    reference_fit_target: bool = False
    reference_max_area: int = 512 * 512
    ground_references: bool = False
    max_grounded_references: int = 1
    grounding_max_px: int = 384
    max_length: int = 1024
    fast_patch_embed: bool = False
    processor_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    steps: int | None = None
    guidance: float | None = None
    lora_scale: float = 1.0
    negative_prompt: str = ""


def resolve(args, *, lora: str | None, log=print) -> Settings:
    """Merge command-line flags with what the checkpoint recorded.

    An explicit flag wins. An *unset* flag takes the training value, not a parser default that
    cannot know what this LoRA was trained with — which is how a checkpoint trained with nine
    grounded references came to be evaluated with one, voiding a full result set.
    """
    recorded: dict = {}
    if lora:
        from dflow.checkpoint.lora_io import load_conditioning

        recorded = load_conditioning(lora)
        if recorded:
            log(f"conditioning from checkpoint: {recorded}")
        else:
            log(
                "no conditioning.json in the LoRA directory (it postdates this checkpoint); "
                "falling back to defaults, so pass the training flags explicitly"
            )

    settings = Settings()
    for name in ("reference_registration", "max_grounded_references", "grounding_max_px",
                 "max_length", "reference_t_scale"):
        given = getattr(args, name, None)
        setattr(settings, name, given if given is not None else recorded.get(name, FALLBACK[name]))
    for name in ("ground_references", "fast_patch_embed", "reference_fit_target"):
        given = bool(getattr(args, name, False))
        if not given and recorded.get(name):
            given = True
            log(f"  enabling --{name.replace('_', '-')} because the checkpoint was trained with it")
        setattr(settings, name, given)
    for name in ("reference_max_area", "processor_path", "steps", "guidance", "lora_scale",
                 "negative_prompt"):
        if getattr(args, name, None) is not None:
            setattr(settings, name, getattr(args, name))
    return settings


def load_image(path, *, max_area: int, multiple: int = 16, fit_inside=None) -> torch.Tensor:
    """Read an image to ``(1, 3, H, W)`` in [-1, 1], mirroring the dataset's transform.

    ``fit_inside=(height, width)`` sizes against the target's extent rather than an area cap, which
    is what `--dataset.reference-fit-target` does during training.
    """
    import numpy as np
    from PIL import Image

    image = path if isinstance(path, Image.Image) else Image.open(path).convert("RGB")
    width, height = image.size
    if fit_inside is not None:
        limit_h, limit_w = fit_inside
        scale = min(1.0, limit_h / height, limit_w / width)
    else:
        scale = min(1.0, (max_area / float(width * height)) ** 0.5)
    width, height = max(multiple, int(width * scale)), max(multiple, int(height * scale))
    image = image.resize((width, height), Image.LANCZOS)

    crop_w, crop_h = width - width % multiple, height - height % multiple
    left, top = (width - crop_w) // 2, (height - crop_h) // 2
    image = image.crop((left, top, left + crop_w, top + crop_h))
    array = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
    return array.unsqueeze(0)


class Krea2Sampler:
    """A loaded pipeline plus its conditioning contract. Build once, render many times."""

    def __init__(self, pipe, settings: Settings, device: torch.device, dtype=torch.bfloat16):
        self.pipe = pipe
        self.settings = settings
        self.device = device
        self.dtype = dtype
        self.transformer = pipe.transformer.eval()
        self.family = Krea2Family(distilled=bool(pipe.config.is_distilled))
        # krea-ai/krea-2 ships different numbers for the two checkpoints — Raw 52 steps at cfg 3.5,
        # Turbo 8 unguided — and running one at the other's settings looks like a broken adapter.
        recommended = self.family.sampling_defaults()
        self.steps = settings.steps or int(recommended["steps"])
        self.guidance = (
            settings.guidance if settings.guidance is not None else float(recommended["guidance"])
        )
        self.vae = Krea2VAEEncoder(
            pipe.vae, VAEConfig(encode_dtype="bfloat16"), device=device,
            patch_size=pipe.patch_size,
        )
        self.encoder = None
        if settings.ground_references:
            from dflow.encoders.text_krea2 import Krea2TextEncoder

            self.encoder = Krea2TextEncoder(
                pipe.text_encoder, pipe.tokenizer,
                TextEncoderConfig(dtype="bfloat16", max_length=settings.max_length),
                device=device, select_layers=tuple(pipe.text_encoder_select_layers),
                processor_path=settings.processor_path,
                grounding_max_px=settings.grounding_max_px,
                grounding_jitter_min=0,          # deterministic at sampling time
                max_grounded=settings.max_grounded_references,
                fast_patch_embed=settings.fast_patch_embed,
            )
        self._negative = None

    @classmethod
    def load(cls, model: str, lora: str | None, settings: Settings, log=print) -> Krea2Sampler:
        from diffusers import Krea2Pipeline

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = Krea2Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to(device)
        pipe.set_progress_bar_config(disable=True)
        if lora:
            from dflow.checkpoint.lora_io import LORA_WEIGHT_NAME

            # ``weight_name`` is explicit: ``load_lora_adapter`` given a bare directory looks for
            # ``pytorch_lora_weights.bin`` and errors on the safetensors we actually write.
            pipe.transformer.load_lora_adapter(
                lora, weight_name=LORA_WEIGHT_NAME, adapter_name="default"
            )
            pipe.transformer.set_adapters(["default"], weights=[settings.lora_scale])
            log(f"loaded LoRA from {lora} at scale {settings.lora_scale}")
        sampler = cls(pipe, settings, device)
        log(
            f"{'turbo' if sampler.family.distilled else 'raw'}: "
            f"{sampler.steps} steps, cfg {sampler.guidance}"
        )
        return sampler

    @torch.no_grad()
    def render(self, prompt: str, references, *, width: int, height: int, seed: int,
               no_refs: bool = False):
        """Render one image. ``references`` are paths or PIL images, in slot order."""
        settings = self.settings
        compression = self.vae.spatial_compression
        height -= height % compression
        width -= width % compression

        pixels = []
        if not no_refs:
            pixels = [
                load_image(
                    r, max_area=settings.reference_max_area,
                    fit_inside=(height, width) if settings.reference_fit_target else None,
                )
                for r in references
            ]

        # Grounding first: when the LoRA was trained with image-grounded encoding the same pixels
        # have to reach the text encoder too, or the render measures the mismatch.
        if self.encoder is not None and pixels:
            grounded = self.encoder.encode_grounded(prompt, pixels)
            embeds, mask = grounded.embeds, grounded.mask
        else:
            embeds, mask = self.pipe.get_text_hidden_states(prompt, device=self.device)

        latents = [self.vae.encode(p.to(self.device), mode="mode").float() for p in pixels]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        target = torch.randn(
            1, self.vae.latent_channels, height // compression, width // compression,
            generator=generator, device=self.device, dtype=torch.float32,
        )
        sequence = build_krea2_sequence(
            target_latents=target, reference_latents=latents,
            registration=settings.reference_registration,
            t_scale=settings.reference_t_scale,
        )

        import numpy as np
        from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift, retrieve_timesteps

        mu = (
            1.15
            if self.pipe.config.is_distilled
            else calculate_shift(
                sequence.target_len,
                self.pipe.scheduler.config.get("base_image_seq_len", 256),
                self.pipe.scheduler.config.get("max_image_seq_len", 6400),
                self.pipe.scheduler.config.get("base_shift", 0.5),
                self.pipe.scheduler.config.get("max_shift", 1.15),
            )
        )
        timesteps, _ = retrieve_timesteps(
            self.pipe.scheduler, self.steps, self.device,
            sigmas=np.linspace(1.0, 1 / self.steps, self.steps), mu=mu,
        )
        self.pipe.scheduler.set_begin_index(0)

        negative_embeds = negative_mask = negative_ids = None
        if self.guidance > 0:
            if self._negative is None:
                self._negative = self.pipe.get_text_hidden_states(
                    settings.negative_prompt, device=self.device
                )
            negative_embeds, negative_mask = self._negative
            negative_ids = build_krea2_text_ids(negative_embeds.shape[1], device=self.device)

        result = denoise(
            model=self.transformer, family=self.family, sequence=sequence,
            text_embeds=embeds.to(self.dtype),
            text_ids=build_krea2_text_ids(embeds.shape[1], device=self.device),
            text_mask=mask, timesteps=timesteps, scheduler=self.pipe.scheduler,
            num_train_timesteps=self.pipe.scheduler.config.num_train_timesteps,
            guidance=self.guidance,
            negative_embeds=None if negative_embeds is None else negative_embeds.to(self.dtype),
            negative_mask=negative_mask, negative_ids=negative_ids,
        )
        return self.vae.decode_to_pil(
            result, height=height // compression, width=width // compression
        )


__all__ = ["FALLBACK", "Krea2Sampler", "Settings", "load_image", "resolve"]

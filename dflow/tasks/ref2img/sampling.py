"""The reference-conditioned denoising loop, in one place.

``SKILL.md`` argues against an in-training sampler on the grounds that it is *a second
implementation of the denoise loop*, free to drift from the one inference runs. That objection is
about duplication, not about sampling, so this module removes the duplication instead of the
feature: ``tools/krea2/sample_krea2.py`` and the in-training preview both call :func:`denoise`, and there
is exactly one loop to keep correct.

It stays a plain function taking the model as an argument — no import of ``models/`` — so a task
module may hold it without inverting the layering. The caller supplies the family adapter, which is
what knows the architecture's calling convention.

Preview sampling during training is nearly free, which is the reason it is worth having: the
transformer, VAE and text encoder are already resident, so a preview costs a handful of forward
passes and no extra weights. Loading them again in a separate process would cost ~34 GB and could
not run alongside training at all.
"""

from __future__ import annotations

import torch

from dflow.tasks.ref2img.conditioning import ReferenceSequence


@torch.no_grad()
def denoise(
    *,
    model,
    family,
    sequence: ReferenceSequence,
    text_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    text_mask: torch.Tensor | None,
    timesteps: torch.Tensor,
    scheduler,
    num_train_timesteps: int = 1000,
    guidance: float = 0.0,
    negative_embeds: torch.Tensor | None = None,
    negative_mask: torch.Tensor | None = None,
    negative_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the velocity field from noise to data, over the target span only.

    ``sequence`` carries the initial noisy target followed by the reference spans; the references
    are clean latents and are re-attached unchanged at every step, exactly as in training. Returns
    the denoised target tokens.

    ``scheduler`` is the inference-side stepper (``FlowMatchEulerDiscreteScheduler``), not the
    training-side one — the pipeline's own object, so the integration matches what a downstream user
    would get.

    ``negative_ids`` is separate from ``text_ids`` because the two branches need not be the same
    length. Grounded conditioning runs at the prompt's natural length while the negative branch is
    padded to ``max_length``, so reusing one set of ids raises ``text_ids has N rows, expected M``
    the moment guidance is switched on — which is exactly what a preview reported once CFG was
    enabled for a non-distilled backbone. It defaults to ``text_ids`` for the common case where both
    branches are padded alike.

    The classifier-free guidance combination is the pipeline's, including its unusual form
    ``pred + scale * (pred - negative)`` rather than ``negative + scale * (pred - negative)``: the
    effective strength is therefore ``1 + scale``. Sampling at a different effective scale than the
    checkpoint expects looks like a training problem, so it is reproduced rather than corrected.
    """
    dtype = text_embeds.dtype
    start, stop = sequence.target_offset, sequence.target_offset + sequence.target_len
    latents = sequence.tokens[:, start:stop].to(dtype)
    head = sequence.tokens[:, :start].to(dtype)
    tail = sequence.tokens[:, stop:].to(dtype)
    do_guidance = guidance > 0 and negative_embeds is not None
    if do_guidance:
        if negative_ids is None:
            negative_ids = text_ids
        if negative_ids.shape[0] != negative_embeds.shape[1]:
            raise ValueError(
                f"negative_ids has {negative_ids.shape[0]} rows for "
                f"{negative_embeds.shape[1]} negative text tokens; pass negative_ids explicitly "
                "when the two branches are encoded at different lengths"
            )

    for t in timesteps:
        timestep = (t / num_train_timesteps).expand(latents.shape[0]).to(dtype)
        tokens = torch.cat([head, latents, tail], dim=1)

        prediction = model(
            **family.prepare_inputs(
                tokens=tokens,
                token_ids=sequence.ids,
                text_embeds=text_embeds,
                text_ids=text_ids,
                timestep=timestep,
                text_mask=text_mask,
            )
        )[0][:, start:stop]

        if do_guidance:
            negative = model(
                **family.prepare_inputs(
                    tokens=tokens,
                    token_ids=sequence.ids,
                    text_embeds=negative_embeds,
                    text_ids=negative_ids,
                    timestep=timestep,
                    text_mask=negative_mask,
                )
            )[0][:, start:stop]
            prediction = prediction + guidance * (prediction - negative)

        latents = scheduler.step(prediction, t, latents, return_dict=False)[0]

    return latents


__all__ = ["denoise"]

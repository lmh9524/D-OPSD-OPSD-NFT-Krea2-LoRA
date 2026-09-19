"""Krea 2 multi-image-reference LoRA — outfit-level virtual try-on.

Read top to bottom. Structurally this is ``flux2_klein_ref2img_lora.py`` with a different backbone;
the shared layer underneath is byte-for-byte the same ``fit()``, which is the point of the split.

    python experiments/krea2_ref2img_lora.py --dataset.root /path/to/Garments2Look/polyvore

``dataset.root`` holds ``train.jsonl``, written by ``tools/data/prepare_garments2look.py``::

    {"target": "looks-resized/women/113753680.jpg",
     "refs": ["images/women/dress/113753680_1.jpg", ...],
     "prompt": "A vibrant red sleeveless skater dress ..."}

**What is different from the FLUX.2 run, and why it matters more than the line count suggests.**

FLUX.2 klein ships multi-reference: its pipeline concatenates reference latents and offsets them on
the RoPE T axis, and the checkpoint was trained that way. A LoRA there adapts a capability that
already works.

Krea 2 does not. ``Krea2Pipeline`` is text-to-image only and emits ``(0, h, w)`` for every image
token — the T axis exists, carries 32 of the 128 head dimensions, and has been pinned at zero for the
model's entire training history. This file builds reference spans anyway, using the same layout
FLUX.2 and Qwen-Image-Edit use, but it is **teaching a new conditioning modality, not adapting one**.
Expect it to need more steps, more rank and more data than the FLUX.2 recipe, and expect early
samples to ignore the references entirely. ``docs/krea2-reference.md`` has the full argument.

Three mechanical differences follow from the architecture, all handled in ``step_fn``:

* the text stream is a **stack** of twelve tapped encoder layers, ``(B, L, 12, 2560)``, and carries an
  attention mask, because Krea 2 pads in the middle of its prompt template;
* position ids are **unbatched** ``(seq, 3)`` covering ``[text | references | target]`` together,
  with the target span **last** and sliced from the tail;
* there is no guidance embedder, so caption dropout is the only thing keeping the unconditional
  branch alive for real CFG at inference.
"""

from __future__ import annotations

import math
import pathlib

import torch
import tyro

from dflow import (
    CheckpointManager,
    Krea2Family,
    Krea2Ref2ImgLoRARecipe,
    Krea2TextEncoder,
    Krea2VAEEncoder,
    TrainLogger,
    apply_lora,
    build_dataloader,
    build_lr_scheduler,
    build_meta,
    build_optimizer,
    build_scheduler,
    destroy_distributed,
    fit,
    flow_mse,
    init_distributed,
    load_weights,
    materialize,
    matmul_precision,
    parameter_summary,
    prepare_model,
    provenance,
    reset_lora_parameters,
    resolve_path,
    seed_everything,
    shift_from_mu,
)
from dflow.tasks.ref2img import Ref2ImgDataset, token_count
from dflow.tasks.ref2img.conditioning import build_krea2_sequence, build_krea2_text_ids
from dflow.tasks.ref2img.sampling import denoise


def make_step_fn(cfg, *, vae, text, family, scheduler, generator, logger=None):
    """The training step, with everything it needs captured.

    Noise goes on the **target span only**. The reference garments are clean latents and read-only
    context: the model is being taught to use them, not to denoise them, and the loss covers only
    the target — the model image wearing the outfit.
    """
    dropout = cfg.dataset.caption_dropout
    ground = cfg.frozen.text.encoders[0].ground_references
    schedule_steps = cfg.flow_match.schedule_steps
    fixed_mu = None if cfg.flow_match.shift is None else math.log(cfg.flow_match.shift)
    seen_shapes: set[tuple[int, int]] = set()

    def step_fn(mesh, model, batch):
        prompts = list(batch["prompt"])
        if dropout > 0.0:
            # Krea 2 embeds no guidance scale and uses real CFG with a negative prompt, so the
            # unconditional branch has to keep being trained.
            keep = torch.rand(len(prompts), generator=generator, device=generator.device)
            prompts = [
                prompt if flag >= dropout else ""
                for prompt, flag in zip(prompts, keep.tolist(), strict=True)
            ]

        with torch.no_grad():
            target_latents = vae.encode(batch["target"], mode="sample", generator=generator)
            # One encode per slot: each garment keeps its own extent, so they cannot be stacked.
            reference_latents = [
                vae.encode(reference, mode="mode") for reference in batch["references"]
            ]
            # Image-grounded when configured: the reference pixels also enter through Qwen3-VL,
            # a pathway that is already pretrained, alongside the in-context VAE tokens. The
            # grounding resolution is jittered per call, so this cannot be cached.
            if ground:
                conditioning = text.encode_grounded(prompts[0], list(batch["references"]))
            else:
                conditioning = text.encode(prompts)

        sequence = build_krea2_sequence(
            target_latents=target_latents.float(),
            reference_latents=[latents.float() for latents in reference_latents],
            registration=cfg.dataset.reference_registration,
            t_scale=cfg.dataset.reference_t_scale,
        )
        # Sliced by ``target_offset``, never by a bare ``[:target_len]``. Krea 2's layout puts the
        # target **last** (`[refs | target]`), so the head slice returns *reference* tokens — the
        # loss then regresses toward `noise - references` and the LoRA learns to destroy the image.
        # It trains, the loss falls, and even plain text-to-image comes out fragmented.
        target_tokens = sequence.tokens[
            :, sequence.target_offset : sequence.target_offset + sequence.target_len
        ]
        batch_size = target_tokens.shape[0]

        noise = torch.randn(
            target_tokens.shape,
            generator=generator,
            device=target_tokens.device,
            dtype=target_tokens.dtype,
        )
        # From the target only. References are context and must not move the noise schedule.
        n_target = token_count(target_latents)
        mu = (
            fixed_mu
            if fixed_mu is not None
            else family.noise_shift_mu(image_tokens=n_target, inference_steps=schedule_steps)
        )

        if logger is not None:
            extent = tuple(target_latents.shape[-2:])
            if extent not in seen_shapes:
                seen_shapes.add(extent)
                n_reference = sum(token_count(r) for r in reference_latents)
                logger.info(
                    f"new shape {extent[0]}x{extent[1]} latent | {n_target} target + "
                    f"{n_reference} reference = {n_target + n_reference} tokens | "
                    f"mu {mu:.4f} (shift {shift_from_mu(mu):.2f})"
                )

        sigmas = scheduler.sample_timesteps(batch_size, mu=mu, generator=generator)
        noisy_target = scheduler.add_noise(target_tokens, noise, sigmas)
        velocity = scheduler.target(target_tokens, noise)

        prediction = model(
            **family.prepare_inputs(
                tokens=sequence.replace_target(noisy_target).tokens.to(model.dtype),
                token_ids=sequence.ids,
                text_embeds=conditioning.embeds.to(model.dtype),
                text_ids=build_krea2_text_ids(
                    conditioning.embeds.shape[1], device=conditioning.embeds.device
                ),
                timestep=sigmas.to(model.dtype),
                text_mask=conditioning.mask,
            )
        )[0]
        prediction = family.take_target_span(
            prediction, sequence.target_len, target_offset=sequence.target_offset
        )

        return flow_mse(prediction, velocity, weights=scheduler.weighting(sigmas))

    return step_fn


def make_preview_fn(cfg, *, dataset, vae, text, family, generator, device, logger):
    """Render one fixed dataset sample after each checkpoint, so progress is visible mid-run.

    Built here rather than inside ``fit()`` for the same reason ``step_fn`` is: the loop may not
    import ``models/`` or ``encoders/``, and a closure hands it everything without inverting that.

    Nearly free, which is what makes it worth doing every 2000 steps: the transformer, VAE and text
    encoder are already resident, so a preview costs a handful of forward passes and no extra
    weights. A separate process would have to load ~34 GB it has no room for while training holds
    the card.

    The sample is **fixed** (``preview_index``) so successive previews are comparable — the question
    is "is the reference span doing more than it was 2000 steps ago", which a different sample each
    time would hide. It renders with references and again without, side by side, because that
    contrast is the only thing that shows whether conditioning is working; a single image cannot.
    """
    if not cfg.checkpoint.preview:
        return None

    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    from PIL import Image

    sample = dataset[cfg.checkpoint.preview_index]
    directory = pathlib.Path(cfg.run_directory) / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    stepper = FlowMatchEulerDiscreteScheduler.from_pretrained(
        resolve_path(cfg.backbone.model), subfolder="scheduler"
    )

    # Encoded once, outside ``render``: the negative branch is the same empty prompt for every
    # preview and every checkpoint, and it is only built when CFG is actually on. Krea 2 embeds no
    # guidance scale, so a non-distilled backbone needs this to follow its conditioning at all.
    # Resolved from the family when the config leaves them unset: Raw and Turbo want different
    # numbers, and the wrong pair makes a working adapter look inert.
    defaults = family.sampling_defaults()
    preview_steps = (
        cfg.checkpoint.preview_steps
        if cfg.checkpoint.preview_steps is not None
        else int(defaults["steps"])
    )
    preview_guidance = (
        cfg.checkpoint.preview_guidance
        if cfg.checkpoint.preview_guidance is not None
        else float(defaults["guidance"])
    )
    negative = text.encode([""]) if preview_guidance > 0 else None

    def render(model, reference_latents, conditioning, grid_shape):
        sequence = build_krea2_sequence(
            target_latents=torch.randn(
                grid_shape, generator=generator, device=device, dtype=torch.float32
            ),
            reference_latents=reference_latents,
            registration=cfg.dataset.reference_registration,
            t_scale=cfg.dataset.reference_t_scale,
        )
        steps = preview_steps
        stepper.set_timesteps(
            sigmas=np.linspace(1.0, 1 / steps, steps).tolist(),
            device=device,
            mu=family.noise_shift_mu(image_tokens=sequence.target_len),
        )
        stepper.set_begin_index(0)
        latents = denoise(
            model=model,
            family=family,
            sequence=sequence,
            text_embeds=conditioning.embeds.to(model.dtype),
            text_ids=build_krea2_text_ids(
                conditioning.embeds.shape[1], device=conditioning.embeds.device
            ),
            text_mask=conditioning.mask,
            timesteps=stepper.timesteps,
            scheduler=stepper,
            num_train_timesteps=stepper.config.num_train_timesteps,
            guidance=preview_guidance,
            negative_embeds=negative.embeds.to(model.dtype) if negative is not None else None,
            negative_mask=negative.mask if negative is not None else None,
            negative_ids=None
            if negative is None
            else build_krea2_text_ids(negative.embeds.shape[1], device=negative.embeds.device),
        )
        return vae.decode_to_pil(latents, height=grid_shape[-2], width=grid_shape[-1])

    def preview_fn(model, step, path):
        model.eval()
        with torch.no_grad():
            references = [
                vae.encode(r.unsqueeze(0).to(device), mode="mode").float()
                for r in sample["references"]
            ]
            # Grounded when training is grounded. A preview conditioned differently from the run
            # it is reporting on measures the difference, not the run -- the same mistake that made
            # an earlier evaluation read as "the references are inert".
            ground = cfg.frozen.text.encoders[0].ground_references
            pixels = [r.unsqueeze(0) for r in sample["references"]]
            conditioning = (
                text.encode_grounded(sample["prompt"], pixels)
                if ground
                else text.encode([sample["prompt"]])
            )
            grid = vae.encode(sample["target"].unsqueeze(0).to(device), mode="mode")
            # Scale the render down to the preview budget, keeping the aspect ratio and the /16
            # grid. The references stay at their own size: what is being previewed is whether the
            # model follows them, not how sharp it is.
            rows, cols = grid.shape[-2], grid.shape[-1]
            budget = cfg.checkpoint.preview_max_area // (vae.spatial_compression**2)
            if rows * cols > budget:
                scale = (budget / (rows * cols)) ** 0.5
                rows, cols = max(1, int(rows * scale)), max(1, int(cols * scale))
                logger.info(f"preview capped to {rows}x{cols} latent ({rows * cols} tokens)")
            grid_shape = (grid.shape[0], grid.shape[1], rows, cols)
            with_refs = render(model, references, conditioning, grid_shape)
            # The un-grounded encode is the honest counterfactual for this branch: with no
            # references there is nothing to ground, and the question it answers is "what does the
            # prompt alone produce".
            without = render(model, [], text.encode([sample["prompt"]]), grid_shape)

        panel = Image.new("RGB", (with_refs.width * 2 + 8, with_refs.height), "white")
        panel.paste(with_refs, (0, 0))
        panel.paste(without, (with_refs.width + 8, 0))
        out = directory / f"step_{step:07d}.png"
        panel.save(out)
        logger.info(f"preview (with refs | without) -> {out}")

    return preview_fn


def run(cfg: Krea2Ref2ImgLoRARecipe) -> None:
    cfg.resolve_paths()

    mesh = init_distributed(cfg.runtime.distributed)
    matmul_precision(cfg.runtime.matmul_precision)
    generator = seed_everything(cfg.training.seed, mesh=mesh)

    logger = TrainLogger(cfg.logging, rank=mesh.rank)
    logger.log_config(cfg)
    logger.info(mesh.describe())
    if mesh.is_master:
        logger.info(provenance.describe(provenance.write(cfg.run_directory)))

    # ---- data (before the backbone, so a bad path fails before 9B of weights load) -------
    dataset = Ref2ImgDataset(cfg.dataset)
    dataloader = build_dataloader(
        dataset, cfg.data, dp_rank=mesh.dp_rank, dp_size=mesh.dp_size, seed=cfg.training.seed
    )
    logger.info(f"{len(dataset)} samples, {len(dataloader)} batches per epoch")

    # ---- backbone: meta -> LoRA -> parallelism -> materialise -> load -------------------
    family = Krea2Family(distilled="turbo" in cfg.backbone.model.family)
    model, architecture = build_meta(cfg.backbone.model, family)
    if cfg.backbone.lora.enabled:
        apply_lora(
            model,
            cfg.backbone.lora,
            targets=cfg.backbone.lora.target_modules or family.default_lora_targets(),
        )
    model = prepare_model(
        model,
        family.parallel_spec(model),
        activation_checkpoint=cfg.runtime.activation_checkpoint,
        compile_config=cfg.runtime.compile,
        fsdp=cfg.runtime.fsdp,
        mesh=mesh.fsdp_mesh,
    )
    materialize(model, mesh.device)
    load_weights(
        model, cfg.backbone.model, is_master=mesh.is_master, strict=not cfg.backbone.lora.enabled
    )
    if cfg.backbone.lora.enabled:
        reset_lora_parameters(model, adapter_name=cfg.backbone.lora.adapter_name)
    logger.info(parameter_summary(model).describe())

    # ---- frozen components -------------------------------------------------------------
    weights = resolve_path(cfg.backbone.model)
    vae = Krea2VAEEncoder.load(cfg.frozen.vae, path=weights, device=mesh.device)
    if vae is None:
        raise ValueError("this experiment encodes images online, so the VAE must be enabled")
    if vae.latent_channels != architecture["in_channels"]:
        raise ValueError(
            f"VAE packs to {vae.latent_channels} channels but the transformer expects "
            f"in_channels={architecture['in_channels']}"
        )
    # No cache path: Krea 2's conditioning is (L, 12, 2560) per prompt, ~63 MB in bf16 at
    # max_length 512 — five times klein's, so precomputing is not the easy win it is there.
    text = Krea2TextEncoder.load(
        cfg.frozen.text.encoders[0],
        path=weights,
        device=mesh.device,
        select_layers=cfg.frozen.text.encoders[0].out_layers,
    )
    text.check_compatibility(
        architecture["text_hidden_dim"], architecture["num_text_layers"]
    )
    logger.info(f"text encoder loaded: {text.num_layers} taps x {text.hidden_size}")

    # ---- schedule ----------------------------------------------------------------------
    scheduler = build_scheduler(cfg.flow_match, device=mesh.device)
    if cfg.flow_match.shift is None:
        logger.info("noise shift: dynamic, mu derived per sample from its target token count")
    else:
        logger.info(
            f"noise shift: fixed at {cfg.flow_match.shift:.4f} "
            f"(mu {math.log(cfg.flow_match.shift):.4f}) for every sample, whatever its size"
        )
    logger.info(
        f"target <= {cfg.dataset.target_max_area // vae.spatial_compression**2} tokens, "
        f"{cfg.dataset.num_references} x reference <= "
        f"{cfg.dataset.reference_max_area // vae.spatial_compression**2} tokens each"
    )
    logger.warning(
        "Krea 2 has no pretrained reference conditioning: the RoPE T axis is pinned at 0 in every "
        "checkpoint. This run is teaching that from scratch, not adapting it."
    )

    # ---- optimisation ------------------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg.optimizer, trainable)
    lr_scheduler = build_lr_scheduler(cfg.lr_scheduler, optimizer, total_steps=cfg.training.steps)
    # Everything a sampler must reproduce to be measuring this checkpoint rather than a mismatch.
    # Recorded rather than documented because documentation has not been enough: this run's own
    # evaluation once sampled one grounded reference against a checkpoint trained with nine.
    encoder = cfg.frozen.text.encoders[0]
    checkpoints = CheckpointManager(
        cfg.checkpoint,
        mesh=mesh,
        conditioning={
            "reference_registration": cfg.dataset.reference_registration,
            "reference_t_scale": cfg.dataset.reference_t_scale,
            "reference_fit_target": cfg.dataset.reference_fit_target,
            "reference_max_area": cfg.dataset.reference_max_area,
            "target_max_area": cfg.dataset.target_max_area,
            "num_references": cfg.dataset.num_references,
            "ground_references": encoder.ground_references,
            "max_grounded_references": encoder.max_grounded_references,
            "grounding_max_px": encoder.grounding_max_px,
            "fast_patch_embed": encoder.fast_patch_embed,
            "max_length": encoder.max_length,
            "family": cfg.backbone.model.family,
        },
    )

    step_fn = make_step_fn(
        cfg, vae=vae, text=text, family=family, scheduler=scheduler, generator=generator,
        logger=logger,
    )
    preview_fn = make_preview_fn(
        cfg, dataset=dataset, vae=vae, text=text, family=family,
        generator=generator, device=mesh.device, logger=logger,
    )
    if preview_fn is not None:
        shown = family.sampling_defaults()
        steps = cfg.checkpoint.preview_steps or int(shown["steps"])
        scale = (
            cfg.checkpoint.preview_guidance
            if cfg.checkpoint.preview_guidance is not None
            else float(shown["guidance"])
        )
        logger.info(
            f"preview on: sample {cfg.checkpoint.preview_index}, {steps} steps, cfg {scale}, "
            f"rendered with and without references at every checkpoint"
        )
    try:
        state = fit(
            mesh=mesh,
            step_fn=step_fn,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dataloader=dataloader,
            checkpoints=checkpoints,
            logger=logger,
            generator=generator,
            steps=cfg.training.steps,
            grad_accum_steps=cfg.training.grad_accum_steps,
            autocast_dtype=cfg.training.autocast_dtype,
            max_grad_norm=cfg.training.max_grad_norm,
            lora=cfg.backbone.lora.enabled,
            on_checkpoint=preview_fn,
        )
        logger.info(f"finished at step {state.global_step}")
    finally:
        logger.close()
        destroy_distributed()


def main() -> None:
    run(tyro.cli(Krea2Ref2ImgLoRARecipe))


if __name__ == "__main__":
    main()

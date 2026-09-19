"""Krea 2 ref2img **D-OPSD** — on-policy self-distillation for try-on, on the step-distilled Turbo.

Read top to bottom. This is ``krea2_ref2img_lora.py`` with a different ``step_fn`` (a few-step
rollout instead of a single noised-target forward) and a second, EMA, LoRA adapter for the teacher.
The shared layer underneath — ``fit()``, the data pipeline, the checkpoint manager — is unchanged;
the EMA teacher rides ``fit``'s existing ``ema=`` slot.

    python experiments/dopsd_krea2_ref2img_lora.py \
        --dataset.root /path/to/tryon_dataset \
        --distill.init-lora-from /mnt/shared/lihaoran/ssd/ckps/krea2-tryon-lora

**What D-OPSD does here.** Plain SFT on a step-distilled model teaches the concept but corrupts the
few-step schedule. D-OPSD instead lets one model play two roles that differ only in conditioning:

* the **student** (adapter ``default``) is conditioned on the deployed try-on context — prompt +
  garment references — and rolls out its own K-step trajectory from noise. This is exactly the
  inference pathway, so it stays fast.
* the **teacher** (adapter ``teacher``, an EMA of the student) additionally sees the ground-truth
  target look, and supervises the student's velocity at each visited state. Stop-gradient.

Loss is the mean over the K rollout states of ``|| x0_student - sg(x0_teacher) ||^2`` (x0-space, the
reference implementation's choice; ``--distill.loss-space velocity`` for the paper's form). States
are detached between steps, so there is no back-prop through sampling and the K terms share one
backward — ``fit`` does it.

**Two Krea-2 facts that shape this (both verified in the framework):**

* Krea 2's own text encoder *is* Qwen3-VL, so grounding the teacher on the target
  (``encode_grounded``) is a pretrained, clean pathway — no VLM-weight surgery, unlike the paper's
  Z-Image recipe.
* Base Krea 2 cannot use garment references (RoPE T axis pinned at 0), so the student must start
  from the phase-1 ref2img LoRA: ``--distill.init-lora-from`` initialises **both** adapters from it.
  Its rank/alpha must match ``--backbone.lora``.
"""

from __future__ import annotations

import pathlib

import torch
import tyro

from dflow import (
    CheckpointManager,
    Krea2DOPSDRecipe,
    Krea2Family,
    Krea2TextEncoder,
    Krea2VAEEncoder,
    LoRATeacherEMA,
    TrainLogger,
    apply_dual_lora,
    apply_time_shift,
    build_dataloader,
    build_lr_scheduler,
    build_meta,
    build_optimizer,
    copy_adapter,
    destroy_distributed,
    fit,
    fit_distill,
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


def _rollout_losses(cfg, model, batch, *, vae, text, family, generator, seen_shapes, logger):
    """Yield one loss per visited rollout state, already divided by K. Shared by both trainers.

    ``fit`` (single backward) sums the yields; ``fit_distill`` (memory-flat) backwards each one and
    frees its graph. The rollout is identical either way — only who consumes the generator differs.

    Per visited state (``batch_size == 1``): a student forward (grad) under the deployed conditioning
    and a teacher forward (no grad, stronger context) at the *same* state, then the x0 (or velocity)
    MSE between them. The state is detached between steps, so the K terms are independent subgraphs —
    which is exactly what lets ``fit_distill`` backward them one at a time.
    """
    d = cfg.distill
    K = d.num_steps
    ground = cfg.frozen.text.encoders[0].ground_references
    student = cfg.backbone.lora.adapter_name
    teacher = d.teacher_adapter_name
    registration = cfg.dataset.reference_registration
    t_scale = cfg.dataset.reference_t_scale

    prompt = batch["prompt"][0]
    references = list(batch["references"])
    target = batch["target"]

    with torch.no_grad():
        reference_latents = [vae.encode(r, mode="mode").float() for r in references]
        target_latents = vae.encode(target, mode="mode").float()
        # Student sees the garments; teacher additionally sees the target. Krea 2's encoder is
        # Qwen3-VL, so grounding is a pretrained pathway (no weight surgery).
        student_cond = (
            text.encode_grounded(prompt, references) if ground else text.encode([prompt])
        )
        teacher_prompt = f"{prompt} {d.edit_suffix}".strip() if d.edit_suffix else prompt
        if d.teacher_ground_target:
            # Target first so it is grounded even at max_grounded_references=1 (encode_grounded
            # only grounds the leading images[:max_grounded]).
            teacher_cond = text.encode_grounded(teacher_prompt, [target, *references])
        elif ground:
            teacher_cond = text.encode_grounded(teacher_prompt, references)
        else:
            teacher_cond = text.encode([teacher_prompt])

    # Krea 2 layout is [refs | target]; the target span is sliced by target_offset, never [:len].
    student_seq = build_krea2_sequence(
        target_latents=target_latents,
        reference_latents=reference_latents,
        registration=registration,
        t_scale=t_scale,
    )
    teacher_refs = reference_latents + ([target_latents] if d.teacher_ref_target else [])
    teacher_seq = build_krea2_sequence(
        target_latents=target_latents,
        reference_latents=teacher_refs,
        registration=registration,
        t_scale=t_scale,
    )
    target_len = student_seq.target_len
    span = student_seq.tokens[
        :, student_seq.target_offset : student_seq.target_offset + target_len
    ]
    batch_size = span.shape[0]

    # Roll out on the *inference* few-step schedule, not the training timestep density: the whole
    # point is to supervise the states the deployed sampler visits. mu is the K-step value.
    n_target = token_count(target_latents)
    mu = family.noise_shift_mu(image_tokens=n_target, inference_steps=K)
    sigmas = apply_time_shift(
        mu, torch.linspace(1.0, 1.0 / K, K, device=span.device, dtype=torch.float32)
    )

    if logger is not None:
        extent = tuple(target_latents.shape[-2:])
        if extent not in seen_shapes:
            seen_shapes.add(extent)
            n_ref = sum(token_count(r) for r in reference_latents)
            logger.info(
                f"new shape {extent[0]}x{extent[1]} latent | {n_target} target + {n_ref} "
                f"reference tokens | K={K} rollout | mu {mu:.4f} (shift {shift_from_mu(mu):.2f})"
            )

    student_ids = build_krea2_text_ids(student_cond.embeds.shape[1], device=span.device)
    teacher_ids = build_krea2_text_ids(teacher_cond.embeds.shape[1], device=span.device)

    state = torch.randn(span.shape, generator=generator, device=span.device, dtype=torch.float32)
    try:
        for k in range(K):
            # Detach the state each step: gradient flows only through this step's student forward,
            # never across the solver (truncated BPTT of length 1). requires_grad keeps activation
            # checkpointing happy when the only other grad-carrying tensors are the adapter weights.
            state = state.detach().requires_grad_(True)
            timestep = sigmas[k].expand(batch_size).to(model.dtype)
            sigma = sigmas[k]

            model.set_adapter(student)
            student_out = model(
                **family.prepare_inputs(
                    tokens=student_seq.replace_target(state).tokens.to(model.dtype),
                    token_ids=student_seq.ids,
                    text_embeds=student_cond.embeds.to(model.dtype),
                    text_ids=student_ids,
                    timestep=timestep,
                    text_mask=student_cond.mask,
                )
            )[0]
            v_student = family.take_target_span(
                student_out, target_len, target_offset=student_seq.target_offset
            )

            with torch.no_grad():
                teacher_kwargs = family.prepare_inputs(
                    tokens=teacher_seq.replace_target(state).tokens.to(model.dtype),
                    token_ids=teacher_seq.ids,
                    text_embeds=teacher_cond.embeds.to(model.dtype),
                    text_ids=teacher_ids,
                    timestep=timestep,
                    text_mask=teacher_cond.mask,
                )
                if d.teacher == "ema":
                    model.set_adapter(teacher)
                    teacher_out = model(**teacher_kwargs)[0]
                else:  # frozen_base: the pretrained model with the adapter disabled
                    with model.disable_adapter():
                        teacher_out = model(**teacher_kwargs)[0]
                v_teacher = family.take_target_span(
                    teacher_out, target_len, target_offset=teacher_seq.target_offset
                )

            # Restore the student adapter BEFORE the loss/yield. Under activation checkpointing the
            # student forward is recomputed during backward, and it must recompute with "default"
            # (the trainable student) active — not "teacher". fit_distill backwards each yield while
            # this generator is suspended here, so if "teacher" were still active the recompute would
            # apply the frozen teacher adapter and the student would receive no gradient (grad_norm
            # == 0, nothing learns). The frozen-base path already restores via disable_adapter()'s
            # context, but the EMA path leaves "teacher" active, so reset it explicitly.
            model.set_adapter(student)

            if d.loss_space == "x0":
                # x_t = (1 - sigma) x0 + sigma * noise  =>  x0 = x_t - sigma * v
                prediction = state - sigma * v_student
                objective = state - sigma * v_teacher
            else:
                prediction, objective = v_student, v_teacher
            yield flow_mse(prediction, objective.detach()) / K

            if k < K - 1:
                # Advance the trajectory by the student's own Euler step, detached (on-policy).
                delta = sigmas[k + 1] - sigmas[k]
                state = (state + delta * v_student.detach()).detach()
    finally:
        model.set_adapter(student)  # leave the student active for saving and the backward pass


def make_step_fn(cfg, *, vae, text, family, generator, logger=None):
    """``step_fn`` for ``fit()``: sum the rollout's per-step losses into one scalar.

    ``fit`` then does a single backward over the sum, which keeps all K forward graphs alive at once
    (O(K) activation memory). Use ``--distill.low-mem`` (``fit_distill`` + :func:`make_rollout_fn`)
    when that does not fit — same result, memory flat in K.
    """
    seen_shapes: set[tuple[int, int]] = set()

    def step_fn(mesh, model, batch):
        total = None
        for loss_term in _rollout_losses(
            cfg, model, batch, vae=vae, text=text, family=family,
            generator=generator, seen_shapes=seen_shapes, logger=logger,
        ):
            total = loss_term if total is None else total + loss_term
        return total

    return step_fn


def make_rollout_fn(cfg, *, vae, text, family, generator, logger=None):
    """``rollout_step_fn`` for ``fit_distill()``: the memory-flat path.

    Hands the generator straight to the loop, which backwards each per-step loss and frees its graph
    before the next forward — peak activation memory O(1) in K instead of O(K).
    """
    seen_shapes: set[tuple[int, int]] = set()

    def rollout_step_fn(mesh, model, batch):
        return _rollout_losses(
            cfg, model, batch, vae=vae, text=text, family=family,
            generator=generator, seen_shapes=seen_shapes, logger=logger,
        )

    return rollout_step_fn


def make_preview_fn(cfg, *, model, dataset, vae, text, family, generator, device, weights, logger):
    """Render a fixed sample each checkpoint: current D-OPSD student vs the ORIGINAL phase-1 LoRA.

    The phase-1 baseline is rendered ONCE here, at construction, while the ``default`` adapter still
    holds the phase-1 weights (before any optimizer step). It is fixed, so it is cached and reused;
    each checkpoint renders the current student from the SAME noise, references and conditioning and
    composites a 2-panel ``[student | phase-1]`` image, saved with the reference images and prompt
    under ``demo/<date>/<time>-<step>/`` for inspection.

    Built here, not in the loop, for the same reason the step closure is: the loop may not import
    models/ or encoders/, and a closure hands it everything. Errors inside the returned hook are
    swallowed by the trainer, so a preview can never end a run.
    """
    import datetime
    import json
    import pathlib
    from dataclasses import replace

    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    from PIL import Image, ImageDraw

    student = cfg.backbone.lora.adapter_name
    # The cached baseline is the `default` adapter BEFORE training: the phase-1 LoRA when
    # init_lora_from is set, or the bare t2i base when it is None (D-OPSD-from-base bootstrap).
    baseline_name = "ori: phase-1 LoRA" if cfg.distill.init_lora_from else "ori: base t2i (pre-train)"
    ground = cfg.frozen.text.encoders[0].ground_references
    registration = cfg.dataset.reference_registration
    t_scale = cfg.dataset.reference_t_scale
    defaults = family.sampling_defaults()
    steps, guidance = int(defaults["steps"]), float(defaults["guidance"])
    demo_root = pathlib.Path(__file__).resolve().parent.parent / "demo"

    stepper = FlowMatchEulerDiscreteScheduler.from_pretrained(weights, subfolder="scheduler")

    # Preview on a HELD-OUT try-on test set if given (real cases, never trained on); else one
    # training sample. Reuse the training dataset's geometry so the test cases are processed the
    # same way, only the root changes.
    if cfg.checkpoint.preview_dataset:
        preview_ds = Ref2ImgDataset(replace(cfg.dataset, root=cfg.checkpoint.preview_dataset))
        count = min(cfg.checkpoint.preview_count, len(preview_ds))
        samples = [preview_ds[i] for i in range(count)]
        logger.info(f"preview: {count} held-out test cases from {cfg.checkpoint.preview_dataset}")
    else:
        samples = [dataset[cfg.checkpoint.preview_index]]

    def _to_pil(t):
        arr = ((t.squeeze(0).clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)

    def _prepare(sample, seed):
        ref_pixels = [r.unsqueeze(0).to(device) for r in sample["references"]]
        with torch.no_grad():
            reference_latents = [vae.encode(r, mode="mode").float() for r in ref_pixels]
            grid = vae.encode(sample["target"].unsqueeze(0).to(device), mode="mode")
            conditioning = (
                text.encode_grounded(sample["prompt"], ref_pixels)
                if ground
                else text.encode([sample["prompt"]])
            )
        grid_shape = tuple(grid.shape)
        # Fixed noise per case so student and baseline start identically (a fair comparison).
        noise = torch.randn(
            grid_shape,
            generator=torch.Generator(device=device).manual_seed(seed),
            device=device,
            dtype=torch.float32,
        )
        text_ids = build_krea2_text_ids(conditioning.embeds.shape[1], device=device)

        def render(m):
            sequence = build_krea2_sequence(
                target_latents=noise.clone(),
                reference_latents=reference_latents,
                registration=registration,
                t_scale=t_scale,
            )
            stepper.set_timesteps(
                sigmas=np.linspace(1.0, 1.0 / steps, steps).tolist(),
                device=device,
                mu=family.noise_shift_mu(image_tokens=sequence.target_len),
            )
            stepper.set_begin_index(0)
            latents = denoise(
                model=m, family=family, sequence=sequence,
                text_embeds=conditioning.embeds.to(m.dtype), text_ids=text_ids,
                text_mask=conditioning.mask, timesteps=stepper.timesteps, scheduler=stepper,
                num_train_timesteps=stepper.config.num_train_timesteps,
                guidance=guidance, negative_embeds=None,
            )
            return vae.decode_to_pil(latents, height=grid_shape[-2], width=grid_shape[-1])

        return {
            "render": render,
            "refs": [_to_pil(r) for r in ref_pixels],
            "gt": _to_pil(sample["target"]),
            "prompt": sample["prompt"],
            "grid": grid_shape,
            "n_refs": len(ref_pixels),
        }

    # Prepare each case and cache its phase-1 baseline now, while "default" still holds phase-1.
    was_training = model.training
    model.eval()
    model.set_adapter(student)
    states = []
    with torch.no_grad():
        for index, sample in enumerate(samples):
            state = _prepare(sample, 20260911 + index)
            state["baseline"] = state["render"](model)
            states.append(state)
    model.train(was_training)
    logger.info(f"preview: cached {len(states)} phase-1 baseline(s) | {steps} steps, cfg {guidance}")

    def preview_fn(model, step, path):
        model.eval()
        model.set_adapter(student)
        now = datetime.datetime.now()
        base = demo_root / now.strftime("%Y-%m-%d") / f"{now.strftime('%H%M%S')}-{step:06d}"
        for index, state in enumerate(states):
            with torch.no_grad():
                current = state["render"](model)
            baseline = state["baseline"]
            gt = state["gt"]
            gap, bar = 8, 30
            # GT | ori | ours  (ori = original phase-1 LoRA; ours = current D-OPSD student).
            cols = [("GT (target)", gt), (baseline_name, baseline), (f"ours: D-OPSD step {step}", current)]
            h = bar + max(im.height for _, im in cols)
            w = sum(im.width for _, im in cols) + gap * (len(cols) - 1)
            panel = Image.new("RGB", (w, h), "white")
            draw = ImageDraw.Draw(panel)
            x = 0
            for name, im in cols:
                draw.text((x + 6, 9), name, fill="black")
                panel.paste(im, (x, bar))
                x += im.width + gap

            out = base / f"case_{index}"
            (out / "refs").mkdir(parents=True, exist_ok=True)
            panel.save(out / "panel.png")
            for j, image in enumerate(state["refs"], start=1):
                image.save(out / "refs" / f"R{j}.png")
            (out / "prompt.txt").write_text(state["prompt"] + "\n", encoding="utf-8")
            (out / "meta.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "case": index,
                        "prompt": state["prompt"],
                        "n_refs": state["n_refs"],
                        "steps": steps,
                        "guidance": guidance,
                        "family": cfg.backbone.model.family,
                        "latent_grid": list(state["grid"]),
                        "panels": ["GT (target)", baseline_name, "ours: D-OPSD (current)"],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        logger.info(f"preview: {len(states)} case(s) (student | phase-1) step {step} -> {base}")

    return preview_fn


def _init_student_from_phase1(model, path: str, *, adapter_name: str, logger) -> int:
    """Load a phase-1 diffusers-format ref2img LoRA into an already-injected adapter.

    Forgiving name+shape match (mirroring the D-OPSD reference's ``load_matching_state_dict``): for
    each ``lora_*`` parameter of ``adapter_name`` in the model, find the source tensor for the same
    module and A/B factor and copy it. Runs *after* ``to_empty`` and ``reset_lora_parameters`` so the
    adapter tensors are real. Requires a **local** directory — download HF repos first (see
    a local checkpoint directory). Raises if nothing matches, so a key-format drift is loud, not a
    silently un-initialised student.
    """
    from safetensors.torch import load_file

    root = pathlib.Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"--distill.init-lora-from={path} is not a local path. Download or copy the phase-1 "
            f"LoRA locally, then point this option at that directory."
        )
    files = [root] if root.is_file() else sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors weights found under {root}")

    source: dict[str, torch.Tensor] = {}
    for f in files:
        source.update(load_file(str(f)))
    # Normalise source keys to "<module>.lora_A" / "<module>.lora_B": diffusers exports name them
    # "transformer.<module>.lora_A.weight" (no adapter segment).
    normalised = {
        k.removeprefix("transformer.").removesuffix(".weight"): v
        for k, v in source.items()
        if "lora_" in k
    }

    params = dict(model.named_parameters())
    matched = total = 0
    marker = f".{adapter_name}."
    with torch.no_grad():
        for name, param in params.items():
            if "lora_" not in name or marker not in name:
                continue
            total += 1
            key = name.replace(marker, ".").removesuffix(".weight")  # -> "<module>.lora_A"
            tensor = normalised.get(key)
            if tensor is not None and tuple(tensor.shape) == tuple(param.shape):
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
                matched += 1
    logger.info(f"phase-1 LoRA: matched {matched}/{total} '{adapter_name}' tensors from {root}")
    if matched == 0:
        raise ValueError(
            f"phase-1 LoRA at {root} matched no adapter tensors. The saved key format may differ "
            f"from the expected 'transformer.<module>.lora_A.weight'; inspect the .safetensors keys "
            f"and adjust _init_student_from_phase1's normalisation."
        )
    return matched


def run(cfg: Krea2DOPSDRecipe) -> None:
    cfg.resolve_paths()

    mesh = init_distributed(cfg.runtime.distributed)
    matmul_precision(cfg.runtime.matmul_precision)
    generator = seed_everything(cfg.training.seed, mesh=mesh)

    logger = TrainLogger(cfg.logging, rank=mesh.rank)
    logger.log_config(cfg)
    logger.info(mesh.describe())
    if mesh.is_master:
        logger.info(provenance.describe(provenance.write(cfg.run_directory)))

    # ---- data (before the backbone, so a bad path fails before weights load) -------------
    dataset = Ref2ImgDataset(cfg.dataset)
    dataloader = build_dataloader(
        dataset, cfg.data, dp_rank=mesh.dp_rank, dp_size=mesh.dp_size, seed=cfg.training.seed
    )
    logger.info(f"{len(dataset)} samples, {len(dataloader)} batches per epoch")

    # ---- backbone: meta -> dual LoRA -> parallelism -> materialise -> load -> init adapters
    if not cfg.backbone.lora.enabled:
        raise ValueError("D-OPSD trains LoRA adapters; set --backbone.lora.enabled")
    family = Krea2Family(distilled="turbo" in cfg.backbone.model.family)
    model, architecture = build_meta(cfg.backbone.model, family)
    apply_dual_lora(
        model,
        cfg.backbone.lora,
        targets=cfg.backbone.lora.target_modules or family.default_lora_targets(),
        teacher_name=cfg.distill.teacher_adapter_name,
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
    # LoRA run: base weights load non-strict, adapter tensors are reset then filled below.
    load_weights(model, cfg.backbone.model, is_master=mesh.is_master, strict=False)
    student_name = cfg.backbone.lora.adapter_name
    teacher_name = cfg.distill.teacher_adapter_name
    if reset_lora_parameters(model, adapter_name=student_name) == 0:
        raise ValueError(f"reset_lora_parameters touched no layers for adapter '{student_name}'")
    if reset_lora_parameters(model, adapter_name=teacher_name) == 0:
        raise ValueError(f"reset_lora_parameters touched no layers for adapter '{teacher_name}'")

    if cfg.distill.init_lora_from:
        _init_student_from_phase1(
            model, cfg.distill.init_lora_from, adapter_name=student_name, logger=logger
        )
    else:
        logger.warning(
            "distill.init_lora_from is None: the student starts from zero-output init and cannot "
            "use garment references yet. For Krea 2 try-on, set it to the phase-1 ref2img LoRA."
        )
    copied = copy_adapter(model, src=student_name, dst=teacher_name)
    logger.info(f"teacher adapter initialised from student ({copied} tensors)")
    logger.info(parameter_summary(model).describe())

    # ---- frozen components -------------------------------------------------------------
    weights = resolve_path(cfg.backbone.model)
    vae = Krea2VAEEncoder.load(cfg.frozen.vae, path=weights, device=mesh.device)
    if vae is None:
        raise ValueError("D-OPSD encodes images online, so the VAE must be enabled")
    if vae.latent_channels != architecture["in_channels"]:
        raise ValueError(
            f"VAE packs to {vae.latent_channels} channels but the transformer expects "
            f"in_channels={architecture['in_channels']}"
        )
    text = Krea2TextEncoder.load(
        cfg.frozen.text.encoders[0],
        path=weights,
        device=mesh.device,
        select_layers=cfg.frozen.text.encoders[0].out_layers,
    )
    text.check_compatibility(architecture["text_hidden_dim"], architecture["num_text_layers"])
    logger.info(f"text encoder loaded: {text.num_layers} taps x {text.hidden_size} (Qwen3-VL)")

    logger.info(
        f"D-OPSD: K={cfg.distill.num_steps} rollout | teacher={cfg.distill.teacher} "
        f"(ground_target={cfg.distill.teacher_ground_target}, ref_target={cfg.distill.teacher_ref_target}) "
        f"| loss={cfg.distill.loss_space} | ema_decay={cfg.distill.ema_decay}"
    )
    logger.warning(
        "Krea 2 has no pretrained reference conditioning; the teacher can only supervise usefully "
        "if it can already use references. Start from a phase-1 ref2img LoRA (init_lora_from) and/or "
        "rely on Qwen3-VL grounding of the target."
    )

    # ---- optimisation ------------------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg.optimizer, trainable)
    lr_scheduler = build_lr_scheduler(cfg.lr_scheduler, optimizer, total_steps=cfg.training.steps)

    # The EMA teacher rides fit()'s ema slot: fit calls ema.step(model, ...) after each optimizer
    # step, and the checkpoint manager round-trips ema.state_dict() through the 'ema' slot — which is
    # how the frozen teacher adapter survives a resume (a LoRA checkpoint would not save it).
    ema = (
        LoRATeacherEMA(
            model,
            src_adapter=student_name,
            dst_adapter=teacher_name,
            decay=cfg.distill.ema_decay,
            update_every=cfg.distill.ema_update_every,
        )
        if cfg.distill.teacher == "ema"
        else None
    )

    encoder = cfg.frozen.text.encoders[0]
    checkpoints = CheckpointManager(
        cfg.checkpoint,
        mesh=mesh,
        conditioning={
            "method": "dopsd",
            "num_steps": cfg.distill.num_steps,
            "teacher": cfg.distill.teacher,
            "teacher_ground_target": cfg.distill.teacher_ground_target,
            "teacher_ref_target": cfg.distill.teacher_ref_target,
            "loss_space": cfg.distill.loss_space,
            "init_lora_from": cfg.distill.init_lora_from,
            "reference_registration": cfg.dataset.reference_registration,
            "reference_t_scale": cfg.dataset.reference_t_scale,
            "num_references": cfg.dataset.num_references,
            "ground_references": encoder.ground_references,
            "max_grounded_references": encoder.max_grounded_references,
            "max_length": encoder.max_length,
            "family": cfg.backbone.model.family,
        },
    )

    # Preview: render the fixed sample every checkpoint as [current student | phase-1 baseline].
    # Built before the loop, while the "default" adapter still holds phase-1, so the baseline can be
    # rendered once and cached. Enable with --checkpoint.preview.
    preview_fn = (
        make_preview_fn(
            cfg, model=model, dataset=dataset, vae=vae, text=text, family=family,
            generator=generator, device=mesh.device, weights=weights, logger=logger,
        )
        if cfg.checkpoint.preview
        else None
    )

    # fit() and fit_distill() take the same keyword args except for the step closure, so share them.
    common = dict(
        mesh=mesh,
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
        ema=ema,
        lora=True,
        on_checkpoint=preview_fn,
    )
    closure_kwargs = dict(
        cfg=cfg, vae=vae, text=text, family=family, generator=generator, logger=logger
    )
    try:
        if cfg.distill.low_mem:
            logger.info("trainer: fit_distill (memory-flat; per-step backward, O(1) in K)")
            state = fit_distill(rollout_step_fn=make_rollout_fn(**closure_kwargs), **common)
        else:
            logger.info("trainer: fit (single backward, O(K) memory; --distill.low-mem if OOM)")
            state = fit(step_fn=make_step_fn(**closure_kwargs), **common)
        logger.info(f"finished at step {state.global_step}")
    finally:
        logger.close()
        destroy_distributed()


def main() -> None:
    run(tyro.cli(Krea2DOPSDRecipe))


if __name__ == "__main__":
    main()

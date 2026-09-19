"""Krea 2 ref2img **OPSD-NFT** — DiffusionNFT reward post-training for try-on, on Turbo.

Read top to bottom. This is the RL sibling of ``dopsd_krea2_ref2img_lora.py``: the same dual-adapter
Krea 2 try-on setup (a trainable ``default`` student plus a frozen second adapter), the same few-step
rollout and the same phase-1 initialisation — but instead of self-distilling against a teacher, it
post-trains with a **reward**, using the fused DiffusionNFT objective.

    python experiments/krea2_ref2img_opsd_nft.py \
        --dataset.root /path/to/tryon_dataset \
        --init-lora-from /mnt/shared/lihaoran/ssd/ckps/krea2-tryon-lora

**What OPSD-NFT does here.** DiffusionNFT is a forward-process, likelihood-free RL method — no PPO
ratio, no SDE window, no log-probability. Per step:

1. the frozen ``old`` adapter (the rollout policy) denoises a *group* of clean try-on images from
   noise, on Turbo's few-step schedule — exactly the inference pathway;
2. a **reference-fidelity reward** scores each clean image against the case's ground-truth target
   (CLIP-I cosine), and the group-relative advantage becomes an *optimality probability* ``r``;
3. for each trajectory, at a freshly sampled ``t``, the re-noised clean latent is pushed by the
   trainable ``default`` policy toward the positive branch (weighted ``r``) and away from the
   negative branch (``1 - r``) in x0-space, with a small KL anchor to the frozen base — the maths in
   ``dflow/rl/objective.py``.

After each optimizer step the ``old`` policy is refreshed from ``default`` (a hard copy at
``nft.old_policy_decay == 0``, fully on-policy), which is what ``LoRATeacherEMA(src="default",
dst="old")`` in ``fit()``'s ``ema`` slot does.

**Two Krea-2 facts that shape this (both verified in the framework):**

* Base Krea 2 cannot use garment references (RoPE T axis pinned at 0), so both the student and the
  ``old`` rollout policy must start from the phase-1 ref2img LoRA: ``--init-lora-from`` initialises
  **both** adapters from it. Its rank/alpha must match ``--backbone.lora``.
* The reference-fidelity reward is the piece ref2img RL was previously held back for: it needs the
  ground-truth image, which the reward reads from ``metadata`` — see :func:`make_closures` for the
  contract.

**GPU-only.** The rollout, the three-forward update and the CLIP reward all need real weights; a CPU
run is out of scope. See ``scripts/train_opsd_nft_krea2.sh``.
"""

from __future__ import annotations

import pathlib

import torch
import tyro

from dflow import (
    CheckpointManager,
    DiffusionNFTConfig,
    Krea2Family,
    Krea2OPSDNFTRecipe,
    Krea2TextEncoder,
    Krea2VAEEncoder,
    LoRATeacherEMA,
    NFTRolloutGroup,
    TrainLogger,
    apply_dual_lora,
    apply_time_shift,
    build_dataloader,
    build_lr_scheduler,
    build_meta,
    build_optimizer,
    build_reward,
    copy_adapter,
    destroy_distributed,
    diffusion_nft_loss,
    fit_opsd_nft,
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


def _init_student_from_phase1(model, path: str, *, adapter_name: str, logger) -> int:
    """Load a phase-1 diffusers-format ref2img LoRA into an already-injected adapter.

    A forgiving name+shape match: for each ``lora_*`` parameter of ``adapter_name`` in the model,
    find the source tensor for the same module and A/B factor and copy it. Runs *after* ``to_empty``
    and ``reset_lora_parameters`` so the adapter tensors are real. Requires a **local** directory.
    Raises if nothing matches, so a key-format drift is loud, not a silently un-initialised student.

    A copy of the D-OPSD experiment's helper, deliberately — ``experiments/`` may import only
    ``dflow`` or ``dflow.tasks.<name>``, and duplicating a ~30-line experiment helper is the feature
    the design intends (a shared abstraction would put task glue in the shared layer).
    """
    from safetensors.torch import load_file

    root = pathlib.Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"--init-lora-from={path} is not a local path. Download or copy the phase-1 LoRA "
            f"locally, then point this option at that directory."
        )
    files = [root] if root.is_file() else sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors weights found under {root}")

    source: dict[str, torch.Tensor] = {}
    for f in files:
        source.update(load_file(str(f)))
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
            key = name.replace(marker, ".").removesuffix(".weight")
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


def _to_uint8(pixels: torch.Tensor) -> torch.Tensor:
    """``(..., 3, H, W)`` in [-1, 1] -> uint8 in [0, 255], the reward's pixel contract."""
    return ((pixels.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)


def _decode_group(vae, tokens: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """``(G, target_len, C)`` packed target tokens -> ``(G, 3, H, W)`` uint8 pixels.

    Mirrors ``Krea2VAEEncoder.decode_to_pil``'s arithmetic but for a whole group at once (that helper
    returns one PIL image). ``height``/``width`` are the latent-grid extents the tokens came from.
    Reaching into ``vae.model`` and ``vae.denormalise`` is fine here — this is L6.
    """
    group, sequence, channels = tokens.shape
    grid = tokens.permute(0, 2, 1).reshape(group, channels, height, width)
    latents = vae.denormalise(grid.to(vae.model.dtype))
    image = vae.model.decode(latents, return_dict=False)[0][:, :, 0]
    return _to_uint8(image.float())


def make_closures(cfg, *, vae, text, family, generator, logger):
    """Build ``rollout_fn``, ``reward_fn`` and ``nft_step_fn``, sharing per-prompt conditioning.

    The shared ``context`` dict is load-bearing, exactly as in ``flux2_klein_t2i_flowgrpo.py``: the
    rollout stores each prompt's sequence geometry, text conditioning and latent grid, and the update
    reads them back rather than re-deriving them — re-encoding would be a second chance to differ
    from what the rollout sampled.

    ## The reward's metadata contract

    ``reward_fn`` decodes each rollout's clean latents to uint8 pixels and scores them with the
    composite reward. The reference-fidelity component compares against the case's **ground-truth
    target**, which cannot be known from the generated pixels, so it is placed per sample in
    ``metadata`` under ``reward.reference_fidelity.reference_key`` as a uint8 ``(3, H, W)`` image —
    the batch's ``target`` converted to the pixel contract. A sample missing it raises rather than
    scoring zero (a silent zero is indistinguishable from a dissimilar image).
    """
    nft: DiffusionNFTConfig = cfg.nft
    student = cfg.backbone.lora.adapter_name
    old = cfg.old_adapter_name
    ground = cfg.frozen.text.encoders[0].ground_references
    registration = cfg.dataset.reference_registration
    t_scale = cfg.dataset.reference_t_scale
    reference_key = nft.reward.reference_fidelity.reference_key
    group_size = cfg.group.size
    K = nft.num_train_timesteps

    reward = build_reward(nft.reward, device=vae.device)
    logger.info(f"reward: {reward.name}")

    context: dict[int, dict] = {}
    logged = False

    def _encode(prompt: str, references: list[torch.Tensor]):
        # One row per group member: the sequence's leading dim is the group, so a group rolls out in
        # one batched forward (Krea 2's unbatched position_ids are shared across the batch).
        if ground:
            return text.encode_grounded(prompt, references)
        return text.encode([prompt])

    # ------------------------------------------------------------------------------ rollout

    @torch.no_grad()
    def rollout_fn(mesh, model, batch):
        nonlocal logged
        context.clear()
        groups: list[NFTRolloutGroup] = []

        for index in range(len(batch["prompt"])):
            prompt = batch["prompt"][index]
            references = [r[index : index + 1].to(mesh.device) for r in batch["references"]]
            target = batch["target"][index : index + 1].to(mesh.device)

            reference_latents = [vae.encode(r, mode="mode").float() for r in references]
            target_latents = vae.encode(target, mode="mode").float()
            grid_h, grid_w = target_latents.shape[-2:]

            conditioning = _encode(prompt, references)
            text_ids = build_krea2_text_ids(conditioning.embeds.shape[1], device=mesh.device)

            # Build the [refs | target] sequence once; the target span is replaced each step.
            sequence = build_krea2_sequence(
                target_latents=target_latents,
                reference_latents=reference_latents,
                registration=registration,
                t_scale=t_scale,
            )
            target_len = sequence.target_len
            n_target = token_count(target_latents)
            mu = family.noise_shift_mu(image_tokens=n_target, inference_steps=K)
            sigmas = apply_time_shift(
                mu, torch.linspace(1.0, 1.0 / K, K, device=mesh.device, dtype=torch.float32)
            )

            # Expand conditioning to the group and roll out G trajectories from independent noise.
            embeds = conditioning.embeds.expand(group_size, -1, -1, -1).to(model.dtype)
            mask = None if conditioning.mask is None else conditioning.mask.expand(group_size, -1)
            state = torch.randn(
                group_size, target_len, sequence.tokens.shape[-1],
                generator=generator, device=mesh.device, dtype=torch.float32,
            )

            model.set_adapter(old)
            for k in range(K):
                timestep = sigmas[k].expand(group_size).to(model.dtype)
                # Refs are shared across the group (the sequence is batch-1) but the target is
                # per-member (`state` is (G, target_len, C)). replace_target can't cat a batch-1 ref
                # span with a batch-G target, so build the batched sequence explicitly: expand the
                # ref spans to G and concatenate the per-member noised targets.
                off = sequence.target_offset
                pre = sequence.tokens[:, :off].expand(group_size, -1, -1)
                post = sequence.tokens[:, off + target_len :].expand(group_size, -1, -1)
                tokens = torch.cat([pre, state, post], dim=1).to(model.dtype)
                velocity = family.take_target_span(
                    model(
                        **family.prepare_inputs(
                            tokens=tokens,
                            token_ids=sequence.ids,
                            text_embeds=embeds,
                            text_ids=text_ids,
                            timestep=timestep,
                            text_mask=mask,
                        )
                    )[0],
                    target_len,
                    target_offset=sequence.target_offset,
                ).float()
                if k < K - 1:
                    state = state + (sigmas[k + 1] - sigmas[k]) * velocity
                else:
                    # x_t + (sigma_next - sigma) v with sigma_next = 0 lands on the clean latent.
                    state = state + (0.0 - sigmas[k]) * velocity

            context[index] = {
                "sequence": sequence,
                "embeds": conditioning.embeds,
                "mask": conditioning.mask,
                "text_ids": text_ids,
                "target_len": target_len,
                "grid": (int(grid_h), int(grid_w)),
                "target_uint8": _to_uint8(target.squeeze(0)).cpu(),
                "prompt": prompt,
                "n_target": n_target,
                "mu": mu,
                "sigmas": sigmas,
            }
            groups.append(
                NFTRolloutGroup(prompt_index=index, final_latents=state.detach())
            )

        if not logged and groups:
            logged = True
            entry = context[groups[0].prompt_index]
            logger.info(
                f"rollout: {entry['grid'][0]}x{entry['grid'][1]} latent | {entry['n_target']} "
                f"target tokens | group {group_size} | K={K} steps | "
                f"mu {entry['mu']:.4f} (shift {shift_from_mu(entry['mu']):.2f})"
            )
        return groups

    # ------------------------------------------------------------------------------- reward

    @torch.no_grad()
    def reward_fn(groups, batch):
        rows = []
        for rolled in groups:
            entry = context[rolled.prompt_index]
            grid_h, grid_w = entry["grid"]
            pixels = _decode_group(vae, rolled.final_latents, height=grid_h, width=grid_w)
            metadata = [{reference_key: entry["target_uint8"]}] * rolled.group_size
            breakdown = reward.score(pixels, [entry["prompt"]] * rolled.group_size, metadata)
            rows.append(breakdown.total)
        return torch.stack(rows)

    # ----------------------------------------------------------------------------- nft step

    def nft_step_fn(mesh, model, sample, reward_prob):
        entry = context[sample.prompt_index]
        sequence = entry["sequence"]
        target_len = entry["target_len"]
        embeds = entry["embeds"].to(model.dtype)
        mask = entry["mask"]
        text_ids = entry["text_ids"]

        x0 = sample.clean_latent.unsqueeze(0).float()  # (1, target_len, C)
        noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=torch.float32)
        # Train on the mu-shifted timesteps the few-step rollout actually visited, not an unshifted
        # uniform draw. DiffusionNFT re-noises the clean endpoint at the schedule the deployed sampler
        # uses; for a step-distilled backbone, training off that grid works against the few-step
        # distribution this method exists to preserve, and verl-omni likewise sub-samples this
        # discrete grid. t == sigma; pick one of the K rollout sigmas (with replacement across the
        # K micro-steps a trajectory takes — close to verl's without-replacement permutation).
        sigmas = entry["sigmas"]
        t = sigmas[torch.randint(len(sigmas), (), generator=generator, device=sigmas.device)]
        xt = (1.0 - t) * x0 + t * noise
        timestep = t.reshape(1).to(model.dtype)
        t_expanded = t.reshape(1, 1, 1)

        def forward(tokens):
            return family.take_target_span(
                model(
                    **family.prepare_inputs(
                        tokens=tokens.to(model.dtype),
                        token_ids=sequence.ids,
                        text_embeds=embeds,
                        text_ids=text_ids,
                        timestep=timestep,
                        text_mask=mask,
                    )
                )[0],
                target_len,
                target_offset=sequence.target_offset,
            )

        tokens = sequence.replace_target(xt).tokens

        # Trainable policy (grad).
        model.set_adapter(student)
        forward_prediction = forward(tokens).float()

        # Frozen old (rollout) policy — the same adapter that produced x0.
        with torch.no_grad():
            model.set_adapter(old)
            old_prediction = forward(tokens).float()
            # Reference: the base model with LoRA disabled. Krea2's diffusers model exposes
            # disable_adapters()/enable_adapters() (methods), not peft's disable_adapter() context
            # manager, so toggle explicitly and re-enable before restoring the student adapter.
            model.disable_adapters()
            ref_prediction = forward(tokens).float()
            model.enable_adapters()
        # Restore the trainable adapter before the loss/backward: under activation checkpointing the
        # grad-carrying forward is recomputed in backward and must recompute with "default" active,
        # or the student receives no gradient (the D-OPSD experiment documents this trap).
        model.set_adapter(student)

        return diffusion_nft_loss(
            forward_prediction=forward_prediction,
            old_prediction=old_prediction,
            ref_forward_prediction=ref_prediction,
            x0=x0,
            xt=xt,
            t_expanded=t_expanded,
            reward_prob=reward_prob.reshape(1),
            config=nft,
        )

    return rollout_fn, reward_fn, nft_step_fn


def run(cfg: Krea2OPSDNFTRecipe) -> None:
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
    if len(dataset) < mesh.dp_size:
        raise ValueError(
            f"{len(dataset)} cases for {mesh.dp_size} data-parallel ranks. Every rank needs at "
            f"least one prompt to roll out, or it contributes no gradient while still joining every "
            f"collective."
        )
    logger.info(f"{len(dataset)} cases, {len(dataloader)} batches per epoch")

    # ---- backbone: meta -> dual LoRA (student + old) -> parallelism -> load -> init ------
    if not cfg.backbone.lora.enabled:
        raise ValueError("OPSD-NFT trains LoRA adapters; set --backbone.lora.enabled")
    family = Krea2Family(distilled="turbo" in cfg.backbone.model.family)
    model, architecture = build_meta(cfg.backbone.model, family)
    apply_dual_lora(
        model,
        cfg.backbone.lora,
        targets=cfg.backbone.lora.target_modules or family.default_lora_targets(),
        teacher_name=cfg.old_adapter_name,
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
    load_weights(model, cfg.backbone.model, is_master=mesh.is_master, strict=False)
    student_name = cfg.backbone.lora.adapter_name
    old_name = cfg.old_adapter_name
    if reset_lora_parameters(model, adapter_name=student_name) == 0:
        raise ValueError(f"reset_lora_parameters touched no layers for adapter '{student_name}'")
    if reset_lora_parameters(model, adapter_name=old_name) == 0:
        raise ValueError(f"reset_lora_parameters touched no layers for adapter '{old_name}'")

    if cfg.init_lora_from:
        _init_student_from_phase1(
            model, cfg.init_lora_from, adapter_name=student_name, logger=logger
        )
    else:
        logger.warning(
            "init_lora_from is None: the student starts from zero-output init and cannot use "
            "garment references yet, so the rollout has nothing worth scoring. For Krea 2 try-on, "
            "set it to the phase-1 ref2img LoRA."
        )
    copied = copy_adapter(model, src=student_name, dst=old_name)
    logger.info(f"old (rollout) adapter initialised from student ({copied} tensors)")
    logger.info(parameter_summary(model).describe())

    # ---- frozen components -------------------------------------------------------------
    weights = resolve_path(cfg.backbone.model)
    vae = Krea2VAEEncoder.load(cfg.frozen.vae, path=weights, device=mesh.device)
    if vae is None:
        raise ValueError("OPSD-NFT encodes and decodes online, so the VAE must be enabled")
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
        f"OPSD-NFT: group {cfg.group.size} | K={cfg.nft.num_train_timesteps} rollout | "
        f"mix_beta {cfg.nft.mix_beta} | ref_kl {cfg.nft.ref_kl_coef} | "
        f"old_policy_decay {cfg.nft.old_policy_decay} (0 = hard copy, on-policy)"
    )
    logger.warning(
        "Krea 2 has no pretrained reference conditioning; the reward can only rise if the rollout "
        "already uses references. Start from a phase-1 ref2img LoRA (init_lora_from)."
    )

    # ---- optimisation ------------------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg.optimizer, trainable)
    lr_scheduler = build_lr_scheduler(cfg.lr_scheduler, optimizer, total_steps=cfg.training.steps)

    # The old-policy refresher rides fit_opsd_nft()'s ema slot: it EMAs (or hard-copies at decay 0)
    # the trainable "default" adapter into the frozen "old" one after each optimizer step, and the
    # checkpoint manager round-trips it through the 'ema' slot — which is how the frozen old adapter
    # survives a resume (a LoRA checkpoint would not save it).
    ema = LoRATeacherEMA(
        model,
        src_adapter=student_name,
        dst_adapter=old_name,
        decay=cfg.nft.old_policy_decay,
        update_every=cfg.nft.old_policy_update_interval,
    )

    encoder = cfg.frozen.text.encoders[0]
    checkpoints = CheckpointManager(
        cfg.checkpoint,
        mesh=mesh,
        conditioning={
            "method": "opsd_nft",
            "num_train_timesteps": cfg.nft.num_train_timesteps,
            "mix_beta": cfg.nft.mix_beta,
            "ref_kl_coef": cfg.nft.ref_kl_coef,
            "adv_clip_max": cfg.nft.adv_clip_max,
            "old_policy_decay": cfg.nft.old_policy_decay,
            "group_size": cfg.group.size,
            "reward": reward_name(cfg.nft),
            "init_lora_from": cfg.init_lora_from,
            "reference_registration": cfg.dataset.reference_registration,
            "reference_t_scale": cfg.dataset.reference_t_scale,
            "num_references": cfg.dataset.num_references,
            "ground_references": encoder.ground_references,
            "max_grounded_references": encoder.max_grounded_references,
            "max_length": encoder.max_length,
            "family": cfg.backbone.model.family,
        },
    )

    rollout_fn, reward_fn, nft_step_fn = make_closures(
        cfg, vae=vae, text=text, family=family, generator=generator, logger=logger
    )

    try:
        state = fit_opsd_nft(
            mesh=mesh,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dataloader=dataloader,
            checkpoints=checkpoints,
            logger=logger,
            generator=generator,
            rollout_fn=rollout_fn,
            reward_fn=reward_fn,
            nft_step_fn=nft_step_fn,
            ema=ema,
            steps=cfg.training.steps,
            group=cfg.group,
            nft=cfg.nft,
            grad_accum_steps=cfg.training.grad_accum_steps,
            autocast_dtype=cfg.training.autocast_dtype,
            max_grad_norm=cfg.training.max_grad_norm,
            lora=True,
        )
        logger.info(f"finished at step {state.global_step}")
    finally:
        logger.close()
        destroy_distributed()


def reward_name(nft: DiffusionNFTConfig) -> str:
    """A short description of the enabled reward, for the checkpoint's conditioning record."""
    names = []
    if nft.reward.reference_fidelity.enabled:
        names.append("reference_fidelity")
    if nft.reward.aesthetic.enabled:
        names.append("aesthetic")
    if nft.reward.ocr.enabled:
        names.append("ocr")
    return "+".join(names)


def main() -> None:
    run(tyro.cli(Krea2OPSDNFTRecipe))


if __name__ == "__main__":
    main()

"""Does the model's loss actually depend on the references?

Sampling-based evaluation answers this only through a long denoise loop, a VAE decode and a
perceptual metric, each of which can hide or manufacture an effect — a broken sampler once made a
working adapter look inert. This asks the question where it is cheapest and least deniable: run the
training forward pass twice on the same batch, same noise, same timestep, once with the real
references and once with them replaced, and compare the loss.

If the reference span carries information the model uses, corrupting it must raise the loss. If the
two numbers agree to within noise, the adapter is not reading its references and no amount of
sampling tuning will change that.

Reported per timestep bucket, because a reference effect lives at high sigma: near sigma=0 the model
already sees the answer in its own input and the references cannot help.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from dflow.config import Ref2ImgConfig, TextEncoderConfig, VAEConfig  # noqa: E402
from dflow.config.diffusion import FlowMatchConfig  # noqa: E402
from dflow.encoders.text_krea2 import Krea2TextEncoder  # noqa: E402
from dflow.encoders.vae_krea2 import Krea2VAEEncoder  # noqa: E402
from dflow.losses import flow_mse  # noqa: E402
from dflow.models.family.krea2 import Krea2Family  # noqa: E402
from dflow.schedulers.flow_matching import FlowMatchScheduler  # noqa: E402
from dflow.tasks.ref2img.conditioning import (  # noqa: E402
    build_krea2_sequence,
    build_krea2_text_ids,
)
from dflow.tasks.ref2img.dataset import Ref2ImgDataset  # noqa: E402
from dflow.tasks.ref2img.transforms import token_count  # noqa: E402

BUCKETS = ((0.0, 0.3), (0.3, 0.6), (0.6, 0.85), (0.85, 1.0))


def _loss(model, family, tokens, sequence, conditioning, text_ids, sigmas, velocity) -> float:
    """One training-shaped forward pass, scored the way the training step scores it."""
    out = model(
        **family.prepare_inputs(
            tokens=tokens.to(model.dtype),
            token_ids=sequence.ids,
            text_embeds=conditioning.embeds.to(model.dtype),
            text_ids=text_ids,
            timestep=sigmas.to(model.dtype),
            text_mask=conditioning.mask,
        )
    )[0]
    predicted = family.take_target_span(
        out, sequence.target_len, target_offset=sequence.target_offset
    )
    return float(flow_mse(predicted, velocity))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", default="edit_test.jsonl")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--max-refs", type=int, default=9)
    parser.add_argument("--target-max-area", type=int, default=262144)
    parser.add_argument("--reference-fit-target", action="store_true")
    parser.add_argument("--ground-references", action="store_true")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fast-patch-embed", action="store_true",
        help="must match how the checkpoint was trained; see TextEncoderConfig.fast_patch_embed",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    from diffusers import Krea2Pipeline

    pipe = Krea2Pipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    if args.lora:
        from dflow.checkpoint.lora_io import LORA_WEIGHT_NAME

        pipe.transformer.load_lora_adapter(
            args.lora, weight_name=LORA_WEIGHT_NAME, adapter_name="default"
        )
        pipe.transformer.set_adapters(["default"], weights=[1.0])
        print(f"loaded LoRA from {args.lora}")
    model = pipe.transformer.eval()
    family = Krea2Family(distilled=bool(pipe.config.is_distilled))
    vae = Krea2VAEEncoder(
        pipe.vae, VAEConfig(encode_dtype="bfloat16"), device=device, patch_size=pipe.patch_size
    )
    text = Krea2TextEncoder(
        pipe.text_encoder, pipe.tokenizer,
        TextEncoderConfig(dtype="bfloat16", max_length=args.max_length),
        device=device, select_layers=tuple(pipe.text_encoder_select_layers),
        grounding_jitter_min=0,
        fast_patch_embed=args.fast_patch_embed,
    )
    scheduler = FlowMatchScheduler(FlowMatchConfig(), device=device)
    dataset = Ref2ImgDataset(
        Ref2ImgConfig(
            root=args.root, manifest=args.manifest, num_references=args.max_refs,
            pad_short_samples=False, pinned_references=1,
            target_max_area=args.target_max_area,
            reference_fit_target=args.reference_fit_target,
        )
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    rows: list[tuple[float, float, float]] = []

    for index in range(min(args.samples, len(dataset))):
        sample = dataset[index]
        with torch.no_grad():
            target = vae.encode(sample["target"].unsqueeze(0).to(device), mode="mode").float()
            references = [
                vae.encode(r.unsqueeze(0).to(device), mode="mode").float()
                for r in sample["references"]
            ]
            if args.ground_references:
                conditioning = text.encode_grounded(
                    sample["prompt"], [r.unsqueeze(0) for r in sample["references"]]
                )
            else:
                conditioning = text.encode([sample["prompt"]])

            sequence = build_krea2_sequence(
                target_latents=target, reference_latents=references
            )
            span = slice(
                sequence.target_offset, sequence.target_offset + sequence.target_len
            )
            clean = sequence.tokens[:, span]
            noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
            mu = family.noise_shift_mu(image_tokens=token_count(target))
            sigmas = scheduler.sample_timesteps(1, mu=mu, generator=generator)
            noisy = scheduler.add_noise(clean, noise, sigmas)
            velocity = scheduler.target(clean, noise)
            text_ids = build_krea2_text_ids(
                conditioning.embeds.shape[1], device=conditioning.embeds.device
            )

            with_refs = sequence.replace_target(noisy).tokens
            # Same shape, same positions, no content: the only thing removed is what the
            # references say. Shuffling would leave the marginal statistics intact and is the
            # weaker probe; zeroing is what "the model was told nothing" looks like.
            blanked = with_refs.clone()
            blanked[:, : sequence.target_offset] = 0.0

            rows.append((
                float(sigmas[0]),
                _loss(model, family, with_refs, sequence, conditioning, text_ids, sigmas, velocity),
                _loss(model, family, blanked, sequence, conditioning, text_ids, sigmas, velocity),
            ))
        print(f"  [{index + 1}/{args.samples}] sigma={rows[-1][0]:.3f} "
              f"refs={rows[-1][1]:.4f} blank={rows[-1][2]:.4f}", flush=True)

    print("\nsigma bucket        n   loss(refs)  loss(blank)   delta")
    for low, high in BUCKETS:
        picked = [r for r in rows if low <= r[0] < high]
        if not picked:
            continue
        a = sum(r[1] for r in picked) / len(picked)
        b = sum(r[2] for r in picked) / len(picked)
        print(f"  [{low:.2f},{high:.2f})  {len(picked):3d}   {a:9.4f}   {b:9.4f}  {b - a:+8.4f}")
    a = sum(r[1] for r in rows) / len(rows)
    b = sum(r[2] for r in rows) / len(rows)
    print(f"  overall       {len(rows):3d}   {a:9.4f}   {b:9.4f}  {b - a:+8.4f}")
    print("\ndelta > 0 means blanking the references costs the model accuracy, i.e. it reads them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sample a Krea 2 multi-reference LoRA.

    python tools/krea2/sample_krea2.py --model <krea2 dir> --lora <ckpt>/lora \
        --prompt "..." --refs a.jpg b.jpg --out out.png

``tools/flux2/sample.py`` samples FLUX.2 through the stock pipeline, which is the right thing to do: it
proves the export format by making the same call a downstream user would. **That is not available
here.** ``Krea2Pipeline`` is text-to-image only — it has no reference-image argument and its
``prepare_position_ids`` pins the RoPE T axis at zero — so there is no upstream call that accepts
what this LoRA was trained on.

This script therefore reconstructs the denoising loop, and reuses everything else from the pipeline
instance rather than reimplementing it: the scheduler, its ``mu``, the VAE, the text encoder, the
prompt template, ``_unpack_latents``. Only the two lines that differ are ours — building the
``[target; references]`` sequence and slicing the target span back out — so drift is confined to
exactly the part that has no upstream equivalent.

The CFG combination is copied verbatim, including its unusual form::

    noise_pred = noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

which is `pos + s*(pos - neg)`, not the more common `neg + s*(pos - neg)`. The effective strength is
therefore `1 + guidance_scale`. Matching it matters: sampling at a different effective scale than
the checkpoint expects looks like a training problem.

``--no-refs`` runs the same prompt with the reference span removed. That A/B is the only honest way
to tell whether the references are being used, because the loss curve cannot: the text-conditioned
part of the task drives it down either way.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _sampler import Krea2Sampler, resolve  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Krea 2 checkpoint directory")
    parser.add_argument("--lora", default=None, help="a checkpoint's lora/ directory")
    parser.add_argument("--lora-scale", type=float, default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--refs", nargs="*", default=[])
    parser.add_argument("--no-refs", action="store_true", help="drop the reference span entirely")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--reference-max-area", type=int, default=None)
    parser.add_argument(
        "--steps", type=int, default=None,
        help="default: the family's own recommendation (Raw 52, Turbo 8)",
    )
    parser.add_argument(
        "--guidance", type=float, default=None,
        help="default: the family's own recommendation (Raw 3.5, Turbo 0.0). 0 disables CFG",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="krea2_sample.png")
    parser.add_argument(
        "--ground-references", action="store_true",
        help="feed the references through Qwen3-VL too; must match how the LoRA was trained",
    )
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--grounding-max-px", type=int, default=None)
    parser.add_argument("--max-grounded-references", type=int, default=None)
    parser.add_argument("--reference-t-scale", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--reference-registration", default=None, choices=("center", "origin", "disjoint"),
        help="must match --dataset.reference-registration the checkpoint was trained with",
    )
    parser.add_argument(
        "--reference-fit-target", action="store_true",
        help="size references against the target extent, matching --dataset.reference-fit-target",
    )
    parser.add_argument(
        "--fast-patch-embed", action="store_true",
        help="must match how the checkpoint was trained; see TextEncoderConfig.fast_patch_embed",
    )
    args = parser.parse_args()

    sampler = Krea2Sampler.load(args.model, args.lora, resolve(args, lora=args.lora))
    image = sampler.render(
        args.prompt, args.refs, width=args.width, height=args.height, seed=args.seed,
        no_refs=args.no_refs,
    )
    image.save(args.out)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

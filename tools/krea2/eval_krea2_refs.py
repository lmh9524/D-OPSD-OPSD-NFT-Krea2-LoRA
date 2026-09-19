"""Does the LoRA use its references? Measured under the run's own training conditions.

An earlier version of this script invented its own prompt and passed a hand-picked subset of a
sample's references. Both were wrong, and together they made a working model look inert:

* the prompt distribution matters. Training captions here are long scene descriptions
  ("Standing front-facing with a slight body turn... a clean and minimalist plain white studio
  background"); a short invented one is out of distribution for a text stream that reaches the
  model through twelve tapped encoder layers and a fusion transformer.
* passing four of a sample's six references is not the condition the model was trained on.

So this script reads everything from the manifest the run trained on — the exact prompt, every
reference, in slot order — and takes ``--reference-max-area`` from the caller so it matches the
training flag. Nothing about the conditioning is invented here.

Reported per sample, all in [0, 1] mean absolute pixel difference:

``delta_refs``   with-references vs without. Near zero means the reference span is inert.
``err_refs``     with-references vs the ground-truth look.
``err_norefs``   without-references vs the same ground truth.
``gain``         ``err_norefs - err_refs``. **The headline number.** Positive means conditioning on
                 the references moves the output toward the right answer; zero or negative means
                 the span is decoration.

``gain`` is what the other two cannot fake: a model that merely perturbs its output when handed
references scores a large ``delta_refs`` and no ``gain`` at all.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _sampler import Krea2Sampler, resolve  # noqa: E402


def difference(a, b, size=(256, 448)) -> float:
    import numpy as np
    from PIL import Image

    def load(path):
        return np.asarray(Image.open(path).convert("RGB").resize(size), dtype=np.float32) / 255.0

    return float(np.abs(load(a) - load(b)).mean())


def contact_sheet(path, *, refs, with_refs, without_refs, truth, caption):
    from PIL import Image, ImageDraw

    def load(source, box):
        image = Image.open(source).convert("RGB")
        image.thumbnail((box, box))
        return image

    inputs = [load(r, 130) for r in refs[:8]]
    outputs = [
        ("[OUT] with refs", load(with_refs, 300)),
        ("[OUT] no refs", load(without_refs, 300)),
        ("GROUND TRUTH", load(truth, 300)),
    ]
    width = max(sum(i.width + 10 for i in inputs) + 10, sum(o[1].width + 10 for o in outputs) + 10)
    sheet = Image.new("RGB", (width, 130 + 300 + 96), "white")
    draw = ImageDraw.Draw(sheet)

    draw.rectangle([0, 0, width, 20], fill=(220, 235, 255))
    draw.text((6, 5), f"INPUTS - {len(refs)} references, slot order left to right", fill="black")
    x = 6
    for index, image in enumerate(inputs):
        sheet.paste(image, (x, 26))
        draw.text((x, 26 + image.height + 2), f"slot{index}", fill=(0, 0, 160))
        x += image.width + 10

    y = 130 + 40
    draw.rectangle([0, y - 4, width, y + 18], fill=(255, 230, 230))
    draw.text((6, y), "OUTPUTS - same prompt and seed; only the reference span differs", fill="black")
    x, y = 6, y + 26
    for label, image in outputs:
        sheet.paste(image, (x, y))
        draw.text((x, y + image.height + 2), label, fill=(160, 0, 0))
        x += image.width + 10
    draw.text((6, sheet.height - 20), caption[:190], fill="black")
    sheet.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--root", required=True, help="the dataset root the manifest resolves against")
    parser.add_argument("--manifest", required=True, help="the manifest the run trained on")
    parser.add_argument("--reference-max-area", type=int, default=None, help="match the training flag")
    parser.add_argument("--max-refs", type=int, default=9, help="match --dataset.num-references")
    parser.add_argument("--out", default="eval_krea2")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0, help="index into the manifest to start at")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--ground-references", action="store_true")
    parser.add_argument("--reference-fit-target", action="store_true")
    parser.add_argument("--fast-patch-embed", action="store_true")
    parser.add_argument(
        "--reference-registration", default=None, choices=("center", "origin", "disjoint"))
    parser.add_argument("--max-grounded-references", type=int, default=None)
    parser.add_argument("--reference-t-scale", type=int, default=None)
    parser.add_argument("--grounding-max-px", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--lora-scale", type=float, default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument(
        "--sampler-flags", default="",
        help=(
            "deprecated: pass the flags directly. Still honoured — one space-separated string "
            "re-parsed into the same options, e.g. "
            "--sampler-flags='--ground-references --reference-fit-target'. A single string rather "
            "than nargs='*' because argparse stops collecting at the first token beginning with "
            "'-', which silently dropped exactly the flags this exists to forward. Whatever the "
            "checkpoint was trained with belongs here: sampling under different conditioning "
            "measures the mismatch, not the model."
        ),
    )
    args = parser.parse_args()

    # Kept working rather than removed: several run scripts pass conditioning this way, and a flag
    # that is quietly ignored is precisely the failure this whole tool keeps tripping over. Re-parse
    # the string into the same namespace so it means what it always meant.
    if args.sampler_flags:
        args = parser.parse_args(sys.argv[1:] + args.sampler_flags.split())
        print(f"--sampler-flags applied: {args.sampler_flags}")

    root = pathlib.Path(args.root)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    entries = [json.loads(line) for line in (root / args.manifest).read_text().splitlines()]
    chosen = entries[args.offset : args.offset + args.samples]
    if not chosen:
        parser.error(f"no entries at offset {args.offset} in {args.manifest}")

    # One loaded pipeline for every render. This used to shell out to the CLI per image, which
    # reloaded a 13B transformer 2 x samples times -- about 25 s of each 30 s render, and five hours
    # across a sweep of eight checkpoints at fifty samples.
    sampler = Krea2Sampler.load(args.model, args.lora, resolve(args, lora=args.lora))

    report = []
    for index, entry in enumerate(chosen):
        refs = [str(root / r) for r in entry["refs"][: args.max_refs]]
        truth = str(root / entry["target"])
        with_refs = out / f"{index:02d}_with_refs.png"
        without = out / f"{index:02d}_no_refs.png"
        # The sample's own caption, verbatim. Inventing one here is what broke an earlier round.
        common = dict(width=args.width, height=args.height, seed=args.seed)
        sampler.render(entry["prompt"], refs, no_refs=False, **common).save(with_refs)
        sampler.render(entry["prompt"], [], no_refs=True, **common).save(without)

        err_refs = difference(with_refs, truth)
        err_norefs = difference(without, truth)
        row = {
            "index": index,
            "refs": len(refs),
            "delta_refs": difference(with_refs, without),
            "err_refs": err_refs,
            "err_norefs": err_norefs,
            "gain": err_norefs - err_refs,
        }
        report.append(row)
        contact_sheet(
            out / f"{index:02d}_sheet.png", refs=refs, with_refs=with_refs,
            without_refs=without, truth=truth, caption=entry["prompt"],
        )
        print(
            f"[{index}] refs={row['refs']}  delta={row['delta_refs']:.4f}  "
            f"err_refs={err_refs:.4f}  err_norefs={err_norefs:.4f}  gain={row['gain']:+.4f}",
            flush=True,
        )

    (out / "report.json").write_text(json.dumps(report, indent=2))
    n = len(report)
    mean_gain = sum(r["gain"] for r in report) / n
    wins = sum(1 for r in report if r["gain"] > 0)
    print(f"\nmean gain = {mean_gain:+.4f}   ({wins}/{n} samples improved by their references)")
    print(f"mean delta_refs = {sum(r['delta_refs'] for r in report) / n:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

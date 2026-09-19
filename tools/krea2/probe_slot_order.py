"""Is a reference followed because of *what* it is, or because of *where* it sits?

On Garments2Look the two are almost perfectly confounded. Slot position is near-deterministic per
category — tops land in slot 1 in 96% of samples, outerwear in slot 2 in 89%, and bags average slot
3.7 — and the order in which items transfer well matches that ordering exactly: top, outerwear,
trousers, shoes, bag. So "lower-body items fail" and "later slots fail" predict the same thing, and
no amount of staring at outputs separates them.

This separates them. Take one entry, render it twice from the same seed: once in manifest order, and
once with a chosen category promoted to slot 1 — the position tops normally occupy — with the
caption's "Image N" numbering rewritten to match, so the text still names the right slot.

    If the promoted item now transfers, the problem is positional: the model privileges early
    frames, and a bag fails because it is late, not because it is a bag.

    If it still does not, the problem is the item: a cutout on white has to become an object worn
    across the body, with a strap and an occlusion relationship, and that is simply harder than
    draping fabric on a torso.

Promotion is deliberately a train/serve mismatch — the model learned slot 1 means "top". That is
what gives the test its power: if slot 1 is privileged, an out-of-distribution bag placed there
should still come out better.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _sampler import Krea2Sampler, Settings  # noqa: E402


def category_of(path: str) -> str:
    parts = path.split("/")
    return parts[-2] if len(parts) >= 2 else "identity"


def renumber(prompt: str, order: list[int]) -> str:
    """Rewrite ``Image N`` so the caption still names the slot each item actually occupies.

    ``order`` maps new slot -> old slot, both 0-based over ``refs``. Captions count from 1, so
    reference ``i`` is "Image i+1". Rewritten in one pass against the *old* numbering, because
    rewriting sequentially would renumber text this function had already produced.
    """
    old_to_new = {old: new for new, old in enumerate(order)}
    lines = {}
    for line in prompt.splitlines():
        match = re.match(r"^- Image (\d+):", line.strip())
        if match:
            lines[int(match.group(1))] = line

    def swap(match: re.Match) -> str:
        old = int(match.group(1)) - 1
        return f"Image {old_to_new.get(old, old) + 1}" if old in old_to_new else match.group(0)

    rewritten = re.sub(r"Image (\d+)", swap, prompt)
    # The per-reference bullet list is now out of order; sort it back so the text reads in slot
    # order, which is how every training caption was written.
    out, block = [], []
    for line in rewritten.splitlines():
        match = re.match(r"^- Image (\d+):", line.strip())
        if match:
            block.append((int(match.group(1)), line))
            continue
        if block:
            out.extend(text for _, text in sorted(block))
            block = []
        out.append(line)
    out.extend(text for _, text in sorted(block))
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", default="edit_test.jsonl")
    parser.add_argument("--promote", default="bag", help="category to move into slot 1")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 2, 3, 5, 6])
    parser.add_argument("--out", default="slot_probe")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--reference-registration", default="disjoint")
    parser.add_argument("--fast-patch-embed", action="store_true")
    parser.add_argument("--max-grounded-references", type=int, default=1)
    args = parser.parse_args()

    root, out = pathlib.Path(args.root), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entries = [json.loads(line) for line in (root / args.manifest).read_text().splitlines()]

    settings = Settings(
        reference_registration=args.reference_registration,
        reference_fit_target=True,
        ground_references=True,
        max_grounded_references=args.max_grounded_references,
        fast_patch_embed=args.fast_patch_embed,
    )
    sampler = Krea2Sampler.load(args.model, args.lora, settings)

    def render(prompt, paths, tag):
        image = sampler.render(
            prompt, [root / p for p in paths],
            width=args.width, height=args.height, seed=args.seed,
        )
        image.save(out / tag)
        return image

    for index in args.indices:
        entry = entries[index]
        refs = entry["refs"]
        categories = [category_of(p) for p in refs]
        if args.promote not in categories[1:]:
            print(f"sample {index}: no {args.promote}, skipped")
            continue
        target_slot = categories.index(args.promote, 1)

        # new slot -> old slot. Slot 0 stays the identity; the promoted item takes slot 1.
        order = [0, target_slot] + [i for i in range(1, len(refs)) if i != target_slot]
        promoted_prompt = renumber(entry["prompt"], order)

        print(f"sample {index}: {args.promote} slot {target_slot} -> 1  ({categories})")
        render(entry["prompt"], refs, f"{index:03d}_original_order.png")
        render(promoted_prompt, [refs[i] for i in order], f"{index:03d}_promoted.png")

        from PIL import Image

        Image.open(root / entry["target"]).convert("RGB").resize(
            (args.width, args.height), Image.LANCZOS
        ).save(out / f"{index:03d}_ground_truth.png")
        (out / f"{index:03d}_promoted_prompt.txt").write_text(promoted_prompt, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

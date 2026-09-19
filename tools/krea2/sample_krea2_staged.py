"""Two-pass reference sampling: dress the upper body, then the lower body.

One pass has to place every garment at once, and it does not place them equally well. Measured on
Garments2Look, references buy +0.161 of accuracy on the head/torso band and +0.121 at the feet,
and the qualitative failure is blunt: a light-blue jeans reference comes out black, tan heeled
sandals come out white sneakers, a clutch does not appear at all, while the top is reproduced
faithfully. The suspected cause is that every reference is spatially registered to the *middle* of
the target grid, so a garment's H coordinate always says "torso height" -- true for a top, twelve
rows wrong for trousers, disjoint from where shoes go.

Splitting the pass sidesteps that. Stage two's first reference is stage one's own render: a
full-size, full-body image at exactly the target's grid, so its registration offset is zero and its
coordinates mean what they say. The remaining references are only the lower-body items, and the
prompt asks for one thing instead of eight.

    python tools/krea2/sample_krea2_staged.py --model <turbo dir> --lora <ckpt>/lora \
        --root <dataset> --manifest edit_test.jsonl --index 184 --out staged/

Costs two denoise passes instead of one. Whether that buys anything is what this exists to answer,
so it also writes the single-pass render for the same entry, seed and settings.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _sampler import Krea2Sampler, Settings  # noqa: E402
from PIL import Image  # noqa: E402

#: Which stage each garment category belongs to. Bags, bracelets, rings and watches sit with the
#: lower body because they hang at hip and hand height, which is the half stage one gets wrong.
UPPER = {"top", "outwear", "neckwear", "necklace", "earrings", "eyewear", "brooch", "hat", "scarf"}
LOWER = {"pants", "skirt", "shoes", "bag", "bracelet", "rings", "watches", "belt", "hosiery"}

#: Where each category sits on a standing full-body figure, as a fraction of frame height, plus a
#: horizontal centre. Used by ``--canvas`` to paste a cutout onto a target-sized white canvas at the
#: height it will actually be worn.
#:
#: This is the same information the caption already carries ("worn on the waist, covering both
#: legs"); the point is to put it where the *position ids* can see it. A cutout pasted onto a
#: full-size canvas is a target-sized reference, so its centring offset is zero and its H
#: coordinates mean what they say -- the one case this checkpoint demonstrably follows, since a
#: full-frame reference is reproduced almost exactly while the same items as loose cutouts are not.
BANDS = {
    "hat":      (0.00, 0.11, 0.50), "eyewear":  (0.05, 0.12, 0.50),
    "earrings": (0.07, 0.15, 0.50), "neckwear": (0.12, 0.24, 0.50),
    "necklace": (0.12, 0.24, 0.50), "brooch":   (0.17, 0.27, 0.42),
    "scarf":    (0.10, 0.30, 0.50), "top":      (0.15, 0.52, 0.50),
    "outwear":  (0.12, 0.64, 0.50), "belt":     (0.40, 0.50, 0.50),
    "skirt":    (0.42, 0.74, 0.50), "pants":    (0.44, 0.93, 0.50),
    "hosiery":  (0.55, 0.93, 0.50), "shoes":    (0.86, 1.00, 0.50),
    "bag":      (0.40, 0.66, 0.74), "bracelet": (0.48, 0.60, 0.22),
    "watches":  (0.48, 0.60, 0.22), "rings":    (0.50, 0.60, 0.20),
}

BODY = (
    "Keep a normal, anatomically correct body: exactly two arms, two legs, two hands and two "
    "feet, with no extra, duplicated, merged or missing limbs."
)


def category_of(path: str) -> str:
    parts = path.split("/")
    return parts[-2] if len(parts) >= 2 else "identity"


def stage_one_prompt(items: list[str]) -> str:
    """Dress the upper body from the person reference; leave the rest deliberately plain.

    The lower body is described as plain and neutral rather than left unmentioned: stage two has to
    replace it, and a plain garment is easier to overwrite than a busy one.
    """
    lines = [
        f"- Image {i + 2}: {c} — put this item on the woman, matching the reference exactly."
        for i, c in enumerate(items)
    ]
    return (
        "Photo-editing task. Compose a photorealistic full-body fashion photograph of the woman "
        "from Image 1, dressed in the referenced items. Keep the exact person from Image 1: same "
        f"face, skin tone, hair, and body shape. {BODY}\n\n"
        "Reference images:\n"
        "- Image 1: the woman — preserve their exact face, skin tone, hair, and body shape; do not "
        "change their identity.\n" + "\n".join(lines) + "\n\n"
        "Dress the upper body from the references. Below the waist, keep it plain and neutral: "
        "simple undecorated trousers and plain shoes, no bag. Full body, standing, plain background."
    )


def stage_two_prompt(items: list[str]) -> str:
    """Change only what is below the waist, and say so first.

    Image 1 is stage one's own render, so the instruction is a genuine edit -- preserve most of the
    frame, replace one region -- which is the shape of task the base checkpoint's edit behaviour was
    adapted for, rather than a composition from eight loose cutouts.
    """
    lines = [
        f"- Image {i + 2}: {c} — put this item on the woman, matching the reference exactly."
        for i, c in enumerate(items)
    ]
    return (
        "Photo-editing task. Image 1 is a photograph of a woman. Keep it exactly as it is from the "
        "waist up: same face, same hair, same skin tone, same upper-body clothing, same pose, same "
        "framing, same lighting, same background. Change only what she wears below the waist and "
        f"what she carries. {BODY}\n\n"
        "Reference images:\n"
        "- Image 1: the photograph to edit — preserve everything above the waist unchanged.\n"
        + "\n".join(lines) + "\n\n"
        "Replace the lower-body clothing, footwear and accessories with the referenced items, "
        "matching their colour, pattern and shape exactly. Full body, standing."
    )


def place_on_canvas(path, category, *, size):
    """Paste one cutout onto a white canvas the size of the target, at its own body height.

    Returns a PIL image at the target's exact extent, so ``build_krea2_latent_ids`` centres it with
    a zero offset and every H coordinate it carries is the row the garment is actually meant to
    occupy.
    """
    from PIL import Image

    width, height = size
    top, bottom, centre = BANDS.get(category, (0.25, 0.75, 0.50))
    band_h = max(1, int((bottom - top) * height))
    band_w = max(1, int(0.92 * width))

    item = Image.open(path).convert("RGB")
    scale = min(band_w / item.width, band_h / item.height)
    item = item.resize((max(1, int(item.width * scale)), max(1, int(item.height * scale))), Image.LANCZOS)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = int(centre * width) - item.width // 2
    y = int(top * height) + (band_h - item.height) // 2
    canvas.paste(item, (max(0, min(x, width - item.width)), max(0, min(y, height - item.height))))

    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", default="edit_test.jsonl")
    parser.add_argument("--index", type=int, default=184)
    parser.add_argument("--out", default="staged")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--canvas", action="store_true",
        help="also render with each cutout pasted onto a target-sized canvas at its body height",
    )
    parser.add_argument(
        "--skip-staged", action="store_true", help="only the single-pass and canvas renders",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entry = json.loads((root / args.manifest).read_text().splitlines()[args.index])

    settings = Settings(
        reference_registration=args.reference_registration,
        reference_fit_target=True,
        ground_references=True,
        max_grounded_references=1,
        fast_patch_embed=args.fast_patch_embed,
    )
    sampler = Krea2Sampler.load(args.model, args.lora, settings)

    identity = entry["refs"][0]
    upper = [p for p in entry["refs"][1:] if category_of(p) in UPPER]
    lower = [p for p in entry["refs"][1:] if category_of(p) in LOWER]
    other = [p for p in entry["refs"][1:] if p not in upper and p not in lower]
    upper += other  # anything uncategorised goes to stage one rather than being dropped
    print(f"sample {args.index}: {len(entry['refs'])} refs -> "
          f"stage1 {[category_of(p) for p in upper]}, stage2 {[category_of(p) for p in lower]}")

    def render(prompt, references, seed, tag):
        image = sampler.render(
            prompt, references, width=args.width, height=args.height, seed=seed
        )
        image.save(out / f"{args.index:03d}_{tag}.png")
        print(f"  wrote {out}/{args.index:03d}_{tag}.png  ({len(references)} refs)")
        return image

    # ---- single pass, for comparison ----------------------------------------------------
    # Paths and PIL images both go straight to the sampler, which sizes them exactly as the dataset
    # does — one implementation of that transform rather than a second one here.
    render(entry["prompt"], [root / p for p in entry["refs"]], args.seed, "single_pass")

    # ---- canvas pass: same references, pasted at the height they are worn ----------------
    if args.canvas:
        canvas = [Image.open(root / identity).convert("RGB")] + [
            place_on_canvas(root / p, category_of(p), size=(args.width, args.height))
            for p in entry["refs"][1:]
        ]
        render(entry["prompt"], canvas, args.seed, "canvas_pass")

    def write_truth():
        Image.open(root / entry["target"]).convert("RGB").resize(
            (args.width, args.height), Image.LANCZOS
        ).save(out / f"{args.index:03d}_ground_truth.png")
        print(f"  wrote {out}/{args.index:03d}_ground_truth.png")

    if args.skip_staged:
        write_truth()
        return 0

    # ---- stage one: upper body ----------------------------------------------------------
    first = render(
        stage_one_prompt([category_of(p) for p in upper]),
        [root / p for p in [identity, *upper]], args.seed, "stage1_upper",
    )

    # ---- stage two: lower body, conditioned on stage one's own render --------------------
    # Stage one's render is a full-size, full-body image at exactly the target's grid, so its
    # registration offset is zero and its coordinates mean what they say.
    render(
        stage_two_prompt([category_of(p) for p in lower]),
        [first, *[root / p for p in lower]], args.seed + 1, "stage2_final",
    )
    write_truth()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

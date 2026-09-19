"""Convert Garments2Look into the ``ref2img`` manifest schema.

    python tools/data/prepare_garments2look.py --root /path/to/Garments2Look --source polyvore

Garments2Look is a genuine many-garments-to-one-look dataset: every outfit pairs 3-12 reference
garment images with one model image wearing that outfit, plus a written description. That is exactly
the ``ref2img`` contract, so the conversion is a re-indexing rather than a transformation::

    {"target": "...", "refs": ["...", "..."], "prompt": "..."}

Layout, read from the dataset's own ``Garments2Look.py`` loader rather than guessed::

    {root}/{source}/looks-resized/{gender}/{outfit_id}.{png,jpg}     the target
    {root}/{source}/images/{gender}/{type}/{garment_id}.jpg          one per reference

``{type}`` is a category folder the loader derives from each garment's name. This tool **indexes the
image tree once** and looks garment ids up in that index instead of re-deriving the category. The
taxonomy is theirs; re-implementing it here would be a second copy to drift, and a miss would look
like a missing file rather than a wrong one.

Two things this writes out that matter more than they look:

**Reference order is sorted by garment id, and stable.** The model distinguishes references purely
by their position on the RoPE T axis, so slot *i* must mean the same thing on every epoch and after
every resume. Dict iteration order in JSON would be stable in practice and undefined in principle;
sorting makes it neither.

**Outfits with missing files are dropped and counted, not silently skipped.** The image tarballs are
~284 GB and often fetched partially, so "my manifest is small" needs to be visible rather than
inferred.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

SOURCES = ("polyvore", "mytheresa")
#: The loader tries .png first, then .jpg.
LOOK_SUFFIXES = (".png", ".jpg")


def index_garment_images(images_root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Map ``garment_id -> path`` by walking the image tree once.

    Avoids re-implementing the dataset's category inference: whatever folder a garment landed in,
    its stem is its id.
    """
    index: dict[str, pathlib.Path] = {}
    duplicates = 0
    for path in images_root.rglob("*.jpg"):
        if path.stem in index:
            duplicates += 1
            continue
        index[path.stem] = path
    if duplicates:
        print(f"  note: {duplicates} duplicate garment ids, kept the first of each", file=sys.stderr)
    return index


def find_look(look_root: pathlib.Path, gender: str, outfit_id: str) -> pathlib.Path | None:
    for suffix in LOOK_SUFFIXES:
        candidate = look_root / gender / f"{outfit_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def build_prompt(outfit_info: dict, items: dict[str, str], *, source: str) -> str:
    """Build the caption. ``source`` decides whether it describes the garments or only the scene.

    **This choice decides whether reference conditioning gets learned at all.**

    ``description`` uses the dataset's ``outfit_description``, which names every garment — "a
    heather grey pocket tee with light wash ripped skinny jeans". A model trained on that can
    minimise the loss from text alone, so the reference span carries no information the caption
    does not already provide and the pathway that reads it never receives gradient. Measured at
    3000 steps: samples with a red dress + black pumps as references produced a grey tank and an
    olive skirt. The references were inert.

    ``scene`` uses ``model_attributes`` — pose and background — which describe the framing and say
    nothing about what the garments look like. Garment identity is then recoverable *only* from the
    reference images, which is the pressure that makes the model learn to read them.

    Ordinals are avoided in both: nothing binds "image 2" to a slot except position, and a caption
    naming one while the loader cycles or truncates slots teaches the wrong correspondence. Keep
    ``--dataset.reference-dropout`` at 0 for the same reason.
    """
    if source == "scene":
        attributes = outfit_info.get("model_attributes") or {}
        # `body` is excluded on purpose: it leaks garment words ("showcasing the dress's
        # silhouette"), which is exactly the shortcut this mode exists to remove.
        parts = [
            (attributes.get("pose") or "").strip(),
            (attributes.get("background") or "").strip(),
        ]
        scene = " ".join(part for part in parts if part)
        if scene:
            return scene
        return "A full body studio photograph of a fashion model on a plain background."

    description = (outfit_info.get("outfit_description") or "").strip()
    if description:
        return description
    return "A model wearing " + ", ".join(sorted(items.values())) + "."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Garments2Look download root")
    parser.add_argument("--source", default="polyvore", choices=SOURCES)
    parser.add_argument(
        "--annotations",
        default=None,
        help="outfit json; defaults to {root}/{source}_outfit_v1.0_2512.json",
    )
    parser.add_argument("--section", default="train", help="train | test | all")
    parser.add_argument("--out", default=None, help="defaults to {root}/{source}/train.jsonl")
    parser.add_argument(
        "--min-refs", type=int, default=2, help="drop outfits with fewer usable references"
    )
    parser.add_argument("--max-refs", type=int, default=0, help="0 keeps every reference")
    parser.add_argument(
        "--prompt-source",
        default="scene",
        choices=("scene", "description"),
        help=(
            "scene (default): pose + background only, so garment identity can come only from the "
            "references. description: the dataset's outfit_description, which names every garment "
            "and lets the model solve the task from text alone."
        ),
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root).expanduser().resolve()
    annotations = pathlib.Path(
        args.annotations or root / f"{args.source}_outfit_v1.0_2512.json"
    )
    if not annotations.exists():
        parser.error(f"annotations not found: {annotations}")

    images_root = root / args.source / "images"
    look_root = root / args.source / "looks-resized"
    for path in (images_root, look_root):
        if not path.exists():
            parser.error(
                f"{path} does not exist. The image tarballs are separate downloads:\n"
                f"    hf download ArtmeScienceLab/Garments2Look --repo-type dataset \\\n"
                f"        --include '{args.source}/*' --local-dir {root}\n"
                f"then extract the .tar.gz parts in place."
            )

    print(f"indexing {images_root} ...", file=sys.stderr)
    index = index_garment_images(images_root)
    print(f"  {len(index)} garment images", file=sys.stderr)

    outfits = json.loads(annotations.read_text())
    out_path = pathlib.Path(args.out or root / args.source / "train.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    #: What ``--dataset.root`` will be at training time: the directory holding the manifest.
    manifest_root = out_path.parent.resolve()

    dropped: collections.Counter[str] = collections.Counter()
    notes: collections.Counter[str] = collections.Counter()
    reference_counts: collections.Counter[int] = collections.Counter()
    written = 0

    with out_path.open("w") as handle:
        for outfit_id, record in outfits.items():
            if args.section != "all" and record.get("section") != args.section:
                dropped["wrong section"] += 1
                continue

            gender = record.get("gender", "")
            target = find_look(look_root, gender, outfit_id)
            if target is None:
                dropped["look image missing"] += 1
                continue

            items: dict[str, str] = record.get("outfit") or {}
            # `U`-prefixed ids carry no image and are skipped by the dataset's own loader
            # (`if garment_id.startswith("U"): continue`, Garments2Look.py:400). They are 9.3% of
            # referenced items and *every* id without a file is one of them, so excluding them here
            # is matching upstream rather than tolerating missing data.
            wanted = [item_id for item_id in sorted(items) if not item_id.startswith("U")]
            skipped_imageless = len(items) - len(wanted)
            # Sorted, so slot i means the same garment on every epoch and after every resume.
            refs = [index[item_id] for item_id in wanted if item_id in index]

            if skipped_imageless:
                notes["outfits with an imageless U-item skipped"] += 1
            if len(refs) < len(wanted):
                # Not expected: every known gap is a U-item. Surfaced separately so a partial
                # extraction does not masquerade as normal.
                notes["outfits missing a garment file (UNEXPECTED)"] += 1
            if len(refs) < args.min_refs:
                dropped["too few references"] += 1
                continue
            if args.max_refs:
                refs = refs[: args.max_refs]

            reference_counts[len(refs)] += 1
            # Relative to the manifest's own directory, not to --root. ``Ref2ImgDataset`` resolves
            # every path against ``dataset.root``, and ``dataset.root`` is the directory holding
            # train.jsonl — so writing paths relative to anything else produces
            # ``.../polyvore/polyvore/...`` at load time.
            handle.write(
                json.dumps(
                    {
                        "target": str(target.relative_to(manifest_root)),
                        "refs": [str(path.relative_to(manifest_root)) for path in refs],
                        "prompt": build_prompt(
                            record.get("outfit_info") or {},
                            items,
                            source=args.prompt_source,
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"\nwrote {written} samples to {out_path}")
    print(f"train with:  --dataset.root {manifest_root}")
    if reference_counts:
        spread = ", ".join(f"{n}x{c}" for n, c in sorted(reference_counts.items()))
        mean = sum(n * c for n, c in reference_counts.items()) / written
        print(f"references per sample: {spread}  (mean {mean:.2f})")
        print(
            "\nset --dataset.num-references to a value you are happy to fix; samples with more "
            "contribute a subset in manifest order, samples with fewer are cycled to fill slots."
        )
    for reason, count in dropped.most_common():
        print(f"dropped: {count:6}  {reason}")
    for reason, count in notes.most_common():
        print(f"kept:    {count:6}  {reason}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())

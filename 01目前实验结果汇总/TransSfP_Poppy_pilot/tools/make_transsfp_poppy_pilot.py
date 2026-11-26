#!/usr/bin/env python3
"""Build a 7-object x 3-frame Poppy-format pilot from the TransSfP test split.

For each held-out real object, pick frames at the 25/50/75 percentiles of the
sequence (1-based, never selected by error), resize all modalities to
512x512, and write the directory layout expected by poppy/run.py:

    <out>/pol000, pol045, pol090, pol135, mask, normal

Frames are named <object>_<index>.png so Poppy's per-frame log can be grouped
back to objects. A manifest.csv records the source files.

Usage:
    python -I baselines/tools/make_transsfp_poppy_pilot.py \
        --src /mnt/workspace/datasets/TransSfP/real-world \
        --out /mnt/workspace/datasets/TransSfP_poppy_pilot_512
"""
import argparse
import csv
import glob
import os

from PIL import Image

OBJECTS = [
    "hemi-sphere-big",
    "hemi-sphere-small",
    "bird-back",
    "cat-back",
    "cat-front",
    "tiny-cup",
    "tiny-cup-edges",
]
ANGLES = [(0, "I-0", "000"), (45, "I-45", "045"), (90, "I-90", "090"), (135, "I-135", "135")]
SUBS = ["pol000", "pol045", "pol090", "pol135", "mask", "normal"]


def pick_indices(n):
    """Three evenly spaced 1-based indices at 25/50/75 percentiles."""
    return [int(q * (n - 1) + 0.5) + 1 for q in (0.25, 0.5, 0.75)]


def find_one(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly 1 match, got {len(hits)}: {pattern}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/workspace/datasets/TransSfP/real-world")
    ap.add_argument("--out", default="/mnt/workspace/datasets/TransSfP_poppy_pilot_512")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"output directory is not empty: {args.out}")
    for sub in SUBS:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    rows = []
    for obj in OBJECTS:
        dolp = sorted(glob.glob(os.path.join(args.src, obj, "params", "DoLP", "*-DoLP.png")))
        if not dolp:
            raise FileNotFoundError(f"no DoLP frames for {obj}")
        indices = pick_indices(len(dolp))
        print(f"{obj:18s} frames={len(dolp):3d} picked={indices}")

        for idx in indices:
            idx_s = f"{idx:03d}"
            name = f"{obj}_{idx_s}.png"
            row = {"object": obj, "frame_index": idx, "name": name}

            for deg, folder, suffix in ANGLES:
                src = find_one(os.path.join(args.src, obj, folder, f"*_{idx_s}_{suffix}.png"))
                dst = os.path.join(args.out, f"pol{deg:03d}", name)
                Image.open(src).convert("L").resize((args.size, args.size), Image.LANCZOS).save(dst)
                row[f"pol{deg:03d}"] = os.path.relpath(src, args.src)

            mask_src = os.path.join(args.src, obj, "masks", f"{idx_s}-Mask.png")
            normal_src = os.path.join(args.src, obj, "normals-png", f"normal-{idx_s}.png")
            for src in (mask_src, normal_src):
                if not os.path.isfile(src):
                    raise FileNotFoundError(src)

            Image.open(mask_src).convert("L").resize(
                (args.size, args.size), Image.NEAREST
            ).save(os.path.join(args.out, "mask", name))
            Image.open(normal_src).convert("RGB").resize(
                (args.size, args.size), Image.BILINEAR
            ).save(os.path.join(args.out, "normal", name))
            row["mask"] = os.path.relpath(mask_src, args.src)
            row["normal"] = os.path.relpath(normal_src, args.src)
            rows.append(row)

    manifest = os.path.join(args.out, "manifest.csv")
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} frames -> {args.out}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()

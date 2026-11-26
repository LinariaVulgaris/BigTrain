#!/usr/bin/env python3
"""Supplementary pixel-level diagnostics for the TransSfP reproduction.

Reads the per-frame `err.npy` / `conf.png` / `dolp.png` written by
`eval_transsfp.py`, together with the dataset masks, and reports:

  * error distribution / tail (how much is catastrophic flipping)
  * rim (<12 px from the mask boundary) vs interior error
  * error by confidence decile and by DoLP decile
  * per-frame confidence-error correlation
  * optionally, epoch-20 vs a second eval dir: per-object mean and how many
    frames improve

These numbers are the basis of Sec. 4.5 of `TransSfP_reproduction.md`.

Usage:
    python -I baselines/tools/analyze_transsfp_errors.py \
        --eval_dir /mnt/workspace/results/transsfp_eval \
        --dataset_dir /mnt/workspace/datasets/TransSfP \
        --compare_dir /mnt/workspace/results/transsfp_eval_epoch49
"""
import argparse
import csv
import glob
import os
import re

import cv2
import numpy as np
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
TAIL_THRESHOLDS = (60, 90, 120)


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"))


def frame_entries(eval_dir, obj):
    """Return (stem, err_path, frame_number) for every saved frame of an object."""
    out = []
    for err_path in sorted(glob.glob(os.path.join(eval_dir, "frames", obj, "*-err.npy"))):
        stem = os.path.basename(err_path)[: -len("-err.npy")]
        # stem is "<index>-<frame>", e.g. "000-001-DoLP"
        frame = stem.split("-", 1)[1]
        num = re.match(r"(\d+)-", frame).group(1)
        out.append((stem, err_path, num))
    return out


def load_mask(dataset_dir, obj, num, shape):
    path = os.path.join(dataset_dir, "real-world", obj, "masks", f"{num}-Mask.png")
    mask = load_gray(path)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def decile_stats(values, errors, n_bins=10):
    edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    bins = np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        sel = bins == k
        n = int(sel.sum())
        rows.append((k, float(edges[k]), float(edges[k + 1]), n, float(errors[sel].mean()) if n else 0.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="/mnt/workspace/results/transsfp_eval")
    ap.add_argument("--dataset_dir", default="/mnt/workspace/datasets/TransSfP")
    ap.add_argument("--compare_dir", default="", help="second eval dir for paired comparison")
    ap.add_argument("--objects", default=",".join(OBJECTS))
    args = ap.parse_args()

    objects = [o for o in args.objects.split(",") if o]
    per_obj = {}
    conf_rows = []
    dolp_rows = []
    rim_n = rim_s = int_n = int_s = 0
    tail = {t: 0 for t in TAIL_THRESHOLDS}
    frame_medians, frame_p90, frame_p99, frame_corr = [], [], [], []
    frame_means, frame_dolp, frame_conf = [], [], []
    all_n = all_s = 0

    for obj in objects:
        entries = frame_entries(args.eval_dir, obj)
        if not entries:
            print(f"[WARN] no frames for {obj} under {args.eval_dir}")
            continue
        obj_n = obj_s = 0
        obj_tail = {t: 0 for t in TAIL_THRESHOLDS}
        for _stem, err_path, num in entries:
            err = np.load(err_path).astype(np.float32)
            valid = load_mask(args.dataset_dir, obj, num, err.shape)
            if not valid.any():
                continue
            conf = load_gray(err_path.replace("-err.npy", "-conf.png")).astype(np.float32)
            dolp = load_gray(err_path.replace("-err.npy", "-dolp.png")).astype(np.float32)
            e, c, d = err[valid], conf[valid], dolp[valid]

            obj_n += e.size
            obj_s += float(e.sum())
            all_n += e.size
            all_s += float(e.sum())
            for t in TAIL_THRESHOLDS:
                obj_tail[t] += int((e > t).sum())

            dt = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
            rim = valid & (dt <= 12)
            inner = valid & (dt > 12)
            rim_n += int(rim.sum())
            rim_s += float(err[rim].sum())
            int_n += int(inner.sum())
            int_s += float(err[inner].sum())

            frame_medians.append(float(np.median(e)))
            frame_p90.append(float(np.percentile(e, 90)))
            frame_p99.append(float(np.percentile(e, 99)))
            frame_corr.append(float(np.corrcoef(c, e)[0, 1]) if c.std() > 0 else np.nan)
            frame_means.append(float(e.mean()))
            frame_conf.append(float(c.mean()))
            frame_dolp.append(float(d.mean()))
            conf_rows.append((c, e))
            dolp_rows.append((d, e))

        per_obj[obj] = (obj_n, obj_s, obj_tail)
        for t in TAIL_THRESHOLDS:
            tail[t] += obj_tail[t]

    if all_n == 0:
        raise SystemExit(f"no valid frames found in {args.eval_dir}")

    print(f"# eval dir: {args.eval_dir}")
    print("\n## Per-object (pixel-weighted within object)")
    print(f"{'object':18s} {'px':>9s} {'mean':>6s} {'>60':>7s} {'>90':>7s} {'>120':>7s}")
    obj_means = []
    for obj, (n, s, t) in per_obj.items():
        mean = s / n
        obj_means.append(mean)
        print(f"{obj:18s} {n:9d} {mean:6.2f} "
              f"{100*t[60]/n:6.2f}% {100*t[90]/n:6.2f}% {100*t[120]/n:6.2f}%")
    print(f"{'object-mean':18s} {'':9s} {np.mean(obj_means):6.2f} "
          f"{100*tail[60]/all_n:6.2f}% {100*tail[90]/all_n:6.2f}% {100*tail[120]/all_n:6.2f}%")
    print(f"pixel-pooled mean: {all_s/all_n:.2f}  (valid px {all_n})")

    print("\n## Error distribution")
    print(f"per-frame median: {np.median(frame_medians):.2f}  "
          f"p90: {np.median(frame_p90):.2f}  p99: {np.median(frame_p99):.2f}")
    print(f"tail >60/>90/>120 deg: {100*tail[60]/all_n:.2f}% / "
          f"{100*tail[90]/all_n:.2f}% / {100*tail[120]/all_n:.2f}%")
    print(f"per-frame corr(confidence, error): mean {np.nanmean(frame_corr):.3f}, "
          f"median {np.nanmedian(frame_corr):.3f}")

    print("\n## Rim (<=12 px from mask boundary) vs interior")
    print(f"rim:      {rim_n:9d} px  mean {rim_s/rim_n:6.2f}")
    print(f"interior: {int_n:9d} px  mean {int_s/int_n:6.2f}")
    print(f"rim share of valid pixels: {100*rim_n/all_n:.1f}%")

    print("\n## Error by confidence decile (pooled)")
    conf = np.concatenate([x[0] for x in conf_rows])
    conf_errs = np.concatenate([x[1] for x in conf_rows])
    for k, lo, hi, n, mean in decile_stats(conf, conf_errs):
        print(f"{k}  conf {lo:6.1f}-{hi:6.1f}  n={n:9d}  mean={mean:6.2f}")

    print("\n## Error by DoLP decile (pooled)")
    dolp = np.concatenate([x[0] for x in dolp_rows])
    dolp_errs = np.concatenate([x[1] for x in dolp_rows])
    for k, lo, hi, n, mean in decile_stats(dolp, dolp_errs):
        print(f"{k}  dolp {lo:6.1f}-{hi:6.1f}  n={n:9d}  mean={mean:6.2f}")

    print("\n## Across-frame correlations")
    print(f"corr(mean error, mean DoLP)        = {np.corrcoef(frame_means, frame_dolp)[0,1]:+.3f}")
    print(f"corr(mean error, mean confidence)  = {np.corrcoef(frame_means, frame_conf)[0,1]:+.3f}")

    if args.compare_dir:
        def load_csv(p):
            return {(r["object"], r["frame"]): float(r["mean_deg"])
                    for r in csv.DictReader(open(os.path.join(p, "metrics_per_frame.csv")))}

        base = load_csv(args.eval_dir)
        other = load_csv(args.compare_dir)
        print(f"\n## Paired comparison: {args.eval_dir} vs {args.compare_dir}")
        by_obj = {}
        for key, v in base.items():
            if key in other:
                by_obj.setdefault(key[0], []).append((v, other[key]))
        for obj, pairs in by_obj.items():
            a = np.array([x[0] for x in pairs])
            b = np.array([x[1] for x in pairs])
            improved = int((b < a).sum())
            print(f"{obj:18s} base={a.mean():6.2f} other={b.mean():6.2f} "
                  f"delta={b.mean()-a.mean():+6.2f} other-better={improved}/{len(pairs)}")


if __name__ == "__main__":
    main()

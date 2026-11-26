#!/usr/bin/env python3
"""Visualize the TransSfP x Poppy 7-object x 3-frame pilot.

Produces:
  figs/overview.png            one representative frame per object, GT/base/Poppy
  figs/per_object/<obj>.png    all 3 frames for each object
  figs/object_bars.png         object-level base / Poppy / TransSfP bars
  figs/per_frame_scatter.png   base vs Poppy per frame
  per_frame_summary.csv        per-frame errors and deltas

Normal convention matches Poppy's loader: RGB/255*2-1, renormalized.
Error is the mean angular error over the mask, in degrees.
"""
import argparse
import collections
import csv
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

OBJECTS = [
    "hemi-sphere-big",
    "hemi-sphere-small",
    "bird-back",
    "cat-back",
    "cat-front",
    "tiny-cup",
    "tiny-cup-edges",
]
COL_TITLES = ["Input (0°)", "GT normal", "MoGe-2 base", "Poppy Full",
              "Base error", "Poppy error"]
ERR_VMAX = 90.0


def load_normal(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    n = rgb * 2.0 - 1.0
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    return n


def load_mask(path):
    m = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return m > 0.5


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def angular_error(pred, gt):
    pred = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    gt = gt / (np.linalg.norm(gt, axis=-1, keepdims=True) + 1e-8)
    cos = np.clip(np.sum(pred * gt, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def normal_rgb(n):
    return np.clip(n * 0.5 + 0.5, 0.0, 1.0)


def read_log(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["object"]] = float(row["mean_loss"])
    return out


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_summary(path):
    with open(path, newline="") as f:
        return {row["object"]: row for row in csv.DictReader(f)}


def show_normal(ax, n, mask, title=None):
    vis = np.ones_like(n, dtype=np.float32)
    vis[mask] = normal_rgb(n[mask])
    ax.imshow(vis)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, pad=4)


def show_input(ax, img, title=None):
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, pad=4)


def show_error(ax, err, mask, title=None):
    show = np.ma.masked_where(~mask, err)
    ax.set_facecolor("white")
    ax.imshow(show, cmap="inferno", vmin=0.0, vmax=ERR_VMAX)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, pad=4)


def annotate(ax, text):
    ax.text(
        0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
        fontsize=8.5, color="black",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2.0),
    )


def render_frame_row(axes, frame, data, show_col_titles):
    obj = frame["object"]
    base, poppy, gt, mask = data["base"], data["poppy"], data["gt"], data["mask"]
    err_b = data["err_base"]
    err_p = data["err_poppy"]

    show_input(axes[0], data["img"], COL_TITLES[0] if show_col_titles else None)
    annotate(axes[0], f"{obj} #{int(frame['frame_index']):03d}\n"
                       f"{data['base_mean']:.1f}° → {data['poppy_mean']:.1f}°")
    show_normal(axes[1], gt, mask, COL_TITLES[1] if show_col_titles else None)
    show_normal(axes[2], base, mask, COL_TITLES[2] if show_col_titles else None)
    show_normal(axes[3], poppy, mask, COL_TITLES[3] if show_col_titles else None)
    show_error(axes[4], err_b, mask, COL_TITLES[4] if show_col_titles else None)
    annotate(axes[4], f"mean {data['base_mean']:.1f}°")
    show_error(axes[5], err_p, mask, COL_TITLES[5] if show_col_titles else None)
    annotate(axes[5], f"mean {data['poppy_mean']:.1f}°")


def add_error_colorbar(fig, rect):
    cax = fig.add_axes(rect)
    sm = ScalarMappable(norm=Normalize(0.0, ERR_VMAX), cmap="inferno")
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("angular error (degrees)", fontsize=9)
    cb.ax.tick_params(labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/workspace/datasets/TransSfP_poppy_pilot_512")
    ap.add_argument("--base_dir", default="/mnt/workspace/results/transsfp_poppy_pilot/base")
    ap.add_argument("--full_dir", default="/mnt/workspace/results/transsfp_poppy_pilot/full_official")
    ap.add_argument("--out", default="/mnt/workspace/results/transsfp_poppy_pilot")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    figs = os.path.join(args.out, "figs")
    os.makedirs(os.path.join(figs, "per_object"), exist_ok=True)

    frames = read_manifest(os.path.join(args.data, "manifest.csv"))
    base_log = read_log(os.path.join(args.base_dir, "log.csv"))
    full_log = read_log(os.path.join(args.full_dir, "log.csv"))

    by_obj = collections.defaultdict(list)
    records = []
    for frame in frames:
        name = frame["name"].removesuffix(".png")
        obj = frame["object"]
        mask = load_mask(os.path.join(args.data, "mask", frame["name"]))
        gt = load_normal(os.path.join(args.data, "normal", frame["name"]))
        base = np.load(os.path.join(args.base_dir, "outputs", name + ".npz"))["normals"]
        poppy = np.load(os.path.join(args.full_dir, "outputs", name + ".npz"))["normals"]
        eb = angular_error(base, gt)
        ep = angular_error(poppy, gt)
        data = {
            "img": load_gray(os.path.join(args.data, "pol000", frame["name"])),
            "gt": gt, "mask": mask, "base": base, "poppy": poppy,
            "err_base": eb, "err_poppy": ep,
            "base_mean": float(eb[mask].mean()),
            "poppy_mean": float(ep[mask].mean()),
        }
        # sanity: recomputed means should match Poppy's log
        assert abs(data["base_mean"] - base_log[name]) < 0.02, (name, data["base_mean"], base_log[name])
        assert abs(data["poppy_mean"] - full_log[name]) < 0.02, (name, data["poppy_mean"], full_log[name])

        records.append({
            "frame": name, "object": obj, "frame_index": int(frame["frame_index"]),
            "base_mean": data["base_mean"], "poppy_mean": data["poppy_mean"],
            "delta": data["poppy_mean"] - data["base_mean"],
        })
        by_obj[obj].append((frame, data))

    records.sort(key=lambda r: (OBJECTS.index(r["object"]), r["frame_index"]))
    with open(os.path.join(args.out, "per_frame_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for r in records:
            writer.writerow({
                "frame": r["frame"], "object": r["object"], "frame_index": r["frame_index"],
                "base_mean": f"{r['base_mean']:.4f}",
                "poppy_mean": f"{r['poppy_mean']:.4f}",
                "delta": f"{r['delta']:+.4f}",
            })

    # 1) Overview: middle frame of each object.
    nrow, ncol = len(OBJECTS), 6
    fig, axes = plt.subplots(nrow, ncol, figsize=(18.0, 15.5),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.10})
    for i, obj in enumerate(OBJECTS):
        frame, data = by_obj[obj][len(by_obj[obj]) // 2]
        render_frame_row(axes[i], frame, data, show_col_titles=(i == 0))
    fig.suptitle("TransSfP × Poppy pilot — one middle frame per object (7 objects × 3 frames)",
                 fontsize=16, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.055)
    add_error_colorbar(fig, [0.35, 0.018, 0.30, 0.010])
    out = os.path.join(figs, "overview.png")
    fig.savefig(out, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print("wrote", out)

    # 2) Per-object grids: all 3 frames.
    for obj in OBJECTS:
        rows = by_obj[obj]
        fig, axes = plt.subplots(len(rows), ncol, figsize=(18.0, 2.05 * len(rows) + 0.9),
                                 gridspec_kw={"wspace": 0.03, "hspace": 0.08})
        for r, (frame, data) in enumerate(rows):
            render_frame_row(axes[r], frame, data, show_col_titles=(r == 0))
        fig.suptitle(f"{obj} — 3 pilot frames", fontsize=15, y=0.995)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.10)
        add_error_colorbar(fig, [0.35, 0.035, 0.30, 0.012])
        out = os.path.join(figs, "per_object", obj + ".png")
        fig.savefig(out, dpi=args.dpi, facecolor="white")
        plt.close(fig)
        print("wrote", out)

    # 3) Object-level bars.
    summary = read_summary(os.path.join(args.out, "pilot_summary.csv"))
    base_vals = [float(summary[o]["base_mean"]) for o in OBJECTS]
    pop_vals = [float(summary[o]["poppy_mean"]) for o in OBJECTS]
    ts_vals = [float(summary[o]["transsfp_mean"]) for o in OBJECTS]
    x = np.arange(len(OBJECTS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    b1 = ax.bar(x - w, base_vals, w, label="MoGe-2 base", color="#9aa0a6")
    b2 = ax.bar(x, pop_vals, w, label="Poppy Full", color="#2f6fd0")
    b3 = ax.bar(x + w, ts_vals, w, label="TransSfP (trained)", color="#e07b00")
    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.6, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(OBJECTS, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("mean angular error (degrees, lower is better)")
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    ax.set_title("Object-level error: MoGe-2 base vs Poppy Full vs trained TransSfP", fontsize=13)
    all5 = summary["5 objects (excl. hemispheres)"]
    ax.text(0.99, 0.97,
            "5 objects (excl. hemispheres):\n"
            f"base {all5['base_mean']}° → Poppy {all5['poppy_mean']}° "
            f"({all5['poppy_minus_base']})\n"
            f"TransSfP {all5['transsfp_mean']}°",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="#cccccc"))
    fig.tight_layout()
    out = os.path.join(figs, "object_bars.png")
    fig.savefig(out, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print("wrote", out)

    # 4) Per-frame scatter.
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(OBJECTS)))
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for c, obj in zip(colors, OBJECTS):
        xs = [d["base_mean"] for _, d in by_obj[obj]]
        ys = [d["poppy_mean"] for _, d in by_obj[obj]]
        ax.scatter(xs, ys, s=55, color=c, label=obj, edgecolor="white", linewidth=0.8, zorder=3)
    lim = 80
    ax.plot([0, lim], [0, lim], "--", color="#666666", linewidth=1.2, label="no change")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("MoGe-2 base error (degrees)")
    ax.set_ylabel("Poppy Full error (degrees)")
    ax.set_title("Per-frame effect of Poppy refinement (21 frames)", fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.text(0.98, 0.02, "below diagonal = improved", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout()
    out = os.path.join(figs, "per_frame_scatter.png")
    fig.savefig(out, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()

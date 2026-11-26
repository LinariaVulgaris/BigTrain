#!/usr/bin/env python3
"""Summarize the 7-object x 3-frame TransSfP/Poppy pilot by object.

Reads the per-frame log.csv written by poppy/run.py for the MoGe-2 base and
official Full runs, groups frames by object, and prints/writes an object-level
table. TransSfP's reference means come from its per-frame metrics CSV, using
the same object-level averaging as the reproduction report.

Usage:
    python -I baselines/tools/summarize_transsfp_poppy_pilot.py \
        --base_dir results/transsfp_poppy_pilot/base \
        --full_dir results/transsfp_poppy_pilot/full_official \
        --transsfp_csv results/transsfp_eval/metrics_summary.csv \
        --out results/transsfp_poppy_pilot
"""
import argparse
import collections
import csv
import os

OBJECTS = [
    "hemi-sphere-big",
    "hemi-sphere-small",
    "bird-back",
    "cat-back",
    "cat-front",
    "tiny-cup",
    "tiny-cup-edges",
]


def read_poppy_log(path):
    """Return {frame_name: mean_deg} from a Poppy run.csv."""
    out = {}
    with open(os.path.join(path, "log.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row.get("mean_loss"):
                out[row["object"]] = float(row["mean_loss"])
    return out


def object_of(frame_name):
    # names are "<object>_<3-digit frame index>"
    return frame_name.rsplit("_", 1)[0]


def mean_by_object(frames):
    groups = collections.defaultdict(list)
    for name, value in frames.items():
        groups[object_of(name)].append(value)
    return {obj: sum(v) / len(v) for obj, v in groups.items()}, groups


def read_transsfp(path):
    with open(path, newline="") as f:
        return {
            row["object"]: float(row["mean_deg"])
            for row in csv.DictReader(f)
            if row["object"] in OBJECTS
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="results/transsfp_poppy_pilot/base")
    ap.add_argument("--full_dir", default="results/transsfp_poppy_pilot/full_official")
    ap.add_argument("--transsfp_csv", default="results/transsfp_eval/metrics_summary.csv")
    ap.add_argument("--out", default="results/transsfp_poppy_pilot")
    args = ap.parse_args()

    base_frames = read_poppy_log(args.base_dir)
    full_frames = read_poppy_log(args.full_dir)
    base_obj, base_groups = mean_by_object(base_frames)
    full_obj, full_groups = mean_by_object(full_frames)
    ts = read_transsfp(args.transsfp_csv)

    rows = []
    for obj in OBJECTS:
        n = len(full_groups.get(obj, []))
        b = base_obj.get(obj)
        p = full_obj.get(obj)
        t = ts.get(obj)
        rows.append({
            "object": obj,
            "n_frames": n,
            "base_mean": "" if b is None else f"{b:.2f}",
            "poppy_mean": "" if p is None else f"{p:.2f}",
            "poppy_minus_base": "" if (b is None or p is None) else f"{p - b:+.2f}",
            "transsfp_mean": "" if t is None else f"{t:.2f}",
            "poppy_minus_transsfp": "" if (p is None or t is None) else f"{p - t:+.2f}",
        })

    def agg(objs, label, n_total):
        b = sum(base_obj[o] for o in objs) / len(objs)
        p = sum(full_obj[o] for o in objs) / len(objs)
        t = sum(ts[o] for o in objs) / len(objs)
        return {
            "object": label,
            "n_frames": n_total,
            "base_mean": f"{b:.2f}",
            "poppy_mean": f"{p:.2f}",
            "poppy_minus_base": f"{p - b:+.2f}",
            "transsfp_mean": f"{t:.2f}",
            "poppy_minus_transsfp": f"{p - t:+.2f}",
        }

    rows.append(agg(OBJECTS, "All (7 objects)", sum(len(full_groups[o]) for o in OBJECTS)))
    non_hemi = [o for o in OBJECTS if not o.startswith("hemi-")]
    rows.append(agg(non_hemi, "5 objects (excl. hemispheres)", sum(len(full_groups[o]) for o in non_hemi)))

    out_csv = os.path.join(args.out, "pilot_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    header = "| Object | Frames | MoGe-2 base | Poppy Full | Δ vs base | TransSfP | Δ vs TransSfP |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['object']} | {r['n_frames']} | {r['base_mean']}° | {r['poppy_mean']}° | "
            f"{r['poppy_minus_base']} | {r['transsfp_mean']}° | {r['poppy_minus_transsfp']} |"
        )
    table = "\n".join(lines)
    out_md = os.path.join(args.out, "pilot_summary.md")
    with open(out_md, "w") as f:
        f.write("# TransSfP × Poppy 7-object × 3-frame pilot\n\n")
        f.write("Mean angular error (lower is better), averaged over 3 frames per object.\n\n")
        f.write(table + "\n")
    print(table)
    print(f"\nwrote {out_csv}\nwrote {out_md}")


if __name__ == "__main__":
    main()

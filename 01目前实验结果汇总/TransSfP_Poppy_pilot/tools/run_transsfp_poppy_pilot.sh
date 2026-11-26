#!/usr/bin/env bash
# Run the 7-object x 3-frame TransSfP pilot with Poppy + MoGe-2.
# Base (MoGe-2 only) is cheap and may already exist; Full is 100-step official.
set -euo pipefail

ROOT=/mnt/workspace
DATA=$ROOT/datasets/TransSfP_poppy_pilot_512
OUT=$ROOT/results/transsfp_poppy_pilot
LOG=$ROOT/logs/transsfp_poppy_pilot.log

mkdir -p "$OUT"
rm -f "$OUT/DONE" "$OUT/FAILED"
trap 'touch "$OUT/FAILED"' ERR

cd "$ROOT/poppy"
export PYTHONPATH="$ROOT/poppy"

if [ ! -s "$OUT/base/log.csv" ]; then
  echo "=== BASE (MoGe-2 only) ==="
  python run.py --backbone moge2 --gpu 0 \
    --data "$DATA" --output "$OUT/base" \
    --num_opt 1 --image_lr 0 --fs_lr 0 --normal_lr 0
fi

echo "=== FULL (official: image_lr=1e-5, fs_lr=1e-2, normal_lr=1e-3, 100 steps) ==="
python run.py --backbone moge2 --gpu 0 \
  --data "$DATA" --output "$OUT/full_official" \
  --num_opt 100 --image_lr 1e-5 --fs_lr 1e-2 --normal_lr 1e-3

cd "$ROOT"
python -I baselines/tools/summarize_transsfp_poppy_pilot.py \
  --base_dir "$OUT/base" \
  --full_dir "$OUT/full_official" \
  --transsfp_csv "$ROOT/results/transsfp_eval/metrics_summary.csv" \
  --out "$OUT"

touch "$OUT/DONE"
echo "=== DONE ==="

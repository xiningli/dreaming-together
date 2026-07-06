#!/bin/bash
# Multi-seed bring-up grid: 3 seeds x {C, B}, sequential, best-by-G6.
# Rationale: diffusion bring-up variance dominates single runs (C's A2
# hit 0.83 and 0.33 on identical configs). Uniform seeds+selection for
# both conditions preserves integrity.
set -u
cd "$(dirname "$0")/.."
for cond in C B; do
  for seed in 1 2 3; do
    echo "=== bring-up $cond seed $seed ==="
    python -m dreaming_together.training.stage2_diffusion \
      --condition $cond --seed $seed --workers 8
    echo "--- $cond s$seed done (exit $?) ---"
  done
done
echo "GRID COMPLETE"
for cond in C B; do
  for seed in 1 2 3; do
    f=runs/stage2_${cond}_diff_s${seed}/G6_RESULT_v2.json
    [ -f "$f" ] && echo "$cond s$seed: $(cat $f | python -c 'import json,sys; d=json.load(sys.stdin); print(d)')"
  done
done

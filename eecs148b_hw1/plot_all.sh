#!/usr/bin/env bash
# Generate all required plots after training runs complete.
set -euo pipefail
mkdir -p plots

echo "=== Plot 1: Baseline learning curve ==="
uv run eecs148b_hw1/plot_curves.py \
    --runs runs/baseline runs/baseline_wr \
    --labels "Baseline (lr=3e-4)" "Warm-restart (lr=5e-4)" \
    --title "Baseline Training — reaching val_loss ≤ 2.0" \
    --out plots/baseline_curve.png

echo "=== Plot 2: LayerNorm ablation ==="
uv run eecs148b_hw1/plot_curves.py \
    --runs runs/no_layernorm_high_lr runs/no_layernorm_low_lr runs/baseline \
    --labels "No-LN lr=3e-4 (unstable)" "No-LN lr=3e-5 (stable)" "Baseline (with LN)" \
    --title "Ablation 1: Effect of LayerNorm" \
    --out plots/layernorm_ablation.png \
    --no-target

echo "=== Plot 3: NoPE vs Sinusoidal PE ==="
uv run eecs148b_hw1/plot_curves.py \
    --runs runs/baseline runs/nope \
    --labels "Sinusoidal PE (baseline)" "NoPE" \
    --title "Ablation 2: Sinusoidal PE vs NoPE" \
    --out plots/nope_ablation.png

echo "=== Plots saved to plots/ ==="
ls -lh plots/

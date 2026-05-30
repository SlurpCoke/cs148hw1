#!/usr/bin/env bash
# Ablation experiments — focused runs to show learning curve trends.
set -euo pipefail
DEVICE=mps

echo "====== WARM RESTART: higher LR from best checkpoint ======"
# Load model weights only (reset optimizer), use lr=5e-4 to push past 2.0
uv run eecs148b_hw1/train.py \
    --checkpoint runs/baseline/best.pt \
    --reset-optimizer \
    --log-dir runs/baseline_wr \
    --device $DEVICE \
    --batch-size 64 --total-steps 2000 \
    --lr 5e-4 --min-lr 5e-5 --warmup-steps 100 \
    --val-interval 200 --log-interval 50 --save-interval 9999 --val-batches 20

echo "====== NO-LN  lr=3e-4 (same as baseline — expect instability) ======"
# 400 steps is enough to see whether training destabilises
uv run eecs148b_hw1/train.py \
    --log-dir runs/no_layernorm_high_lr \
    --no-layernorm \
    --device $DEVICE \
    --batch-size 64 --total-steps 400 \
    --lr 3e-4 --min-lr 3e-5 --warmup-steps 100 \
    --val-interval 200 --log-interval 50 --save-interval 9999 --val-batches 10

echo "====== NO-LN  lr=3e-5 (lower LR for stability) ======"
uv run eecs148b_hw1/train.py \
    --log-dir runs/no_layernorm_low_lr \
    --no-layernorm \
    --device $DEVICE \
    --batch-size 64 --total-steps 1500 \
    --lr 3e-5 --min-lr 3e-6 --warmup-steps 100 \
    --val-interval 250 --log-interval 50 --save-interval 9999 --val-batches 10

echo "====== NOPE (No Positional Embeddings) ======"
uv run eecs148b_hw1/train.py \
    --log-dir runs/nope \
    --no-pos-emb \
    --device $DEVICE \
    --batch-size 64 --total-steps 2500 \
    --lr 3e-4 --min-lr 3e-5 --warmup-steps 200 \
    --val-interval 250 --log-interval 50 --save-interval 9999 --val-batches 20

echo "====== ALL ABLATIONS DONE ======"

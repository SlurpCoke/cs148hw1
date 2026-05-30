#!/usr/bin/env bash
# Run all experiments sequentially.  Each prints logs to stdout which is captured.
set -euo pipefail
DEVICE=mps

echo "====== BASELINE ======"
uv run eecs148b_hw1/train.py \
    --log-dir runs/baseline \
    --device $DEVICE \
    --batch-size 64 --total-steps 2500 \
    --lr 3e-4 --min-lr 3e-5 --warmup-steps 200 \
    --val-interval 250 --log-interval 50 --save-interval 500 --val-batches 20

echo "====== NO-LN  lr=3e-4 (same as baseline) ======"
uv run eecs148b_hw1/train.py \
    --log-dir runs/no_layernorm_high_lr \
    --no-layernorm \
    --device $DEVICE \
    --batch-size 64 --total-steps 1000 \
    --lr 3e-4 --min-lr 3e-5 --warmup-steps 200 \
    --val-interval 200 --log-interval 50 --save-interval 999 --val-batches 10

echo "====== NO-LN  lr=3e-5 (lower LR for stability) ======"
uv run eecs148b_hw1/train.py \
    --log-dir runs/no_layernorm_low_lr \
    --no-layernorm \
    --device $DEVICE \
    --batch-size 64 --total-steps 2000 \
    --lr 3e-5 --min-lr 3e-6 --warmup-steps 200 \
    --val-interval 250 --log-interval 50 --save-interval 999 --val-batches 10

echo "====== NOPE (No Positional Embeddings) ======"
uv run eecs148b_hw1/train.py \
    --log-dir runs/nope \
    --no-pos-emb \
    --device $DEVICE \
    --batch-size 64 --total-steps 2500 \
    --lr 3e-4 --min-lr 3e-5 --warmup-steps 200 \
    --val-interval 250 --log-interval 50 --save-interval 500 --val-batches 20

echo "====== ALL EXPERIMENTS DONE ======"

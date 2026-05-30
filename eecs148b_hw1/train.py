"""
Problem (training together): Training loop for TransformerLM on TinyStories.

Usage:
    uv run eecs148b_hw1/train.py [OPTIONS]

Examples:
    # Baseline run
    uv run eecs148b_hw1/train.py --log-dir runs/baseline

    # LayerNorm ablation
    uv run eecs148b_hw1/train.py --no-layernorm --log-dir runs/no_layernorm

    # NoPE ablation
    uv run eecs148b_hw1/train.py --no-pos-emb --log-dir runs/no_pos_emb
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from eecs148b_hw1.model import TransformerLM


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup followed by cosine annealing to min_lr."""
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ── Loss ──────────────────────────────────────────────────────────────────────

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Numerically stable cross-entropy loss. logits: (N, V), targets: (N,)."""
    max_vals = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_vals
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1)) + max_vals.squeeze(-1)
    target_logits = logits[torch.arange(logits.shape[0], device=logits.device), targets]
    return (log_sum_exp - target_logits).mean()


def perplexity(loss: float) -> float:
    return math.exp(loss)


# ── Data ──────────────────────────────────────────────────────────────────────

def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (inputs, targets) from a token array."""
    starts = np.random.randint(0, len(data) - context_length, size=batch_size)
    x = np.stack([np.array(data[s : s + context_length]) for s in starts])
    y = np.stack([np.array(data[s + 1 : s + context_length + 1]) for s in starts])
    return (
        torch.from_numpy(x).long().to(device),
        torch.from_numpy(y).long().to(device),
    )


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str | Path,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    config: dict,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0), ckpt.get("val_loss", float("inf"))


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_val_loss(
    model: TransformerLM,
    val_data: np.ndarray,
    context_length: int,
    batch_size: int,
    num_batches: int,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        total += cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return total / num_batches


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = args.device

    # Memory-mapped data loading
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")
    print(f"Train tokens: {len(train_data):,}   Val tokens: {len(val_data):,}")

    # Model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        use_layernorm=not args.no_layernorm,
        use_pos_emb=not args.no_pos_emb,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Optimizer (torch.optim usage explicitly allowed in §5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    best_val_loss = float("inf")
    if args.checkpoint and Path(args.checkpoint).exists():
        if getattr(args, "reset_optimizer", False):
            # Load model weights only; fresh optimizer (warm restart)
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            start_step = 0
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"Warm restart from checkpoint (model only), prev val_loss={best_val_loss:.4f}")
        else:
            start_step, best_val_loss = load_checkpoint(args.checkpoint, model, optimizer)
            print(f"Resumed from step {start_step}, best_val_loss={best_val_loss:.4f}")

    # Experiment log
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args)
    log_path = log_dir / "run_log.jsonl"

    # Write config once
    with open(log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    model.train()
    t0 = time.time()

    for step in range(start_step, args.total_steps):
        # Update LR
        lr = get_lr(step, args.warmup_steps, args.total_steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Forward + backward
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Logging
        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            log_entry: dict = {
                "step": step + 1,
                "train_loss": round(loss.item(), 5),
                "lr": lr,
                "wall_time": round(elapsed, 1),
            }

            if (step + 1) % args.val_interval == 0:
                val_loss = estimate_val_loss(
                    model, val_data, args.context_length,
                    args.batch_size, args.val_batches, device,
                )
                val_ppl = perplexity(val_loss)
                log_entry["val_loss"] = round(val_loss, 5)
                log_entry["val_ppl"] = round(val_ppl, 3)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(log_dir / "best.pt", model, optimizer, step + 1, val_loss, config)
                    log_entry["saved_best"] = True

                print(
                    f"step={step+1:5d}  train={loss.item():.4f}  val={val_loss:.4f}"
                    f"  ppl={val_ppl:.2f}  lr={lr:.2e}  t={elapsed:.0f}s"
                )
            else:
                print(f"step={step+1:5d}  train={loss.item():.4f}  lr={lr:.2e}  t={elapsed:.0f}s")

            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        # Periodic checkpoint
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(
                log_dir / f"ckpt_{step+1:06d}.pt",
                model, optimizer, step + 1, loss.item(), config,
            )

    # Final checkpoint
    save_checkpoint(log_dir / "final.pt", model, optimizer, args.total_steps, loss.item(), config)
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}  ppl={perplexity(best_val_loss):.2f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train TransformerLM on TinyStories")

    # Data / IO
    p.add_argument("--train-data", default="data/train_tokens.npy")
    p.add_argument("--val-data", default="data/valid_tokens.npy")
    p.add_argument("--log-dir", default="runs/baseline")
    p.add_argument("--checkpoint", default=None, help="Path to resume checkpoint")
    p.add_argument("--reset-optimizer", action="store_true",
                   help="Load model weights only; reset optimizer for a warm restart")

    # Model architecture
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=2048)

    # Ablation flags
    p.add_argument("--no-layernorm", action="store_true", help="Remove LayerNorm (ablation 1)")
    p.add_argument("--no-pos-emb", action="store_true", help="Remove positional embeddings / NoPE (ablation 2)")

    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--total-steps", type=int, default=2500,
                   help="64 * 2500 * 256 = 40,960,000 tokens processed")
    p.add_argument("--device", default="cpu")

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Logging / checkpointing
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--val-interval", type=int, default=250)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=500)

    return p


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()

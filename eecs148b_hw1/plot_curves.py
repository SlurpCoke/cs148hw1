"""
Plot training and validation loss curves from run_log.jsonl files.

Usage:
    uv run eecs148b_hw1/plot_curves.py --runs runs/baseline runs/no_layernorm runs/nope --out plots/curves.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(run_dir: Path) -> dict[str, list]:
    log_path = run_dir / "run_log.jsonl"
    steps, train_losses, val_losses = [], [], []
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            steps.append(entry["step"])
            train_losses.append(entry["train_loss"])
            if "val_loss" in entry:
                val_losses.append((entry["step"], entry["val_loss"]))
    return {"steps": steps, "train": train_losses, "val": val_losses}


def plot_runs(
    run_dirs: list[Path],
    labels: list[str],
    title: str,
    out_path: Path,
    target_val_loss: float | None = 2.0,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        if not (run_dir / "run_log.jsonl").exists():
            print(f"  [skip] {run_dir} — no log yet")
            continue
        data = load_log(run_dir)
        c = colors[i % len(colors)]

        # Training loss (smoothed with EMA)
        ema, alpha = [], 0.9
        s = None
        for v in data["train"]:
            s = v if s is None else alpha * s + (1 - alpha) * v
            ema.append(s)
        axes[0].plot(data["steps"], ema, color=c, label=label, linewidth=1.5)

        # Validation loss
        if data["val"]:
            vsteps, vlosses = zip(*data["val"])
            axes[1].plot(vsteps, vlosses, "o-", color=c, label=label, linewidth=1.5, markersize=4)

    if target_val_loss is not None:
        axes[1].axhline(target_val_loss, color="gray", linestyle="--", linewidth=1, label=f"target={target_val_loss}")

    for ax, ylabel, ttl in zip(axes, ["Loss", "Loss"], ["Train loss (EMA)", "Validation loss"]):
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(ttl)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--title", default="Loss curves")
    p.add_argument("--out", default="plots/curves.png")
    p.add_argument("--no-target", action="store_true")
    args = p.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    labels = args.labels if args.labels else [r.name for r in run_dirs]
    target = None if args.no_target else 2.0

    plot_runs(run_dirs, labels, args.title, Path(args.out), target_val_loss=target)


if __name__ == "__main__":
    main()

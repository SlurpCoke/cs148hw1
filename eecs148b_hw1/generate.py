"""
Problem (decoding): Text generation from a trained TransformerLM.

Supports temperature scaling and top-p (nucleus) sampling.

Usage:
    uv run eecs148b_hw1/generate.py --checkpoint runs/baseline/best.pt --prompt "Once upon a time"
    uv run eecs148b_hw1/generate.py --checkpoint runs/baseline/best.pt --temperature 0.8 --top-p 0.95
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from eecs148b_hw1.model import TransformerLM
from eecs148b_hw1.tokenizer import Tokenizer

VOCAB_PATH = "data/tinystories_vocab.json"
MERGES_PATH = "data/tinystories_merges.txt"
SPECIAL_TOKENS = ["<|endoftext|>"]
EOT = "<|endoftext|>"


# ── Sampling helpers ──────────────────────────────────────────────────────────

def _softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature and return a probability distribution."""
    scaled = logits / temperature
    scaled = scaled - scaled.max()
    exp_scaled = torch.exp(scaled)
    return exp_scaled / exp_scaled.sum()


def _top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: zero out tokens outside the smallest set whose
    cumulative probability >= p, then renormalize.
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Shift right: keep the token that first crosses the threshold
    mask = (cumulative - sorted_probs) >= p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()
    # Scatter back to original ordering
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_indices, sorted_probs)
    return filtered


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_id: int | None = None,
    device: str = "cpu",
) -> list[int]:
    """
    Autoregressively sample from the model.

    Args:
        model:          Trained TransformerLM (already on device).
        prompt_ids:     Token IDs for the prompt prefix.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature:    Softmax temperature (< 1 = sharper, > 1 = flatter).
        top_p:          Nucleus-sampling threshold in (0, 1].  1.0 = no filtering.
        eot_id:         Token ID of <|endoftext|>; generation stops on this token.
        device:         PyTorch device string.

    Returns:
        Full token ID list (prompt + generated).
    """
    model.eval()
    tokens = list(prompt_ids)
    context_length = model.context_length

    for _ in range(max_new_tokens):
        # Truncate to context window
        context = tokens[-context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)             # (1, seq_len, vocab_size)
        logits = logits[0, -1, :]     # last position logits: (vocab_size,)

        probs = _softmax_with_temperature(logits, temperature)
        if top_p < 1.0:
            probs = _top_p_filter(probs, top_p)

        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_token)

        if eot_id is not None and next_token == eot_id:
            break

    return tokens


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Generate text from a trained TransformerLM")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--vocab", default=VOCAB_PATH)
    p.add_argument("--merges", default=MERGES_PATH)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # Tokenizer
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=SPECIAL_TOKENS)
    eot_id = tokenizer.encode(EOT)[0]

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]

    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        use_layernorm=not cfg.get("no_layernorm", False),
        use_pos_emb=not cfg.get("no_pos_emb", False),
    ).to(args.device)
    model.load_state_dict(ckpt["model"])

    step = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"Loaded checkpoint: step={step}  val_loss={val_loss:.4f}")
    print(f"Temperature={args.temperature}  top_p={args.top_p}\n")

    prompt_ids = tokenizer.encode(args.prompt)
    output_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eot_id=eot_id,
        device=args.device,
    )

    print(tokenizer.decode(output_ids))
    n_new = len(output_ids) - len(prompt_ids)
    print(f"\n[{n_new} new tokens generated]")


if __name__ == "__main__":
    main()

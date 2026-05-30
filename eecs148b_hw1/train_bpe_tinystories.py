"""
Train a byte-level BPE tokenizer on the TinyStories dataset.

Usage:
    uv run eecs148b_hw1/train_bpe_tinystories.py \
        --input  data/TinyStoriesV2-GPT4-train.txt \
        --output data/
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from eecs148b_hw1.tokenizer import train_bpe

SPECIAL_TOKENS = ["<|endoftext|>"]
VOCAB_SIZE = 10_000


def save_vocab(vocab: dict[int, bytes], path: Path) -> None:
    """Save vocab as a JSON file mapping token-string → token-ID (GPT-2 style)."""
    # Use a printable representation for inspection; store raw hex bytes for safety.
    serialisable = {token_bytes.hex(): token_id for token_id, token_bytes in vocab.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, ensure_ascii=False, indent=2)


def save_merges(merges: list[tuple[bytes, bytes]], path: Path) -> None:
    """Save merges as a text file, one merge per line (hex-encoded bytes)."""
    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--output", default="data/")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Download with:\n"
            "  mkdir -p data && cd data\n"
            "  wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
        )

    print(f"Training BPE on {input_path} (vocab_size={VOCAB_SIZE}) …")
    t0 = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )
    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.1f}s  |  {len(vocab)} tokens  |  {len(merges)} merges")

    vocab_path = output_dir / "tinystories_vocab.json"
    merges_path = output_dir / "tinystories_merges.txt"
    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)
    print(f"Saved vocab  → {vocab_path}")
    print(f"Saved merges → {merges_path}")

    # Report the longest token
    longest_id, longest_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    print(f"\nLongest token  id={longest_id}  len={len(longest_bytes)}  bytes={longest_bytes!r}")
    try:
        print(f"  decoded: {longest_bytes.decode('utf-8')!r}")
    except UnicodeDecodeError:
        print("  (not valid UTF-8)")


if __name__ == "__main__":
    main()

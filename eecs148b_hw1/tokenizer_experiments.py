"""
Problem (tokenizer_experiments): Experiments with tokenizers.

(a) Sample 10 documents from TinyStories, encode them, report compression ratio.
(b) Encode full train + validation splits and serialize as uint16 NumPy arrays.

Usage:
    uv run eecs148b_hw1/tokenizer_experiments.py
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np

from eecs148b_hw1.tokenizer import Tokenizer

VOCAB_PATH = Path("data/tinystories_vocab.json")
MERGES_PATH = Path("data/tinystories_merges.txt")
TRAIN_PATH = Path("data/TinyStoriesV2-GPT4-train.txt")
VALID_PATH = Path("data/TinyStoriesV2-GPT4-valid.txt")
SPECIAL_TOKENS = ["<|endoftext|>"]

EOT = "<|endoftext|>"


def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_files(
        vocab_filepath=str(VOCAB_PATH),
        merges_filepath=str(MERGES_PATH),
        special_tokens=SPECIAL_TOKENS,
    )


# ── (a) Compression ratio on 10 sampled documents ─────────────────────────────

def sample_documents(path: Path, n: int = 10, seed: int = 42) -> list[str]:
    """Split corpus on <|endoftext|> and return n randomly sampled documents."""
    print(f"Reading {path} …")
    text = path.read_text(encoding="utf-8")
    docs = [d.strip() for d in text.split(EOT) if d.strip()]
    rng = random.Random(seed)
    return rng.sample(docs, min(n, len(docs)))


def compression_ratio_experiment(tokenizer: Tokenizer) -> None:
    print("\n── (a) Compression ratio on 10 sampled documents ──────────────────")
    docs = sample_documents(TRAIN_PATH, n=10)

    total_bytes = 0
    total_tokens = 0
    for i, doc in enumerate(docs):
        doc_bytes = len(doc.encode("utf-8"))
        ids = tokenizer.encode(doc)
        n_tokens = len(ids)
        ratio = doc_bytes / n_tokens if n_tokens else float("inf")
        print(f"  doc {i+1:2d}: {doc_bytes:6d} bytes  {n_tokens:5d} tokens  ratio={ratio:.2f}")
        total_bytes += doc_bytes
        total_tokens += n_tokens

    overall = total_bytes / total_tokens if total_tokens else 0
    print(f"\n  Total: {total_bytes} bytes / {total_tokens} tokens")
    print(f"  Overall compression ratio: {overall:.3f} bytes/token")


# ── (b) Encode full splits and save as uint16 arrays ──────────────────────────

def encode_split(tokenizer: Tokenizer, path: Path, out_path: Path) -> None:
    print(f"\nEncoding {path.name} …")
    t0 = time.time()
    all_ids: list[int] = []

    with open(path, encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            all_ids.append(token_id)

    elapsed = time.time() - t0
    arr = np.array(all_ids, dtype=np.uint16)
    np.save(out_path, arr)

    print(f"  {len(all_ids):,} tokens  |  max id = {arr.max()}  |  {elapsed:.1f}s")
    print(f"  Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  dtype=uint16 uses {arr.nbytes / 1e6:.1f} MB  (vs {arr.nbytes * 2 / 1e6:.1f} MB for uint32)")


def encoding_experiment(tokenizer: Tokenizer) -> None:
    print("\n── (b) Encode full train + validation splits ───────────────────────")
    print(
        "Why uint16? Vocab size is 10,000, which fits in uint16 (max 65,535).\n"
        "uint16 uses 2 bytes/token vs 4 for int32, halving storage and memory."
    )
    encode_split(tokenizer, TRAIN_PATH, Path("data/train_tokens.npy"))
    encode_split(tokenizer, VALID_PATH, Path("data/valid_tokens.npy"))


def main() -> None:
    tokenizer = load_tokenizer()
    compression_ratio_experiment(tokenizer)
    encoding_experiment(tokenizer)
    print("\nDone.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    Split text on special tokens first (so no merge crosses a special token boundary),
    then apply the GPT-2 regex pre-tokenizer to each chunk. Returns a mapping from
    each pre-token (as a tuple of single-byte bytes objects) to its frequency.
    """
    if special_tokens:
        # Sort longest-first so overlapping specials (e.g. "<|eot|>" vs "<|eot|><|eot|>") match correctly
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        split_pat = "|".join(regex.escape(st) for st in sorted_specials)
        chunks = regex.split(split_pat, text)
    else:
        chunks = [text]

    counts: dict[tuple[bytes, ...], int] = {}
    for chunk in chunks:
        for match in regex.finditer(PAT, chunk):
            # Represent each pre-token as a tuple of individual UTF-8 bytes
            byte_seq: tuple[bytes, ...] = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            counts[byte_seq] = counts.get(byte_seq, 0) + 1

    return counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the text at input_path.

    Returns:
        vocab:  mapping from token ID → token bytes
        merges: list of (bytes_a, bytes_b) pairs in the order they were merged
    """
    # ── 1. Initialise vocabulary with every possible byte ──────────────────────
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ── 2. Pre-tokenise the corpus ─────────────────────────────────────────────
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    word_freq = pretokenize(text, special_tokens)

    # Represent each unique pre-token as a mutable list of bytes objects so we
    # can replace individual elements during merging.
    word_items = list(word_freq.items())
    words: list[list[bytes]] = [list(seq) for seq, _ in word_items]
    word_counts: list[int] = [cnt for _, cnt in word_items]

    # ── 3. Build initial pair-frequency index ──────────────────────────────────
    # pair_counts[p]   = total weighted frequency of adjacent pair p across all words
    # pair_to_words[p] = set of word indices whose current token list contains p
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = {}

    for i, word in enumerate(words):
        freq = word_counts[i]
        for j in range(len(word) - 1):
            pair = (word[j], word[j + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(i)

    # ── 4. Iteratively merge the most-frequent pair ────────────────────────────
    merges: list[tuple[bytes, bytes]] = []

    # vocab_size = 256 initial bytes + num_merges + len(special_tokens)
    num_merges = max(0, vocab_size - 256 - len(special_tokens))

    for _ in range(num_merges):
        if not pair_counts:
            break

        # Highest count wins; ties broken by lexicographically greater pair (max()).
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        a, b = best_pair
        ab = a + b

        vocab[len(vocab)] = ab
        merges.append((a, b))

        # Update every word that contains the merged pair.
        for i in list(pair_to_words.get(best_pair, set())):
            word = words[i]
            freq = word_counts[i]

            # Count every pair present in the current (pre-merge) word.
            old_pair_cnts: dict[tuple[bytes, bytes], int] = {}
            for j in range(len(word) - 1):
                p = (word[j], word[j + 1])
                old_pair_cnts[p] = old_pair_cnts.get(p, 0) + 1

            # Build the new word, merging every occurrence of (a, b) → ab.
            new_word: list[bytes] = []
            j = 0
            while j < len(word):
                if j + 1 < len(word) and word[j] == a and word[j + 1] == b:
                    new_word.append(ab)
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            words[i] = new_word

            # Count every pair present in the updated word.
            new_pair_cnts: dict[tuple[bytes, bytes], int] = {}
            for j in range(len(new_word) - 1):
                p = (new_word[j], new_word[j + 1])
                new_pair_cnts[p] = new_pair_cnts.get(p, 0) + 1

            # Remove old pair contributions from the global index.
            for p, cnt in old_pair_cnts.items():
                pair_counts[p] -= cnt * freq
                if pair_counts[p] == 0:
                    del pair_counts[p]
                if p in pair_to_words:
                    pair_to_words[p].discard(i)
                    if not pair_to_words[p]:
                        del pair_to_words[p]

            # Add new pair contributions to the global index.
            for p, cnt in new_pair_cnts.items():
                pair_counts[p] = pair_counts.get(p, 0) + cnt * freq
                if p not in pair_to_words:
                    pair_to_words[p] = set()
                pair_to_words[p].add(i)

    # ── 5. Append special tokens (they do not influence BPE training) ──────────
    existing_bytes = set(vocab.values())
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        if st_bytes not in existing_bytes:
            vocab[len(vocab)] = st_bytes
            existing_bytes.add(st_bytes)

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges = merges

        # Reverse mapping: bytes → token ID
        self._bytes_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # Merge rank: (bytes_a, bytes_b) → index in merges list (lower = earlier)
        self._merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

        # Add any special tokens that aren't already in the vocab
        for st in (special_tokens or []):
            st_bytes = st.encode("utf-8")
            if st_bytes not in self._bytes_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = st_bytes
                self._bytes_to_id[st_bytes] = new_id

        # Pre-compile a splitting pattern for special tokens.
        # Sorting longest-first ensures overlapping specials (e.g. "<|x|><|x|>" vs "<|x|>")
        # are matched greedily by the longer one.
        self._special_tokens: list[str] = list(special_tokens or [])
        if self._special_tokens:
            sorted_specials = sorted(self._special_tokens, key=len, reverse=True)
            # Capturing group keeps the delimiters in the split result.
            self._split_pat: regex.Pattern | None = regex.compile(
                "(" + "|".join(regex.escape(st) for st in sorted_specials) + ")"
            )
            self._special_set: set[str] = set(self._special_tokens)
        else:
            self._split_pat = None
            self._special_set = set()

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Load a Tokenizer from the hex-encoded vocab/merges files written by train_bpe_tinystories."""
        with open(vocab_filepath, encoding="utf-8") as f:
            raw = json.load(f)
        vocab: dict[int, bytes] = {v: bytes.fromhex(k) for k, v in raw.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((bytes.fromhex(parts[0]), bytes.fromhex(parts[1])))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    # ── internal helpers ───────────────────────────────────────────────────────

    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        """
        Greedily apply BPE merges to a list of single-byte tokens.

        At each step, find the adjacent pair with the lowest merge rank
        (= the merge that was learned first during training) and merge it.
        Repeat until no more merges apply.
        """
        while len(tokens) > 1:
            best_rank = len(self.merges)  # sentinel: larger than any valid rank
            best_i = -1
            for i in range(len(tokens) - 1):
                rank = self._merge_rank.get((tokens[i], tokens[i + 1]), len(self.merges))
                if rank < best_rank:
                    best_rank = rank
                    best_i = i

            if best_i == -1:
                break

            merged = tokens[best_i] + tokens[best_i + 1]
            tokens = tokens[:best_i] + [merged] + tokens[best_i + 2:]

        return tokens

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode a plain-text chunk (guaranteed to contain no special tokens)."""
        ids: list[int] = []
        for match in regex.finditer(PAT, text):
            word_bytes = list(bytes([b]) for b in match.group().encode("utf-8"))
            for token in self._apply_merges(word_bytes):
                ids.append(self._bytes_to_id[token])
        return ids

    # ── public API ─────────────────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        if not text:
            return []

        if self._split_pat is None:
            return self._encode_chunk(text)

        ids: list[int] = []
        for piece in self._split_pat.split(text):
            if not piece:
                continue
            if piece in self._special_set:
                ids.append(self._bytes_to_id[piece.encode("utf-8")])
            else:
                ids.extend(self._encode_chunk(piece))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings (e.g. an open file handle), yielding
        one token ID at a time without materialising the full token sequence.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

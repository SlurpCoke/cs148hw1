"""
Transformer Language Model — built from scratch.

No torch.nn or torch.nn.functional ops are used except:
  - torch.nn.Parameter
  - torch.nn container classes (Module, ModuleList)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── Linear ────────────────────────────────────────────────────────────────────

class Linear(nn.Module):
    """y = x W^T  (no bias).  Stores W of shape (out_features, in_features)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


# ── Embedding ─────────────────────────────────────────────────────────────────

class Embedding(nn.Module):
    """Token-ID → dense vector lookup.  weight shape: (num_embeddings, embedding_dim)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


# ── LayerNorm ─────────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """Standard LayerNorm normalising the last dimension."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        result = x_norm * self.weight + self.bias
        return result.to(in_dtype)


# ── Feed-Forward Network ──────────────────────────────────────────────────────

class FFN(nn.Module):
    """Position-wise 2-layer ReLU FFN: FFN(x) = W2 ReLU(W1 x)."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.fc1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.fc2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = h * (h > 0)  # ReLU without torch.relu / nn.functional.relu
        return self.fc2(h)


# ── Sinusoidal Positional Encoding ────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional embeddings (Vaswani et al. 2017).
    PE(p, 2i)   = sin(p / 10000^(2i/d_model))
    PE(p, 2i+1) = cos(p / 10000^(2i/d_model))
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1).float()  # (L, 1)
        i = torch.arange(d_model // 2, device=device).float()                      # (d/2,)
        div = torch.exp(-i * (2.0 * math.log(10000.0) / d_model))                  # (d/2,)

        pe = torch.zeros(max_seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, token_positions: torch.Tensor) -> torch.Tensor:
        return self.pe[token_positions]


# ── Softmax ───────────────────────────────────────────────────────────────────

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax: subtract the max before exponentiating."""
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


# ── Scaled Dot-Product Attention ─────────────────────────────────────────────

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Q, K: (..., seq, d_k)   V: (..., seq, d_v)
    mask: (..., queries, keys)  True = attend, False = block
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (..., queries, keys)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    return softmax(scores, dim=-1) @ V


# ── Multi-Head Self-Attention ─────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention (Vaswani et al. 2017, §3.2.2)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch_dims, seq_len, _ = x.shape

        Q = self.q_proj(x)   # (..., seq, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads: (..., seq, d_model) → (..., heads, seq, d_k)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            *b, s, _ = t.shape
            return t.view(*b, s, self.num_heads, self.d_k).transpose(-3, -2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Causal mask: lower-triangular, shape (seq, seq)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        # (..., heads, seq, d_k)
        attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Merge heads: (..., heads, seq, d_k) → (..., seq, d_model)
        attn_out = attn_out.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, self.d_model)

        return self.output_proj(attn_out)


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block:
        y = x + MHA(LN1(x))
        y = y + FFN(LN2(y))
    When use_layernorm=False the LayerNorms are omitted (ablation).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device=None,
        dtype=None,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln1 = LayerNorm(d_model, device=device, dtype=dtype)
            self.ln2 = LayerNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ffn = FFN(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layernorm:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            x = x + self.attn(x)
            x = x + self.ffn(x)
        return x


# ── Transformer Language Model ────────────────────────────────────────────────

class TransformerLM(nn.Module):
    """
    Decoder-only Transformer language model:
        token_embeddings → (+ sinusoidal_PE) → N × TransformerBlock → LayerNorm → LM head

    use_pos_emb=False removes positional embeddings (NoPE ablation).
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        device=None,
        dtype=None,
        use_layernorm: bool = True,
        use_pos_emb: bool = True,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.use_pos_emb = use_pos_emb
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        if use_pos_emb:
            self.pos_enc = SinusoidalPositionalEncoding(d_model, context_length, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype, use_layernorm=use_layernorm)
             for _ in range(num_layers)]
        )
        self.ln_final = LayerNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        # in_indices: (batch, seq_len)
        seq_len = in_indices.shape[-1]
        x = self.token_embeddings(in_indices)                                   # (batch, seq, d_model)
        if self.use_pos_emb:
            positions = torch.arange(seq_len, device=in_indices.device)
            x = x + self.pos_enc(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)                                                  # (batch, seq, vocab_size)

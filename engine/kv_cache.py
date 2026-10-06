"""KV caches.

Why a KV cache at all: generating token n needs attention over the keys/values of tokens
0..n-1. Without a cache we would re-run the whole prefix through every layer for each new
token (O(n^2) total work). With it, each step only computes K/V for the one new token and
appends them, so decode does O(n) attention work per token and O(1) projection work.

Phase 1: `KVCache`, a contiguous, preallocated cache for a batch of sequences that advance
in lockstep. Phase 4 adds a paged (block-based) cache next to it.
"""

from __future__ import annotations

import torch

from engine.config import ModelConfig


class KVCache:
    """Contiguous preallocated KV cache.

    Storage: one tensor each for K and V, shaped

        [num_layers, batch, num_kv_heads, max_seq_len, head_dim]

    Design choices:
    * **Preallocate to max_seq_len** (instead of `torch.cat`-ing a growing tensor every step,
      which is what HF's DynamicCache does). Appending is then a slice write: no reallocation,
      no copying old entries (cat would copy the whole history every step: O(n^2) bytes moved).
      The cost: every sequence reserves max_seq_len slots up front, even if it stops after
      20 tokens. That waste is exactly what the paged cache in Phase 4 eliminates.
    * **Heads before sequence** ([.., H, S, D]) is the layout attention kernels consume, so
      reading the cache needs no transpose: `k[layer, :, :, :n]` is passed straight to SDPA.
    * **One tensor for all layers** = two allocations total; per-layer access is a view.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if max_seq_len > cfg.max_position_embeddings:
            raise ValueError(
                f"max_seq_len {max_seq_len} exceeds the model's max_position_embeddings "
                f"({cfg.max_position_embeddings})"
            )
        shape = (cfg.num_layers, batch_size, cfg.num_kv_heads, max_seq_len, cfg.head_dim)
        # torch.empty, not zeros: slots beyond seq_len are never read, so no need to memset.
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty(shape, dtype=dtype, device=device)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.seq_len = 0  # number of positions filled (identical for every sequence in the batch)

    def update(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write K/V for new positions [seq_len, seq_len + T) of `layer`; return the full history.

        k_new, v_new: [B, num_kv_heads, T, head_dim] (already rotated by RoPE).
        Returns views k, v: [B, num_kv_heads, seq_len + T, head_dim]. Views, not copies.

        Does not advance `seq_len`: every layer writes at the same offset during one forward
        pass, and the model calls `advance(T)` once after the last layer.
        """
        t = k_new.shape[2]
        end = self.seq_len + t
        if end > self.max_seq_len:
            raise RuntimeError(f"KV cache overflow: need {end} positions, have {self.max_seq_len}")
        self.k[layer, :, :, self.seq_len : end] = k_new
        self.v[layer, :, :, self.seq_len : end] = v_new
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, n: int) -> None:
        self.seq_len += n

    def reset(self) -> None:
        """Forget the contents (reuse the allocation for a new sequence)."""
        self.seq_len = 0

    @property
    def nbytes(self) -> int:
        """Bytes reserved (not just used): 2 * L * B * H_kv * max_seq_len * D * dtype_bytes."""
        return self.k.numel() * self.k.element_size() * 2

    @staticmethod
    def bytes_needed(
        cfg: ModelConfig, batch_size: int, max_seq_len: int, dtype: torch.dtype
    ) -> int:
        """How much memory a cache would reserve, without allocating it."""
        return cfg.kv_bytes_per_token(dtype) * batch_size * max_seq_len

    def utilization(self) -> float:
        """Fraction of reserved slots actually holding data (1.0 = no waste)."""
        return self.seq_len / self.max_seq_len

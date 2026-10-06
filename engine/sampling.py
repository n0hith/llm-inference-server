"""Token sampling: greedy, temperature, top-k, top-p (nucleus).

Everything is vectorized over the batch with **per-row** parameters, because in Phase 3 one
decode step serves many requests that each asked for different settings. A request with
temperature 0 (greedy) and one with temperature 0.8 / top-p 0.9 share one `sample()` call.

Order of operations (same as HF's default logits warpers, so behaviour matches):
    1. temperature: logits / T      (T < 1 sharpens the distribution, T > 1 flattens it)
    2. top-k: keep only the k highest-scoring tokens
    3. top-p: of those, keep the smallest set whose probability mass reaches p
    4. renormalize (softmax over survivors) and draw one token
Temperature 0 means greedy argmax (it's the T -> 0 limit, and avoids dividing by zero).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    """Per-request generation settings (mirrors the OpenAI API fields we'll expose in Phase 5)."""

    temperature: float = 1.0
    top_k: int = 0  # 0 = disabled (consider the whole vocabulary)
    top_p: float = 1.0  # 1.0 = disabled
    max_tokens: int = 128
    seed: int | None = None
    ignore_eos: bool = False  # keep generating past EOS (benchmarks need fixed output lengths)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0 (0 disables it)")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def apply_top_k_top_p(
    logits: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
) -> torch.Tensor:
    """Mask (set to -inf) every token outside each row's top-k / top-p set.

    logits: [B, V] float32 (already temperature-scaled); top_k: [B] int (0 = off);
    top_p: [B] float (1.0 = off). Returns masked logits in the original vocabulary order.

    One descending sort serves both filters: in sorted order, top-k is "rank < k" and top-p
    is "the probability mass of all strictly better tokens is < p". The best token always
    survives (its better-mass is 0), so a row is never fully masked.
    """
    vocab = logits.shape[-1]
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    rank = torch.arange(vocab, device=logits.device)

    k = torch.where(top_k > 0, top_k, vocab)  # 0 -> keep everything
    sorted_logits = sorted_logits.masked_fill(rank[None, :] >= k[:, None], float("-inf"))

    # Top-p is applied to the distribution *after* top-k (renormalized over the survivors).
    probs = sorted_logits.softmax(dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs  # mass of strictly higher-ranked tokens
    sorted_logits = sorted_logits.masked_fill(mass_before >= top_p[:, None], float("-inf"))

    # Undo the sort: put each (possibly masked) logit back at its token id.
    return torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)


def sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one token per row. logits: [B, V] (any float dtype); the rest are [B] tensors.

    Returns token ids [B] (int64). Rows with temperature 0 take the argmax.
    """
    logits = logits.float()  # fp16 softmax over 32K-128K entries loses too much precision
    greedy_tokens = logits.argmax(dim=-1)
    is_greedy = temperature == 0
    if bool(is_greedy.all()):  # fast path (also the common case in tests and benchmarks)
        return greedy_tokens

    # Divide by 1 for greedy rows: their sampled result is discarded below anyway.
    t = torch.where(is_greedy, torch.ones_like(temperature), temperature)
    scaled = logits / t[:, None]
    masked = apply_top_k_top_p(scaled, top_k, top_p)
    probs = masked.softmax(dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    return torch.where(is_greedy, greedy_tokens, sampled)


def params_to_tensors(
    params: Sequence[SamplingParams], device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack per-request settings into the [B] tensors `sample()` takes."""
    temperature = torch.tensor([p.temperature for p in params], dtype=torch.float32, device=device)
    top_k = torch.tensor([p.top_k for p in params], dtype=torch.int64, device=device)
    top_p = torch.tensor([p.top_p for p in params], dtype=torch.float32, device=device)
    return temperature, top_k, top_p


def make_generator(seed: int | None, device: torch.device | str) -> torch.Generator | None:
    """Seeded RNG for reproducible sampling (None = use torch's global RNG)."""
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g

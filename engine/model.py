"""Our own Llama-architecture decoder, written from scratch in PyTorch.

Data flow for one forward call (B = batch, T = new tokens, S = total context incl. cache):

    input_ids [B, T]
      -> embed_tokens                          [B, T, hidden]
      -> N x DecoderLayer:
           h = h + Attention(RMSNorm(h))       (pre-norm residual block)
           h = h + MLP(RMSNorm(h))
      -> final RMSNorm -> lm_head              [B, T or 1, vocab] logits

Prefill (processing the prompt) and decode (one new token) are the *same* function;
they differ only in T and in how much is already in the KV cache. Prefill is
compute-bound (big matmuls over T tokens); decode is memory-bound (T = 1, so every
weight is read from DRAM to do a single matrix-vector product).

Numerics deliberately mirror HF's implementation where precision matters (RMSNorm and
softmax in fp32, RoPE angles in fp32), so our outputs can be compared against HF in
the parity tests. Module/parameter names also mirror HF's checkpoint names, so the
loader (`engine/weights.py`) is a straightforward rename.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from engine.config import ModelConfig
from engine.kv_cache import KVCache


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight.

    Compared with LayerNorm there's no mean subtraction and no bias: cheaper, and it works
    just as well for LLMs. The mean of squares is computed in fp32 because squaring fp16
    activations overflows once |x| > 256 (fp16 max is 65504), and Llama residual streams get
    close: we measured |x| up to ~180 entering TinyLlama's norms on ordinary prompts.
    (At normal magnitudes the upcast's effect on rounding is small; it's overflow insurance.)
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(dtype)


def rope_inv_freq(cfg: ModelConfig) -> torch.Tensor:
    """RoPE inverse frequencies theta_i = base^(-2i/d), one per pair of head dimensions.

    Dimension pair i rotates by angle `position * theta_i`. Small i -> high frequency (fast
    rotation, encodes fine local position); large i -> low frequency (slow rotation, long range).

    For Llama 3 ("llama3" rope_scaling) the frequencies are adjusted to extend the context:
    low-frequency dims (wavelength longer than original_context / low_freq_factor) are slowed
    down by `factor`, high-frequency dims (wavelength shorter than original_context /
    high_freq_factor) are left alone, and dims in between are linearly blended. Same formula
    as Meta's reference code and HF's `_compute_llama3_parameters`.
    """
    d = cfg.head_dim
    # Computed on CPU in fp32 regardless of the current default device, so construction
    # under `torch.device("meta")` (see weights.py) still produces real values.
    exponents = torch.arange(0, d, 2, dtype=torch.int64, device="cpu").float() / d
    inv_freq = 1.0 / (cfg.rope_theta**exponents)

    rs = cfg.rope_scaling
    if rs is None:
        return inv_freq
    low_freq_wavelen = rs.original_max_position_embeddings / rs.low_freq_factor
    high_freq_wavelen = rs.original_max_position_embeddings / rs.high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / rs.factor, inv_freq)
    smooth = (rs.original_max_position_embeddings / wavelen - rs.low_freq_factor) / (
        rs.high_freq_factor - rs.low_freq_factor
    )
    smoothed = (1 - smooth) * scaled / rs.factor + smooth * scaled
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)


class RotaryEmbedding(nn.Module):
    """Produces cos/sin tables for the given absolute positions.

    The angles are computed on the fly from `inv_freq` (in fp32) instead of being read from
    a precomputed [max_pos, head_dim] table. It costs a few tiny kernels per forward (shared
    by all layers), but matches HF's computation exactly, including on-device cos/sin, which
    can differ from CPU cos/sin in the last bit.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.register_buffer("inv_freq", rope_inv_freq(cfg), persistent=False)

    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: [B, T] -> angles: [B, T, head_dim/2]
        angles = positions[..., None].float() * self.inv_freq
        # HF-format checkpoints use the "rotate_half" layout: dimension i is paired with
        # dimension i + d/2 (not with i+1 as in Meta's original interleaved layout; HF
        # permutes the q/k weights at conversion time). So the table is [angles, angles].
        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] -> [-x2, x1] on the last dim (the 90-degree rotation of each (i, i+d/2) pair)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each 2D pair of q/k by its position-dependent angle.

    x: [B, H, T, D]; cos/sin: [B, T, D] (broadcast over heads). Because q at position m and
    k at position n are rotated by m*theta and n*theta, their dot product depends only on
    (m - n): relative position falls out of absolute rotations. It also means cached keys
    never need re-rotating: K is stored *after* RoPE.
    """
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


def causal_mask(q_start: int, t_q: int, t_k: int, device: torch.device) -> torch.Tensor:
    """Boolean [t_q, t_k] mask, True = may attend. Query i sits at absolute position q_start + i
    and may see keys 0 .. q_start + i."""
    q_pos = torch.arange(q_start, q_start + t_q, device=device)
    k_pos = torch.arange(t_k, device=device)
    return k_pos[None, :] <= q_pos[:, None]


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_start: int, impl: str = "sdpa"
) -> torch.Tensor:
    """Causal grouped-query attention.

    q: [B, Hq, T, D] queries for the new tokens at positions q_start .. q_start+T-1
    k, v: [B, Hkv, S, D] keys/values for ALL positions 0 .. S-1 (cache + new), S = q_start + T
    returns [B, Hq, T, D]
    """
    t_q, t_k = q.shape[2], k.shape[2]
    scale = q.shape[-1] ** -0.5

    if impl == "sdpa":
        # PyTorch's fused attention (FlashAttention / memory-efficient kernels on GPU): never
        # materializes the [T, S] score matrix in global memory.
        #
        # Masking subtlety: SDPA's `is_causal=True` aligns the causal mask to the TOP-LEFT,
        # i.e. it assumes query i corresponds to key i. That's only right when t_q == t_k
        # (prefill from an empty cache). When a chunk of queries comes after cached keys,
        # query i must see keys up to q_start + i (bottom-right alignment), so we pass an
        # explicit mask. Single-token decode needs no mask: the new token may see everything.
        if t_q == 1:
            mask, is_causal = None, False
        elif q_start == 0 and t_q == t_k:
            mask, is_causal = None, True
        else:
            mask, is_causal = causal_mask(q_start, t_q, t_k, q.device), False
        # enable_gqa lets SDPA broadcast each KV head to its group of query heads without
        # us materializing repeated K/V copies.
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal, scale=scale, enable_gqa=True
        )

    if impl == "naive":
        # Textbook attention; readable reference. Memory is O(Hq * T * S) for the scores.
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)  # [B, Hq, S, D]: KV head j serves q heads j*g..
        v = v.repeat_interleave(groups, dim=1)
        scores = (q @ k.transpose(-2, -1)) * scale  # [B, Hq, T, S]
        if t_q > 1:
            mask = causal_mask(q_start, t_q, t_k, q.device)
            scores = scores.masked_fill(~mask, float("-inf"))
        # Softmax in fp32: exp() of fp16 scores and the normalizing sum are precision-sensitive.
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return probs @ v

    raise ValueError(f"unknown attention impl {impl!r}")


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, attn_impl: str) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads, self.num_kv_heads, self.head_dim = (
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
        )
        self.attn_impl = attn_impl
        # Kept as four separate projections (not one fused QKV matmul) for readability and so
        # each maps 1:1 to a checkpoint tensor. Fusing them is a known decode optimization
        # (fewer kernel launches); we'll decide based on profiling data in Phase 7.
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        q_start: int,
        cache: KVCache | None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        # [B, T, H*D] -> [B, T, H, D] -> [B, H, T, D] (heads first, the layout attention wants)
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:
            # Write this step's K/V into the cache and get back views of the whole history.
            k, v = cache.update(self.layer_idx, k, v)

        out = attention(q, k, v, q_start, self.attn_impl)  # [B, Hq, T, D]
        out = out.transpose(1, 2).reshape(b, t, self.num_heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU feed-forward: down( silu(gate(x)) * up(x) ).

    The gate branch, passed through SiLU, multiplicatively controls the up branch. Three
    matrices instead of two, which is why Llama's intermediate size is ~2.7x hidden rather
    than 4x: it keeps the parameter count similar to a classic 4x MLP.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, attn_impl: str) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, layer_idx, attn_impl)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        q_start: int,
        cache: KVCache | None,
    ) -> torch.Tensor:
        # Pre-norm residual blocks: normalize the *input* of each sublayer and add its output
        # back to the un-normalized residual stream. This keeps gradients/activations stable
        # in deep stacks (and is what the checkpoint was trained with).
        h = h + self.self_attn(self.input_layernorm(h), cos, sin, q_start, cache)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class LlamaModel(nn.Module):
    """The full causal LM. Construct via `engine.weights.load_model`, not directly."""

    def __init__(self, cfg: ModelConfig, attn_impl: str = "sdpa") -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i, attn_impl) for i in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(cfg)

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        logits: str = "last",
    ) -> torch.Tensor:
        """Run T new tokens through the model.

        input_ids: [B, T]. All B sequences are at the same position (Phase 1 = lockstep batch;
            ragged batches come with the Phase 3 scheduler).
        cache: if given, new K/V are appended at `cache.seq_len` and attention covers the full
            cached history; the cache is then advanced by T. If None, the T tokens are the
            entire context (positions 0..T-1) and nothing is stored.
        logits: "last" -> [B, 1, vocab] for the final position only (all that sampling needs;
            skipping the other T-1 positions saves a [T, vocab] matmul and its memory, which
            for a 128K vocab and a 2K prompt would be ~1 GB in fp32). "all" -> [B, T, vocab].
        """
        b, t = input_ids.shape
        q_start = cache.seq_len if cache is not None else 0
        positions = torch.arange(q_start, q_start + t, device=input_ids.device).expand(b, t)

        h = self.embed_tokens(input_ids)
        cos, sin = self.rotary(positions, h.dtype)  # computed once, shared by every layer
        for layer in self.layers:
            h = layer(h, cos, sin, q_start, cache)
        if cache is not None:
            cache.advance(t)

        if logits == "last":
            h = h[:, -1:, :]  # RMSNorm is per-token, so slicing before the norm is equivalent
        elif logits != "all":
            raise ValueError(f"logits must be 'last' or 'all', got {logits!r}")
        return self.lm_head(self.norm(h))

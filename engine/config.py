"""Model and engine configuration.

`ModelConfig` is our own, minimal view of a HuggingFace `config.json`: only the fields
the forward pass needs. Parsing it ourselves (instead of passing a `transformers.LlamaConfig`
around) keeps the engine independent of transformers at runtime; transformers is only used
to load the tokenizer and as the correctness reference in tests.

It also owns the two pieces of arithmetic the rest of the system plans around:
* `num_params()` / `weight_bytes()`: how much memory the weights take.
* `kv_bytes_per_token()`: how much KV-cache memory each token of context costs. This
  decides how many sequences fit in VRAM (Phase 3) and the block size math (Phase 4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

DTYPE_BYTES: dict[torch.dtype, int] = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}


@dataclass(frozen=True)
class RopeScaling:
    """Llama 3.x "llama3" RoPE frequency scaling (used by Llama-3.1/3.2; TinyLlama has none).

    Llama 3 was pre-trained at 8K context and then extended. Rather than scaling all RoPE
    frequencies equally, it leaves high-frequency dimensions (short wavelengths, which
    encode local position) untouched, divides low-frequency dimensions (long wavelengths)
    by `factor`, and smoothly interpolates in between. See `engine.model.rope_inv_freq`.
    """

    rope_type: str
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    bos_token_id: int | None = None
    eos_token_ids: tuple[int, ...] = ()
    rope_scaling: RopeScaling | None = None
    name: str = "unknown"

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be a multiple of num_kv_heads for GQA")

    # ------------------------------------------------------------------ derived sizes
    @property
    def num_kv_groups(self) -> int:
        """How many query heads share each K/V head (1 = MHA, num_heads = MQA)."""
        return self.num_heads // self.num_kv_heads

    def kv_bytes_per_token(self, dtype: torch.dtype = torch.float16) -> int:
        """KV-cache bytes per token of context, summed over all layers.

            2 (K and V) * num_layers * num_kv_heads * head_dim * bytes_per_element

        Only the K/V heads count (not query heads), which is exactly why GQA matters for
        serving: TinyLlama has 32 query heads but 4 KV heads, so its cache is 8x smaller
        than an equivalent multi-head-attention model.
        """
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * DTYPE_BYTES[dtype]

    def decoder_linear_params(self) -> int:
        """Parameters in the decoder blocks' Linear layers (the ones 4-bit quantization hits)."""
        q = self.hidden_size * self.num_heads * self.head_dim
        kv = 2 * self.hidden_size * self.num_kv_heads * self.head_dim
        o = self.num_heads * self.head_dim * self.hidden_size
        mlp = 3 * self.hidden_size * self.intermediate_size  # gate, up, down
        return self.num_layers * (q + kv + o + mlp)

    def embedding_params(self) -> int:
        """Input embedding plus the LM head (a separate matrix unless tied)."""
        emb = self.vocab_size * self.hidden_size
        return emb if self.tie_word_embeddings else 2 * emb

    def num_params(self) -> int:
        norms = self.num_layers * 2 * self.hidden_size + self.hidden_size
        return self.decoder_linear_params() + self.embedding_params() + norms

    def weight_bytes(self, dtype: torch.dtype = torch.float16) -> int:
        return self.num_params() * DTYPE_BYTES[dtype]

    def weight_bytes_4bit(self, bits_per_param: float = 4.5) -> int:
        """Estimated size with decoder Linear layers in 4-bit and everything else in 16-bit.

        4.5 bits/param = 4-bit codes + one fp32 scale per block of 64 weights (32/64 = 0.5 bit),
        which is bitsandbytes NF4 without double quantization. Embeddings and the LM head
        usually stay 16-bit (they're quality-sensitive), which is why 128K-vocab models
        shrink much less than 4x.
        """
        linear = self.decoder_linear_params() * bits_per_param / 8
        rest = (self.num_params() - self.decoder_linear_params()) * 2
        return int(linear + rest)

    # ------------------------------------------------------------------ parsing
    @classmethod
    def from_hf_dict(cls, d: dict[str, Any], name: str = "unknown") -> ModelConfig:
        """Build from a HF Llama `config.json` dict.

        Handles both the classic layout (`rope_theta` + `rope_scaling`) found in hub
        checkpoints and the transformers-5 layout (`rope_parameters`).
        """
        if d.get("model_type", "llama") != "llama":
            raise ValueError(f"only Llama-architecture models are supported, got {d['model_type']}")
        num_heads = d["num_attention_heads"]
        head_dim = d.get("head_dim") or d["hidden_size"] // num_heads

        rope = d.get("rope_parameters") or d.get("rope_scaling") or {}
        rope_theta = d.get("rope_theta", rope.get("rope_theta", 10000.0))
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        scaling: RopeScaling | None = None
        if rope_type == "llama3":
            scaling = RopeScaling(
                rope_type="llama3",
                factor=rope["factor"],
                low_freq_factor=rope["low_freq_factor"],
                high_freq_factor=rope["high_freq_factor"],
                original_max_position_embeddings=rope["original_max_position_embeddings"],
            )
        elif rope_type != "default":
            raise NotImplementedError(f"RoPE type {rope_type!r} not supported")

        eos = d.get("eos_token_id")
        eos_ids = tuple(eos) if isinstance(eos, list) else (() if eos is None else (eos,))
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_layers=d["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=d.get("num_key_value_heads", num_heads),
            head_dim=head_dim,
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=float(rope_theta),
            max_position_embeddings=d["max_position_embeddings"],
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            bos_token_id=d.get("bos_token_id"),
            eos_token_ids=eos_ids,
            rope_scaling=scaling,
            name=name,
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> ModelConfig:
        """Read `config.json`, merging extra EOS ids from `generation_config.json`.

        Chat models often stop on a token that isn't the base EOS (Llama-3 instruct ends
        turns with <|eot_id|>), and that list lives in generation_config.json.
        """
        model_dir = Path(model_dir)
        cfg = cls.from_hf_dict(json.loads((model_dir / "config.json").read_text()), model_dir.name)
        gen_path = model_dir / "generation_config.json"
        if gen_path.exists():
            gen_eos = json.loads(gen_path.read_text()).get("eos_token_id")
            extra = gen_eos if isinstance(gen_eos, list) else ([] if gen_eos is None else [gen_eos])
            merged = tuple(dict.fromkeys([*cfg.eos_token_ids, *extra]))  # ordered, de-duplicated
            cfg = replace(cfg, eos_token_ids=merged)
        return cfg


@dataclass
class EngineConfig:
    """Runtime knobs that are not properties of the model itself."""

    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    # Attention backend: "sdpa" (PyTorch fused kernel, default) or "naive" (explicit
    # matmul + softmax; slower and O(T^2) memory, but easy to read and used in tests).
    attn_impl: str = "sdpa"
    max_seq_len: int = 2048
    extra: dict[str, Any] = field(default_factory=dict)


# Public config.json values of models we might run, so memory planning (env_check, docs)
# works without downloading anything. Llama models are gated, so their configs are
# reproduced here; only the fields ModelConfig needs.
REFERENCE_CONFIGS: dict[str, dict[str, Any]] = {
    "TinyLlama-1.1B": dict(
        vocab_size=32000, hidden_size=2048, intermediate_size=5632, num_hidden_layers=22,
        num_attention_heads=32, num_key_value_heads=4, rms_norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=2048, tie_word_embeddings=False,
    ),
    "Llama-3.2-1B": dict(
        vocab_size=128256, hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
        num_attention_heads=32, num_key_value_heads=8, head_dim=64, rms_norm_eps=1e-5,
        rope_theta=500000.0, max_position_embeddings=131072, tie_word_embeddings=True,
    ),
    "Llama-3.2-3B": dict(
        vocab_size=128256, hidden_size=3072, intermediate_size=8192, num_hidden_layers=28,
        num_attention_heads=24, num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-5,
        rope_theta=500000.0, max_position_embeddings=131072, tie_word_embeddings=True,
    ),
    "Llama-3.1-8B": dict(
        vocab_size=128256, hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-5,
        rope_theta=500000.0, max_position_embeddings=131072, tie_word_embeddings=False,
    ),
    # No GQA: included to show how much bigger the KV cache is with plain multi-head attention.
    "Llama-2-7B": dict(
        vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32,
        num_attention_heads=32, num_key_value_heads=32, rms_norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=4096, tie_word_embeddings=False,
    ),
}  # fmt: skip

"""KV cache tests: memory formula, write/read semantics, and bounds checking.
(Cache *correctness through the model* is covered in test_model_parity.py.)"""

from __future__ import annotations

import pytest
import torch

from engine.config import REFERENCE_CONFIGS, ModelConfig
from engine.kv_cache import KVCache


def _cfg(name: str) -> ModelConfig:
    return ModelConfig.from_hf_dict(REFERENCE_CONFIGS[name], name)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("TinyLlama-1.1B", 2 * 22 * 4 * 64 * 2),  # 22,528 B = 22 KiB
        ("Llama-3.2-1B", 2 * 16 * 8 * 64 * 2),  # 32 KiB
        ("Llama-3.2-3B", 2 * 28 * 8 * 128 * 2),  # 112 KiB
        ("Llama-2-7B", 2 * 32 * 32 * 128 * 2),  # 512 KiB (no GQA)
    ],
)
def test_kv_bytes_per_token(name: str, expected: int) -> None:
    assert _cfg(name).kv_bytes_per_token(torch.float16) == expected


def test_tinyllama_param_count_formula() -> None:
    # Known exact parameter count of TinyLlama-1.1B (also checked against real weights
    # in test_model_parity.py::test_param_count_matches_config).
    assert _cfg("TinyLlama-1.1B").num_params() == 1_100_048_384


def _small_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=10, hidden_size=16, intermediate_size=32, num_layers=3, num_heads=4,
        num_kv_heads=2, head_dim=4, rms_norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=64, tie_word_embeddings=False,
    )  # fmt: skip


def test_reserved_bytes_match_formula() -> None:
    cfg = _small_cfg()
    cache = KVCache(cfg, batch_size=2, max_seq_len=16, dtype=torch.float16, device="cpu")
    assert cache.nbytes == KVCache.bytes_needed(cfg, 2, 16, torch.float16)
    assert cache.nbytes == cfg.kv_bytes_per_token(torch.float16) * 2 * 16


def test_update_writes_at_offset_and_returns_history() -> None:
    cfg = _small_cfg()
    cache = KVCache(cfg, batch_size=1, max_seq_len=8, dtype=torch.float32, device="cpu")
    first_k, first_v = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    for layer in range(cfg.num_layers):
        k, v = cache.update(layer, first_k, first_v)
        assert k.shape == (1, 2, 3, 4)
    cache.advance(3)

    new_k, new_v = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
    k, v = cache.update(1, new_k, new_v)
    assert k.shape == (1, 2, 4, 4)
    torch.testing.assert_close(k[:, :, :3], first_k)
    torch.testing.assert_close(k[:, :, 3:], new_k)
    torch.testing.assert_close(v[:, :, 3:], new_v)
    # Returned tensors are views into the cache, not copies.
    assert k.untyped_storage().data_ptr() == cache.k.untyped_storage().data_ptr()


def test_overflow_raises() -> None:
    cache = KVCache(_small_cfg(), batch_size=1, max_seq_len=4, dtype=torch.float32, device="cpu")
    cache.update(0, torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))
    cache.advance(4)
    with pytest.raises(RuntimeError, match="overflow"):
        cache.update(0, torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))


def test_reset_and_utilization() -> None:
    cache = KVCache(_small_cfg(), batch_size=1, max_seq_len=8, dtype=torch.float32, device="cpu")
    cache.advance(2)
    assert cache.utilization() == 0.25
    cache.reset()
    assert cache.seq_len == 0


def test_max_seq_len_beyond_model_context_rejected() -> None:
    with pytest.raises(ValueError, match="max_position_embeddings"):
        KVCache(_small_cfg(), batch_size=1, max_seq_len=65, dtype=torch.float32, device="cpu")

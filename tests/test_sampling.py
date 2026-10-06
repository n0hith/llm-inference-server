"""Sampling tests: exact semantics (masks vs HF's warpers) plus statistical behaviour."""

from __future__ import annotations

import pytest
import torch
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

from engine.sampling import (
    SamplingParams,
    apply_top_k_top_p,
    make_generator,
    params_to_tensors,
    sample,
)


def _row(values: list[float]) -> torch.Tensor:
    return torch.tensor([values], dtype=torch.float32)


def _sample_many(logits: torch.Tensor, n: int, **kw) -> torch.Tensor:
    """Draw n samples for one row by repeating it n times in a batch."""
    p = SamplingParams(**kw)
    t, k, tp = params_to_tensors([p] * n, "cpu")
    return sample(logits.expand(n, -1), t, k, tp, make_generator(0, "cpu"))


def test_greedy_is_argmax_and_ignores_filters() -> None:
    logits = torch.randn(4, 100)
    t, k, p = params_to_tensors([SamplingParams(temperature=0.0, top_k=5, top_p=0.3)] * 4, "cpu")
    assert torch.equal(sample(logits, t, k, p), logits.argmax(-1))


def test_top_k_1_equals_greedy() -> None:
    logits = torch.randn(3, 50)
    t, k, p = params_to_tensors([SamplingParams(temperature=1.0, top_k=1)] * 3, "cpu")
    assert torch.equal(sample(logits, t, k, p, make_generator(0, "cpu")), logits.argmax(-1))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("top_k,top_p", [(0, 0.9), (10, 1.0), (20, 0.7), (5, 0.5), (0, 0.05)])
def test_mask_matches_hf_warpers(seed: int, top_k: int, top_p: float) -> None:
    """Same surviving token set as HF's TopK then TopP warpers (HF's default order)."""
    torch.manual_seed(seed)
    logits = torch.randn(2, 1000) * 3
    expected = logits.clone()
    if top_k > 0:
        expected = TopKLogitsWarper(top_k)(None, expected)
    if top_p < 1.0:
        expected = TopPLogitsWarper(top_p)(None, expected)
    got = apply_top_k_top_p(logits, torch.full((2,), top_k), torch.full((2,), top_p))
    assert torch.equal(torch.isfinite(got), torch.isfinite(expected))
    kept = torch.isfinite(got)
    assert torch.equal(got[kept], logits[kept])  # survivors are untouched


def test_top_p_boundary() -> None:
    # probs exactly [0.5, 0.3, 0.15, 0.05]
    logits = torch.log(_row([0.5, 0.3, 0.15, 0.05]))
    keep = lambda p: torch.isfinite(apply_top_k_top_p(logits, torch.tensor([0]), torch.tensor([p])))  # noqa: E731
    # The 3rd token's "mass of better tokens" is 0.8: kept only while that is < p.
    assert keep(0.8).tolist() == [[True, True, False, False]]
    assert keep(0.81).tolist() == [[True, True, True, False]]
    assert keep(0.01).tolist() == [[True, False, False, False]]  # best token always survives


def test_samples_stay_inside_top_k() -> None:
    logits = torch.randn(1, 200)
    allowed = set(logits[0].topk(7).indices.tolist())
    draws = _sample_many(logits, 2000, temperature=2.0, top_k=7)
    assert set(draws.tolist()) <= allowed
    assert len(set(draws.tolist())) == 7  # high temperature: all 7 should show up


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_temperature_matches_softmax_distribution(temperature: float) -> None:
    logits = _row([2.0, 1.0, 0.5, 0.0, -1.0])
    n = 40_000
    draws = _sample_many(logits, n, temperature=temperature)
    freq = torch.bincount(draws, minlength=5).float() / n
    expected = torch.softmax(logits[0] / temperature, -1)
    # Std of a frequency estimate is sqrt(p(1-p)/n) <= 0.0025 here; 0.01 is ~4 sigma.
    torch.testing.assert_close(freq, expected, atol=0.01, rtol=0)


def test_per_row_parameters_in_one_batch() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 300)
    params = [
        SamplingParams(temperature=0.0),  # greedy
        SamplingParams(temperature=1.0, top_k=1),  # effectively greedy
        SamplingParams(temperature=1.5, top_k=3),  # one of the top 3
    ]
    t, k, p = params_to_tensors(params, "cpu")
    for seed in range(50):
        out = sample(logits, t, k, p, make_generator(seed, "cpu"))
        assert out[0] == logits[0].argmax()
        assert out[1] == logits[1].argmax()
        assert out[2] in logits[2].topk(3).indices


def test_seeded_sampling_is_reproducible() -> None:
    logits = torch.randn(1, 500)
    a = _sample_many(logits, 100, temperature=1.0, seed=None)
    b = _sample_many(logits, 100, temperature=1.0, seed=None)
    assert torch.equal(a, b)  # _sample_many seeds its generator with 0 both times


def test_fp16_logits_are_upcast() -> None:
    logits = torch.randn(2, 64).half()
    t, k, p = params_to_tensors([SamplingParams(temperature=0.0)] * 2, "cpu")
    assert torch.equal(sample(logits, t, k, p), logits.float().argmax(-1))


@pytest.mark.parametrize(
    "kwargs",
    [dict(temperature=-0.1), dict(top_k=-1), dict(top_p=0.0), dict(top_p=1.1), dict(max_tokens=0)],
)
def test_invalid_params_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)

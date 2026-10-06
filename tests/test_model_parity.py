"""Parity tests: our forward pass vs HuggingFace transformers (the correctness reference).

This file is the project's safety net. Every later phase (quantization, batching, paged
cache, fused kernels) must keep these passing.

Two tiers:

A. **Fast, CPU, no download**: tiny randomly-initialized Llama models built with HF's own
   `LlamaForCausalLM`, saved to disk, loaded by our loader. One is TinyLlama-shaped (GQA,
   untied head), one is Llama-3.2-shaped (GQA, tied embeddings, "llama3" RoPE scaling), so
   the Llama-3.2 code paths are verified without downloading a gated model. fp32, tight tolerance.

B. **Slow, GPU, real TinyLlama-1.1B**, in three precisions:

   * **fp32, tight tolerance (max |logit diff| <= 1e-4).** fp32 has ~7 significant digits
     (unit roundoff 6e-8). The only legitimate source of difference is a different
     summation order in matmul/attention kernels, which for logits of magnitude ~20 gives
     errors around 1e-5 (measured: 0 for our SDPA path, 2.5e-5 for the naive path).
     1e-4 leaves headroom for that while catching any real bug, which shows up as O(0.1-10).

   * **fp16 / bf16, tolerance derived from HF's own error.** In 16-bit, every op rounds to
     ~3 (fp16: 11-bit mantissa, u = 4.9e-4) or ~2 (bf16: 8-bit, u = 3.9e-3) significant
     digits, and those errors compound over 22 layers. So "ours == HF fp16" is the wrong
     question: two correct implementations may round differently. The right question is:
     *is our fp16 model as accurate as HF's fp16 model?* We use HF **fp32** as ground truth
     and require:

         err(ours_16bit vs HF_fp32)  <=  1.5 x  err(HF_16bit vs HF_fp32)

     for both the max and the mean absolute logit error. Measured HF noise floors on our
     prompts: fp16 max ~0.21 / mean ~8e-3; bf16 max ~0.73 / mean ~3e-2. The 1.5x slack covers
     a different-but-valid rounding order (our naive path measured up to 1.26x on the mean in
     bf16); a real bug (wrong RoPE, missing norm) is orders of magnitude larger.

   * **Greedy decoding, token for token**, over 48 new tokens per prompt, for the paths where
     exactness is expected: fp32 (both attention impls) and the default SDPA path in
     fp16/bf16 (bit-identical to HF, since it runs the same kernels in the same order). It is
     *not* asserted for the naive path in 16-bit: with different rounding, a near-tie between
     the top two logits can legitimately flip (we measured one such flip in bf16), and after
     that the sequences diverge. That path is covered by the noise-floor test instead.

What these tests can and can't catch (verified by injecting bugs, see docs/PROGRESS.md):
logic bugs (wrong RoPE pairing, lost position offset, top-left causal mask in chunked
prefill) fail the fast tier immediately. A *precision* bug whose effect is comparable to
16-bit rounding noise (e.g. RMSNorm without the fp32 upcast: 1.0-1.3x HF's error on these
prompts) passes the noise-floor test by construction; only the bit-exact SDPA comparisons
catch it, and its real failure mode (fp16 overflow) has its own direct test.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

from engine.config import ModelConfig
from engine.generate import generate
from engine.kv_cache import KVCache
from engine.model import LlamaModel, rope_inv_freq
from engine.sampling import SamplingParams
from engine.weights import load_model, resolve_model_dir

# ============================================================================ tier A: tiny models

TINY_CONFIGS: dict[str, dict] = {
    # GQA (8 query heads, 2 KV heads), untied LM head, plain RoPE.
    "tinyllama_like": dict(
        vocab_size=256, hidden_size=64, intermediate_size=176, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=False, rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    ),
    # GQA, tied embeddings, llama3 RoPE scaling. With head_dim 16, theta 5e5 and
    # original context 64, the 8 frequency pairs land in all three llama3 regimes
    # (unchanged / smoothly interpolated / divided by factor).
    "llama32_like": dict(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        tie_word_embeddings=True, rms_norm_eps=1e-5,
        rope_parameters={
            "rope_type": "llama3", "rope_theta": 500000.0, "factor": 32.0,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 64,
        },
    ),
}  # fmt: skip

TINY_ATOL = 1e-5  # fp32 on CPU, tiny model: differences are pure summation-order noise


def _make_tiny_hf_model(name: str, save_dir: Path) -> LlamaForCausalLM:
    """Random HF Llama with weights large enough that outputs are far from uniform.

    HF's default init (std 0.02) gives near-uniform logits on a tiny model, which would make
    almost any implementation look "correct". Larger random weights make every component
    matter.
    """
    torch.manual_seed(0)
    config = LlamaConfig(
        **TINY_CONFIGS[name], bos_token_id=None, eos_token_id=None, pad_token_id=None
    )
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for pname, p in model.named_parameters():
            if "norm" in pname:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(0.15 * torch.randn_like(p))
    model.save_pretrained(save_dir)
    return model


@pytest.fixture(scope="module", params=list(TINY_CONFIGS))
def tiny_pair(request, tmp_path_factory) -> tuple[LlamaForCausalLM, Path]:
    save_dir = tmp_path_factory.mktemp(request.param)
    return _make_tiny_hf_model(request.param, save_dir), save_dir


def _tokens(n: int, seed: int = 1, batch: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (batch, n), generator=g)


@pytest.mark.parametrize("attn_impl", ["sdpa", "naive"])
def test_tiny_logits_match_hf(tiny_pair, attn_impl: str) -> None:
    hf, path = tiny_pair
    ours = load_model(path, dtype=torch.float32, device="cpu", attn_impl=attn_impl)
    ids = _tokens(24)
    with torch.inference_mode():
        expected = hf(ids).logits
        got = ours(ids, logits="all")
    assert expected.abs().max() > 1.0, "test model is degenerate (near-uniform logits)"
    torch.testing.assert_close(got, expected, atol=TINY_ATOL, rtol=0)


def test_tiny_greedy_matches_hf_generate(tiny_pair) -> None:
    hf, path = tiny_pair
    ours = load_model(path, dtype=torch.float32, device="cpu")
    prompt = _tokens(10, seed=2)
    with torch.inference_mode():
        hf_out = hf.generate(prompt, max_new_tokens=30, do_sample=False)[0, 10:].tolist()
    result = generate(ours, prompt[0].tolist(), SamplingParams(temperature=0.0, max_tokens=30))
    assert result.output_ids == hf_out


@pytest.mark.parametrize("attn_impl", ["sdpa", "naive"])
def test_cached_decode_matches_full_forward(tiny_pair, attn_impl: str) -> None:
    """Prefill 8 tokens, then feed the remaining tokens one at a time through the KV cache.
    Each step's logits must equal the no-cache forward over the full sequence at that position."""
    _, path = tiny_pair
    model = load_model(path, dtype=torch.float32, device="cpu", attn_impl=attn_impl)
    ids = _tokens(20, seed=3)
    with torch.inference_mode():
        full = model(ids, logits="all")
        cache = KVCache(model.cfg, 1, 32, torch.float32, "cpu")
        steps = [model(ids[:, :8], cache=cache, logits="all")]
        for i in range(8, 20):
            steps.append(model(ids[:, i : i + 1], cache=cache, logits="all"))
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=TINY_ATOL, rtol=0)
    assert cache.seq_len == 20


@pytest.mark.parametrize("attn_impl", ["sdpa", "naive"])
def test_chunked_prefill_matches_full_forward(tiny_pair, attn_impl: str) -> None:
    """Prefill in chunks of 5 after a cached prefix: exercises the bottom-right-aligned causal
    mask (query chunk after cached keys), the case SDPA's is_causal flag gets wrong."""
    _, path = tiny_pair
    model = load_model(path, dtype=torch.float32, device="cpu", attn_impl=attn_impl)
    ids = _tokens(20, seed=4)
    with torch.inference_mode():
        full = model(ids, logits="all")
        cache = KVCache(model.cfg, 1, 32, torch.float32, "cpu")
        chunks = [model(ids[:, s : s + 5], cache=cache, logits="all") for s in range(0, 20, 5)]
    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=TINY_ATOL, rtol=0)


def test_lockstep_batch_matches_single(tiny_pair) -> None:
    """Two different same-length prompts in one batch == each run alone."""
    _, path = tiny_pair
    model = load_model(path, dtype=torch.float32, device="cpu")
    ids = _tokens(12, seed=5, batch=2)
    with torch.inference_mode():
        batched = model(ids, logits="all")
        singles = torch.cat([model(ids[i : i + 1], logits="all") for i in range(2)])
    torch.testing.assert_close(batched, singles, atol=TINY_ATOL, rtol=0)


def test_logits_last_equals_last_of_all(tiny_pair) -> None:
    _, path = tiny_pair
    model = load_model(path, dtype=torch.float32, device="cpu")
    ids = _tokens(9, seed=6)
    with torch.inference_mode():
        torch.testing.assert_close(model(ids)[:, -1], model(ids, logits="all")[:, -1])


@pytest.mark.parametrize("name", list(TINY_CONFIGS))
def test_rope_inv_freq_matches_hf(name: str) -> None:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    hf_config = LlamaConfig(**TINY_CONFIGS[name])
    rope_type = hf_config.rope_parameters["rope_type"]
    if rope_type == "default":
        expected, _ = LlamaRotaryEmbedding.compute_default_rope_parameters(hf_config)
    else:
        expected, _ = ROPE_INIT_FUNCTIONS[rope_type](hf_config)
    ours = rope_inv_freq(ModelConfig.from_hf_dict(hf_config.to_dict()))
    torch.testing.assert_close(ours, expected, atol=0, rtol=0)


def test_weight_loader_tied_embeddings_share_storage(tmp_path: Path) -> None:
    _make_tiny_hf_model("llama32_like", tmp_path)
    model = load_model(tmp_path, dtype=torch.float32, device="cpu")
    assert model.lm_head.weight is model.embed_tokens.weight


def test_rmsnorm_fp16_large_activations_do_not_overflow() -> None:
    """x^2 overflows fp16 (max 65504) once |x| > 256. TinyLlama's residual stream reaches
    |x| ~ 180 on ordinary prompts (measured), so this is a real margin, not a hypothetical.
    The noise-floor test can't catch a missing upcast (its effect at normal magnitudes is
    within rounding noise), so this checks the failure mode directly."""
    from engine.model import RMSNorm

    norm = RMSNorm(64, eps=1e-5).half()
    x = torch.full((1, 64), 300.0).half()
    x[0, ::2] = -300.0
    y = norm(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.float(), torch.sign(x.float()), atol=1e-3, rtol=0)


def test_model_meta_construction_has_no_memory() -> None:
    """Sanity check of the loader's trick: building on 'meta' allocates nothing."""
    cfg = ModelConfig.from_hf_dict({**TINY_CONFIGS["tinyllama_like"], "model_type": "llama"})
    with torch.device("meta"):
        model = LlamaModel(cfg)
    assert model.embed_tokens.weight.is_meta
    assert not model.rotary.inv_freq.is_meta  # computed explicitly on CPU


# ============================================================================ tier B: TinyLlama

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NEW_TOKENS = 48
FP32_ATOL = 1e-4
NOISE_SLACK = 1.5
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

RAW_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "1, 1, 2, 3, 5, 8, 13,",
]
CHAT_PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a haiku about GPUs.",
]


def _free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def prompts() -> list[list[int]]:
    tok = AutoTokenizer.from_pretrained(resolve_model_dir(MODEL_ID))
    ids = [tok(p).input_ids for p in RAW_PROMPTS]
    for p in CHAT_PROMPTS:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
        )
        # TinyLlama's chat template doesn't emit <s>, so the tokenizer adds BOS like for raw text.
        ids.append(tok(text).input_ids)
    return [list(map(int, x)) for x in ids]


@pytest.fixture(scope="module")
def hf_reference(prompts) -> dict[str, dict[str, list]]:
    """HF logits (full prompt, as fp32 on CPU) and greedy continuations for each dtype.

    Models are loaded one at a time and freed: an fp32 1.1B model is 4.4 GB, and HF's copy
    plus ours would not fit in 6 GB together.
    """
    ref: dict[str, dict[str, list]] = {}
    model_dir = resolve_model_dir(MODEL_ID)
    for name, dtype in DTYPES.items():
        hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).cuda().eval()
        logits, greedy = [], []
        with torch.inference_mode():
            for ids in prompts:
                x = torch.tensor([ids], device="cuda")
                logits.append(hf(x).logits[0].float().cpu())
                out = hf.generate(x, max_new_tokens=NEW_TOKENS, do_sample=False)
                greedy.append(out[0, len(ids) :].tolist())
        ref[name] = {"logits": logits, "greedy": greedy}
        del hf
        _free_gpu()
    return ref


_model_slot: dict[tuple[str, str], LlamaModel] = {}


def _our_model(dtype_name: str, attn_impl: str) -> LlamaModel:
    """Keep only one of our models on the GPU at a time (same memory reason as above)."""
    key = (dtype_name, attn_impl)
    if key not in _model_slot:
        _model_slot.clear()
        _free_gpu()
        _model_slot[key] = load_model(MODEL_ID, dtype=DTYPES[dtype_name], attn_impl=attn_impl)
    return _model_slot[key]


def _our_logits(model: LlamaModel, prompts: list[list[int]]) -> list[torch.Tensor]:
    with torch.inference_mode():
        return [
            model(torch.tensor([ids], device="cuda"), logits="all")[0].float().cpu()
            for ids in prompts
        ]


@pytest.mark.gpu
@pytest.mark.slow
def test_param_count_matches_config(hf_reference) -> None:
    model = _our_model("fp16", "sdpa")
    n = sum(p.numel() for p in model.parameters())
    assert n == model.cfg.num_params() == 1_100_048_384


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("attn_impl", ["sdpa", "naive"])
def test_fp32_logits_match_hf(hf_reference, prompts, attn_impl: str) -> None:
    ours = _our_logits(_our_model("fp32", attn_impl), prompts)
    for got, expected in zip(ours, hf_reference["fp32"]["logits"], strict=True):
        torch.testing.assert_close(got, expected, atol=FP32_ATOL, rtol=0)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("attn_impl", ["sdpa", "naive"])
@pytest.mark.parametrize("dtype_name", ["fp16", "bf16"])
def test_low_precision_error_within_hf_noise_floor(
    hf_reference, prompts, dtype_name: str, attn_impl: str
) -> None:
    ours = _our_logits(_our_model(dtype_name, attn_impl), prompts)
    truth = hf_reference["fp32"]["logits"]
    hf_low = hf_reference[dtype_name]["logits"]
    for i, (o, h, g) in enumerate(zip(ours, hf_low, truth, strict=True)):
        ours_err, hf_err = (o - g).abs(), (h - g).abs()
        assert ours_err.max() <= NOISE_SLACK * hf_err.max(), (
            f"prompt {i}: max err {ours_err.max():.3e} vs HF {dtype_name} {hf_err.max():.3e}"
        )
        assert ours_err.mean() <= NOISE_SLACK * hf_err.mean(), (
            f"prompt {i}: mean err {ours_err.mean():.3e} vs HF {dtype_name} {hf_err.mean():.3e}"
        )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "dtype_name,attn_impl",
    [("fp32", "sdpa"), ("fp32", "naive"), ("fp16", "sdpa"), ("bf16", "sdpa")],
)
def test_greedy_matches_hf_token_for_token(hf_reference, prompts, dtype_name, attn_impl) -> None:
    model = _our_model(dtype_name, attn_impl)
    params = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS)
    for ids, expected in zip(prompts, hf_reference[dtype_name]["greedy"], strict=True):
        result = generate(model, ids, params, eos_token_ids=model.cfg.eos_token_ids)
        assert result.output_ids == expected

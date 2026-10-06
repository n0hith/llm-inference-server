# Progress Log

One entry per phase: what was built, key design decisions, and numbers actually measured
on the dev machine. Any number not measured here is explicitly labelled as an estimate.

| Phase | Status |
|---|---|
| 0. Setup | done |
| 1. Own forward pass + KV cache | done |
| 2. Quantization (NF4, own int4) | not started |
| 3. Continuous batching scheduler | not started |
| 4. Paged KV cache + prefix sharing | not started |
| 5. OpenAI-compatible server | not started |
| 6. Fused int4 Triton kernel | not started |
| 7. Benchmarks, profiling, write-up | not started |

---

## Phase 0: Setup (2026-10-05)

### What was built
- Repo scaffolding: `engine/`, `scheduler/`, `server/`, `kernels/` packages, `benchmarks/results/`, docs.
- `pyproject.toml` with exact pins for top-level deps, plus `requirements.lock` (full `pip freeze`).
- `Makefile`: `install`, `env`, `test`, `test-fast`, `lint`, `format`, `lock`, `clean`.
- `scripts/env_check.py`: prints the hardware/software environment, measures memory bandwidth
  and fp16 matmul throughput, and runs Triton + bitsandbytes NF4 smoke tests. `collect_env()`
  will be embedded in every benchmark result file.
- `tests/conftest.py` auto-skips `gpu`-marked tests when CUDA is absent; `tests/test_env.py` smoke tests.

### Key design decisions
- **WSL2 instead of native Windows.** vLLM is Linux-only, Triton officially supports Linux
  only, and the Phase 5 Docker image needs Linux GPU passthrough. WSL2 uses the Windows NVIDIA
  driver directly, so there's no GPU performance penalty for kernels. (Host-device copies and
  file I/O on `/mnt/c` are slower, which is why the repo lives in the Linux filesystem.)
- **Project-local `.venv` and exact pins.** Logit-parity tests (Phase 1) can shift across
  torch/transformers releases, so versions are frozen. torch comes from the cu128 index and
  brings its own matched Triton, so Triton is intentionally not pinned separately.
- **Baselines get their own venv (Phase 7).** vLLM pins its own torch version; installing it
  into the engine venv would silently change the torch our numbers are measured with.
- **Model scope for 6 GB VRAM.** 1B-class models for development and benchmarks; Llama-3.2-3B
  in 4-bit as an optional larger point. See the fit table below for the numbers behind this.

### Environment and measured numbers

| Item | Value |
|---|---|
| GPU | RTX 4050 Laptop, sm_89, 20 SMs, 24 MB L2, bf16 supported |
| VRAM | 6.44 GB total, **5.32 GB free** at startup (desktop compositor + CUDA context use the rest) |
| Driver / CUDA runtime | 617.14 / 12.8 (torch) |
| OS / Python | WSL2 (kernel 6.18) Ubuntu 24.04 / 3.12.3 |
| torch / triton / transformers / bitsandbytes | 2.11.0+cu128 / 3.6.0 / 5.18.0 / 0.50.2 |
| **Measured** read-only bandwidth | **192.4 GB/s** (fp16 sum over random data) |
| **Measured** copy bandwidth | 173.6-174.5 GB/s (read + write bytes both counted) |
| **Measured** fp16 GEMM throughput | **25.9 TFLOPS** (4096x4096x4096) |

### Correction (before Phase 1): which bandwidth sets the decode ceiling

The Phase 0 copy-bandwidth number was already counting both directions (`2 * numel *
element_size`, i.e. N bytes read + N written). But a copy is the wrong access pattern for
decode, which **reads** every weight and writes almost nothing. A read-only measurement
(a reduction over a 512 MB random fp16 buffer) gives 192.4 GB/s, matching the card's spec
peak (96-bit GDDR6 at 16 Gbps = 192 GB/s). It stays 191.8-192.7 GB/s for buffers from
256 MB to 2 GB (all much bigger than the 24 MB L2), so it's DRAM bandwidth, not a cache artifact.

Corrected batch-1 decode ceiling for TinyLlama-1.1B in fp16:

```
tokens/s <= read_bandwidth / bytes_read_per_token
bytes_read_per_token = weight_bytes + context_len * kv_bytes_per_token
                     ~ weight_bytes                 (short contexts: 22 KiB/token of KV is tiny)
weight_bytes = 1,100,048,384 params * 2 B = 2.200 GB
ceiling = 192.4 GB/s / 2.200 GB = 87.5 tokens/s     (was 79 with copy bandwidth)
```

### Model fit on this GPU (estimates from `make env`, not measurements)

KV budget = 5.32 GB free - weights - 0.75 GB assumed runtime reserve; KV cache in fp16;
"4-bit" = decoder Linear layers at 4.5 bits/param (NF4 + fp32 block scales), embeddings and
LM head kept 16-bit.

| Model | Format | Weights | KV/token | Max KV tokens | Verdict |
|---|---|---|---|---|---|
| TinyLlama-1.1B | fp16 | 2.20 GB | 22 KiB | 105,198 | comfortable |
| TinyLlama-1.1B | 4-bit | 0.81 GB | 22 KiB | 167,022 | comfortable |
| Llama-3.2-1B | fp16 | 2.47 GB | 32 KiB | 64,037 | comfortable |
| Llama-3.2-1B | 4-bit | 1.07 GB | 32 KiB | 106,725 | comfortable |
| Llama-3.2-3B | fp16 | 6.43 GB | 112 KiB | 0 | does not fit |
| Llama-3.2-3B | 4-bit | 2.37 GB | 112 KiB | 19,149 | tight (little batching) |
| Llama-3.1-8B | 4-bit | 6.03 GB | 128 KiB | 0 | does not fit |
| Llama-2-7B (no GQA) | 4-bit | 4.17 GB | 512 KiB | 767 | does not fit |

("comfortable" = KV for >= 16 sequences x 2K context; "tight" = >= 4K tokens total.)
Takeaways: the 1B models leave room for real batching experiments in either format; 3B is
only viable in 4-bit; 8B doesn't fit even in 4-bit because its untied 128K-vocab embedding
and LM head (1.05 GB each in fp16) aren't quantized. Llama-2-7B shows what GQA buys:
512 KiB/token of KV vs 128 KiB for Llama-3.1-8B.

---

## Phase 1: Own forward pass + KV cache (2026-10-05)

### What was built
- `engine/config.py`: `ModelConfig` parsed from HF `config.json` (classic and transformers-5
  RoPE layouts), with memory arithmetic: `num_params()`, `kv_bytes_per_token()`,
  `weight_bytes_4bit()`. `REFERENCE_CONFIGS` for planning without downloads.
- `engine/model.py`: Llama decoder from scratch: RMSNorm, RoPE (plus Llama-3 "llama3"
  frequency scaling), grouped-query attention (SDPA path and a naive reference path),
  SwiGLU MLP, final norm, LM head; one `forward` for both prefill and decode.
- `engine/weights.py`: safetensors loader (meta-device construction, strict key/shape checks,
  tied embeddings).
- `engine/kv_cache.py`: contiguous preallocated KV cache.
- `engine/sampling.py`: greedy / temperature / top-k / top-p, vectorized with per-row parameters.
- `engine/generate.py`: single-request prefill + decode loop with TTFT and decode-rate timing.
- `scripts/generate.py`: CLI (streams text) and `--bench` mode (roofline fraction, optional
  HF `generate()` baseline, results to `benchmarks/results/phase1_generate.jsonl`).
- Tests: `test_model_parity.py` (two tiers), `test_sampling.py`, `test_kv_cache.py`.

### Key design decisions
- **Mirror HF numerics where precision matters.** RMSNorm in fp32, softmax in fp32, RoPE
  angles in fp32 then cast. Combined with using the same SDPA kernel, our default path is
  **bit-identical** to HF in fp32, fp16 and bf16, which makes the parity test maximally sharp.
- **RoPE `rotate_half` layout.** HF checkpoints pair dimension i with i + d/2 (HF permuted
  Meta's q/k weights at conversion), so we must too. Injecting Meta's interleaved pairing
  instead fails 6 tests immediately.
- **Causal mask alignment.** SDPA's `is_causal=True` assumes query i aligns with key i
  (top-left). That's wrong when new queries follow cached keys (chunked prefill), so we pass
  an explicit bottom-right-aligned mask there; decode (1 query) needs no mask at all.
- **Preallocated contiguous cache, layout [L, B, H_kv, S, D].** Appends are slice writes (HF's
  DynamicCache `torch.cat`s, copying the full history each step); reads are views fed to SDPA
  without transposes. The cost, reserving max_seq_len per sequence, is what Phase 4 removes.
- **Meta-device loading.** Building the model on `meta` then attaching checkpoint tensors
  avoids random init and a second copy of the weights; fp32 TinyLlama (4.4 GB) would not
  otherwise fit next to a copy on a 6 GB card.
- **Last-position logits only** during generation (`logits="last"`): skips a [T, vocab] matmul
  and its memory during prefill.
- **Per-row sampling parameters** from day one, so Phase 3 can batch requests with different
  settings in one `sample()` call. Mask semantics are checked against HF's TopK/TopP warpers.
- **Kept q/k/v/o and gate/up/down as separate matmuls** for readability and 1:1 checkpoint
  mapping; fusing is a candidate optimization for Phase 7, to be decided with profiler data.
- **Correction to the plan:** TinyLlama already uses GQA (32 query heads, 4 KV heads). What
  Llama-3.2-1B adds is llama3 RoPE scaling, tied embeddings and a 128K vocab; all three are
  implemented and tested on a tiny random Llama-3.2-shaped model (the real one is gated).

### Parity results (TinyLlama-1.1B-Chat, 5 prompts: 3 raw, 2 chat-templated)

Logit error, max / mean absolute difference over all prompt positions:

| Path | dtype | ours vs HF same dtype | ours vs HF fp32 | HF same dtype vs HF fp32 |
|---|---|---|---|---|
| sdpa (default) | fp32 | 0 (bit-identical) | 0 | 0 |
| naive | fp32 | 2.5e-5 max | 2.5e-5 max | 0 |
| sdpa | fp16 | 0 (bit-identical) | 0.21 / 8.1e-3 | 0.21 / 8.1e-3 |
| naive | fp16 | 3.1e-2 max | 0.22 / 8.7e-3 | 0.21 / 8.1e-3 |
| sdpa | bf16 | 0 (bit-identical) | 0.73 / 3.9e-2 | 0.73 / 3.9e-2 |
| naive | bf16 | 0.29 max | 0.73 / 4.0e-2 | 0.73 / 3.9e-2 |

(Logits have magnitude up to ~22.) Test tolerances and why:
- **fp32: max |diff| <= 1e-4.** Only summation order may differ; measured worst case 2.5e-5.
- **fp16/bf16: error vs fp32 <= 1.5x HF's own error vs fp32** (max and mean). Two correct
  16-bit implementations may round differently; the question is whether ours is as accurate
  as HF's. Measured worst ratio 1.26x (naive, bf16, mean).
- **Greedy token-for-token (48 new tokens x 5 prompts)** for fp32 (both paths) and SDPA in
  fp16/bf16: all match. Not asserted for naive 16-bit: one near-tie top-1 flip was observed
  there in bf16, which legitimately changes everything after it.

**Testing the tests** (bugs injected deliberately, then the suite run):

| Injected bug | Caught by |
|---|---|
| Meta-style interleaved RoPE pairing | 6 fast tests (tiny-model logits + greedy) |
| RoPE positions restart at 0 after prefill | 10 fast tests (cached decode, chunked prefill, greedy) |
| `is_causal=True` for chunked prefill (top-left mask) | 4 fast tests (chunked prefill) |
| RMSNorm without fp32 upcast | only bf16 greedy exact-match (slow) + new direct overflow test |

The last row is an honest limitation: on TinyLlama the missing upcast changes error by
1.0-1.34x, inside 16-bit noise, so a tolerance test *cannot* see it. Its real failure mode is
fp16 overflow (x^2 > 65504 when |x| > 256; we measured |x| up to 179.5 entering TinyLlama's
norms), so `test_rmsnorm_fp16_large_activations_do_not_overflow` tests that directly.

### Performance (batch 1, 256 new tokens, median of 5 runs after warmup)

| Config | TTFT | Decode tok/s | Roofline | Fraction | HF `generate()` decode |
|---|---|---|---|---|---|
| fp16, 32-token prompt | 15.1 ms | **71.6** | 87.5 | 82% | 67.8 |
| fp16, 512-token prompt | 51.0 ms | 69.4 | 87.5 | 79% | 63.4 |
| bf16, 32-token prompt | 16.5 ms | 67.9 | 87.4 | 78% | 50.6 |
| fp32, 32-token prompt | 29.2 ms | 39.2 | 43.7 | 90% | 37.6 |

Peak GPU memory: 2.22 GB (fp16), 4.43 GB (fp32). HF's TTFT is a separately timed prefill
forward; its decode rate is the rest of `generate()`'s time (an approximation; noted in the script).

**Analysis.**
- **Decode is memory-bound and we're within ~20% of the bandwidth roofline.** The gap is a
  near-constant ~2.5 ms per token in every dtype (fp16: 13.97 ms measured vs 11.43 ms ideal;
  fp32: 25.5 ms vs 22.9 ms). A fixed cost that doesn't scale with bytes read is CPU-side:
  Python dispatch and ~300 kernel launches per token (22 layers x ~14 ops), plus the
  per-token `.item()` sync. That's why fp32 reaches a higher fraction: same overhead,
  twice the memory time. Candidate fixes (Phase 7, profiler-driven): CUDA graphs, fused QKV
  and gate/up matmuls.
- **Prefill is compute-bound.** 512 tokens in 51 ms is 2 x 1.03B (non-LM-head) params x 512
  tokens / 0.051 s, about 21 TFLOPS: ~80% of the measured 25.9 TFLOPS GEMM peak.
- **vs HF generate():** ours is 5-9% faster in fp16/fp32 decode. HF's bf16 decode (50.6 tok/s)
  was much slower than its fp16; I didn't investigate why, so no claim about the cause.
- Longer context costs little at this scale: 512-token context decodes at 69.4 vs 71.6 tok/s,
  since 512 x 22 KiB = 11 MB of KV per step is small next to 2.2 GB of weights.

### How to run
```bash
make test-fast                       # 75 tests, ~5 s, no GPU/download
make test                            # 89 tests, ~60 s (downloads TinyLlama once, ~2.2 GB)
python scripts/generate.py --chat --prompt "Write a haiku about GPUs."
python scripts/generate.py --bench --prompt-len 512 --max-tokens 256 --compare-hf --save
```

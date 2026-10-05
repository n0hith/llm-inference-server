# Progress Log

One entry per phase: what was built, key design decisions, and numbers actually measured
on the dev machine. Any number not measured here is explicitly labelled as an estimate.

| Phase | Status |
|---|---|
| 0. Setup | done |
| 1. Own forward pass + KV cache | not started |
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
  in 4-bit as an optional larger point. A 7B model would fit only in 4-bit, with almost
  no room left for KV cache, so batching benchmarks on it would be meaningless.

### Environment and measured numbers

| Item | Value |
|---|---|
| GPU | RTX 4050 Laptop, 6.0 GB, sm_89, 20 SMs, bf16 supported |
| Driver / CUDA runtime | 617.14 / 12.8 (torch) |
| OS / Python | WSL2 (kernel 6.18) Ubuntu 24.04 / 3.12.3 |
| torch / triton / transformers / bitsandbytes | 2.11.0+cu128 / 3.6.0 / 5.18.0 / 0.50.2 |
| **Measured** device-to-device copy bandwidth | **173.6 GB/s** (512 MB fp16 copy) |
| **Measured** fp16 GEMM throughput | **25.9 TFLOPS** (4096x4096x4096) |
| Tests | 5 passed (`make test`), 2 passed / 3 GPU-deselected (`make test-fast`) |

**Why these two numbers matter.** Batch-1 decode reads every weight once per generated token,
so it is memory-bandwidth-bound. *Estimate (not a measurement):* TinyLlama-1.1B in fp16 is
about 2.2 GB of weights, so the decode ceiling is about 173.6 / 2.2 ≈ **79 tokens/s**. Phase 1
will report what our engine actually achieves as a fraction of that roofline.

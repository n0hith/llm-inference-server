# Low-Latency Quantized LLM Inference Server

A from-scratch LLM inference engine and OpenAI-compatible server: my own Llama forward
pass and KV cache, 4-bit quantization, a continuous-batching scheduler, a paged KV cache
with prefix sharing, and a fused int4 Triton kernel, all benchmarked honestly against
vLLM and llama.cpp.

> **Status:** Phase 0 (setup) complete. See [docs/PROGRESS.md](docs/PROGRESS.md) for the
> phase-by-phase log. Results, the architecture diagram, and design notes get added as
> each phase lands.

## Hardware this was developed on

| | |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop, 6 GB VRAM (Ada, sm_89) |
| CPU / RAM | Intel i7-13700HX / 16 GB |
| OS | Windows 11 + WSL2 Ubuntu 24.04, Python 3.12 |

The 6 GB VRAM limit is why development and benchmarks use 1B-class models
(TinyLlama-1.1B, Llama-3.2-1B), with Llama-3.2-3B in 4-bit as an optional larger data point.

## Quick start (WSL2 / Linux with an NVIDIA GPU)

```bash
make install     # creates .venv, installs pinned deps (torch from the CUDA 12.8 index)
make env         # prints GPU / CUDA / library versions and runs GPU smoke tests
make test-fast   # tests that need no GPU and no model download
make test        # full suite
```

## Repository layout

```
engine/       our model, KV caches, sampling, quantization
scheduler/    request state, continuous batching, paged-KV block manager
server/       FastAPI OpenAI-compatible API + SSE streaming
kernels/      Triton fused int4 dequant+matmul
benchmarks/   load tests, kernel microbenchmarks, baseline comparisons (+ results/)
scripts/      env_check.py and other utilities
tests/        parity, unit, property-based, and server tests
docs/         PROGRESS.md (phase log), ARCHITECTURE.md (request flow)
```

# Architecture

> Filled in progressively. Each phase adds its component below; Phase 7 adds the
> full end-to-end request-flow walkthrough and diagram.

## Components (planned)

| Component | Module | Phase |
|---|---|---|
| Llama forward pass (RMSNorm, RoPE, GQA, SwiGLU) | `engine/model.py` | 1 |
| HF weight loading | `engine/weights.py` | 1 |
| Contiguous KV cache | `engine/kv_cache.py` | 1 |
| Sampling (greedy / temperature / top-k / top-p) | `engine/sampling.py` | 1 |
| NF4 / own int4 quantization | `engine/quantization/` | 2 |
| Continuous batching scheduler | `scheduler/scheduler.py` | 3 |
| Paged KV cache + prefix sharing | `scheduler/block_manager.py`, `engine/kv_cache.py` | 4 |
| OpenAI-compatible server + SSE | `server/` | 5 |
| Fused int4 dequant+matmul | `kernels/triton_int4_matmul.py` | 6 |

## Request flow

_To be written in Phase 5/7._

# Architecture

> Filled in progressively. Each phase adds its component below; Phase 7 adds the
> full end-to-end request-flow walkthrough and diagram.

## Components

| Component | Module | Phase |
|---|---|---|
| Model / engine config, memory arithmetic | `engine/config.py` | 1 |
| Llama forward pass (RMSNorm, RoPE, GQA, SwiGLU) | `engine/model.py` | 1 |
| HF safetensors weight loading | `engine/weights.py` | 1 |
| Contiguous KV cache | `engine/kv_cache.py` | 1 |
| Sampling (greedy / temperature / top-k / top-p) | `engine/sampling.py` | 1 |
| Single-request generation loop | `engine/generate.py` | 1 |
| NF4 / own int4 quantization | `engine/quantization/` | 2 |
| Continuous batching scheduler | `scheduler/scheduler.py` | 3 |
| Paged KV cache + prefix sharing | `scheduler/block_manager.py`, `engine/kv_cache.py` | 4 |
| OpenAI-compatible server + SSE | `server/` | 5 |
| Fused int4 dequant+matmul | `kernels/triton_int4_matmul.py` | 6 |

## KV-cache memory math

During generation, every layer keeps the key and value vectors of every past token so they
don't have to be recomputed. Per token of context, summed over all layers:

```
kv_bytes_per_token = 2 (K and V) * num_layers * num_kv_heads * head_dim * bytes_per_element
```

Only the **KV heads** count, not the query heads. With grouped-query attention (GQA),
several query heads share one K/V head, so the cache shrinks by `num_heads / num_kv_heads`.
Implemented in `ModelConfig.kv_bytes_per_token()`.

### TinyLlama-1.1B (values from its `config.json`)

`num_layers = 22`, `num_kv_heads = 4` (with 32 query heads, so 8 query heads per KV head),
`head_dim = 2048 / 32 = 64`, cache dtype fp16 (2 bytes):

```
2 * 22 * 4 * 64 * 2 B = 22,528 B = 22 KiB per token
```

| Quantity | Value |
|---|---|
| One token, one layer (K+V) | 2 * 4 * 64 * 2 B = 1,024 B = 1 KiB |
| One token, all 22 layers | **22 KiB** |
| One 2,048-token sequence (max context) | 22,528 * 2,048 = 46.1 MB (44 MiB) |
| Same model with full multi-head attention (32 KV heads) | 176 KiB/token, 8x more |
| A 16-token paged block (Phase 4), all layers | 16 * 22,528 B = 352 KiB |

### Other models (from `REFERENCE_CONFIGS`)

| Model | Layers | KV heads | head_dim | KV/token (fp16) |
|---|---|---|---|---|
| TinyLlama-1.1B | 22 | 4 | 64 | 22 KiB |
| Llama-3.2-1B | 16 | 8 | 64 | 32 KiB |
| Llama-3.2-3B | 28 | 8 | 128 | 112 KiB |
| Llama-3.1-8B | 32 | 8 | 128 | 128 KiB |
| Llama-2-7B (no GQA) | 32 | 32 | 128 | 512 KiB |

### Why this drives the Phase 3/4 design

The KV budget is what's left after weights and runtime overhead:

```
kv_budget  = free_vram - weight_bytes - runtime_reserve
max_tokens = kv_budget / kv_bytes_per_token      (shared across ALL running sequences)
```

On the dev GPU (5.32 GB free, 0.75 GB assumed reserve) TinyLlama in fp16 leaves about
2.37 GB, or about **105K tokens** of cache in total (estimate; see `make env`).

* **Contiguous cache (Phase 1):** each sequence reserves `max_seq_len` slots up front.
  At 2,048 tokens that is 44 MiB per sequence, so about 51 concurrent sequences, *even if*
  most finish after 200 tokens and the reserved memory sits ~90% empty.
* **Paged cache (Phase 4):** memory is allocated in fixed blocks (e.g. 16 tokens = 352 KiB)
  as a sequence grows, so waste is at most one partly-filled block per sequence, and the
  same 105K-token budget serves many more concurrent sequences of realistic length.
  Prefix sharing goes further: sequences with a common prompt prefix point at the same
  blocks instead of storing duplicate copies.

## Request flow

_To be written in Phase 5/7._

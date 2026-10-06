"""Single-request generation loop: prefill once, then decode one token at a time.

    prompt ids --prefill--> logits for the last prompt position --sample--> token 1
    token 1    --decode---> logits --sample--> token 2 ... until EOS or max_tokens

This is the Phase 1 "engine": one sequence, one contiguous KV cache, no batching. The
Phase 3 scheduler replaces this loop with one that interleaves many sequences.

Timing: `time_to_first_token_s` covers prefill + first sample (what a user waits before
anything appears); `decode_tokens_per_s` covers the remaining tokens only, so the two
phases' very different performance characteristics aren't blended into one number.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from engine.kv_cache import KVCache
from engine.model import LlamaModel
from engine.sampling import SamplingParams, make_generator, params_to_tensors, sample


@dataclass
class GenerationResult:
    prompt_ids: list[int]
    output_ids: list[int] = field(default_factory=list)
    finish_reason: str = ""  # "stop" (EOS) or "length" (max_tokens / context limit)
    time_to_first_token_s: float = 0.0
    decode_time_s: float = 0.0

    @property
    def decode_tokens_per_s(self) -> float:
        """Tokens/s over the decode phase (excludes the first token, which prefill produced)."""
        n = len(self.output_ids) - 1
        return n / self.decode_time_s if n > 0 and self.decode_time_s > 0 else 0.0


@torch.inference_mode()
def stream_generate(
    model: LlamaModel,
    prompt_ids: list[int],
    params: SamplingParams,
    eos_token_ids: tuple[int, ...] = (),
    result: GenerationResult | None = None,
) -> Iterator[int]:
    """Yield generated token ids one by one (streaming-friendly).

    If `result` is given, it's filled with outputs, finish reason and timings as we go.
    """
    cfg, device = model.cfg, model.device
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    max_seq_len = min(len(prompt_ids) + params.max_tokens, cfg.max_position_embeddings)
    if len(prompt_ids) >= max_seq_len:
        raise ValueError(f"prompt ({len(prompt_ids)} tokens) leaves no room in the context window")

    res = result if result is not None else GenerationResult(prompt_ids=list(prompt_ids))
    cache = KVCache(cfg, batch_size=1, max_seq_len=max_seq_len, dtype=model.dtype, device=device)
    temperature, top_k, top_p = params_to_tensors([params], device)
    generator = make_generator(params.seed, device)
    stop_ids = set() if params.ignore_eos else set(eos_token_ids)

    t0 = time.perf_counter()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # prefill: whole prompt
    while True:
        logits = model(input_ids, cache=cache, logits="last")  # [1, 1, vocab]
        next_id = sample(logits[:, -1, :], temperature, top_k, top_p, generator)
        # .item() copies the token to the host, which also waits for the GPU to finish the
        # step. Unavoidable here: we need the value to check EOS and to stream it out.
        token = int(next_id.item())

        now = time.perf_counter()
        if not res.output_ids:
            res.time_to_first_token_s = now - t0
            t_decode_start = now
        else:
            res.decode_time_s = now - t_decode_start
        res.output_ids.append(token)
        yield token

        if token in stop_ids:
            res.finish_reason = "stop"
            return
        if len(res.output_ids) >= params.max_tokens or cache.seq_len >= max_seq_len:
            res.finish_reason = "length"
            return
        input_ids = next_id.view(1, 1)  # decode: just the new token; history lives in the cache


def generate(
    model: LlamaModel,
    prompt_ids: list[int],
    params: SamplingParams,
    eos_token_ids: tuple[int, ...] = (),
) -> GenerationResult:
    """Non-streaming convenience wrapper around `stream_generate`."""
    result = GenerationResult(prompt_ids=list(prompt_ids))
    for _ in stream_generate(model, prompt_ids, params, eos_token_ids, result):
        pass
    return result

"""Single-request generation with our engine: stream text, or benchmark tokens/sec.

Examples:
    python scripts/generate.py --prompt "Write a haiku about GPUs." --chat
    python scripts/generate.py --prompt "Once upon a time" --temperature 0.8 --top-p 0.9 --seed 0
    python scripts/generate.py --bench --prompt-len 512 --max-tokens 256 --compare-hf

Benchmark mode (--bench) forces fixed-length outputs (ignore EOS), does a warmup run, then
reports the median over --runs of:
  * TTFT: prefill + first token latency
  * decode tok/s: tokens 2..N divided by their wall time
  * fraction of the bandwidth roofline: decode tok/s / (read_bandwidth / weight_bytes)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from engine.generate import GenerationResult, stream_generate
from engine.sampling import SamplingParams
from engine.weights import load_model, resolve_model_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_check  # noqa: E402

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
RESULTS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "results"


def build_prompt_ids(tok, args: argparse.Namespace) -> list[int]:
    if args.prompt_len:  # synthetic prompt of an exact token length (benchmarks)
        sentence = "The quick brown fox jumps over the lazy dog. "
        filler = tok(sentence, add_special_tokens=False).input_ids
        body = (filler * (args.prompt_len // len(filler) + 1))[: args.prompt_len - 1]
        return [tok.bos_token_id, *body]
    if args.chat:
        text = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True, tokenize=False
        )
        return tok(text).input_ids
    return tok(args.prompt).input_ids


def run_once(model, prompt_ids, params, eos, tok=None) -> GenerationResult:
    """Generate once; if `tok` is given, stream decoded text to stdout as it arrives."""
    result = GenerationResult(prompt_ids=list(prompt_ids))
    printed = ""
    for _ in stream_generate(model, prompt_ids, params, eos, result):
        if tok is not None:
            # Decode the whole output each step and print only the new suffix: a token can be
            # a partial UTF-8 character or merge with the previous one, so decoding tokens
            # one at a time would garble text. (O(n^2) in output length; fine for a CLI.)
            text = tok.decode(result.output_ids, skip_special_tokens=True)
            if not text.endswith("�"):  # wait until a multi-byte char is complete
                print(text[len(printed) :], end="", flush=True)
                printed = text
    if tok is not None:
        print()
    return result


def time_hf_generate(
    model_dir: Path, dtype: torch.dtype, prompt_ids: list[int], n_new: int, runs: int
):
    """Baseline only (never used by the engine): HF generate() on the same prompt and length.

    HF doesn't expose TTFT directly, so we time a bare prefill forward separately and
    attribute the rest of generate()'s time to decoding the remaining n_new - 1 tokens.
    """
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).cuda().eval()
    x = torch.tensor([prompt_ids], device="cuda")
    kw = dict(max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False)
    ttfts, decode_rates = [], []
    with torch.inference_mode():
        hf.generate(x, **kw)  # warmup
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            hf(x, logits_to_keep=1).logits.argmax(-1).item()
            ttft = time.perf_counter() - t0
            t0 = time.perf_counter()
            hf.generate(x, **kw)
            torch.cuda.synchronize()
            total = time.perf_counter() - t0
            ttfts.append(ttft)
            decode_rates.append((n_new - 1) / (total - ttft))
    del hf
    torch.cuda.empty_cache()
    return statistics.median(ttfts), statistics.median(decode_rates)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate text with our engine (single request).")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--prompt", default="Explain what a KV cache is in two sentences.")
    p.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", choices=list(DTYPES), default="fp16")
    p.add_argument("--attn", choices=["sdpa", "naive"], default="sdpa")
    p.add_argument("--bench", action="store_true", help="benchmark mode (fixed lengths, no text)")
    p.add_argument("--prompt-len", type=int, default=0, help="synthetic prompt of N tokens")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--compare-hf", action="store_true", help="also time HF generate() (baseline)")
    p.add_argument("--save", action="store_true", help="append results to benchmarks/results/")
    args = p.parse_args()

    model_dir = resolve_model_dir(args.model)
    tok = AutoTokenizer.from_pretrained(model_dir)
    dtype = DTYPES[args.dtype]
    model = load_model(model_dir, dtype=dtype, attn_impl=args.attn)
    prompt_ids = build_prompt_ids(tok, args)
    eos = model.cfg.eos_token_ids
    # Bytes decode must read per token. parameters() yields a tied embedding/LM-head matrix
    # once, which is right: it's one tensor in memory.
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    if not args.bench:
        params = SamplingParams(
            args.temperature, args.top_k, args.top_p, args.max_tokens, args.seed
        )
        r = run_once(model, prompt_ids, params, eos, tok)
        print(f"\n[{len(prompt_ids)} prompt tokens, {len(r.output_ids)} generated, "
              f"finish={r.finish_reason}] TTFT {r.time_to_first_token_s * 1e3:.1f} ms, "
              f"decode {r.decode_tokens_per_s:.1f} tok/s")  # fmt: skip
        return 0

    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    run_once(model, prompt_ids, params, eos)  # warmup: cuBLAS heuristics, allocator, SDPA dispatch
    torch.cuda.reset_peak_memory_stats()
    runs = [run_once(model, prompt_ids, params, eos) for _ in range(args.runs)]
    ttft = statistics.median(r.time_to_first_token_s for r in runs)
    tps = statistics.median(r.decode_tokens_per_s for r in runs)
    peak_mem = torch.cuda.max_memory_allocated()

    read_bw = env_check.measure_read_bandwidth_gbs()
    ceiling = read_bw * 1e9 / weight_bytes
    out = {
        "model": args.model,
        "dtype": args.dtype,
        "attn": args.attn,
        "prompt_tokens": len(prompt_ids),
        "new_tokens": args.max_tokens,
        "runs": args.runs,
        "ttft_ms": round(ttft * 1e3, 2),
        "decode_tok_s": round(tps, 1),
        "weight_gb": round(weight_bytes / 1e9, 3),
        "read_bandwidth_gbs": round(read_bw, 1),
        "roofline_tok_s": round(ceiling, 1),
        "roofline_fraction": round(tps / ceiling, 3),
        "peak_mem_gb": round(peak_mem / 1e9, 3),
    }
    print(f"ours   : TTFT {out['ttft_ms']:.1f} ms | decode {tps:.1f} tok/s "
          f"({out['roofline_fraction']:.0%} of {ceiling:.1f} tok/s roofline) | "
          f"peak mem {out['peak_mem_gb']:.2f} GB")  # fmt: skip

    if args.compare_hf:
        del model
        torch.cuda.empty_cache()
        hf_ttft, hf_tps = time_hf_generate(model_dir, dtype, prompt_ids, args.max_tokens, args.runs)
        out["hf_ttft_ms"], out["hf_decode_tok_s"] = round(hf_ttft * 1e3, 2), round(hf_tps, 1)
        print(f"HF gen : TTFT {hf_ttft * 1e3:.1f} ms | decode {hf_tps:.1f} tok/s  (baseline)")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "phase1_generate.jsonl"
        out["env"] = env_check.collect_env()
        with path.open("a") as f:
            f.write(json.dumps(out) + "\n")
        print(f"appended to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

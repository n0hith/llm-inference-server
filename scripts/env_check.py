"""Environment check: print hardware + library versions and run GPU smoke tests.

Why this exists: inference performance numbers are meaningless without the exact
hardware/software context, so every benchmark in this repo embeds the output of
`collect_env()`. It also measures the numbers we keep coming back to:

* **Read bandwidth (GB/s)**: batch-1 decode reads every weight once per token and writes
  almost nothing, so its speed ceiling is

      tokens/s  <=  read_bandwidth / bytes_read_per_token
      bytes_read_per_token = weight_bytes + context_len * kv_bytes_per_token

  For short contexts the weights dominate. This is the "roofline" we compare our decode
  speed against.
* **Copy bandwidth (GB/s)**: counts BOTH directions (a copy of N bytes reads N and writes
  N, so 2N bytes moved). Reported for reference; read bandwidth is the relevant one for decode.
* **fp16 matmul TFLOPS**: the compute ceiling that matters for prefill and large-batch decode.
* **Free VRAM + model fit table**: which models fit with room left for KV cache.

Run: `make env` (or `python scripts/env_check.py [--json]`).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from engine.config import REFERENCE_CONFIGS, ModelConfig

LIBRARIES = [
    "torch",
    "triton",
    "transformers",
    "safetensors",
    "tokenizers",
    "huggingface_hub",
    "bitsandbytes",
    "datasets",
    "fastapi",
    "uvicorn",
    "numpy",
]

GB = 1e9  # decimal GB everywhere (bandwidths are quoted in decimal GB/s)

# Memory held back from the KV-cache budget for everything that isn't weights or KV:
# activations (we only compute last-token logits during prefill, so no T x vocab tensor),
# cuBLAS workspace, and caching-allocator fragmentation. This is an ASSUMPTION for planning,
# to be replaced with a measured peak once the engine exists.
RUNTIME_RESERVE_BYTES = int(0.75 * GB)

# A model "fits comfortably" if the KV budget left over holds 16 sequences of 2K context
# (enough for meaningful batching experiments), and "fits tight" if it holds at least one
# 4K-token sequence's worth of cache.
COMFORTABLE_KV_TOKENS = 16 * 2048
TIGHT_KV_TOKENS = 4096


def _version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _is_wsl() -> bool:
    return "microsoft" in platform.uname().release.lower()


def _driver_version() -> str:
    """The driver comes from the Windows host under WSL2; nvidia-smi still reports it."""
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi not found"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def collect_env() -> dict[str, Any]:
    """Return a JSON-serializable description of the environment (embedded in benchmark results)."""
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "wsl": _is_wsl(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "libraries": {lib: _version(lib) for lib in LIBRARIES},
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        # mem_get_info asks the driver, so "free" excludes our own CUDA context and anything
        # else using the GPU (on Windows/WSL2 that includes the desktop compositor).
        free, total = torch.cuda.mem_get_info()
        env["gpu"] = {
            "name": props.name,
            "vram_total_gb": round(total / GB, 2),
            "vram_free_gb": round(free / GB, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "torch_cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "driver": _driver_version(),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    return env


def _time_cuda(fn, iters: int) -> float:
    """Average milliseconds per call, timed with CUDA events (host timers would only
    measure kernel *launch*, since CUDA kernels run asynchronously)."""
    for _ in range(3):  # warmup: first calls pay for cuBLAS heuristics / allocator growth
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def measure_copy_bandwidth_gbs(size_mb: int = 512) -> float:
    """Device-to-device copy bandwidth, counting read + write: a copy of N bytes moves 2N."""
    n = size_mb * 1024**2 // 2  # fp16 elements
    src = torch.empty(n, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    ms = _time_cuda(lambda: dst.copy_(src), iters=20)
    bytes_moved = 2 * src.numel() * src.element_size()  # read src + write dst
    return bytes_moved / (ms / 1e3) / GB


def measure_read_bandwidth_gbs(size_mb: int = 512) -> float:
    """Read-only bandwidth: a sum reduction reads N bytes and writes one scalar.

    This is the access pattern of decode (stream the weights, write a tiny output), so it's
    the right denominator for the decode roofline. 512 MB is far larger than the L2 cache,
    so we're measuring DRAM, not cache. Random data (not zeros/ones) so no hardware
    compression or other constant-data shortcut can inflate the number.
    """
    n = size_mb * 1024**2 // 2
    src = torch.randn(n, dtype=torch.float16, device="cuda")
    ms = _time_cuda(lambda: src.sum(), iters=20)
    return src.numel() * src.element_size() / (ms / 1e3) / GB


def measure_matmul_tflops(n: int = 4096) -> float:
    """fp16 GEMM throughput. An (n x n) @ (n x n) matmul is 2*n^3 FLOPs (multiply + add)."""
    a = torch.randn(n, n, dtype=torch.float16, device="cuda")
    b = torch.randn(n, n, dtype=torch.float16, device="cuda")
    ms = _time_cuda(lambda: a @ b, iters=20)
    return 2 * n**3 / (ms / 1e3) / 1e12


def model_fit_table(free_bytes: int, read_bw_gbs: float | None) -> list[dict[str, Any]]:
    """For each reference model and weight format: does it fit, and how much KV is left?

    kv_budget = free_vram - weights - RUNTIME_RESERVE_BYTES
    max_kv_tokens = kv_budget / kv_bytes_per_token(fp16)   (summed over all sequences)
    decode_ceiling = read_bw / weight_bytes                  (batch 1, short context)
    """
    rows = []
    for name, hf_dict in REFERENCE_CONFIGS.items():
        cfg = ModelConfig.from_hf_dict(hf_dict, name)
        kv_per_tok = cfg.kv_bytes_per_token(torch.float16)
        for fmt, wbytes in [("fp16", cfg.weight_bytes()), ("4-bit", cfg.weight_bytes_4bit())]:
            budget = free_bytes - wbytes - RUNTIME_RESERVE_BYTES
            kv_tokens = max(budget, 0) // kv_per_tok
            if kv_tokens >= COMFORTABLE_KV_TOKENS:
                verdict = "comfortable"
            elif kv_tokens >= TIGHT_KV_TOKENS:
                verdict = "tight (little batching)"
            else:
                verdict = "does not fit"
            rows.append(
                {
                    "model": name,
                    "format": fmt,
                    "params_b": round(cfg.num_params() / 1e9, 2),
                    "weights_gb": round(wbytes / GB, 2),
                    "kv_kib_per_token": round(kv_per_tok / 1024, 1),
                    "max_kv_tokens": int(kv_tokens),
                    "decode_ceiling_tok_s": (
                        round(read_bw_gbs * GB / wbytes, 1) if read_bw_gbs else None
                    ),
                    "verdict": verdict,
                }
            )
    return rows


# A Triton kernel must live in a real file (Triton reads the source with `inspect`),
# so we write this tiny vector-add to a temp module and import it.
_TRITON_SMOKE = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
"""


def check_triton() -> str:
    import triton

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "triton_smoke.py"
        path.write_text(_TRITON_SMOKE)
        spec = importlib.util.spec_from_file_location("triton_smoke", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        x = torch.randn(10_000, device="cuda")
        out = torch.empty_like(x)
        n_blocks = triton.cdiv(x.numel(), 1024)
        mod.add_kernel[(n_blocks,)](x, x, out, x.numel(), BLOCK=1024)
        torch.cuda.synchronize()
    if not torch.allclose(out, 2 * x):
        raise AssertionError("Triton vector-add produced wrong results")
    return "ok"


def check_bitsandbytes_nf4() -> str:
    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(256, 256, compute_dtype=torch.float16, quant_type="nf4").cuda()
    y = layer(torch.randn(4, 256, dtype=torch.float16, device="cuda"))
    if y.shape != (4, 256) or not torch.isfinite(y).all():
        raise AssertionError("bitsandbytes NF4 layer produced bad output")
    return "ok"


def run_smoke_tests() -> dict[str, str]:
    results: dict[str, str] = {}
    for name, fn in [("triton", check_triton), ("bitsandbytes_nf4", check_bitsandbytes_nf4)]:
        try:
            results[name] = fn()
        except Exception as e:  # report every failure instead of stopping at the first
            results[name] = f"FAILED: {type(e).__name__}: {e}"
    return results


def _print_report(env: dict[str, Any]) -> None:
    print(f"Platform     : {env['platform']}  (WSL2: {env['wsl']})")
    print(f"Python       : {env['python']}   CPUs: {env['cpu_count']}")
    if env["cuda_available"]:
        g = env["gpu"]
        print(f"GPU          : {g['name']}  sm_{g['compute_capability']}, {g['sm_count']} SMs")
        print(f"VRAM         : {g['vram_total_gb']} GB total, {g['vram_free_gb']} GB free")
        print(f"CUDA runtime : {g['torch_cuda_runtime']} (torch)  driver {g['driver']}")
        print(f"cuDNN / bf16 : {g['cudnn']} / {g['bf16_supported']}")
        m = env["measured"]
        print(
            f"Bandwidth    : {m['read_bandwidth_gbs']} GB/s read-only, "
            f"{m['copy_bandwidth_gbs']} GB/s copy (read+write counted)"
        )
        print(f"fp16 GEMM    : {m['fp16_matmul_tflops']} TFLOPS")
    else:
        print("GPU          : none (CUDA not available) -- only CPU paths will work")
    print("Libraries    :")
    for lib, ver in env["libraries"].items():
        print(f"  {lib:<16} {ver}")
    if env.get("smoke_tests"):
        print("Smoke tests  :")
        for name, res in env["smoke_tests"].items():
            print(f"  {name:<16} {res}")
    if env.get("model_fit"):
        print(
            f"\nModel fit (free VRAM - weights - {RUNTIME_RESERVE_BYTES / GB:.2f} GB reserve "
            "= KV budget; KV in fp16; estimates, not measurements):"
        )
        hdr = (
            f"  {'model':<15}{'fmt':<7}{'weights':>9}{'KV/token':>11}"
            f"{'max KV tok':>12}{'ceil tok/s':>12}  verdict"
        )
        print(hdr)
        for r in env["model_fit"]:
            ceil = r["decode_ceiling_tok_s"] if r["verdict"] != "does not fit" else "-"
            print(
                f"  {r['model']:<15}{r['format']:<7}{r['weights_gb']:>7.2f}GB"
                f"{r['kv_kib_per_token']:>7.1f}KiB{r['max_kv_tokens']:>12,}{ceil!s:>12}"
                f"  {r['verdict']}"
            )
        print("  ceil tok/s = read bandwidth / weight bytes (batch-1 decode upper bound).")
    # fmt: on


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    env = collect_env()
    if env["cuda_available"]:
        env["measured"] = {
            "read_bandwidth_gbs": round(measure_read_bandwidth_gbs(), 1),
            "copy_bandwidth_gbs": round(measure_copy_bandwidth_gbs(), 1),
            "fp16_matmul_tflops": round(measure_matmul_tflops(), 1),
        }
        env["smoke_tests"] = run_smoke_tests()
        free_bytes = int(env["gpu"]["vram_free_gb"] * GB)  # measured before our test tensors
        env["model_fit"] = model_fit_table(free_bytes, env["measured"]["read_bandwidth_gbs"])

    if args.json:
        print(json.dumps(env, indent=2))
    else:
        _print_report(env)

    failed = [k for k, v in env.get("smoke_tests", {}).items() if v != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

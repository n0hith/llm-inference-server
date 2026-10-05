"""Environment check: print hardware + library versions and run GPU smoke tests.

Why this exists: inference performance numbers are meaningless without the exact
hardware/software context, so every benchmark in this repo embeds the output of
`collect_env()`. It also measures two numbers we'll keep coming back to:

* **Achieved memory bandwidth (GB/s)** -- single-sequence decode reads every weight
  once per token, so tokens/sec is bounded by roughly bandwidth / model_bytes.
  This is the "roofline" we compare our decode speed against in later phases.
* **Achieved fp16 matmul TFLOPS** -- the compute ceiling that matters for prefill and
  large-batch decode.

Run: `make env` (or `python scripts/env_check.py [--json]`).
"""

from __future__ import annotations

import argparse
import importlib.metadata
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
        env["gpu"] = {
            "name": props.name,
            "vram_gb": round(props.total_memory / 1024**3, 2),
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


def measure_bandwidth_gbs(size_mb: int = 512) -> float:
    """Device-to-device copy bandwidth. A copy reads N bytes and writes N bytes, so it moves 2N."""
    n = size_mb * 1024**2 // 2  # fp16 elements
    src = torch.empty(n, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    ms = _time_cuda(lambda: dst.copy_(src), iters=20)
    return 2 * src.numel() * src.element_size() / (ms / 1e3) / 1e9


def measure_matmul_tflops(n: int = 4096) -> float:
    """fp16 GEMM throughput. An (n x n) @ (n x n) matmul is 2*n^3 FLOPs (multiply + add)."""
    a = torch.randn(n, n, dtype=torch.float16, device="cuda")
    b = torch.randn(n, n, dtype=torch.float16, device="cuda")
    ms = _time_cuda(lambda: a @ b, iters=20)
    return 2 * n**3 / (ms / 1e3) / 1e12


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
    import importlib.util

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    env = collect_env()
    if env["cuda_available"]:
        env["measured"] = {
            "d2d_bandwidth_gbs": round(measure_bandwidth_gbs(), 1),
            "fp16_matmul_tflops": round(measure_matmul_tflops(), 1),
        }
        env["smoke_tests"] = run_smoke_tests()

    if args.json:
        print(json.dumps(env, indent=2))
    else:
        print(f"Platform     : {env['platform']}  (WSL2: {env['wsl']})")
        print(f"Python       : {env['python']}   CPUs: {env['cpu_count']}")
        if env["cuda_available"]:
            g = env["gpu"]
            print(f"GPU          : {g['name']}  {g['vram_gb']} GB  sm_{g['compute_capability']}")
            print(f"             : {g['sm_count']} SMs, bf16={g['bf16_supported']}")
            print(f"CUDA runtime : {g['torch_cuda_runtime']} (torch)  driver {g['driver']}")
            print(f"cuDNN        : {g['cudnn']}")
            m = env["measured"]
            print(
                f"Measured     : {m['d2d_bandwidth_gbs']} GB/s copy bandwidth, "
                f"{m['fp16_matmul_tflops']} fp16 TFLOPS"
            )
        else:
            print("GPU          : none (CUDA not available) -- only CPU paths will work")
        print("Libraries    :")
        for lib, ver in env["libraries"].items():
            print(f"  {lib:<16} {ver}")
        if env.get("smoke_tests"):
            print("Smoke tests  :")
            for name, res in env["smoke_tests"].items():
                print(f"  {name:<16} {res}")

    failed = [k for k, v in env.get("smoke_tests", {}).items() if v != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

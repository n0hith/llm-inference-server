"""Phase 0 smoke tests: the toolchain the later phases depend on actually works.

These are deliberately cheap; their job is to fail loudly and early (e.g. a torch
wheel without CUDA, or a triton/torch mismatch) instead of deep inside a parity test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import env_check  # noqa: E402


def test_packages_importable() -> None:
    import engine  # noqa: F401
    import engine.quantization  # noqa: F401
    import kernels  # noqa: F401
    import scheduler  # noqa: F401
    import server  # noqa: F401


def test_collect_env_is_json_serializable() -> None:
    import json

    env = env_check.collect_env()
    json.dumps(env)
    assert env["libraries"]["torch"] != "not installed"


@pytest.mark.gpu
def test_cuda_matmul_matches_cpu() -> None:
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    gpu = (a.cuda() @ b.cuda()).cpu()
    # TF32 may be used for fp32 matmul on Ampere+, so allow a loose tolerance.
    torch.testing.assert_close(gpu, a @ b, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
def test_triton_kernel_runs() -> None:
    assert env_check.check_triton() == "ok"


@pytest.mark.gpu
def test_bitsandbytes_nf4_runs() -> None:
    assert env_check.check_bitsandbytes_nf4() == "ok"

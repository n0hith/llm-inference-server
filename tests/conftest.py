"""Shared pytest configuration.

Tests marked `gpu` are skipped automatically on machines without CUDA, so
`make test` still works (with skips) on a CPU-only laptop or CI runner.
"""

from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA GPU not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)

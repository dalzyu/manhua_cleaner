"""Tests for OCR worker pool defaults."""

import multiprocessing as mp

from ..core.worker import OCRWorkerPool


def test_worker_pool_default_cap() -> None:
    """Default worker count should be capped to avoid excessive memory use."""
    pool = OCRWorkerPool()
    assert pool.num_workers >= 1
    assert pool.num_workers <= max(1, mp.cpu_count() - 1)
    assert pool.num_workers <= 4

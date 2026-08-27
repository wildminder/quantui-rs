#!/usr/bin/env python
"""Phase 11.1 — Python reference CPU-bound kernel harness.

Times ONLY the per-tensor quantization compute (``LearnedRoundingConverter.convert``)
on an in-memory 4096x4096 weight — no file I/O, no manifest, no writer. This is
the compute-only counterpart of the Rust ``quantize_int8_kernel`` criterion bench,
isolating the CPU-bound quantization math the plan's >=5x target refers to.

Run with the ctq venv:  python tools/bench_python_kernel.py [--runs N]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "docs", "ref"))

import torch  # noqa: E402

from quantui.tensor_quant import QuantConfig, make_converter  # noqa: E402

M = N = 4096
BS = 128
CALIB_SAMPLES = 3072  # stream_quant.CALIB_SAMPLES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10, help="timed convert() calls")
    args = ap.parse_args()

    torch.manual_seed(20260827)
    torch.set_num_threads(os.cpu_count() or 1)

    config = QuantConfig()  # block, bs=128, simple, heur — same as Rust bench
    converter = make_converter(config)

    weight = torch.randn(M, N, dtype=torch.float32).to(torch.bfloat16)
    calib = torch.randn(CALIB_SAMPLES, N, dtype=torch.float32).to(torch.bfloat16)

    # Warm-up (allocator + kernel caches).
    converter.convert(weight, key="warm.weight", calibration_data=calib, has_bias=False)

    times: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        converter.convert(weight, key="bench.weight", calibration_data=calib, has_bias=False)
        times.append(time.perf_counter() - t0)

    mib = M * N * 2 / 2**20  # bf16 input bytes
    best, med = min(times), statistics.median(times)
    print(f"tensor: {M}x{N} bf16 ({mib:.0f} MiB)  block bs={BS} simple  runs={args.runs}")
    print(f"best:   {best * 1000:8.1f} ms   {mib / 1024 / best:7.3f} GiB/s")
    print(f"median: {med * 1000:8.1f} ms   {mib / 1024 / med:7.3f} GiB/s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 11.1 — Python reference throughput harness.

Times the reference streaming quantizer (``docs/ref/quantui/stream_quant.py``)
on the SAME ~1 GB fixture the Rust criterion bench uses, so the GB/s numbers
are directly comparable. Run with the ctq venv (needs torch + convert_to_quant):

    python tools/bench_python_ref.py [--runs N]

Prints wall-clock seconds and GB/s per run plus the best/median.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time

# Make the reference package importable (Textual-free __init__).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "docs", "ref"))

from quantui.stream_quant import stream_quantize  # noqa: E402
from quantui.tensor_quant import QuantConfig  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "tests", "bench", "bench_1gb.safetensors")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="timed runs (default 3)")
    args = ap.parse_args()

    fixture = os.path.abspath(FIXTURE)
    if not os.path.exists(fixture):
        print(f"[skip] fixture not found: {fixture}")
        print("       generate it first with: python tools/gen_bench_fixture.py")
        sys.exit(0)

    size = os.path.getsize(fixture)
    gib = size / 2**30
    # Same config as the Rust bench / golden parity: block, bs=128, simple, heur.
    config = QuantConfig()

    print(f"fixture: {fixture}")
    print(f"size: {size:,} bytes ({gib:.3f} GiB)  runs: {args.runs}")

    times: list[float] = []
    for i in range(args.runs):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.safetensors")
            t0 = time.perf_counter()
            stream_quantize(fixture, out, config)
            dt = time.perf_counter() - t0
        gbps = gib / dt
        times.append(dt)
        print(f"  run {i + 1}: {dt:8.2f} s   {gbps:7.3f} GiB/s")

    best = min(times)
    med = statistics.median(times)
    print(f"best:   {best:8.2f} s   {gib / best:7.3f} GiB/s")
    print(f"median: {med:8.2f} s   {gib / med:7.3f} GiB/s")


if __name__ == "__main__":
    main()

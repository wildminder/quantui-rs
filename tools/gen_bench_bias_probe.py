#!/usr/bin/env python
"""PROBE-ONLY fixture generator: like gen_bench_fixture.py but emits a `.bias`
sibling for every 2D `.weight`, so the bias-correction path actually runs.

The committed 1 GB fixture has NO `.bias` tensors, which means
`stream_quantize` skips `correct_bias` entirely (it is only reachable via
`calib.get(n)` inside the `.bias` branch of the main loop). Timing that
fixture measures the quantize kernel alone and UNDERSTATES a real run, where
every quantized Linear has a bias.

Not a benchmark fixture: this file exists to answer "where does wall-clock
actually go". Layout mirrors a dense transformer post-attention projection,
which is where a `.bias` is both present and large (out_features = 4096).

Run:  python tools/gen_bench_bias_probe.py
Writes: tests/bench/bench_bias_probe.safetensors (gitignored, small on purpose)
"""

from __future__ import annotations

import os

import torch
from safetensors.torch import save_file

OUT = os.path.join(
    os.path.dirname(__file__), "..", "tests", "bench", "bench_bias_probe.safetensors"
)
SEED = 20260827
# 8 x [4096, 4096] bf16 = 8 * 4096 * 4096 * 2 = 268 MB. Small enough to iterate
# on, large enough that the per-tensor GEMM dominates any fixed startup cost.
N_LAYERS = 8
DIM = 4096


def main() -> None:
    torch.manual_seed(SEED)
    tensors: dict[str, torch.Tensor] = {}

    for i in range(N_LAYERS):
        base = f"model.layers.{i}.mlp.down_proj"
        w = torch.randn(DIM, DIM, dtype=torch.float32).to(torch.bfloat16)
        tensors[f"{base}.weight"] = w
        # The tensor that makes bias correction reachable: 4096 is both
        # in_features for the GEMM's K and out_features for its N.
        tensors[f"{base}.bias"] = torch.randn(DIM, dtype=torch.float32).to(torch.bfloat16)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT)
    size = os.path.getsize(OUT)
    print(f"wrote {os.path.abspath(OUT)}")
    print(f"tensors: {len(tensors)}  size: {size:,} bytes ({size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
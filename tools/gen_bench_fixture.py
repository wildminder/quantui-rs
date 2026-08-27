#!/usr/bin/env python
"""Generate the shared ~1 GB benchmark fixture for Phase 11.1.

Both the Rust criterion bench and the Python reference harness quantize THIS
exact file, so the GB/s numbers are directly comparable. Run once:

    python tools/gen_bench_fixture.py

Writes ``tests/bench/bench_1gb.safetensors`` (gitignored). Deterministic via a
fixed torch seed. Layout mirrors a dense transformer: many 2D ``.weight``
tensors whose dims are divisible by the INT8 block size (128) so they are all
quantized (not skipped by the heur), plus a few small non-quantizable tensors
(1-D norms, an embedding) to exercise the copy path.
"""

from __future__ import annotations

import os

import torch
from safetensors.torch import save_file

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "bench", "bench_1gb.safetensors")
SEED = 20260827
# 32 x [4096, 4096] bf16 = 32 * 4096 * 4096 * 2 bytes = 1.0737 GB (~1 GiB).
N_LAYERS = 32
DIM = 4096


def main() -> None:
    torch.manual_seed(SEED)
    tensors: dict[str, torch.Tensor] = {}

    for i in range(N_LAYERS):
        # One 2D quantizable weight per layer; dims divisible by block_size=128.
        w = torch.randn(DIM, DIM, dtype=torch.float32).to(torch.bfloat16)
        tensors[f"model.layers.{i}.mlp.down_proj.weight"] = w

    # Small non-quantizable tensors to exercise the copy path (negligible size).
    tensors["model.norm.weight"] = torch.randn(DIM, dtype=torch.float32).to(torch.bfloat16)
    tensors["model.embed_tokens.weight"] = torch.randn(512, DIM, dtype=torch.float32).to(
        torch.bfloat16
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT)
    size = os.path.getsize(OUT)
    print(f"wrote {os.path.abspath(OUT)}")
    print(f"tensors: {len(tensors)}  size: {size:,} bytes ({size / 2**30:.3f} GiB)")


if __name__ == "__main__":
    main()

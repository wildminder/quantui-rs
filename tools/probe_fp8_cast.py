"""Phase 3.1 probe: capture torch's f32 -> float8_e4m3fn cast bytes.

Generates a committed fixture pair under tests/golden/fp8_cast/:

    fp8_cast_inputs.bin   f32 LE, the cast inputs
    fp8_cast_outputs.bin  u8, torch's `.to(torch.float8_e4m3fn)` bytes

The Rust test crates/quant-core/tests/fp8_parity.rs loads both and asserts
bit-exact equality against `f32_to_fp8_e4m3_bits` (a verbatim port of c10's
fp8e4m3fn_from_fp32_value).

Coverage (>= 1000 random vectors requirement from plan 3.1):
  * 1024 randn vectors x 128      - normal-range rounding
  * 256 uniform vectors x 128     - different mantissa distribution
  * 256 log-uniform vectors x 128 - full exponent sweep incl. subnormals,
                                    underflow to zero, overflow saturation
  * grid vectors near k*2^-9      - subnormal ties / RNE boundaries
  * explicit edge list            - +-448, +-464, +-480, +-inf, NaN, +-0,
                                    2^-6, 2^-9, halfway cases

Run with the managed CPU torch interpreter:
    python tools/probe_fp8_cast.py
"""

from __future__ import annotations

import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir))
OUT_DIR = os.path.join(WS_ROOT, "tests", "golden", "fp8_cast")


def main() -> int:
    import torch

    torch.manual_seed(233983427)  # parity contract seed

    parts: list[torch.Tensor] = []

    # 1) 1024 randn vectors of 128: standard normal, exercises normal range.
    parts.append(torch.randn(1024, 128, dtype=torch.float32))

    # 2) 256 uniform [-1, 1) vectors.
    parts.append(torch.rand(256, 128, dtype=torch.float32) * 2.0 - 1.0)

    # 3) 256 log-uniform vectors spanning 2^-150 .. 2^10: subnormals,
    #    underflow, overflow, saturation all hit here.
    exp = torch.empty(256, 128).uniform_(-150.0, 10.0)
    mant = torch.empty(256, 128).uniform_(1.0, 2.0)
    parts.append(torch.pow(2.0, exp) * mant)
    parts.append(-torch.pow(2.0, exp) * mant)  # negative mirror

    # 4) Grid vectors around the subnormal/normal boundary: k * 2^-9 for
    #    k in [-20, 20] plus half-ulp offsets to hit RNE ties exactly.
    k = torch.arange(-20, 21, dtype=torch.float32)
    base = (k * (2.0 ** -9)).reshape(1, -1)
    offsets = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
    grid = (base + (offsets.reshape(-1, 1) * (2.0 ** -9))).reshape(-1)
    parts.append(grid.reshape(1, -1))

    # 5) Explicit edge values.
    edges = [
        0.0, -0.0,
        448.0, -448.0,
        464.0, -464.0,        # rounds up into NaN pattern -> saturates 0x7E
        480.0, -480.0,        # first unrepresentable value
        447.99997, -447.99997,
        float("inf"), float("-inf"),
        float("nan"), -float("nan"),      # NaN with sign bit set
        2.0 ** -6, -(2.0 ** -6),          # smallest normal
        2.0 ** -9, -(2.0 ** -9),          # smallest subnormal
        1.5 * 2.0 ** -9, 1.25 * 2.0 ** -9, 0.5 * 2.0 ** -9,
        2.0 ** -10, 2.0 ** -20, 2.0 ** -100, 2.0 ** -149,
        1.1754944e-38,                     # near f32 min normal
        3.4028235e38,                      # f32 max -> saturates
        1.0, -1.0, 1.0625, 1.1875,         # normal-range RNE ties
        0.99999994, 1.0000001,             # neighbors of 1.0
    ]
    parts.append(torch.tensor(edges, dtype=torch.float32).reshape(1, -1))

    x = torch.cat([p.reshape(-1) for p in parts]).contiguous()
    y = x.to(torch.float8_e4m3fn).view(torch.uint8)

    os.makedirs(OUT_DIR, exist_ok=True)
    in_path = os.path.join(OUT_DIR, "fp8_cast_inputs.bin")
    out_path = os.path.join(OUT_DIR, "fp8_cast_outputs.bin")
    with open(in_path, "wb") as f:
        f.write(x.numpy().tobytes())
    with open(out_path, "wb") as f:
        f.write(y.numpy().tobytes())

    n = x.numel()
    n_vec = n // 128
    print(f"wrote {n} values ({n_vec} full 128-vectors + edges)")
    print(f"  inputs : {in_path} ({os.path.getsize(in_path)} bytes)")
    print(f"  outputs: {out_path} ({os.path.getsize(out_path)} bytes)")

    # Sanity: distribution of output codes.
    import collections

    hist = collections.Counter(y.tolist())
    print(f"  distinct output codes: {len(hist)} / 256")
    print(f"  zeros: {hist[0] + hist[0x80]}, max(0x7E): {hist[0x7E]}, "
          f"NaN(0x7F/0xFF): {hist[0x7F] + hist[0xFF]}")
    assert n_vec >= 1000, "must satisfy >=1000 random vectors"
    return 0


if __name__ == "__main__":
    sys.exit(main())

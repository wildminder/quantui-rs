#!/usr/bin/env python
"""Generate GGUF quant-parity goldens: gguf-py encoded bytes for deterministic
f32 test vectors (plan Phase 10.4).

The Rust parity test re-quantizes the same f32 inputs with rlx-gguf and
byte-compares against these goldens. Parity contract (plan 10.1 spike):
legacy schemes (F16/BF16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1) are byte-identical between
gguf-py and rlx-gguf; K-quants are NOT (rlx uses a simpler min/max search) and
are therefore excluded here.

Run with a torch environment that has gguf + numpy:
    python tools/gen_golden_gguf.py

Output: tests/golden/gguf_quants/
    <case>.f32.bin            raw little-endian f32 input vector
    <case>.<FORMAT>.bin       gguf-py encoded bytes for that format
    manifest.json             case list + shapes + formats
"""

import json
import os

import numpy as np
import gguf
from gguf import GGMLQuantizationType as QT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "tests", "golden", "gguf_quants")

# Legacy formats only — the byte-parity contract (see module docstring).
FORMATS = ["F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1"]


def make_cases() -> dict[str, np.ndarray]:
    """Deterministic f32 test vectors covering normal + edge distributions.

    All lengths are multiples of 256 (divides every legacy block size: 32 for
    the Q*_0/1 families, and keeps K-quant-free alignment trivial).
    """
    rng = np.random.default_rng(233983427)  # project-wide pinned seed
    cases: dict[str, np.ndarray] = {}

    cases["randn_256"] = rng.standard_normal(256).astype(np.float32)
    cases["randn_1024"] = rng.standard_normal(1024).astype(np.float32)
    # Wide dynamic range (exercises per-block amax scaling).
    cases["wide_range"] = (rng.standard_normal(512) * np.logspace(-3, 3, 512)).astype(np.float32)
    # All zeros (amax == 0 edge: scale must not be NaN/inf).
    cases["zeros"] = np.zeros(256, dtype=np.float32)
    # Tiny values near the subnormal boundary.
    cases["tiny"] = (rng.standard_normal(256) * 1e-30).astype(np.float32)
    # Constant vector (every element identical — ties in rounding).
    cases["constant"] = np.full(256, 0.375, dtype=np.float32)
    # Mixed signs with exact half-values (round-ties behavior).
    t = np.linspace(-4.0, 4.0, 512).astype(np.float32)
    cases["linspace"] = t
    return cases


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    manifest = {"seed": 233983427, "formats": FORMATS, "cases": {}}

    for name, vec in make_cases().items():
        assert vec.dtype == np.float32
        assert vec.size % 256 == 0
        vec.tofile(os.path.join(OUT, f"{name}.f32.bin"))
        entry = {"n": int(vec.size)}
        for fmt in FORMATS:
            qt = getattr(QT, fmt)
            enc = gguf.quantize(vec, qt)
            enc = np.ascontiguousarray(enc).view(np.uint8)
            fn = f"{name}.{fmt}.bin"
            enc.tofile(os.path.join(OUT, fn))
            entry[fmt] = {"bytes": int(enc.size)}
        manifest["cases"][name] = entry
        print(f"  {name}: n={vec.size} " + ", ".join(f"{f}={manifest['cases'][name][f]['bytes']}B" for f in FORMATS))

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['cases'])} cases x {len(FORMATS)} formats -> {OUT}")


if __name__ == "__main__":
    main()

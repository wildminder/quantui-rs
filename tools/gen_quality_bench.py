#!/usr/bin/env python
"""Generate the Tier 2 quality-mode reconstruction-error fixture.

Output (BOTH files under `crates/`, never under `docs/`):
    crates/quant-core/tests/fixtures/quality/weights.f32
    crates/quant-core/tests/fixtures/quality/manifest.json

WHY `crates/` AND NOT `docs/`
-----------------------------
`.gitignore` line 16 is a bare `docs/`, which ignores the whole tree; this repo
has 0 tracked files under `docs/`. A previous phase shipped an `include_str!`
pointing into `docs/`, so a fresh clone could not compile the test binary at
all. These are benchmark *fixtures*, consumed by a build input
(`include_bytes!`), so they belong with the code that compiles them. Never point
a build input at a `docs/` path.

DETERMINISM
-----------
Every value comes from `random.Random(seed).random()` -- CPython's Mersenne
Twister, whose stream is a documented, version-stable property. Distributions
are built from that ONE primitive by explicit inverse-CDF / Box-Muller math
below rather than by `random.gauss` / `random.paretovariate`, whose internal
state handling (notably `gauss_next` caching) is an implementation detail we
would rather not depend on for a byte-reproducible fixture.

Consequence: two runs produce byte-identical `weights.f32` and
`manifest.json`. `diff -q` on two runs is the proof, and `manifest.json`
carries the blob's SHA-256 so the bench can detect drift.

Usage:
    python tools/gen_quality_bench.py
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # tools/ -> repo root
OUT_DIR = ROOT / "crates" / "quant-core" / "tests" / "fixtures" / "quality"
BLOB = OUT_DIR / "weights.f32"
MANIFEST = OUT_DIR / "manifest.json"

BLOB_FORMAT = "f32le-rowmajor-c-contiguous"

# (name, rows, cols) -- deliberately mixes aligned and UNALIGNED shapes.
#
# NVFP4 pads to a multiple of 16, MXFP8 to a multiple of 32. `ragged_37x53`
# (37 and 53 are prime, so neither dimension is a multiple of either) and
# `ragged_17x129` (17 is prime, 129 = 3*43) force the zero-fill padding path in
# BOTH formats. Without them a padding bug -- wrong padded stride, a scale
# computed over padding, a cropped region off by one block -- is invisible,
# because every aligned shape would still produce the right answer.
SHAPES: list[tuple[str, int, int]] = [
    ("aligned_64x64", 64, 64),
    ("aligned_48x96", 48, 96),
    ("ragged_37x53", 37, 53),
    ("ragged_17x129", 17, 129),
]

DISTRIBUTIONS = ["uniform", "gaussian", "heavy_tail", "single_spike"]

# Fixed seed base. The per-case seed is SEED_BASE + 1000*dist_idx + shape_idx,
# so every (distribution, shape) cell has its own independent, fixed stream and
# adding a shape never perturbs an existing case's numbers.
SEED_BASE = 0x5EED_0000


def _uniform(rng: random.Random, half_width: float) -> float:
    """Uniform on [-half_width, +half_width]."""
    return (2.0 * rng.random() - 1.0) * half_width


def _gaussian(rng: random.Random, sigma: float) -> float:
    """N(0, sigma) via Box-Muller on the raw uniform stream.

    Spelled out instead of `random.gauss` so the only RNG primitive in this
    file is `random()`. See the module docstring.
    """
    # Guard the log against u1 == 0.0 (probability ~2^-53, but a fixture
    # generator that can raise is a fixture generator that will, eventually).
    u1 = rng.random()
    if u1 < 1e-300:
        u1 = 1e-300
    u2 = rng.random()
    return sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _heavy_tail(rng: random.Random, mean: float) -> float:
    """Symmetric Pareto (Student-t-like) tail with unit mean, scaled by `mean`.

    Inverse CDF of Pareto(alpha): x = (1-u)^(-1/alpha). Subtracting the Pareto
    mode and multiplying by (alpha-1) makes the mean exactly 1, so `mean` is a
    real scale knob rather than a median. Symmetrized by the sign of a second
    uniform draw, which is what makes it heavy-TAILED rather than one-sided:
    the extreme magnitudes that break a shared-exponent format are the point.
    """
    alpha = 2.5
    u = rng.random()
    # 1-u in (0, 1]; with u == 0.0 this is exactly 1.0 -> x == 0.0, no blowup.
    tail = ((1.0 - u) ** (-1.0 / alpha) - 1.0) * (alpha - 1.0)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return sign * mean * tail


def _single_spike(rng: random.Random, density: float, magnitude: float) -> float:
    """Sparse impulses: mostly structural zeros with rare large values.

    The regime where a single shared per-tensor scale is dominated by a handful
    of outliers while every other element in the block is crushed toward zero.
    """
    if rng.random() < density:
        return _uniform(rng, magnitude)
    return 0.0


def _value(dist: str, rng: random.Random) -> float:
    """Draw one value from `dist` using only `rng.random()`."""
    if dist == "uniform":
        return _uniform(rng, 1.0)
    if dist == "gaussian":
        return _gaussian(rng, 0.5)
    if dist == "heavy_tail":
        return _heavy_tail(rng, 0.5)
    if dist == "single_spike":
        return _single_spike(rng, 0.005, 20.0)
    raise ValueError(f"unknown distribution: {dist!r}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, object]] = []
    blob = bytearray()
    offset = 0  # in f32 elements

    for d_idx, dist in enumerate(DISTRIBUTIONS):
        for s_idx, (shape_name, rows, cols) in enumerate(SHAPES):
            seed = SEED_BASE + 1000 * d_idx + s_idx
            rng = random.Random(seed)
            count = rows * cols
            vals = [_value(dist, rng) for _ in range(count)]
            # Narrow to f32 here, once, so the bytes on disk are exactly what
            # the bench will read back -- the manifest offset/length arithmetic
            # is in f32 ELEMENTS and must match the f32-encoded byte count.
            for v in vals:
                blob += struct.pack("<f", v)
            cases.append(
                {
                    "id": f"{dist}/{shape_name}",
                    "distribution": dist,
                    "shape_name": shape_name,
                    "rows": rows,
                    "cols": cols,
                    "seed": seed,
                    "offset_elements": offset,
                    "length_elements": count,
                }
            )
            offset += count

    BLOB.write_bytes(bytes(blob))

    manifest = {
        "schema": "quantui.quality_bench.fixture.v1",
        "generator": "tools/gen_quality_bench.py",
        "blob_format": BLOB_FORMAT,
        "note": (
            "Committed under crates/ on purpose: .gitignore:16 ignores docs/ "
            "wholesale, and this file is a build input for "
            "benches/quality_error.rs via include_bytes!. See the generator's "
            "module docstring."
        ),
        "distributions": DISTRIBUTIONS,
        "shapes": [
            {"name": n, "rows": r, "cols": c} for n, r, c in SHAPES
        ],
        "blob_file": BLOB.name,
        "blob_bytes": len(blob),
        "blob_sha256": hashlib.sha256(bytes(blob)).hexdigest(),
        "cases": cases,
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"wrote {BLOB.relative_to(ROOT)}  ({len(blob)} bytes)")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    print(f"  blob_sha256: {manifest['blob_sha256']}")
    print(f"  cases: {len(cases)}  total f32 elements: {offset}")


if __name__ == "__main__":
    main()

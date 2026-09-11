"""Per-shard ctq goldens for the NEW formats over the sharded fixture (plan E.2).

The reference streaming quantizer only implements INT8, so byte-parity for the
new formats in SHARDED mode is defined against ctq's whole-file path run
INDEPENDENTLY on each input shard (mirroring `stream_quantize_sharded`, which
processes each shard as its own self-contained file with its own calibration
cache).

For each format and each of the 3 committed input shards this writes:

    tests/golden/sharded_model/output_sharded_<fmt>/<shard>.safetensors

The input shards are the COMMITTED fixtures under
``tests/golden/sharded_model/input/`` and are NOT regenerated here.

Determinism: pinned calib seed 233983427, CUDA hidden (eager backend), simple
mode, heur on — identical to ``gen_golden_formats.py``.

Run with a torch environment that has the reference package (ctq), CUDA
hidden:

    CUDA_VISIBLE_DEVICES=-1 \\
    python tools/gen_golden_sharded_formats.py

Optional ``--only <fmt> [fmt ...]`` restricts generation to a subset.
"""

from __future__ import annotations

import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir))
CASE_DIR = os.path.join(WS_ROOT, "tests", "golden", "sharded_model")
INPUT_DIR = os.path.join(CASE_DIR, "input")

SEED = 233983427  # parity contract, MUST stay pinned

SHARDS = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]

# Same per-format ctq kwargs as gen_golden_formats.py (simple, heur, seeded).
FORMATS: dict[str, dict] = {
    "fp8": dict(
        comfy_quant=True, simple=True, heur=True,
        scaling_mode="block", block_size=128,
    ),
    "fp8_tensor": dict(
        comfy_quant=True, simple=True, heur=True,
        scaling_mode="tensor",
    ),
    "fp8_row": dict(
        comfy_quant=True, simple=True, heur=True,
        scaling_mode="row",
    ),
    "mxfp8": dict(
        mxfp8=True, comfy_quant=True, simple=True, heur=True,
    ),
    "nvfp4": dict(
        nvfp4=True, comfy_quant=True, simple=True, heur=True,
    ),
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import argparse

    from convert_to_quant import quantize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="+", default=None, metavar="FMT",
        help="restrict generation to these format keys (default: all)",
    )
    args = parser.parse_args()

    formats = FORMATS
    if args.only:
        unknown = [f for f in args.only if f not in FORMATS]
        if unknown:
            print(f"unknown format(s): {unknown}; valid: {sorted(FORMATS)}")
            return 2
        formats = {k: v for k, v in FORMATS.items() if k in args.only}
        print(f"restricted to formats: {sorted(formats)}")

    for shard in SHARDS:
        if not os.path.exists(os.path.join(INPUT_DIR, shard)):
            print(f"missing input shard {shard}; run gen_golden_sharded.py first")
            return 1

    results = 0
    for fmt, kwargs in formats.items():
        out_dir = os.path.join(CASE_DIR, f"output_sharded_{fmt}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {fmt} ===")
        for shard in SHARDS:
            src = os.path.join(INPUT_DIR, shard)
            dst = os.path.join(out_dir, shard)
            quantize(src, dst, manual_seed=SEED, device="cpu", **kwargs)
            if not os.path.exists(dst):
                print(f"FAIL: {fmt}/{shard}: ctq produced no output")
                return 1
            print(f"  {shard}: {os.path.getsize(dst)} bytes sha256 {sha256_file(dst)[:16]}")
            results += 1

    print(f"\nDONE: {results} per-shard format goldens generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase E.5 golden fixture: sharded model with deliberately UNSORTED 2D
``.weight`` names, locking down the calibration-cache DRAW ORDER.

Background (plan 3.4 / finding #1)
----------------------------------
The ctq calibration cache draws ``randn(3072, in_features)`` from ONE shared
generator for every unique ``in_features`` among all 2D ``.weight`` tensors
(quantizable or not). The DRAW ORDER is format-family dependent:

* INT8 / FP8  -> ``CalibOrder::FileOrderAll2D``: draw in FILE order.
* MXFP8/NVFP4 -> ``CalibOrder::SortedWeightsOnly``: collect the 2D weights,
  SORT them by name, then draw.

Skipping a draw, or drawing in the wrong order, shifts the shared RNG stream
and corrupts every subsequent bias correction.

The committed ``sharded_model`` fixture CANNOT discriminate the two orders:
its 2D weights appear in alphabetical order within every shard (file order ==
sorted order), so both strategies produce identical RNG streams there. That
gap is what this fixture closes.

How this fixture discriminates
------------------------------
``safetensors.torch.save_file`` does NOT preserve dict insertion order - it
groups tensors by dtype (larger dtype first) and sorts ALPHABETICALLY within
each group (confirmed by ``tools/probe_st_order.py``). A fixture written with
``save_file`` therefore always has same-dtype 2D weights in alphabetical
order. To get a file order that is NOT alphabetical we write the shards with
an EXPLICIT hand-built safetensors header (``write_safetensors_ordered``).
That is fully valid: readers resolve tensors by ``data_offsets`` and never
assume any particular key order.

Shard 1 lists ``zzz.weight`` (in_features=256) BEFORE ``aaa.weight`` (=128):

    file order of 2D weights : [zzz(256), aaa(128)]
    sorted-by-name order     : [aaa(128), zzz(256)]

so FileOrderAll2D draws 256-then-128 while SortedWeightsOnly draws
128-then-256 -> different RNG streams -> different corrected biases. Both
``zzz.bias`` and ``aaa.bias`` are present, so the divergence is visible in the
output bytes. Shard 2 carries ``bbb.weight``/``bbb.bias`` (one 2D weight, no
ordering signal) so the fixture is genuinely multi-shard.

Shapes are multiples of 128 rows/cols so every format quantizes them
(INT8/FP8 block 128, MXFP8 block 32, NVFP4 block 16).

Formats generated (per shard, CUDA hidden, seed 233983427):

    int8  -> quantui REFERENCE streaming (docs/ref/quantui/stream_quant.py).
             It reads the header RAW (read_safetensors_header), so it walks
             names in FILE order -> draws zzz(256) first.
    fp8   -> ctq whole-file. ctq builds its key list from safe_open.keys(),
             which returns names ALWAYS SORTED (see probe_safe_open_keys.py)
             -> draws aaa(128) first.
    mxfp8 -> ctq whole-file, sorted -> aaa(128) first.
    nvfp4 -> ctq whole-file, sorted -> aaa(128) first.

Generating INT8 *and* the ctq formats on the SAME discriminating fixture is
what makes this a real parity lock on the SPLIT: INT8 must draw in file order
while fp8/mxfp8/nvfp4 must draw in sorted order. A port that collapsed the two
orders into a single one mismatches at least one golden, whichever way it
collapses.

Run with the ctq venv interpreter, CUDA hidden:

    CUDA_VISIBLE_DEVICES=-1 ^
    <DEV-TREE>\\Python\\<LOCAL-VENV>\\Scripts\\python.exe ^
        tools\\gen_golden_sharded_unsorted.py

Optional ``--only <fmt> [fmt ...]`` restricts generation to a subset.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir))
# The INT8 goldens come from the reference streaming quantizer, which lives in
# docs/ref (same path setup as tools/gen_golden.py).
_REF_DIR = os.path.normpath(os.path.join(WS_ROOT, "docs", "ref"))
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)

CASE_DIR = os.path.join(WS_ROOT, "tests", "golden", "sharded_unsorted")
INPUT_DIR = os.path.join(CASE_DIR, "input")

SEED = 233983427        # ctq parity contract, MUST stay pinned
DATA_SEED = 60606       # fixture data (arbitrary but fixed)

SHARDS = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]

# (name, shape, dtype) in the EXACT on-disk order we want.
SHARD1_SPEC = [
    ("zzz.weight", (128, 256), "BF16"),   # in_features=256, file-first
    ("aaa.weight", (128, 128), "BF16"),   # in_features=128, sorts-first
    ("zzz.bias", (128,), "F32"),
    ("aaa.bias", (128,), "F32"),
]
SHARD2_SPEC = [
    ("bbb.weight", (128, 128), "BF16"),
    ("bbb.bias", (128,), "F32"),
]

# Same per-format ctq kwargs as gen_golden_formats.py (simple, heur, seeded).
FORMATS: dict[str, dict] = {
    "fp8": dict(
        comfy_quant=True, simple=True, heur=True,
        scaling_mode="block", block_size=128,
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


def header_keys(path: str) -> list[str]:
    """Tensor names in on-disk header order (file order)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return [k for k in hdr if k != "__metadata__"]


def write_safetensors_ordered(path: str, items: list[tuple[str, object, str]]) -> None:
    """Write a safetensors file whose header order is EXACTLY ``items`` order.

    ``safetensors.torch.save_file`` reorders keys (dtype group + alphabetical),
    so we build the container by hand. This is valid: consumers index tensors
    by ``data_offsets`` and never rely on key order.
    """
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, tensor, dtype in items:
        if dtype == "BF16":
            blob = tensor.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tobytes()
        elif dtype == "F32":
            blob = tensor.to(torch.float32).contiguous().numpy().tobytes()
        else:
            raise ValueError(f"unsupported dtype {dtype}")
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)

    raw = json.dumps(header).encode("utf-8")
    pad = (8 - len(raw) % 8) % 8
    raw += b" " * pad
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        for blob in blobs:
            fh.write(blob)


def build_shards() -> dict[str, str]:
    """Write the 2 input shards + index json + config.json.

    Returns the weight_map. Also asserts the fixture actually discriminates
    (shard1's 2D-weight file order != its sorted order).
    """
    gen = torch.Generator().manual_seed(DATA_SEED)

    def rand(*shape):
        return torch.randn(*shape, generator=gen)

    def materialize(spec):
        out = []
        for name, shape, dtype in spec:
            t = rand(*shape)
            t = t * (0.05 if dtype == "BF16" else 0.01)
            out.append((name, t, dtype))
        return out

    os.makedirs(INPUT_DIR, exist_ok=True)
    for shard, spec in ((SHARDS[0], SHARD1_SPEC), (SHARDS[1], SHARD2_SPEC)):
        write_safetensors_ordered(
            os.path.join(INPUT_DIR, shard), materialize(spec)
        )

    weight_map: dict[str, str] = {}
    for shard, spec in ((SHARDS[0], SHARD1_SPEC), (SHARDS[1], SHARD2_SPEC)):
        for name, _shape, _dtype in spec:
            weight_map[name] = shard

    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    with open(
        os.path.join(INPUT_DIR, "model.safetensors.index.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(index, fh, indent=2)

    with open(os.path.join(INPUT_DIR, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"model_type": "synthetic", "hidden_size": 128}, fh)

    # ---- the discriminating property ---------------------------------- #
    shard1 = os.path.join(INPUT_DIR, SHARDS[0])
    file_order = [k for k in header_keys(shard1) if k.endswith(".weight")]
    sorted_order = sorted(file_order)
    print(f"shard1 2D weights, file order : {file_order}")
    print(f"shard1 2D weights, sorted     : {sorted_order}")
    assert file_order != sorted_order, (
        "fixture does NOT discriminate: shard1's 2D weights are in "
        f"alphabetical file order {file_order}"
    )

    return weight_map


def generate_int8_reference() -> int:
    """Per-shard INT8 goldens from the quantui REFERENCE streaming quantizer.

    This is the other half of the draw-order lock: the reference reads the
    header RAW and walks names in FILE order, so it draws zzz(256) before
    aaa(128) — the opposite of the ctq formats below.
    """
    from quantui.stream_quant import stream_quantize
    from quantui.tensor_quant import QuantConfig

    out_dir = os.path.join(CASE_DIR, "output_sharded_int8")
    os.makedirs(out_dir, exist_ok=True)
    print("\n=== int8 (reference streaming, FILE order) ===")

    config = QuantConfig(
        target_format="int8",
        int8=True,
        scaling_mode="block",
        block_size=128,
        no_learned_rounding=True,   # --simple
        convrot=False,
        device="cpu",
        orig_dtype="bfloat16",
        skip_inefficient=True,      # --heur
        calib_seed=SEED,
        exclude_layers=None,
    )

    count = 0
    for shard in SHARDS:
        src = os.path.join(INPUT_DIR, shard)
        dst = os.path.join(out_dir, shard)
        # One shard at a time: stream_quantize's "[path]" form is a union, so
        # passing both shards would merge them into a single output.
        stream_quantize([src], dst, config)
        if not os.path.exists(dst):
            print(f"FAIL: int8/{shard}: reference streaming produced no output")
            raise SystemExit(1)
        print(f"  {shard}: {os.path.getsize(dst)} bytes sha256 {sha256_file(dst)[:16]}")
        count += 1
    return count


def main() -> int:
    global torch
    import argparse

    import torch  # noqa: F811  (bound for write_safetensors_ordered)
    from convert_to_quant import quantize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="+", default=None, metavar="FMT",
        help="restrict generation to these format keys (default: all)",
    )
    args = parser.parse_args()

    valid = sorted(FORMATS) + ["int8"]
    formats = FORMATS
    if args.only:
        unknown = [f for f in args.only if f not in valid]
        if unknown:
            print(f"unknown format(s): {unknown}; valid: {valid}")
            return 2
        formats = {k: v for k, v in FORMATS.items() if k in args.only}
        print(f"restricted to formats: {sorted(formats)}")

    weight_map = build_shards()
    print(f"built {len(weight_map)} tensors across {len(SHARDS)} shards in {INPUT_DIR}")

    results = 0

    # ---- int8: quantui REFERENCE streaming (file order) -------------------- #
    if not args.only or "int8" in args.only:
        results += generate_int8_reference()

    # ---- ctq whole-file formats (sorted order) ----------------------------- #
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

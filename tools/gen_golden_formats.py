"""Golden fixture generator for Phase 3 formats (FP8 / MXFP8 / NVFP4).

Runs the REFERENCE whole-file ``convert_to_quant.quantize`` (ctq) simple path
over the SAME tiny synthetic inputs used by the INT8 goldens and commits, per
input case and per format:

    tests/golden/<case>/output_<format>.safetensors

The inputs (``tests/golden/<case>/input.safetensors``) are shared with the INT8
goldens and are NOT regenerated here.

Why whole-file ctq (not the quantui streaming path)?
    The reference ``quantui/stream_quant.py`` only implements INT8. FP8/MXFP8/
    NVFP4 have no streaming reference, so byte-parity for those kernels is
    defined against ctq's whole-file ``convert_to_quant`` simple path. Note ctq
    writes tensors in its own processing order (not input file order), so Phase
    3 parity tests byte-compare individual tensor PAYLOADS extracted from these
    goldens rather than whole-file bytes (see plan §3.2-3.4).

Determinism:
    * ``manual_seed`` pinned to 233983427 (same calibration seed contract)
    * ``device='cpu'`` forced for reproducibility (CUDA would be nondeterministic)
    * ``simple=True`` (no learned rounding), ``heur=True`` (skip-inefficient)
    * double-run verified byte-identical for all formats (2026-08-27)

Provenance note (2026-08-27): the format modules hardcode
``device = "cuda" if torch.cuda.is_available() else "cpu"`` (fp8_conversion.py:118,
mxfp8_conversion.py:106, nvfp4_conversion.py:108), silently overriding the
``device="cpu"`` passed here. The first golden batch was therefore generated on
CUDA. Per-tensor diffing showed the FP8/MXFP8 CUDA kernels are numerically
identical to eager for the quantized payloads (only bias-correction matmuls
differed), but the NVFP4 CUDA kernel uses reciprocal-multiply data division and
its quantized payloads differ from eager at E2M1 tie points. All 20 goldens
were regenerated with CUDA hidden (``CUDA_VISIBLE_DEVICES=-1``) so the eager
backend runs — matching this script's documented CPU intent and the plan's
parity target ("All bit-exact portable" against the eager/PyTorch algorithm).

Run with the ctq venv interpreter, CUDA hidden so the format modules'
``device = "cuda" if torch.cuda.is_available() else "cpu"`` fallback lands on
CPU/eager (the modules ignore the ``device="cpu"`` we pass; see
nvfp4_conversion.py:108):

    CUDA_VISIBLE_DEVICES=-1 \\
    <DEV-TREE>\\Python\\<LOCAL-VENV>\\Scripts\\python.exe tools/gen_golden_formats.py

Optional ``--only <fmt> [fmt ...]`` restricts generation to a subset of
FORMATS (e.g. ``--only nvfp4``).
"""

from __future__ import annotations

import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir))
GOLDEN_DIR = os.path.join(WS_ROOT, "tests", "golden")

SEED = 233983427  # parity contract, MUST stay pinned

# Reuse the existing INT8 input fixtures (shared inputs).
CASES = ["linear_basic_bf16", "odd_shapes", "conv_net"]

# Extra case built by THIS script (not shared with INT8): a weight containing
# all-zero 128-blocks to exercise MXFP8 zero-block (scale=0/data=0) and NVFP4
# zero-mask semantics.
EXTRA_CASES = ["zero_blocks"]


def build_zero_blocks_input(path: str) -> None:
    """[256,256] bf16 weight with rows 64..127 forced to zero (four full
    128-blocks of zeros across those rows) + f32 bias."""
    import numpy as np
    import torch
    from safetensors.torch import save_file as st_save_file_torch

    rng = np.random.default_rng(4004)
    w = (rng.standard_normal((256, 256)).astype(np.float32) * 0.05)
    w[64:128, :] = 0.0  # zero rows -> zero blocks in every column-group
    wt = torch.from_numpy(w).to(torch.bfloat16)
    b = torch.from_numpy((rng.standard_normal(256).astype(np.float32) * 0.01))
    st_save_file_torch({"blk.weight": wt, "blk.bias": b}, path)

# Per-format ctq kwargs. All simple-mode, CPU, seeded. The dict key becomes the
# output suffix: output_<suffix>.safetensors. FP8 gets all three scaling modes
# (plan §3.2); MXFP8/NVFP4 are single-mode formats.
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


def _files_equal(a: str, b: str) -> bool:
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            ca = fa.read(1 << 20)
            cb = fb.read(1 << 20)
            if ca != cb:
                return False
            if not ca:
                return True


def _publish(src: str, dst: str) -> bool:
    """Copy src over dst; returns True if dst changed. No-op when identical."""
    if os.path.exists(dst) and _files_equal(src, dst):
        return False
    if os.path.exists(dst):
        os.remove(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as fs, open(dst, "wb") as fdst:
        while True:
            chunk = fs.read(1 << 20)
            if not chunk:
                break
            fdst.write(chunk)
    return True


def main() -> int:
    import argparse
    import tempfile

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

    work_root = tempfile.mkdtemp(prefix="gen_golden_fmt_")
    print(f"work dir: {work_root}")

    results: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for case in CASES + EXTRA_CASES:
        input_path = os.path.join(GOLDEN_DIR, case, "input.safetensors")
        if case in EXTRA_CASES and not os.path.exists(input_path):
            os.makedirs(os.path.dirname(input_path), exist_ok=True)
            build_zero_blocks_input(input_path)
            print(f"built extra input: {input_path}")
        if not os.path.exists(input_path):
            skipped.append((case, "input.safetensors missing (run gen_golden.py first)"))
            continue

        for fmt, kwargs in formats.items():
            tag = f"{case}/{fmt}"
            out_name = f"output_{fmt}.safetensors"
            dst = os.path.join(GOLDEN_DIR, case, out_name)
            wout = os.path.join(work_root, f"{case}_{fmt}.safetensors")
            print(f"\n=== {tag} ===")
            try:
                quantize(input_path, wout, manual_seed=SEED, device="cpu", **kwargs)
                if not os.path.exists(wout):
                    raise RuntimeError("ctq produced no output file")
                changed = _publish(wout, dst)
                results.append({
                    "case": case, "format": fmt,
                    "size": os.path.getsize(dst),
                    "sha256": sha256_file(dst),
                    "changed": changed,
                })
                print(f"OK: {tag} ({os.path.getsize(dst)} bytes"
                      f"{', updated' if changed else ', unchanged'})")
            except Exception as exc:  # noqa: BLE001
                skipped.append((tag, f"{type(exc).__name__}: {exc}"))
                print(f"FAIL: {tag}: {exc}")

    try:
        import shutil

        shutil.rmtree(work_root, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass

    print("\n================ FORMAT GOLDEN SUMMARY ================")
    hdr = f"{'case':<20} {'format':<7} {'bytes':>9}  sha256[:16]"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['case']:<20} {r['format']:<7} {r['size']:>9}  {r['sha256'][:16]}")

    if skipped:
        print("\nSKIPPED/FAILED:")
        for n, why in skipped:
            print(f"  {n}: {why}")
        return 1

    print(f"\nAll {len(results)} format goldens generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

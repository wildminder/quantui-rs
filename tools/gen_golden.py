"""Golden fixture generator for quantui-rust byte-parity tests (Phase 0.2/0.3).

Runs the REFERENCE Python streaming quantizer (the quantui reference
package's stream_quant.py, torch path, simple mode) over tiny synthetic
safetensors models and commits:

    tests/golden/<case>/input.safetensors
    tests/golden/<case>/output.safetensors
    tests/golden/<case>/output.safetensors.quant-manifest.json

Determinism:
  * model weights drawn from np.random.default_rng(fixed per-case seed)
  * calibration data for bias correction uses the reference's pinned calib_seed
    (233983427) via torch CPU generator inside stream_quantize itself
  * all torch work forced to device="cpu" for reproducibility

The script is idempotent (same seed -> same bytes), prints a summary table of
cases + file sizes + sha256 of each golden output, and exits nonzero if any case
fails. Cases that cannot run on this machine are recorded in a SKIPPED list.

Run with a Python that has torch + safetensors + numpy installed, and
QUANTUI_REF_DIR set to your reference quantui checkout:

    python tools/gen_golden.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

# --- make the reference package importable --------------------------------- #
# The reference checkout lives outside the published repo — set
# QUANTUI_REF_DIR to the folder containing the quantui package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REF_DIR = os.environ.get("QUANTUI_REF_DIR")
if not _REF_DIR:
    raise SystemExit("set QUANTUI_REF_DIR to your reference quantui checkout")
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)

import numpy as np
from safetensors.numpy import save_file as st_save_file
from safetensors.torch import save_file as st_save_file_torch

from quantui.stream_quant import stream_quantize
from quantui.tensor_quant import QuantConfig

WS_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir))
GOLDEN_DIR = os.path.join(WS_ROOT, "tests", "golden")

CALIB_SEED = 233983427  # reference default, MUST stay pinned (parity contract)


def build_config(exclude_layers: str | None = None) -> QuantConfig:
    """Direct QuantConfig construction mirroring worker_ctq defaults for INT8 simple mode."""
    return QuantConfig(
        target_format="int8",
        int8=True,
        scaling_mode="block",
        block_size=128,
        no_learned_rounding=True,   # --simple
        convrot=False,
        device="cpu",               # force CPU for determinism
        orig_dtype="bfloat16",
        skip_inefficient=True,      # --heur semantics
        calib_seed=CALIB_SEED,
        exclude_layers=exclude_layers,
    )


# --------------------------------------------------------------------------- #
# Fixture builders: each returns {name: np.ndarray} written to input.safetensors
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
    """Cast float32 array to bfloat16 raw uint16 view (round-to-nearest-even via torch)."""
    import torch

    t = torch.from_numpy(arr_f32.astype(np.float32)).to(torch.bfloat16)
    return t.view(np.uint16).numpy()


def _f16_from_f32(arr_f32: np.ndarray) -> np.ndarray:
    import torch

    t = torch.from_numpy(arr_f32.astype(np.float32)).to(torch.float16)
    return t.view(np.uint16).numpy()


def _bf16_from_f32(arr_f32: np.ndarray) -> np.ndarray:
    """Cast float32 array to bfloat16 raw uint16 bytes (round-to-nearest-even via torch)."""
    import torch

    t = torch.from_numpy(arr_f32.astype(np.float32)).to(torch.bfloat16)
    return t.view(torch.uint16).numpy()


def _f16_from_f32(arr_f32: np.ndarray) -> np.ndarray:
    """Cast float32 array to float16 raw uint16 bytes."""
    import torch

    t = torch.from_numpy(arr_f32.astype(np.float32)).to(torch.float16)
    return t.view(torch.uint16).numpy()


def _save_model(tensors_u16_views: dict[str, np.ndarray], path: str) -> None:
    """Write a model dict to a safetensors file with PROPER dtype headers.

    bf16/f16 tensors are carried as raw uint16 numpy arrays by the builders
    (numpy has no bfloat16); here they are re-viewed to real torch dtypes so the
    header records BF16/F16 -- matching what a real HF safetensors input looks
    like. Writing the uint16 views directly would produce a U16/UINT16 header,
    which the reference streaming path would then copy verbatim as UINT16.
    """
    import torch

    out: dict = {}
    for name, arr in tensors_u16_views.items():
        if arr.dtype == np.uint16:
            t16 = torch.from_numpy(arr)
            # Distinguish f16 vs bf16 by the tensor-name suffix convention used in
            # the case builders; both are 2-byte floats with identical storage.
            raise RuntimeError(
                "ambiguous uint16 payload; use _save_model_tagged instead"
            )
        out[name] = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
    st_save_file_torch(out, path)


def _save_model_tagged(tensors: dict[str, object], path: str) -> None:
    """Write a mixed model dict where bf16/f16 entries are (kind, uint16-array).

    ``kind`` is "bf16" or "f16". All other values must be numpy arrays.
    """
    import torch

    out: dict = {}
    for name, val in tensors.items():
        if isinstance(val, tuple) and len(val) == 2 and val[0] in ("bf16", "f16"):
            kind, arr = val
            t16 = torch.from_numpy(np.ascontiguousarray(arr)).view(torch.uint16)
            out[name] = (
                t16.view(torch.bfloat16) if kind == "bf16" else t16.view(torch.float16)
            )
        else:
            out[name] = torch.from_numpy(val)  # numpy -> torch (shares memory)
    st_save_file_torch(out, path)


def case_linear_basic_bf16_tensors() -> dict[str, object]:
    """(a) Linear weights [256,128] & [128,64] bf16 (torch convention: (out, in)),
    biases f32 with length out_features, one LayerNorm weight 1-D passthrough.
    [128,64]: cols < block_size -> skip-heur copies it (cast to bf16 output dtype).
    bf16 entries are ("bf16", uint16-array) tags for _save_model_tagged."""
    rng = np.random.default_rng(1001)
    return {
        "blocks.0.weight": ("bf16", _bf16_from_f32(rng.standard_normal((256, 128)).astype(np.float32) * 0.05)),
        "blocks.0.bias": rng.standard_normal(256).astype(np.float32) * 0.01,
        "blocks.1.weight": ("bf16", _bf16_from_f32(rng.standard_normal((128, 64)).astype(np.float32) * 0.05)),
        "blocks.1.bias": rng.standard_normal(128).astype(np.float32) * 0.01,
        "norm.weight": rng.uniform(0.5, 1.5, size=256).astype(np.float32),
    }


def case_odd_shapes_tensors() -> dict[str, object]:
    """(b) [384,256] divisible by 128 (quantized), [130,130] non-divisible (skip-heur
    copy + cast to bf16 output dtype), plus attn_norm exclusion test.

    Note: the excluded attn_norm weight stays at its ORIGINAL dtype in the output."""
    rng = np.random.default_rng(2002)
    return {
        "model.diffusion_model.double_block.0.weight":
            rng.standard_normal((384, 256)).astype(np.float32) * 0.04,
        "model.diffusion_model.double_block.0.bias":
            rng.standard_normal(384).astype(np.float32) * 0.01,
        "odd.weight":
            ("bf16", _bf16_from_f32(rng.standard_normal((130, 130)).astype(np.float32) * 0.08)),
        "attn_norm.weight":
            rng.uniform(0.9, 1.1, size=256).astype(np.float32),  # excluded -> original dtype kept
    }


def case_conv_net_tensors() -> dict[str, object]:
    """(c) conv2d weight [32,16,3,3] f32 (4-D -> passthrough per heuristics),
    linear [128,128] f16 (divisible by 128 -> quantized)."""
    rng = np.random.default_rng(3003)
    return {
        "conv.net.weight": (rng.standard_normal((32, 16, 3, 3)) * 0.1).astype(np.float32),
        "conv.bias": rng.standard_normal(32).astype(np.float32) * 0.01,
        "head.weight": ("f16", _f16_from_f32(rng.standard_normal((128, 128)).astype(np.float32) * 0.05)),
    }


def case_nonf32_bias_tensors() -> dict[str, object]:
    """(d) Bias correction with NON-F32 biases (regression for the VibeVoice
    BF16-bias panic). Two quantizable 2D weights whose sibling biases are
    bfloat16 and float16 respectively. The reference upcasts each bias to f32
    for the correction math, then casts the corrected value back to the bias's
    original dtype (stream_quant.py:145) — the golden output therefore carries
    BF16 / F16 corrected biases, exercising the Rust decode->correct->re-encode
    round trip for non-F32 bias dtypes."""
    rng = np.random.default_rng(4004)
    return {
        # bf16 weight + bf16 bias (the exact VibeVoice shape of the bug).
        "blk.0.weight": ("bf16", _bf16_from_f32(rng.standard_normal((256, 128)).astype(np.float32) * 0.05)),
        "blk.0.bias": ("bf16", _bf16_from_f32(rng.standard_normal(256).astype(np.float32) * 0.01)),
        # f16 weight + f16 bias (covers the other 2-byte float path).
        "blk.1.weight": ("f16", _f16_from_f32(rng.standard_normal((128, 128)).astype(np.float32) * 0.05)),
        "blk.1.bias": ("f16", _f16_from_f32(rng.standard_normal(128).astype(np.float32) * 0.01)),
    }


CASES: list[dict] = [
    {
        "name": "linear_basic_bf16",
        "seed_note": "rng seed 1001",
        "build": case_linear_basic_bf16_tensors,
        "exclude_layers": None,
    },
    {
        "name": "odd_shapes",
        "seed_note": "rng seed 2002",
        "build": case_odd_shapes_tensors,
        "exclude_layers": "attn_norm",
    },
    {
        "name": "conv_net",
        "seed_note": "rng seed 3003",
        "build": case_conv_net_tensors,
        "exclude_layers": None,
    },
    {
        "name": "nonf32_bias",
        "seed_note": "rng seed 4004",
        "build": case_nonf32_bias_tensors,
        "exclude_layers": None,
    },
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files_equal(a: str, b: str) -> bool:
    """Byte-compare two files (size short-circuit then content)."""
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


def _publish(src: str, dst: str) -> None:
    """Copy src over dst, skipping the copy when bytes already match.

    Idempotent by construction: identical regeneration never rewrites committed
    golden files. Some environments guard deletes of protected paths behind
    interactive confirmation; never rewriting avoids tripping that entirely.
    If a stale file must be replaced but cannot be removed, an error is raised.
    """
    if os.path.exists(dst):
        if _files_equal(src, dst):
            return
        try:
            os.remove(dst)
        except SystemExit as exc:  # guarded-delete hooks may raise SystemExit
            raise RuntimeError(
                f"cannot replace changed golden file {dst!r} "
                f"(delete blocked by environment): {exc}"
            ) from exc
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # copy2 preserves nothing we need; plain byte copy is enough.
    with open(src, "rb") as fs, open(dst, "wb") as fdst:
        while True:
            chunk = fs.read(1 << 20)
            if not chunk:
                break
            fdst.write(chunk)


def main() -> int:
    import tempfile

    results: list[dict] = []
    skipped: list[tuple[str, str]] = []

    # Generate everything into a scratch dir first; publish only when new/changed.
    work_root = tempfile.mkdtemp(prefix="gen_golden_")
    print(f"work dir: {work_root}")

    for case in CASES:
        name = case["name"]
        case_dir = os.path.join(GOLDEN_DIR, name)
        input_path = os.path.join(case_dir, "input.safetensors")
        output_path = os.path.join(case_dir, "output.safetensors")
        manifest_path = output_path + ".quant-manifest.json"

        wdir = os.path.join(work_root, name)
        win = os.path.join(wdir, "input.safetensors")
        wout = os.path.join(wdir, "output.safetensors")
        wman = wout + ".quant-manifest.json"

        print(f"\n=== case: {name} ===")
        try:
            tensors = case["build"]()
            os.makedirs(wdir, exist_ok=True)

            _save_model_tagged(tensors, win)

            config = build_config(exclude_layers=case["exclude_layers"])
            manifest = stream_quantize(win, wout, config, on_progress=None,
                                       use_numpy_fallback=False)

            if not os.path.exists(wout):
                raise RuntimeError("stream_quantize produced no output file")

            with open(wman, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, separators=(",", ":"))

            # Publish to tests/golden/ (no-op when bytes are identical).
            for src, dst in ((win, input_path), (wout, output_path),
                             (wman, manifest_path)):
                _publish(src, dst)

            results.append({
                "case": name,
                "tensors": len(tensors),                "input_size": os.path.getsize(input_path),
                "output_size": os.path.getsize(output_path),
                "manifest_size": os.path.getsize(manifest_path),
                "input_sha256": sha256_file(input_path),
                "output_sha256": sha256_file(output_path),
            })
            print(f"OK: {name} ({len(tensors)} tensors)")
        except Exception as exc:  # noqa: BLE001
            skipped.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL: {name}: {exc}")

    # Scratch cleanup is best-effort; failures here are non-fatal.
    try:
        import shutil

        shutil.rmtree(work_root, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass

    print("\n================ GOLDEN SUMMARY ================")
    hdr = f"{'case':<22} {'tensors':>7} {'in_bytes':>10} {'out_bytes':>10} {'manifest':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['case']:<22} {r['tensors']:>7} {r['input_size']:>10} "
            f"{r['output_size']:>10} {r['manifest_size']:>9}"
        )
    print("\nsha256:")
    for r in results:
        print(f"  {r['case']}/input.safetensors          {r['input_sha256']}")
        print(f"  {r['case']}/output.safetensors         {r['output_sha256']}")

    if skipped:
        print("\nSKIPPED/FAILED cases:")
        for n, why in skipped:
            print(f"  {n}: {why}")

    if not results and not skipped:
        print("\nNo cases defined!")
        return 1
    if skipped:
        return 1
    print("\nAll golden fixtures generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

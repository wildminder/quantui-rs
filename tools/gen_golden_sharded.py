"""Golden fixture generator for Phase 6 (sharded input/output).

Builds a 3-shard synthetic HuggingFace-style model (index json + sidecar) and
runs the REFERENCE streaming quantizer both ways:

    tests/golden/sharded_model/input/            (3 shards + index + config.json)
    tests/golden/sharded_model/output_single.safetensors            (+ manifest)
    tests/golden/sharded_model/output_sharded/   (3 shards + index + sidecars
                                                  + per-shard manifests
                                                  + global .quant-manifest.json)

Single mode   = ``stream_quantize([shard1, shard2, shard3], out)`` — union
              header, first-wins shard mapping, one output file.
Sharded mode  = ``stream_quantize_sharded(model, out_dir)`` — one output shard
              per input shard, index/sidecars copied verbatim, global manifest.

Design notes (parity traps this fixture exercises):
  * shard1 carries ``aaa_skipped.weight`` [128,192] — a 2D weight SKIPPED by
    the heuristics that sorts BEFORE the quantized ``blocks.0.weight``. The
    reference calibration cache draws RNG for EVERY 2D ``.weight`` (skipped
    ones included), so the draw order here is 192-then-128; a port that only
    draws for quantizable weights shifts the RNG stream and corrupts bias
    correction. (Confirmed by probe 2026-08-27.)
  * ``blocks.1.bias`` lives in shard2 but ``blocks.1.weight`` in shard3: in
    SINGLE mode the bias is reached before its weight across shards (deferred
    weight processing); in SHARDED mode each shard is independent, so the bias
    copies unchanged and the weight quantizes without bias correction. The two
    output modes therefore legitimately differ on ``blocks.1.bias``.
  * ``small.weight`` [128,64] is skipped (cols < block) and cast to the output
    dtype; ``norm.weight`` is a 1-D passthrough.

Determinism: pinned calib seed 233983427, device cpu (CUDA hidden so the
format modules' hardcoded device fallback lands on CPU), simple mode, heur on.

Run with the ctq venv interpreter, CUDA hidden:

    CUDA_VISIBLE_DEVICES=-1 \\
    <DEV-TREE>\\Python\\<LOCAL-VENV>\\Scripts\\python.exe tools/gen_golden_sharded.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF_DIR = os.path.normpath(os.path.join(_HERE, os.pardir, "docs", "ref"))
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)

WS_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir))
GOLDEN_DIR = os.path.join(WS_ROOT, "tests", "golden")
CASE_DIR = os.path.join(GOLDEN_DIR, "sharded_model")

CALIB_SEED = 233983427  # parity contract, MUST stay pinned

SHARDS = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]


def _bf16(arr):
    import numpy as np
    import torch

    t = torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    return t


def build_shards(input_dir: str) -> dict[str, list[str]]:
    """Write the 3 input shards + index json + config.json.

    Returns the weight_map (tensor -> shard filename).
    """
    import numpy as np
    import torch
    from safetensors.torch import save_file as st_save

    os.makedirs(input_dir, exist_ok=True)

    rng = np.random.default_rng(6006)

    shard1 = {
        # SKIPPED 2D weight (192 % 128 != 0) that sorts FIRST alphabetically:
        # forces the calibration cache to draw for in_features=192 before 128.
        "aaa_skipped.weight": _bf16(rng.standard_normal((128, 192)) * 0.05),
        "blocks.0.weight": _bf16(rng.standard_normal((256, 128)) * 0.05),
        "blocks.0.bias": torch.from_numpy(
            rng.standard_normal(256).astype(np.float32) * 0.01
        ),
        "norm.weight": torch.from_numpy(
            rng.uniform(0.5, 1.5, size=256).astype(np.float32)
        ),
    }
    shard2 = {
        # Bias whose weight lives in the NEXT shard (cross-shard ordering trap).
        "blocks.1.bias": torch.from_numpy(
            rng.standard_normal(128).astype(np.float32) * 0.01
        ),
        # Skipped 2D weight (cols 64 < block 128) -> copy + cast to out dtype.
        "small.weight": _bf16(rng.standard_normal((128, 64)) * 0.05),
    }
    shard3 = {
        "blocks.1.weight": _bf16(rng.standard_normal((128, 128)) * 0.05),
        "head.weight": _bf16(rng.standard_normal((256, 256)) * 0.05),
        "head.bias": torch.from_numpy(
            rng.standard_normal(256).astype(np.float32) * 0.01
        ),
    }

    st_save(shard1, os.path.join(input_dir, SHARDS[0]))
    st_save(shard2, os.path.join(input_dir, SHARDS[1]))
    st_save(shard3, os.path.join(input_dir, SHARDS[2]))

    weight_map: dict[str, str] = {}
    for shard, tensors in ((SHARDS[0], shard1), (SHARDS[1], shard2), (SHARDS[2], shard3)):
        for name in tensors:
            weight_map[name] = shard

    index = {
        "metadata": {"total_size": 123456},
        "weight_map": weight_map,
    }
    with open(os.path.join(input_dir, "model.safetensors.index.json"), "w",
              encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)

    with open(os.path.join(input_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"model_type": "synthetic", "hidden_size": 128}, fh)

    return weight_map


def build_config():
    from quantui.tensor_quant import QuantConfig

    return QuantConfig(
        target_format="int8",
        int8=True,
        scaling_mode="block",
        block_size=128,
        no_learned_rounding=True,   # --simple
        convrot=False,
        device="cpu",
        orig_dtype="bfloat16",
        skip_inefficient=True,      # --heur
        calib_seed=CALIB_SEED,
        exclude_layers=None,
    )


class _Model:
    """Duck-typed ShardedModel for stream_quantize_sharded."""

    def __init__(self, model_dir: str):
        from quantui.worker_ctq import discover_shards

        m = discover_shards(model_dir)
        self.model_dir = m.model_dir
        self.index_path = m.index_path
        self.weight_map = m.weight_map
        self.shard_files = m.shard_files
        self.non_weight_files = m.non_weight_files


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    from quantui.stream_quant import stream_quantize, stream_quantize_sharded

    input_dir = os.path.join(CASE_DIR, "input")
    weight_map = build_shards(input_dir)
    print(f"built {len(weight_map)} tensors across {len(SHARDS)} shards in {input_dir}")

    model = _Model(input_dir)
    assert model.shard_files == SHARDS, model.shard_files
    config = build_config()

    # ---- single mode: all shards -> one output file ------------------------ #
    # stream_quantize writes <output>.quant-manifest.json itself (per-tensor +
    # final save), so no manual manifest write is needed here.
    shard_paths = [os.path.join(input_dir, s) for s in model.shard_files]
    single_out = os.path.join(CASE_DIR, "output_single.safetensors")
    manifest = stream_quantize(shard_paths, single_out, config)
    assert os.path.exists(single_out + ".quant-manifest.json")
    print(f"single mode: {single_out} ({os.path.getsize(single_out)} bytes) "
          f"sha256 {sha256_file(single_out)[:16]} done={len(manifest['done'])}")

    # ---- sharded mode: one output shard per input shard -------------------- #
    sharded_out = os.path.join(CASE_DIR, "output_sharded")
    global_manifest = stream_quantize_sharded(model, sharded_out, config)
    print(f"sharded mode: {sharded_out}")
    for s in model.shard_files:
        p = os.path.join(sharded_out, s)
        print(f"  {s}: {os.path.getsize(p)} bytes sha256 {sha256_file(p)[:16]}")
    print(f"  global manifest shards: {sorted(global_manifest['shards'])}")

    print("\nDONE: sharded goldens generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

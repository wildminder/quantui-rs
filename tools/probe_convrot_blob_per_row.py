"""Probe: does the reference emit `per_row` for INT8 row-wise WITHOUT ConvRot?

Phase 7.1 finding (2026-09-02), recorded because it CONTRADICTS the Phase 7.1
work order and must be revisited in Phase 7.2 (golden generation + byte parity).

Run the three configs below over `tests/golden/linear_basic_bf16/input.safetensors`
(in_features 128 and 64 — neither divisible by 256) and print every
`.comfy_quant` blob:

    CUDA_VISIBLE_DEVICES=-1 \
    python tools/probe_convrot_blob_per_row.py

Observed output:

    [int8_row_plain]     blocks.0.comfy_quant:
        {"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "per_row": true}
    [int8_row_convrot64] blocks.0.comfy_quant:
        {"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "convrot": true, "convrot_groupsize": 64, "per_row": true}
    [int8_row_convrot256] blocks.0.comfy_quant:
        {"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "per_row": true}

Conclusions:

1. `per_row: true` is emitted for **every** INT8 row-wise layer, rotated or
   not — `fp8_conversion.py:583-584` sets `per_row = True` whenever
   `converter.scaling_mode == "row"`, and `:594` passes it through. It is a
   property of the SCALING MODE, not of ConvRot.
2. A ConvRot run whose `in_features` is not divisible by the group size emits
   `per_row: true` but NO `convrot` key (see `int8_row_convrot256` above).
3. `weight_scale` is `[m, 1]` and there is no `input_scale` for row-wise INT8,
   rotated or not.

Divergence: `stream.rs` currently emits the family-A blob with no `per_row`
for non-ConvRot INT8 row-wise (pre-Phase-7.1 behaviour, explicitly frozen by
the Phase 7.1 work order). That is a real byte-parity gap for plain
`--format int8 --scaling-mode row`, and for ConvRot-skipped layers. No existing
golden covers INT8 row-wise, so no test catches it yet — Phase 7.2 must decide.
"""

from __future__ import annotations

import os

from convert_to_quant import quantize

_HERE = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir))
GOLDEN = os.path.join(WS_ROOT, "tests", "golden", "linear_basic_bf16")
SEED = 233983427

CONFIGS: dict[str, dict] = {
    # Plain INT8 row-wise: no convrot at all.
    "int8_row_plain": dict(
        int8=True, scaling_mode="row", comfy_quant=True, simple=True, heur=True
    ),
    # gs=64 divides BOTH 128 and 64 -> every layer rotates.
    "int8_row_convrot64": dict(
        int8=True,
        scaling_mode="row",
        convrot=True,
        convrot_group_size=64,
        comfy_quant=True,
        simple=True,
        heur=True,
    ),
    # gs=256 divides neither 128 nor 64 -> no layer rotates.
    "int8_row_convrot256": dict(
        int8=True,
        scaling_mode="row",
        convrot=True,
        convrot_group_size=256,
        comfy_quant=True,
        simple=True,
        heur=True,
    ),
}


def main() -> None:
    from safetensors import safe_open

    inp = os.path.join(GOLDEN, "input.safetensors")
    out_dir = os.path.join(WS_ROOT, "target")
    os.makedirs(out_dir, exist_ok=True)

    for tag, kwargs in CONFIGS.items():
        out = os.path.join(out_dir, f"probe_{tag}.safetensors")
        if os.path.exists(out):
            os.remove(out)
        quantize(inp, out, manual_seed=SEED, device="cpu", **kwargs)
        with safe_open(out, framework="pt") as f:
            for key in sorted(f.keys()):
                if key.endswith(".comfy_quant"):
                    blob = f.get_tensor(key)
                    print(f"[{tag}] {key}: {bytes(blob.tolist()).decode('utf-8')}")
        print()


if __name__ == "__main__":
    main()

"""Probe NVFP4 simple-path pipeline for the Rust port (Phase 3.4).

Replicates the exact ctq simple path:
  nvfp4_conversion.py (simple=True):
    tensor_gpu = tensor.to(f32)
    NVFP4Converter.quantize(tensor_gpu):
      amax over the F32 tensor (BEFORE bf16 cast)
      per_tensor_scale = amax / (448*6.0)          [tensor/scalar = IEEE div]
      ck.quantize_nvfp4(tensor.to(bf16), pts, pad_16x=True)
  comfy_kitchen eager quantize_nvfp4 (CPU golden path):
    pad 16x; blocks of 16
    block_scale = bf16_block_max.to(f32) / 6.0     [tensor/scalar = IEEE div]
    scaled = block_scale / per_tensor_scale        [tensor/tensor = IEEE div]
    scaled_fp8 = clamp(scaled, max=448)            [max clamp only]
    scaled_f32 = _float8_round(scaled_fp8)         [E4M3 round-trip]
    total = pts * scaled_f32
    zero_mask = total == 0; total_safe = where(mask, 1, total)
    data = x.float() / total_safe                  [tensor/tensor = IEEE div]
    data = where(zero_mask, +0.0, data); clamp(+-6)
    codes = _f32_to_floatx_unpacked(data, 2, 1); pack_uint4(hi_first=True)
    scales_out = to_blocked(scaled_fp8.to(e4m3), flatten=False)

Verifies byte-exactness of qdata (packed U8), weight_scale (F8_E4M3 tiled),
and weight_scale_2 (F32 scalar) against all four golden cases.

IMPORTANT — golden provenance: the goldens MUST be generated with CUDA hidden
(``CUDA_VISIBLE_DEVICES=-1``), because ``nvfp4_conversion.py:108`` hardcodes
``device = "cuda" if torch.cuda.is_available() else "cpu"`` and ignores the
``device="cpu"`` the gen script passes. With CUDA visible the compiled CUDA
kernel runs instead of the eager backend, and its data division is
reciprocal-multiply (``x * (1/total_scale)``) rather than the eager IEEE
division this probe (and the Rust port) replicates — qdata then mismatches at
E2M1 tie points and weight_scale_2 differs by 1 ulp. The CPU/eager goldens are
the plan's parity target ("NVFP4: nvfp4_converter.py::_quantize_pytorch ...
All bit-exact portable") and match the gen script's documented CPU intent.

Run with the ctq venv interpreter:
    CUDA_VISIBLE_DEVICES=-1 \\
    <DEV-TREE>\\Python\\<LOCAL-VENV>\\Scripts\\python.exe tools/probe_nvfp4.py
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import torch

from comfy_kitchen.float_utils import (
    _f32_to_floatx_unpacked,
    pack_uint4,
    to_blocked,
)

ROOT = r"<REPO-DIR>\tests\golden"
F8_E4M3_MAX = 448.0
F4_E2M1_MAX = 6.0


def load_tensor(path: str, nm: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        off = 8 + n
        meta = hdr[nm]
        f.seek(off + meta["data_offsets"][0])
        data = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])
        return data, meta["dtype"], meta["shape"]


def to_f32(data: bytes, dt: str, shp) -> torch.Tensor:
    if dt == "BF16":
        return torch.frombuffer(bytearray(data), dtype=torch.uint16).view(torch.bfloat16).float().reshape(shp)
    if dt == "F16":
        return torch.frombuffer(bytearray(data), dtype=torch.uint16).view(torch.float16).float().reshape(shp)
    return torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(shp)


def roundup(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def nvfp4_ref(w_f32: torch.Tensor):
    """Replicate ctq simple-path NVFP4 on an f32 weight tensor."""
    # per-tensor scale from the F32 tensor (before bf16 cast)
    amax = torch.amax(torch.abs(w_f32))
    per_tensor_scale = amax / (F8_E4M3_MAX * F4_E2M1_MAX)
    per_tensor_scale = per_tensor_scale.to(dtype=torch.float32)

    # kernel input: bf16-rounded
    x = w_f32.to(torch.bfloat16)

    rows, cols = x.shape
    prows, pcols = roundup(rows, 16), roundup(cols, 16)
    if (prows, pcols) != (rows, cols):
        x = torch.nn.functional.pad(x, (0, pcols - cols, 0, prows - rows))

    xb = x.reshape(prows, -1, 16)
    max_abs = torch.amax(torch.abs(xb), dim=-1)
    block_scale = max_abs.to(torch.float32) / F4_E2M1_MAX
    scaled = block_scale / per_tensor_scale
    scaled_fp8 = torch.clamp(scaled, max=F8_E4M3_MAX)
    scaled_f32 = scaled_fp8.to(torch.float8_e4m3fn).to(torch.float32)
    total = per_tensor_scale * scaled_f32
    zero_mask = total == 0
    total_safe = torch.where(zero_mask, torch.ones_like(total), total)

    data = xb.float() / total_safe.unsqueeze(-1)
    data = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(data), data)
    data = torch.clamp(data, -F4_E2M1_MAX, F4_E2M1_MAX)
    data = data.view(prows, pcols)

    codes = _f32_to_floatx_unpacked(data.float(), 2, 1)
    packed = pack_uint4(codes, hi_first=True)
    scales_out = to_blocked(scaled_fp8.to(torch.float8_e4m3fn), flatten=False)
    return packed, scales_out, per_tensor_scale


def main() -> None:
    all_ok = True
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"]:
        ip = os.path.join(ROOT, case, "input.safetensors")
        op = os.path.join(ROOT, case, "output_nvfp4.safetensors")
        with open(op, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            ohdr = json.loads(f.read(n))
        # quantized weights are packed U8 with a matching .weight_scale (F8_E4M3)
        qnames = [
            k for k, v in ohdr.items()
            if k != "__metadata__" and v["dtype"] == "U8"
            and k.endswith(".weight")
            and f"{k[:-len('.weight')]}.weight_scale" in ohdr
        ]
        for name in qnames:
            din, dt_in, shp = load_tensor(ip, name)
            w = to_f32(din, dt_in, shp)
            packed, scales, pts = nvfp4_ref(w)

            dq, _, shq = load_tensor(op, name)
            qg = torch.frombuffer(bytearray(dq), dtype=torch.uint8).reshape(shq)
            ds, _, shs = load_tensor(op, name.replace(".weight", ".weight_scale"))
            sg = torch.frombuffer(bytearray(ds), dtype=torch.uint8).reshape(shs)
            d2, _, sh2 = load_tensor(op, name.replace(".weight", ".weight_scale_2"))
            pts_g = struct.unpack("<f", d2)[0]

            okq = torch.equal(packed.view(torch.uint8), qg)
            oks = torch.equal(scales.view(torch.uint8), sg)
            okp = pts.item() == pts_g
            all_ok &= okq and oks and okp
            print(
                f"{case:18} {name[:42]:42} in={dt_in:4} "
                f"q={'OK' if okq else 'XX'} scale={'OK' if oks else 'XX'} "
                f"pts={'OK' if okp else 'XX'} (rust? {pts.item():.10g} vs {pts_g:.10g})"
            )
            if not okq:
                p = packed.view(torch.uint8).numpy().ravel()
                g = qg.numpy().ravel()
                mm = np.nonzero(p != g)[0]
                print(f"    qdata first diffs at {mm[:8].tolist()}: "
                      f"got {p[mm[:8]].tolist()} want {g[mm[:8]].tolist()}")
            if not oks:
                p = scales.view(torch.uint8).numpy().ravel()
                g = sg.numpy().ravel()
                mm = np.nonzero(p != g)[0]
                print(f"    scale first diffs at {mm[:8].tolist()}: "
                      f"got {p[mm[:8]].tolist()} want {g[mm[:8]].tolist()}")
    print("\nALL OK" if all_ok else "\nMISMATCHES FOUND")


if __name__ == "__main__":
    main()

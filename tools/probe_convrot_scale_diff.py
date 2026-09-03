"""Phase 7.2 diagnosis probe: WHERE do our convrot goldens' scales/biases
diverge from the reference? Reads the actual golden tensors and recomputes
the reference chain step by step for one tensor.

head.weight (conv_net) is F16 [128,128]:
  - gs256: NOT rotated (128 % 256 != 0) -> plain row-wise INT8
  - gs64:  rotated (128 % 64 == 0)

The parity test shows head.weight_scale differing at BOTH group sizes, and
every corrected bias differing too. int8 weight payloads PASS everywhere.

Run:
    CUDA_VISIBLE_DEVICES=-1 <LOCAL-PYTHON> tools/probe_convrot_scale_diff.py
"""

from __future__ import annotations

import json
import struct

import numpy as np
import torch

GOLDEN = "tests/golden/conv_net"
INPUT = f"{GOLDEN}/input.safetensors"


def load_tensor(path: str, name: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        v = h[name]
        f.seek(8 + n + v["data_offsets"][0])
        return f.read(v["data_offsets"][1] - v["data_offsets"][0])


def main() -> None:
    # ---- reference recomputation, exactly as the batch driver does ------- #
    w_raw = load_tensor(INPUT, "head.weight")
    w_f16 = torch.from_numpy(
        np.frombuffer(w_raw, dtype="<f2").reshape(128, 128).copy()
    )
    w32 = w_f16.to(torch.float32)

    # gs256: plain row-wise (TensorWiseINT8Layout auto, is_weight=True)
    row_max = w32.abs().amax(dim=1, keepdim=True)
    quant_scale = 127.0 / row_max.clamp_min(1e-12)
    scaled = (w32 * quant_scale).clamp(-127.0, 127.0)
    qdata = scaled.round().to(torch.int8)
    scale_ref = (1.0 / quant_scale).to(torch.float32)

    g = load_tensor(f"{GOLDEN}/output_int8_convrot.safetensors", "head.weight_scale")
    scale_golden = torch.from_numpy(
        np.frombuffer(g, dtype="<f4").reshape(128, 1).copy()
    )

    print("scale ref  [:6]:", scale_ref.flatten()[:6].tolist())
    print("scale gold [:6]:", scale_golden.flatten()[:6].tolist())
    diff = (scale_ref != scale_golden).sum().item()
    print(f"ref-vs-golden scale mismatches: {diff}/128")
    if diff:
        idx = (scale_ref != scale_golden).flatten().nonzero().flatten()[:5]
        for i in idx.tolist():
            print(
                f"  row {i}: ref {scale_ref[i,0].item()!r} golden {scale_golden[i,0].item()!r}"
            )

    # qdata compare too
    q = load_tensor(f"{GOLDEN}/output_int8_convrot.safetensors", "head.weight")
    q_golden = torch.from_numpy(np.frombuffer(q, dtype="<i1").reshape(128, 128).copy())
    md = (qdata != q_golden).sum().item()
    print(f"qdata ref-vs-golden mismatches: {md}/{128*128}")

    # row_max values of interest
    print("row_max [:6]:", row_max.flatten()[:6].tolist())

    # ---- what does 1/quant_scale vs row_max/127 give? -------------------- #
    alt = row_max / 127.0
    md2 = (alt.to(torch.float32) != scale_golden).sum().item()
    print(f"row_max/127 vs golden mismatches: {md2}/128")
    alt2 = 1.0 / (127.0 / row_max.clamp_min(1e-12))
    md3 = (alt2.to(torch.float32) != scale_golden).sum().item()
    print(f"1/(127/clamp(row_max)) vs golden mismatches: {md3}/128")

    # torch double vs single precision path?
    w64 = w_f16.to(torch.float64)
    rm64 = w64.abs().amax(dim=1, keepdim=True)
    qs64 = 127.0 / rm64.clamp_min(1e-12)
    s64 = (1.0 / qs64).to(torch.float32)
    md4 = (s64 != scale_golden).sum().item()
    print(f"f64 chain -> f32 vs golden mismatches: {md4}/128")


if __name__ == "__main__":
    main()

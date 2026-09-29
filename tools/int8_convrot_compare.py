#!/usr/bin/env python
"""Compare two int8_convrot safetensors: bytes, scales, and dequantized error.

Reads the comfy_quant blob, dequantizes `weight` (int8 * weight_scale) with the
inverse of whatever rotation the blob declares, and compares against the
BF16 source. Reports per-tensor and aggregate relative error so "which is
better" is a measurement, not an opinion.
"""
import json
import struct
import sys

import numpy as np

try:
    from safetensors import safe_open
except Exception:
    safe_open = None


def read_hdr(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
        meta = h.pop("__metadata__", None)
        base = 8 + n
    return h, meta, base


def raw(fh, base, entry, shape, dtype):
    a, b = entry["data_offsets"]
    n = int(np.prod(shape)) if shape else 1
    dt = {"U8": np.uint8, "I8": np.int8, "F32": np.float32,
          "BF16": np.uint16, "F16": np.float16}[dtype]
    fh.seek(base + a)
    return np.frombuffer(fh.read(b - a), dtype=dt).reshape(shape)


def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def rel_err(got, ref):
    num = np.linalg.norm((got - ref).astype(np.float64).ravel())
    den = np.linalg.norm(ref.astype(np.float64).ravel())
    return num / den if den else float("nan")


def collect(path):
    h, meta, base = read_hdr(path)
    fh = open(path, "rb")
    out = {}
    for k in list(h):
        if not k.endswith(".comfy_quant"):
            continue
        name = k[: -len(".comfy_quant")]
        blob = raw(fh, base, h[k], h[k]["shape"], h[k]["dtype"])
        cfg = json.loads(bytes(blob).decode())
        wkey = name + ".weight"
        skey = name + ".weight_scale"
        if wkey not in h or skey not in h:
            continue
        w = raw(fh, base, h[wkey], h[wkey]["shape"], h[wkey]["dtype"])
        s = raw(fh, base, h[skey], h[skey]["shape"], h[skey]["dtype"])
        out[name] = (cfg, w, s)
    fh.close()
    return out, meta


def main():
    ours_p, ref_p, src_p = sys.argv[1], sys.argv[2], sys.argv[3]
    A, ma = collect(ours_p)
    B, mb = collect(ref_p)

    print(f"ours: {len(A)} quantized tensors, meta keys={list(ma or {})}")
    print(f"ref : {len(B)} quantized tensors, meta keys={list(mb or {})}\n")

    shared = sorted(set(A) & set(B))
    print(f"shared quantized tensors: {len(shared)}")
    print(f"ours-only: {sorted(set(A)-set(B))}")
    print(f"ref-only : {sorted(set(B)-set(A))}\n")

    # ---- blob config comparison ----
    print("--- comfy_quant blob config ---")
    ca = A[shared[0]][0]
    cb = B[shared[0]][0]
    for k in sorted(set(ca) | set(cb)):
        va, vb = ca.get(k, "<absent>"), cb.get(k, "<absent>")
        flag = "" if va == vb else "   <-- DIFFERS"
        print(f"  {k:22s} ours={va!r:28s} ref={vb!r}{flag}")

    # ---- byte / scale comparison ----
    same_w = same_s = 0
    scale_ratio = []
    for n in shared:
        _, wa, sa = A[n]
        _, wb, sb = B[n]
        if np.array_equal(wa, wb):
            same_w += 1
        if np.array_equal(sa, sb):
            same_s += 1
        else:
            nz = sa != 0
            if nz.any():
                scale_ratio.append(float(np.max(np.abs(sb[nz] / sa[nz]))))
    print(f"\n--- payload ---")
    print(f"  identical int8 weight payloads : {same_w}/{len(shared)}")
    print(f"  identical weight_scale arrays   : {same_s}/{len(shared)}")
    if scale_ratio:
        print(f"  scale max|ref/ours| ratio      : "
              f"{min(scale_ratio):.6f} .. {max(scale_ratio):.6f}")

    # ---- numeric accuracy vs source ----
    print("\n--- dequantized relative error vs BF16 source ---")
    sh = safe_open(src_p, framework="np")
    rows = []
    for n in shared:
        key = n + ".weight"
        if key not in sh.keys():
            continue
        ref_f = sh.get_tensor(key).astype(np.float32)
        sa = A[n][1].astype(np.float32) * A[n][2].astype(np.float32)
        sb = B[n][1].astype(np.float32) * B[n][2].astype(np.float32)
        rows.append((n, rel_err(sa, ref_f), rel_err(sb, ref_f)))
    if rows:
        a = np.array([r[1] for r in rows])
        b = np.array([r[2] for r in rows])
        print(f"  tensors compared      : {len(rows)}")
        print(f"  ours  mean rel-L2     : {a.mean():.6f}   max {a.max():.6f}")
        print(f"  ref   mean rel-L2     : {b.mean():.6f}   max {b.max():.6f}")
        print(f"  ours worse on {int((a > b * 1.001).sum())}/{len(rows)} tensors")
        print("\n  worst 10 by (ours - ref):")
        for n, x, y in sorted(rows, key=lambda r: r[1] - r[2], reverse=True)[:10]:
            print(f"    {n:66s} ours={x:.5f} ref={y:.5f}")


if __name__ == "__main__":
    main()

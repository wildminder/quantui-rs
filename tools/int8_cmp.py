#!/usr/bin/env python
"""Correct, self-contained int8_convrot comparison.

Handles the three real-world gotchas:
  * safetensors `data_offsets` are relative to the end of the header (8+N).
  * numpy has no bfloat16 -> decode uint16 pairs manually.
  * convrot blobs differ in key set between producers.

Usage: int8_cmp.py <ours.st> <ref.st> [<bf16-source.st>]
"""
import json
import struct
import sys

import numpy as np


class ST:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            self.h = json.loads(fh.read(n))
            self.meta = self.h.pop("__metadata__", None)
            self.base = 8 + n
        self.fh = open(path, "rb")

    DT = {"U8": np.uint8, "I8": np.int8, "F32": np.float32,
          "BF16": np.uint16, "F16": np.float16, "U32": np.uint32}

    def arr(self, key):
        e = self.h[key]
        a, z = e["data_offsets"]
        self.fh.seek(self.base + a)
        raw = self.fh.read(z - a)
        dt = self.DT[e["dtype"]]
        x = np.frombuffer(raw, dtype=dt)
        return x.reshape(e["shape"]) if e["shape"] else x

    def f32(self, key):
        e = self.h[key]
        a, z = e["data_offsets"]
        self.fh.seek(self.base + a)
        raw = self.fh.read(z - a)
        if e["dtype"] == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16)
            out = (u.astype(np.uint32) << 16).view(np.float32)
        elif e["dtype"] == "F16":
            out = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif e["dtype"] == "F32":
            out = np.frombuffer(raw, dtype=np.float32)
        else:
            raise ValueError(e["dtype"])
        return out.reshape(e["shape"]) if e["shape"] else out

    def blob(self, key):
        e = self.h[key]
        a, z = e["data_offsets"]
        self.fh.seek(self.base + a)
        return json.loads(self.fh.read(z - a).decode())


def rel_err(got, ref):
    num = np.linalg.norm((got - ref).astype(np.float64).ravel())
    den = np.linalg.norm(ref.astype(np.float64).ravel())
    return num / den if den else float("nan")


def main():
    ours_p, ref_p = sys.argv[1], sys.argv[2]
    src_p = sys.argv[3] if len(sys.argv) > 3 else None
    A, B = ST(ours_p), ST(ref_p)

    qa = sorted(k[:-len(".comfy_quant")] for k in A.h if k.endswith(".comfy_quant"))
    qb = sorted(k[:-len(".comfy_quant")] for k in B.h if k.endswith(".comfy_quant"))
    print(f"ours: {len(qa)} quantized, meta={list(A.meta or {})}")
    print(f"ref : {len(qb)} quantized, meta={list(B.meta or {})}")
    print(f"ours-only: {sorted(set(qa)-set(qb))}")
    print(f"ref-only : {sorted(set(qb)-set(qa))}")

    shared = sorted(set(qa) & set(qb))
    print(f"shared: {len(shared)}\n")

    print("--- comfy_quant blob config ---")
    ca, cb = A.blob(qa[0] + ".comfy_quant"), B.blob(qb[0] + ".comfy_quant")
    for k in sorted(set(ca) | set(cb)):
        va, vb = ca.get(k, "<absent>"), cb.get(k, "<absent>")
        print(f"  {k:20s} ours={va!r:26s} ref={vb!r}"
              + ("" if va == vb else "   <-- DIFFERS"))

    # payload / scale
    same_w = same_s = 0
    sratio = []
    for n in shared:
        wa, sa = A.arr(n + ".weight"), A.arr(n + ".weight_scale")
        wb, sb = B.arr(n + ".weight"), B.arr(n + ".weight_scale")
        if np.array_equal(wa, wb):
            same_w += 1
        if np.array_equal(sa, sb):
            same_s += 1
        else:
            nz = sa != 0
            if nz.any():
                sratio.append(float(np.abs(sb[nz] / sa[nz]).max()))
    print(f"\n--- payload ---\n  identical int8 payloads : {same_w}/{len(shared)}")
    print(f"  identical scale arrays   : {same_s}/{len(shared)}")
    if sratio:
        print(f"  scale max|ref/ours|      : {min(sratio):.6f} .. {max(sratio):.6f}")

    if src_p:
        S = ST(src_p)
        print("\n--- dequantized relative error vs BF16 source ---")
        rows = []
        for n in shared:
            key = n + ".weight"
            if key not in S.h:
                continue
            ref_f = S.f32(key)
            wa = A.arr(n + ".weight")
            wb = B.arr(n + ".weight")
            sca = A.arr(n + ".weight_scale")
            scb = B.arr(n + ".weight_scale")
            da = wa.astype(np.float32) * sca.astype(np.float32)
            db = wb.astype(np.float32) * scb.astype(np.float32)
            # the bf16 source may store the same matrix transposed; rel-L2 is
            # invariant to that, so flatten both to 1-D before comparing.
            rows.append((n, rel_err(da.ravel(), ref_f.ravel()),
                         rel_err(db.ravel(), ref_f.ravel())))
        if rows:
            a = np.array([r[1] for r in rows])
            b = np.array([r[2] for r in rows])
            print(f"  tensors        : {len(rows)}")
            print(f"  ours mean relL2: {a.mean():.6f}   max {a.max():.6f}")
            print(f"  ref  mean relL2: {b.mean():.6f}   max {b.max():.6f}")
            print(f"  ours worse on {int((a > b * 1.001).sum())}/{len(rows)} tensors")
            print("\n  top-10 by (ours - ref):")
            for n, x, y in sorted(rows, key=lambda r: r[1] - r[2], reverse=True)[:10]:
                print(f"    {n:64s} ours={x:.5f} ref={y:.5f}")


if __name__ == "__main__":
    main()

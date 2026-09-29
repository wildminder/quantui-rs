#!/usr/bin/env python
"""Invert the ConvRot Hadamard rotation and measure TRUE reconstruction error.

The stored int8 payload lives in ROTATED space, so comparing it directly to the
BF16 source is meaningless (rel-L2 ~ sqrt(2) for both producers). The only
honest metric is: dequantize -> un-rotate -> compare against the source.

W_rot = W_grouped @ H^T   (convrot.rs::rotate_weight)
H is symmetric and orthogonal, so the inverse is W = W_rot @ H.

Usage: convrot_verify.py <file.st> <bf16-source.st> [--label NAME]
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from int8_cmp import ST  # noqa: E402

H4 = np.array([[1, 1, 1, -1],
               [1, 1, -1, 1],
               [1, -1, 1, 1],
               [-1, 1, 1, 1]], dtype=np.float32)


def build_hadamard(size):
    """Port of convrot.rs::build_hadamard — kron build, normalize by sqrt(size)."""
    cur = H4.copy()
    s = 4
    while s < size:
        cur = np.kron(cur, H4)
        s *= 4
    if s != size:
        raise ValueError(f"size {size} not a power of 4")
    return cur / np.float32(np.sqrt(size))


def unrotate(w_rot, h, gs):
    """W = W_rot_grouped @ H, applied per gs-wide column group."""
    rows, n = w_rot.shape
    out = np.empty_like(w_rot)
    for g in range(n // gs):
        b = g * gs
        # (W_rot[:, b:b+gs] @ H) == (W_rot_grouped @ H) with H symmetric
        out[:, b:b + gs] = w_rot[:, b:b + gs] @ h
    return out


def main():
    path, src = sys.argv[1], sys.argv[2]
    label = path
    if "--label" in sys.argv:
        label = sys.argv[sys.argv.index("--label") + 1]

    A = ST(path)
    S = ST(src)
    gs = 256
    h = build_hadamard(gs)

    qa = sorted(k[: -len(".comfy_quant")] for k in A.h if k.endswith(".comfy_quant"))
    rows = []
    for n in qa:
        wkey = n + ".weight"
        if wkey not in S.h:
            continue
        src_f = S.f32(wkey).astype(np.float32)
        w = A.arr(wkey).astype(np.float32) * A.arr(n + ".weight_scale").astype(np.float32)
        rec = unrotate(w, h, gs)

        # the source may be stored transposed; try both orientations and keep
        # the better one (a wrong orientation gives rel-L2 ~ sqrt(2), not ~0.05)
        best = None
        for cand in (rec, rec.T):
            if cand.shape != src_f.shape:
                continue
            e = np.linalg.norm((cand - src_f).ravel()) / np.linalg.norm(src_f.ravel())
            if best is None or e < best[0]:
                best = (e, "direct" if cand is rec else "transposed")
        if best is None:
            continue
        rows.append((n, best[0], best[1]))

    errs = np.array([r[1] for r in rows])
    orient = {}
    for r in rows:
        orient[r[2]] = orient.get(r[2], 0) + 1
    print(f"=== {label} ===")
    print(f"  tensors verified      : {len(rows)}")
    print(f"  mean rel-L2 (unrotated): {errs.mean():.6f}")
    print(f"  median                : {np.median(errs):.6f}")
    print(f"  max                   : {errs.max():.6f}  ({rows[int(errs.argmax())][0]})")
    print(f"  min                   : {errs.min():.6f}")
    print(f"  orientation used      : {orient}")
    print("\n  worst 8:")
    for n, e, o in sorted(rows, key=lambda r: -r[1])[:8]:
        print(f"    {e:.5f}  [{o:10s}] {n}")
    print("\n  best 4:")
    for n, e, o in sorted(rows, key=lambda r: r[1])[:4]:
        print(f"    {e:.5f}  [{o:10s}] {n}")


if __name__ == "__main__":
    main()

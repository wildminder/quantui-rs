#!/usr/bin/env python3
"""Is the ours-vs-official ConvRot payload difference a DEFECT or ULP noise?

The int8 codes differ on ~20 of 16.7M elements and the F32 row scales differ
by ~1 ULP. That is only meaningful once you undo the rotation, because the
stored payload lives in rotated space.

Method (the only honest one for ConvRot):
    dequantize(int8, scale) -> un-rotate with H^T -> compare in SOURCE space

Then report, for the same tensors:
    ours vs official   (the two producers)
    ours   vs bf16 source
    official vs bf16 source

If ours ~= official ~= source to ~1e-6, the 1-ULP scale delta and the +-1
code flips are rounding noise, not a quantizer defect.

Usage: convrot_ulp_verdict.py <ours.st> <official.st> <bf16-source.st> [n_tensors]
"""
import json
import struct
import sys

import numpy as np

H4 = np.array([[1, 1, 1, -1],
               [1, 1, -1, 1],
               [1, -1, 1, 1],
               [-1, 1, 1, 1]], dtype=np.float32)


def build_hadamard(size):
    cur = H4.copy()
    s = 4
    while s < size:
        cur = np.kron(cur, H4).astype(np.float32)
        s *= 4
    return (cur / np.sqrt(size)).astype(np.float32)


class ST:
    def __init__(self, path):
        self.path = path
        self.fh = open(path, "rb")
        (n,) = struct.unpack("<Q", self.fh.read(8))
        self.hdr = json.loads(self.fh.read(n))
        self.hdr.pop("__metadata__", None)
        self.base = 8 + n

    def raw(self, key):
        o = self.hdr[key]["data_offsets"]
        self.fh.seek(self.base + o[0])
        return self.fh.read(o[1] - o[0])

    def i8(self, key):
        return np.frombuffer(self.raw(key), dtype=np.int8).copy()

    def f32(self, key):
        return np.frombuffer(self.raw(key), dtype=np.float32).copy()

    def numeric(self, key):
        """bf16/f16/f32 -> f32. BF16 is the top 2 bytes of an f32, so widen
        by shifting — reading BF16 bytes as f32 (the first attempt) silently
        halves the element count and reshape then fails."""
        dt = self.hdr[key]["dtype"]
        raw = self.raw(key)
        if dt == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            return (u16 << 16).view(np.float32).copy()
        if dt == "F16":
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        if dt == "F32":
            return np.frombuffer(raw, dtype=np.float32).copy()
        raise ValueError(f"{key}: unsupported dtype {dt}")

    def shape(self, key):
        return tuple(self.hdr[key]["shape"])


def dequant_unrotate(st, key, H):
    """int8 + per-row scale -> f32 -> un-rotate -> (n, in_features)."""
    q = st.i8(key).reshape(st.shape(key)).astype(np.float32)
    sk = key.replace(".weight", ".weight_scale")
    s = st.f32(sk)
    if s.size == 1:
        w = q * s.reshape(())
    else:
        w = q * s.reshape(-1, 1)
    gs = H.shape[0]
    return (w.reshape(w.shape[0], -1, gs) @ H).reshape(w.shape)


def rel_l2(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    n = np.linalg.norm(d)
    den = np.linalg.norm(b.astype(np.float64))
    return float(n / den) if den else float("nan")


def main():
    ours_p, off_p, src_p = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 6

    A, B, S = ST(ours_p), ST(off_p), ST(src_p)
    H = build_hadamard(256)

    # Tensors both sides quantized AND the source has as bf16.
    keys = [k for k in sorted(A.hdr)
            if k.endswith(".weight")
            and A.hdr[k]["dtype"] == "I8"
            and B.hdr.get(k, {}).get("dtype") == "I8"
            and S.hdr.get(k, {}).get("dtype") == "BF16"]
    print(f"comparable tensors: {len(keys)}\n")
    print(f"{'tensor':<46} {'ours~off':>10} {'ours~src':>10} {'off~src':>10}")
    print("-" * 80)
    rows = []
    for k in keys[:limit]:
        wa = dequant_unrotate(A, k, H)
        wb = dequant_unrotate(B, k, H)
        src = S.numeric(k).reshape(S.shape(k))
        a_b, a_s, b_s = rel_l2(wa, wb), rel_l2(wa, src), rel_l2(wb, src)
        rows.append((a_b, a_s, b_s))
        print(f"{k:<46} {a_b:>10.3e} {a_s:>10.3e} {b_s:>10.3e}")
    if rows:
        r = np.array(rows)
        print("-" * 80)
        print(f"{'MEAN':<46} {r[:,0].mean():>10.3e} {r[:,1].mean():>10.3e} {r[:,2].mean():>10.3e}")
        print(f"{'MAX':<46} {r[:,0].max():>10.3e} {r[:,1].max():>10.3e} {r[:,2].max():>10.3e}")
        print()
        # The verdict must be RELATIVE to the int8 quantization noise floor,
        # not against a magic absolute constant. Each producer's own distance
        # to the bf16 source IS the floor: two producers that each sit
        # ~8.9e-3 from the source and ~3e-5 from EACH OTHER are the same
        # quantizer to far better than the quantization error itself.
        #
        # An earlier version of this script hardcoded `ours~off < 1e-5` and
        # printed "REAL DIVERGENCE", and separately declared a winner by
        # comparing means that were equal to 4 significant figures. Both were
        # wrong: the first ignored the noise floor, the second manufactured a
        # winner out of a difference below print precision.
        floor = max(r[:, 1].mean(), r[:, 2].mean())
        gap = r[:, 0].mean()
        ratio = gap / floor if floor else float("inf")
        if ratio < 0.01:
            verdict = "EQUIVALENT (difference is far below the int8 noise floor)"
        elif ratio < 0.25:
            verdict = "EQUIVALENT for practical purposes"
        else:
            verdict = "REAL DIVERGENCE — investigate"
        print(f"ours-vs-official gap        : {gap:.6e}")
        print(f"int8 noise floor (to source): {floor:.6e}")
        print(f"gap / floor                 : {ratio:.2e}  ({1/ratio:.0f}x below the floor)")
        print(f"VERDICT: {verdict}")
        print()
        # Full precision, and only a "winner" if the difference is material.
        d_ours = r[:, 1].mean() - r[:, 2].mean()
        rel = abs(d_ours) / floor if floor else 0.0
        if rel < 0.01:
            print("accuracy vs bf16 source: STATISTICALLY TIED "
                  f"(delta {d_ours:+.3e}, {rel*100:.3f}% of the floor)")
        else:
            print(f"accuracy vs bf16 source: ours is "
                  f"{'CLOSER' if d_ours < 0 else 'FARTHER'} "
                  f"({d_ours:+.3e}, {rel*100:.2f}% of the floor)")


if __name__ == "__main__":
    main()

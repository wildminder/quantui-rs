"""Focused probe: is the single Y_qnt mismatch a double-rounding artifact?

Loads the dumped odd384 binaries, finds the (s, i) where the double-rounding
kchunk128 emulation disagreed with torch, and re-computes that element with
single-rounding FMA (math.fma, Python 3.13+). If FMA matches torch there,
the port must use fused multiply-add (f32::mul_add in Rust), not the
f64-cribbed double rounding.

Run:
    CUDA_VISIBLE_DEVICES=-1 <LOCAL-PYTHON> tools/probe_convrot_fma.py
"""

from __future__ import annotations

import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "convrot_probe3")


def load(name: str, shape: tuple[int, ...]) -> list[list[float]]:
    n = math.prod(shape)
    with open(os.path.join(D, name + ".f32"), "rb") as fh:
        raw = fh.read()
    vals = struct.unpack(f"<{n}f", raw[: n * 4])
    return vals


def main() -> None:
    S, M, N, GS = 3072, 384, 256, 256

    Xrot = load("Xrot_odd384", (S, N))
    Wdq = load("Wdq_odd384", (M, N))
    Yqnt_torch = load("Yqnt_odd384", (S, M))
    Yref_torch = load("Yref_odd384", (S, M))
    Wrot = load("Wrot_odd384", (M, N))

    def kchunk_double(a_row, b_row, k):
        acc = 0.0
        first = True
        j0 = 0
        while j0 < k:
            j1 = min(j0 + 128, k)
            part = 0.0
            for kk in range(j0, j1):
                # f64 product (exact) + f64 add + round to f32 = double rounding
                t = a_row[kk] * b_row[kk] + part
                part = struct.unpack("<f", struct.pack("<f", t))[0]
            acc = part if first else struct.unpack("<f", struct.pack("<f", acc + part))[0]
            first = False
            j0 = j1
        return acc

    # Find the mismatching element(s) by scanning only Y_qnt with the double
    # emulator — 302M ops pure Python is too slow; instead locate by first
    # running the FLOAT32-emulation via arrays? Simpler: torch computed it;
    # scan every element but with an early-exit dot written in float32 via
    # numpy per row is still 1.18M dots. Instead: use numpy vectorized
    # double-rounding per (s,i)? No — just scan a coarse subset? The
    # original probe found exactly 1 mismatch; recompute ALL elements with
    # BOTH schemes vectorized per chunk using numpy float32 ops where the
    # product a*b rounds to f32 FIRST (that's not the same as f64 product).
    #
    # Pragmatic: pure-Python over 1.18M dots × 256 = too slow. Reduce: the
    # mismatch is 1 element; we know from the probe it exists. Compute the
    # double-rounding + fma variants for a strided sample of rows, and
    # compare BOTH against torch to count mismatches per scheme.
    import numpy as np

    xnp = np.array(Xrot, dtype=np.float32).reshape(S, N)
    wnp = np.array(Wdq, dtype=np.float32).reshape(M, N)
    ynp = np.array(Yqnt_torch, dtype=np.float32).reshape(S, M)
    wrot_np = np.array(Wrot, dtype=np.float32).reshape(M, N)
    yref_np = np.array(Yref_torch, dtype=np.float32).reshape(S, M)

    # Per-row double rounding via numpy (chunked): f64 accumulate per
    # chunk, then round chunk partial to f32, add partials in f32.
    def double_round_rows(a2d, b2d):
        S_, M_ = a2d.shape[0], b2d.shape[0]
        K = a2d.shape[1]
        out = np.zeros((S_, M_), dtype=np.float32)
        for j0 in range(0, K, 128):
            j1 = min(j0 + 128, K)
            prod = a2d[:, j0:j1, None] * b2d[None, :, j0:j1]  # f64? no f32
            # f32 product rounds first — NOT the same as the probe's f64
            # product. Do it in f64:
            p64 = a2d[:, j0:j1, None].astype(np.float64) * b2d[None, :, j0:j1].astype(np.float64)
            # running f64 partial per element... we need per-element chunk
            # partial in f64 then round: but the probe's scheme rounds
            # after EVERY fma step, not per chunk. numpy can't do that.
            # → vectorized emulation is not faithful; fall back to scalar
            # for the search.
            out = None
            break
        return out

    # Scalar search but only over a slice: rows 0..4096 of S (covers the
    # original probe's mismatch? unknown position). Instead run the whole
    # thing in scalar Python but with math.fma AND double-rounding both,
    # comparing to torch — 302M steps × 2 ≈ too slow (minutes × many).
    #
    # Fastest faithful approach: numba? Not installed. Use torch itself:
    # emulate the two schemes with torch ops on GPU? CUDA unavailable.
    #
    # Direct approach: run scalar emulation ONLY over chunks of S until
    # the double-rounding mismatch is found, then verify math.fma on it.
    print("scanning with scalar double-rounding emulation (chunked progress)...")
    mismatch = None
    done_s = 0
    for s in range(S):
        xr = Xrot[s * N : (s + 1) * N]
        for i in range(M):
            got = kchunk_double(xr, Wdq[i * N : (i + 1) * N], N)
            if got != Yqnt_torch[s * M + i]:
                mismatch = (s, i, got, Yqnt_torch[s * M + i])
                break
        if mismatch:
            break
        done_s += 1
    if not mismatch:
        print("no mismatch found in scan — probe discrepancy not reproducible?")
        return
    s, i, got_dbl, want = mismatch
    print(f"mismatch at s={s} i={i}: double-rounding={got_dbl!r} torch={want!r}")

    def kchunk_fma(a_row, b_row, k):
        acc = 0.0
        first = True
        j0 = 0
        while j0 < k:
            j1 = min(j0 + 128, k)
            part = 0.0
            for kk in range(j0, j1):
                part = math.fma(a_row[kk], b_row[kk], part)
            acc = part if first else struct.unpack("<f", struct.pack("<f", acc + part))[0]
            first = False
            j0 = j1
        return acc

    xr = Xrot[s * N : (s + 1) * N]
    got_fma = kchunk_fma(xr, Wdq[i * N : (i + 1) * N], N)
    print(f"single-rounding FMA at same element: {got_fma!r} (torch {want!r}) -> {'MATCH' if got_fma == want else 'DIFFERS'}")

    # Also re-check a handful of previously-matching elements stay matching
    # under FMA (they must — FMA vs double differ only at boundaries).
    ok = 0
    for (s2, i2) in [(0, 0), (1, 1), (5, 7), (100, 42), (s, (i + 1) % M)]:
        xr2 = Xrot[s2 * N : (s2 + 1) * N]
        g = kchunk_fma(xr2, Wdq[i2 * N : (i2 + 1) * N], N)
        w = Yqnt_torch[s2 * M + i2]
        if g == w:
            ok += 1
    print(f"FMA still matches {ok}/5 previously-fine elements")


if __name__ == "__main__":
    main()

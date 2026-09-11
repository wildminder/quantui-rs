"""Probe v2: find torch CPU's accumulation order for the gs=256 rotation matmul.

torch.matmul((M,1,256) f32, (256,256) f32) on CPU eager — candidates:
  a) oneDNN-style K-chunks of size C (fma per element in f64->f32, partials
     summed left-to-right), C in {32, 64, 128, 256}
  b) vectorized 8-lane: 8 independent f32 chains, k strided by 8, summed at end
     (various lane-sum orders)
  c) 4-way ILP chains strided by 4
  d) naive sequential (known mismatch)

Run:
    CUDA_VISIBLE_DEVICES=-1 python tools/probe_convrot_gs256.py
"""

from __future__ import annotations

import itertools

import numpy as np
import torch


def build_H(gs: int) -> np.ndarray:
    H4 = np.array(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=np.float32,
    )
    H = H4
    cur = 4
    while cur < gs:
        H = np.kron(H, H4).astype(np.float32)
        cur *= 4
    return (H / np.float32(np.sqrt(np.float32(gs)))).astype(np.float32)


def f32(x) -> np.float32:
    return np.float32(x)


def cand_kchunk(W, Ht, M, gs, chunk) -> np.ndarray:
    """oneDNN-style: per 128(=chunk) K-chunk, fma f64->f32 chain; partials
    added left-to-right as f32 (f64 add of two f32s rounded once)."""
    out = np.zeros((M, gs), dtype=np.float32)
    for i in range(M):
        for j in range(gs):
            acc = f32(0.0)
            first = True
            j0 = 0
            while j0 < gs:
                j1 = min(j0 + chunk, gs)
                part = f32(0.0)
                for k in range(j0, j1):
                    part = f32(np.float64(W[i, k]) * np.float64(Ht[k, j]) + np.float64(part))
                acc = part if first else f32(np.float64(acc) + np.float64(part))
                first = False
                j0 = j1
            out[i, j] = acc
    return out


def cand_lanes(W, Ht, M, gs, lanes, lane_order) -> np.ndarray:
    """vectorized: `lanes` independent f32 chains over k strided by `lanes`
    (k = q*lanes + lane), each a plain f32 multiply-add chain; final sum of
    lane partials in the given order (f32 adds)."""
    out = np.zeros((M, gs), dtype=np.float32)
    for i in range(M):
        for j in range(gs):
            acc = [f32(0.0)] * lanes
            for k in range(gs):
                acc[k % lanes] = f32(acc[k % lanes] + f32(W[i, k] * Ht[k, j]))
            s = acc[lane_order[0]]
            for li in lane_order[1:]:
                s = f32(s + acc[li])
            out[i, j] = s
    return out


def cand_blocked_lanes(W, Ht, M, gs, block, lanes, lane_order) -> np.ndarray:
    """blocked+lanes: outer accumulation over blocks of `block` K; inside a
    block, `lanes` chains; block partials (summed in lane_order) added
    left-to-right."""
    out = np.zeros((M, gs), dtype=np.float32)
    for i in range(M):
        for j in range(gs):
            bacc = f32(0.0)
            first = True
            for b0 in range(0, gs, block):
                acc = [f32(0.0)] * lanes
                for k in range(b0, b0 + block):
                    acc[(k - b0) % lanes] = f32(
                        acc[(k - b0) % lanes] + f32(W[i, k] * Ht[k, j])
                    )
                s = acc[lane_order[0]]
                for li in lane_order[1:]:
                    s = f32(s + acc[li])
                bacc = s if first else f32(bacc + s)
                first = False
            out[i, j] = bacc
    return out


def main() -> None:
    gs = 256
    M = 8
    rng = np.random.default_rng(4321)
    W = rng.standard_normal((M, gs)).astype(np.float32)
    H = build_H(gs)
    Ht = np.ascontiguousarray(H.T)  # values equal H (symmetric)

    tW = torch.from_numpy(W)
    tWg = tW.view(M, 1, gs)
    Ht_t = torch.from_numpy(Ht)

    # The exact reference call shape: matmul(W_grouped, H.T-view)
    ref_view = torch.matmul(tWg, Ht_t.T).reshape(M, gs).numpy()  # .T view
    ref_contig = torch.matmul(tWg, Ht_t).reshape(M, gs).numpy()  # contiguous
    ref_2d = (tW @ Ht_t.T).numpy()  # plain 2-D mm with transposed view
    print("view vs contiguous identical:", np.array_equal(ref_view, ref_contig))
    print("3d vs 2d identical:", np.array_equal(ref_view, ref_2d))
    ref = ref_view

    cands = {}
    for chunk in (32, 64, 128, 256):
        cands[f"kchunk{chunk}"] = cand_kchunk(W, Ht, M, gs, chunk)
    for lanes in (4, 8, 16):
        for order in (tuple(range(lanes)), tuple(reversed(range(lanes)))):
            cands[f"lanes{lanes}o{''.join(map(str, order[:3]))}"] = cand_lanes(
                W, Ht, M, gs, lanes, list(order)
            )
    for block, lanes in itertools.product((64, 128), (4, 8)):
        cands[f"blk{block}ln{lanes}"] = cand_blocked_lanes(
            W, Ht, M, gs, block, lanes, list(range(lanes))
        )

    print(f"{'candidate':24s} exact-mismatch count / {M*gs}")
    best = None
    for name, got in cands.items():
        mm = int((got != ref).sum())
        print(f"{name:24s} {mm}")
        if mm == 0:
            best = name
    print("WINNER:", best)


if __name__ == "__main__":
    main()

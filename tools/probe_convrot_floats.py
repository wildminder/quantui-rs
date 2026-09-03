"""Probe: torch CPU matmul float semantics for the ConvRot port (Phase 7.1/7.2).

Three float ops must be emulated bit-exactly in Rust:

1. `rotate_weight`:  W_grouped @ H^T  where W_grouped is (M, N/gs, gs) and
   H^T is (gs, gs) — torch.einsum/bmm-style batched GEMM over M*(N/gs) rows,
   K = gs (64/256/1024).
2. `Y_ref  = X_rot @ W_rot.T` — GEMM (S=3072, K=N, M=N rows).
3. `Y_quant = X_rot @ W_dq_rot.T` then `bias_adj = (Y_ref - Y_quant).mean(0)`
   then `b_new = b + bias_adj` (convrot path ADDS; the generic path SUBTRACTS
   `mean(X @ err.T)`).

For each: dump inputs + outputs as raw f32 LE binaries, and print whether
simple dot-product with f64-accumulated-then-f32-chunks (the oneDNN K-chunk
scheme used in bias_correction.rs) reproduces torch bit-exactly.

Run with the ctq venv, CUDA hidden:
    CUDA_VISIBLE_DEVICES=-1 <LOCAL-PYTHON> tools/probe_convrot_floats.py
"""

from __future__ import annotations

import os

import numpy as np
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "convrot_probe")
os.makedirs(OUT, exist_ok=True)


def dump(name: str, t: torch.Tensor) -> None:
    """Dump a float32 tensor as raw LE f32 bytes."""
    a = t.detach().cpu().to(torch.float32).contiguous().numpy()
    with open(os.path.join(OUT, name + ".f32"), "wb") as fh:
        fh.write(a.tobytes())
    print(f"  dumped {name}: shape={tuple(a.shape)}")


def main() -> None:
    torch.manual_seed(1234)
    rng = np.random.default_rng(4321)

    # ---- 1. rotate_weight: (M, gs) group @ H^T -------------------------- #
    print("[1] rotate_weight semantics")
    for gs in (4, 16, 64, 256):
        M = 8
        # Reference build_hadamard: H4 Kronecker, normalized by sqrt(size).
        H4 = np.array(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=np.float64,
        )
        H = H4
        cur = 4
        while cur < gs:
            H = np.kron(H, H4)
            cur *= 4
        H = torch.from_numpy(H).to(torch.float32) / (gs**0.5)

        W = torch.from_numpy(rng.standard_normal((M, gs)).astype(np.float32))
        Wg = W.view(M, 1, gs)
        Ht = H.T.contiguous()
        W_rot = torch.matmul(Wg, Ht).reshape(M, gs)
        dump(f"W_{gs}", W)
        dump(f"H_{gs}", H)
        dump(f"Wrot_{gs}", W_rot)
        # plain loop comparison: f32 dot with sequential f32 adds
        Hnp = H.numpy()
        Wnp = W.numpy()
        ref = W_rot.numpy()
        mismatch = 0
        for i in range(M):
            for j in range(gs):
                acc = np.float32(0.0)
                for k in range(gs):
                    acc = np.float32(acc + np.float32(Wnp[i, k] * Hnp.T[k, j]))
                if acc != ref[i, j]:
                    mismatch += 1
        print(f"  gs={gs}: naive f32 sequential-dot mismatches = {mismatch}/{M*gs}")

    # ---- 2. Y_ref / Y_quant GEMM + mean + add ---------------------------- #
    print("[2] convrot bias-correction chain")
    S, K, N = 512, 256, 256  # S small probe; golden uses 3072
    X = torch.from_numpy(rng.standard_normal((S, K)).astype(np.float32))
    W1 = torch.from_numpy(rng.standard_normal((N, K)).astype(np.float32) * 0.05)
    W2 = torch.from_numpy(rng.standard_normal((N, K)).astype(np.float32) * 0.05)
    b = torch.from_numpy(rng.standard_normal(N).astype(np.float32) * 0.01)

    Y_ref = X @ W1.T
    Y_qnt = X @ W2.T
    diff = Y_ref - Y_qnt
    mean = diff.mean(dim=0)
    b_new = b + mean

    dump("X", X)
    dump("W1", W1)
    dump("W2", W2)
    dump("b", b)
    dump("Y_ref", Y_ref)
    dump("Y_qnt", Y_qnt)
    dump("diff", diff)
    dump("mean", mean)
    dump("b_new", b_new)

    # (b + mean) association test: is b_new == b + (sum/S) with f64 sum?
    Xn = X.numpy()
    W1n = W1.numpy()
    W2n = W2.numpy()
    bad = 0
    for j in range(N):
        # GEMM row: sum_k X[s,k]*(W1-W2)[j,k]  via f64 accumulation in K-chunks
        # then f32 chunk partials (oneDNN style), then cascade mean.
        outs = np.zeros(S, dtype=np.float32)
        for s in range(S):
            acc = np.float32(0.0)
            first = True
            j0 = 0
            while j0 < K:
                j1 = min(j0 + 128, K)
                part = np.float32(0.0)
                for k in range(j0, j1):
                    part = np.float32(
                        np.float64(Xn[s, k])
                        * np.float64(W1n[j, k] - W2n[j, k])
                        + np.float64(part)
                    )
                acc = part if first else np.float32(np.float64(acc) + np.float64(part))
                first = False
                j0 = j1
            outs[s] = acc
        # mean via naive /S after f64 sum
        m = np.float32(np.float32(np.sum(outs.astype(np.float64))) / S)
        if np.float32(b.numpy()[j] + m) != b_new.numpy()[j]:
            bad += 1
    print(f"  chunked-GEMM + naive-mean mismatches = {bad}/{N}")


if __name__ == "__main__":
    main()

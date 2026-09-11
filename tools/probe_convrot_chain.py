"""Probe v3: verify the exact ConvRot bias-correction chain bit-exactly.

Reference structure (learned_rounding.py::_convert_int8_tensorwise, simple
mode + fp8_conversion.py driver):

    H    = build_hadamard(gs)                     # H4-Kronecker / sqrt(gs)
    Wrot = rotate_weight(W, H, gs)                # (m, n/gs, gs) @ H^T
    q, s = TensorWiseINT8Layout.quantize(Wrot)    # row-wise INT8
    Wdq  = dequantize(q, s)                       # q.to(f32) * s  (row bcast)
    Xrot = rotate_activation(X, H, gs)            # (S, n/gs, gs) @ H
    Yref = Xrot @ Wrot.T                          # GEMM K=n
    Yqnt = Xrot @ Wdq.T                           # GEMM K=n
    adj  = (Yref - Yqnt).mean(dim=0)              # cascade sum / S
    bnew = b + adj                                # f32 add

Emulation candidates (each GEMM): oneDNN kchunk128 fma scheme (the same one
validated in bias_correction.rs at S=3072, K in {64,128,256}).
Mean: cascade_sum (multi_row_sum/row_sum dispatch) then f32 /S.

Dumps every stage to tools/convrot_probe3/ for the Rust unit tests.

Run:
    CUDA_VISIBLE_DEVICES=-1 python tools/probe_convrot_chain.py
"""

from __future__ import annotations

import os

import numpy as np
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "convrot_probe3")
os.makedirs(OUT, exist_ok=True)

SEED = 97531


def dump(name: str, t) -> None:
    a = np.ascontiguousarray(t.detach().numpy(), dtype=np.float32)
    with open(os.path.join(OUT, name + ".f32"), "wb") as fh:
        fh.write(a.tobytes())
    print(f"  dumped {name:12s} shape={a.shape}")


def f32(x) -> np.float32:
    return np.float32(x)


def build_H(gs: int) -> torch.Tensor:
    H4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
    )
    H = H4
    cur = 4
    while cur < gs:
        H = torch.kron(H, H4)
        cur *= 4
    return H / (gs**0.5)


def kchunk_gemm_rowmajor(a_row: np.ndarray, b_row: np.ndarray, k: int | None = None) -> np.float32:
    """out = sum_k a_row[k] * b_row[k]; oneDNN kchunk128: per 128-chunk fma
    chain in f64->f32, chunk partials added left-to-right."""
    if k is None:
        k = len(a_row)
    acc = f32(0.0)
    first = True
    j0 = 0
    while j0 < k:
        j1 = min(j0 + 128, k)
        part = f32(0.0)
        for kk in range(j0, j1):
            part = f32(np.float64(a_row[kk]) * np.float64(b_row[kk]) + np.float64(part))
        acc = part if first else f32(np.float64(acc) + np.float64(part))
        first = False
        j0 = j1
    return acc


def emulate_gemm(x: np.ndarray, w: np.ndarray, m: int, n: int, s: int) -> np.ndarray:
    """Y[s, i] = sum_j X[s, j] * W[i, j]  (i.e. X @ W.T), kchunk128."""
    out = np.zeros((s, m), dtype=np.float32)
    for si in range(s):
        xr = x[si]
        for i in range(m):
            out[si, i] = kchunk_gemm_rowmajor(xr, w[i], n)
    return out


def rotate_rows(w: np.ndarray, h: np.ndarray, gs: int, rows: int, n: int) -> np.ndarray:
    """Grouped rotation: out[r, j] = sum_k W[r, g*gs+k] * H.T[k, j - g*gs].
    H symmetric so H.T values == H; emulate as (rows, n/gs, gs)@(gs, gs)."""
    out = np.zeros((rows, n), dtype=np.float32)
    n_groups = n // gs
    h_t = np.ascontiguousarray(h.T)
    for r in range(rows):
        for g in range(n_groups):
            base = g * gs
            for j in range(gs):
                out[r, base + j] = kchunk_gemm_rowmajor(w[r, base : base + gs], h_t[:, j], gs)
    return out


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    for tag, gs, m, n, S in [
        ("odd384", 256, 384, 256, 3072),
        ("small64", 64, 128, 128, 512),
    ]:
        print(f"[case {tag}] gs={gs} m={m} n={n} S={S}")
        H = build_H(gs)
        W = torch.from_numpy((rng.standard_normal((m, n)).astype(np.float32) * 0.05))
        b = torch.from_numpy(rng.standard_normal(m).astype(np.float32) * 0.01)
        X = torch.from_numpy(rng.standard_normal((S, n)).astype(np.float32))

        # ---- reference pipeline (exactly as learned_rounding.py) ---------- #
        Wg = W.view(m, n // gs, gs)
        H_t = H.T.to(dtype=W.dtype)
        W_rot = torch.matmul(Wg, H_t).reshape(m, n)

        # TensorWiseINT8Layout.quantize row-wise, is_weight=True
        row_max = W_rot.abs().amax(dim=1, keepdim=True)
        quant_scale = 127.0 / row_max.clamp_min(1e-12)
        scaled = (W_rot * quant_scale).clamp(-127.0, 127.0)
        qdata = scaled.round().to(torch.int8)
        scale = (1.0 / quant_scale).to(torch.float32)  # [m, 1]

        W_dq = qdata.to(torch.float32) * scale  # row broadcast dequant

        Xg = X.view(S, n // gs, gs)
        X_rot = torch.matmul(Xg, H.to(dtype=X.dtype)).view(S, n)

        Y_ref = X_rot @ W_rot.T
        Y_qnt = X_rot @ W_dq.T
        adj = (Y_ref - Y_qnt).mean(dim=0)
        b_new = b + adj

        dump(f"W_{tag}", W)
        dump(f"H_{tag}", H)
        dump(f"X_{tag}", X)
        dump(f"b_{tag}", b)
        dump(f"Wrot_{tag}", W_rot)
        dump(f"scale_{tag}", scale)
        dump(f"Wdq_{tag}", W_dq)
        dump(f"Xrot_{tag}", X_rot)
        dump(f"Yref_{tag}", Y_ref)
        dump(f"Yqnt_{tag}", Y_qnt)
        dump(f"adj_{tag}", adj)
        dump(f"bnew_{tag}", b_new)

        # ---- emulation checks ---------------------------------------------- #
        Hn = H.numpy()
        Wn = W.numpy()
        Xn = X.numpy()
        bn = b.numpy()

        W_rot_e = rotate_rows(Wn, Hn, gs, m, n)
        mm_w = int((W_rot_e != W_rot.numpy()).sum())
        print(f"  W_rot  kchunk128 mismatches: {mm_w}/{m*n}")

        X_rot_e = rotate_rows(Xn, Hn, gs, S, n)
        mm_x = int((X_rot_e != X_rot.numpy()).sum())
        print(f"  X_rot  kchunk128 mismatches: {mm_x}/{S*n}")

        Y_ref_e = emulate_gemm(X_rot_e, W_rot_e, m, n, S)
        mm_yr = int((Y_ref_e != Y_ref.numpy()).sum())
        print(f"  Y_ref  kchunk128 mismatches: {mm_yr}/{S*m}")

        Y_qnt_e = emulate_gemm(X_rot_e, W_dq.numpy(), m, n, S)
        mm_yq = int((Y_qnt_e != Y_qnt.numpy()).sum())
        print(f"  Y_qnt  kchunk128 mismatches: {mm_yq}/{S*m}")

        # diff exact (f32 sub), then cascade mean — reuse probe-validated
        # cascade from bias_correction; approximate here with naive f64 sum
        # then /S to see how close; exact check needs the cascade port.
        diff = (Y_ref_e.astype(np.float64) - Y_qnt_e.astype(np.float64))
        # elementwise f32 subtraction first (torch does f32 sub):
        diff32 = (Y_ref_e - Y_qnt_e).astype(np.float32)
        d_ref = (Y_ref.numpy() - Y_qnt.numpy()).astype(np.float32)
        mm_d = int((diff32 != d_ref).sum())
        print(f"  diff   f32-sub mismatches:   {mm_d}/{S*m}")


if __name__ == "__main__":
    main()

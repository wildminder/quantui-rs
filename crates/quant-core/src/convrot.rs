//! ConvRot — group-wise Hadamard rotation for INT8 row-wise quantization
//! (plan Phase 7.1/7.2, decision Q3).
//!
//! Port of `convert_to_quant/utils/convrot.py` (originally from
//! ComfyUI-ZImage-Triton, MIT; based on QuaRot 2024 / ConvRot 2025):
//! - `build_hadamard`: the Theorem-3.3 base H4 (every row/column sums to
//!   exactly 2 — avoids the all-1s column of Sylvester matrices, which
//!   amplifies row-wise outliers) Kronecker-built up to `size`, normalized
//!   by `sqrt(size)` to make it orthogonal. Size must be a power of 4
//!   (4, 16, 64, 256, 1024); non-power-of-4 powers of two fall back to
//!   scipy's regular Sylvester Hadamard — NOT ported (our group sizes are
//!   the power-of-4 ladder; a fallback entry point is rejected at config
//!   time, mirroring the quantui UI which only offers 64/256/1024).
//! - `rotate_weight`: `W_grouped @ H^T` per group of `group_size` columns.
//! - `rotate_activation`: `X_grouped @ H` per group (same math, H
//!   symmetric so the transpose is a layout detail only).
//!
//! FLOAT SEMANTICS (parity contract, probe
//! `tools/probe_convrot_gs256.py` + `tools/probe_convrot_chain.py`):
//! torch CPU eager matmul over these shapes is reproduced bit-exactly by
//! the oneDNN-style K-chunk-128 accumulation used for the bias-correction
//! GEMM (`bias_correction.rs`, validated vs torch 2.13.0+cpu at S=3072 for
//! K in {64, 128, 256}): per 128-element K-chunk a correctly-rounded FMA
//! chain (product exact in f64, added to the running f32 partial, rounded
//! once), chunk partials summed left-to-right as f32. H entries are
//! ±1/sqrt(gs) — products are exact in f64 — so only the addition order
//! matters, and kchunk128 matched with ZERO mismatches at gs=256.
//!
//! The bias-correction chain for ConvRot differs from the generic path
//! (learned_rounding.py Phase 4): it runs TWO separate GEMMs on the
//! ROTATED operands — `Y_ref = X_rot @ W_rot.T`, `Y_qnt = X_rot @ W_dq.T`
//! — subtracts, and ADDS the mean to the bias (`b + mean(Y_ref - Y_qnt)`),
//! where the generic streaming path subtracts `mean(X @ err.T)` from b.

use crate::quant::{dequantize_int8, quantize_int8_weight, ScalingMode};

/// Errors from the ConvRot rotation.
#[derive(Debug, thiserror::Error)]
pub enum ConvRotError {
    /// Group size not on the power-of-4 ladder (convrot.py:35-36 accepts any
    /// power of two via the scipy Sylvester fallback; we port the
    /// Theorem-3.3 Kronecker path only, like quantui's UI choices).
    #[error(
        "ConvRot group size must be a power of 4 (4, 16, 64, 256, 1024), got {0} \
         (the scipy Sylvester fallback for other powers of two is not ported)"
    )]
    NotPowerOf4(u32),
    /// in_features not divisible by the group size (the caller should have
    /// checked and skipped rotation for this tensor).
    #[error("in_features {n} not divisible by ConvRot group size {gs} — tensor stays unrotated")]
    Indivisible { n: usize, gs: u32 },
}

/// Build the normalized regular orthogonal Hadamard matrix of `size`
/// (power-of-4 ladder), row-major `(size, size)` f32.
///
/// Port of `build_hadamard` (convrot.py:18-64): H4 from Theorem 3.3,
/// Kronecker construction `H_{4^{k+1}} = H_{4^k} ⊗ H_4`, normalized by
/// `1/sqrt(size)`. Returned row-major: `H[r*gs + c]`.
pub fn build_hadamard(size: u32) -> Result<Vec<f32>, ConvRotError> {
    if size < 4 || (size & (size - 1)) != 0 {
        return Err(ConvRotError::NotPowerOf4(size));
    }
    // Power of 4 check: size == 4^k. (size & 3) == 0 for every 4^k, and a
    // power of two that is not a power of 4 has (size & 3) == 0 too — no:
    // 2^(2k) vs 2^(2k+1); 2^(2k+1) has bit 0 set. Correct test: a power of
    // two is a power of 4 iff bit-count of the exponent is even, i.e.
    // ((size.trailing_zeros() & 1) == 0).
    if size.trailing_zeros() & 1 != 0 {
        return Err(ConvRotError::NotPowerOf4(size));
    }

    const H4: [[f32; 4]; 4] = [
        [1.0, 1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, 1.0],
    ];

    // Kronecker build in f32 values, then normalize. The reference builds
    // the integer matrix in the input dtype (f32 here) and divides by
    // size**0.5 once at the end — values are exactly ±1/sqrt(gs) after
    // rounding, identical to dividing H4 by sqrt(4)=2 first and building
    // Kronecker products of that (products of dyadic rationals are exact).
    // We keep the reference order: build ±1 entries, divide once.
    let mut cur: Vec<f32> = H4.iter().flatten().copied().collect();
    let mut cur_size: usize = 4;
    while cur_size < size as usize {
        // kron(cur, H4): rows of blocks cur[r1][c1] * H4
        let next_size = cur_size * 4;
        let mut next = vec![0.0f32; next_size * next_size];
        for r1 in 0..cur_size {
            for c1 in 0..cur_size {
                let v = cur[r1 * cur_size + c1];
                for r2 in 0..4 {
                    for c2 in 0..4 {
                        next[(r1 * 4 + r2) * next_size + (c1 * 4 + c2)] = v * H4[r2][c2];
                    }
                }
            }
        }
        cur = next;
        cur_size = next_size;
    }
    // Normalize once (reference: H / (size**0.5) in f32).
    let inv = (size as f32).sqrt();
    for v in cur.iter_mut() {
        *v /= inv;
    }
    Ok(cur)
}

/// oneDNN-style K-chunk-128 dot: `sum_k a[k]*b[k]`.
///
/// Per 128-element chunk: correctly-rounded FMA chain (product exact in
/// f64, added to the running f32 partial, rounded once back to f32); chunk
/// partials summed left-to-right in f32 (f64 add of two f32s, rounded
/// once). Bit-exact vs torch CPU eager matmul for the shapes ConvRot uses
/// (probes `probe_convrot_gs256.py`, `probe_convrot_chain.py`; the same
/// scheme is validated for the bias GEMM at S=3072).
fn dot_kchunk128(a: &[f32], b: &[f32]) -> f32 {
    let k = a.len();
    let mut acc = 0.0f32;
    let mut first = true;
    let mut j0 = 0usize;
    while j0 < k {
        let j1 = (j0 + 128).min(k);
        let mut part = 0.0f32;
        for j in j0..j1 {
            part = ((a[j] as f64) * (b[j] as f64) + (part as f64)) as f32;
        }
        acc = if first {
            part
        } else {
            ((acc as f64) + (part as f64)) as f32
        };
        first = false;
        j0 = j1;
    }
    acc
}

/// Rotate a weight matrix offline: `W_rot = W_grouped @ H^T` per
/// `group_size`-column group (convrot.py `rotate_weight`, :66-94).
///
/// `w`: `(rows, n)` row-major f32; `h`: `(gs, gs)` row-major (from
/// [`build_hadamard`]). H is symmetric, so `H^T` values equal H — but the
/// reference multiplies by the TRANSPOSED matrix, so index it transposed
/// (`h[c*gs + r]`) to keep the byte-parity structure obvious.
pub fn rotate_weight(
    w: &[f32],
    h: &[f32],
    rows: usize,
    n: usize,
    gs: u32,
) -> Result<Vec<f32>, ConvRotError> {
    let gs = gs as usize;
    if n % gs != 0 {
        return Err(ConvRotError::Indivisible { n, gs: gs as u32 });
    }
    let n_groups = n / gs;
    let mut out = vec![0.0f32; rows * n];
    // (out[r, g*gs + j]) = sum_k W[r, g*gs + k] * H^T[k, j] = sum_k W[..] * H[j, k]
    for r in 0..rows {
        for g in 0..n_groups {
            let base = g * gs;
            let wgrp = &w[r * n + base..r * n + base + gs];
            for j in 0..gs {
                // H^T[k, j] == H[j, k] == h[j*gs + k]
                let h_row = &h[j * gs..j * gs + gs];
                out[r * n + base + j] = dot_kchunk128(wgrp, h_row);
            }
        }
    }
    Ok(out)
}

/// Rotate activation data online: `x_rot = x_grouped @ H` per group
/// (convrot.py `rotate_activation`, :97-126). Same math as
/// [`rotate_weight`] — H symmetric — kept as a separate entry point for
/// documentation parity with the reference module.
pub fn rotate_activation(
    x: &[f32],
    h: &[f32],
    rows: usize,
    n: usize,
    gs: u32,
) -> Result<Vec<f32>, ConvRotError> {
    // x_grouped @ H: out[.., g*gs + j] = sum_k x[.., g*gs + k] * H[k, j]
    let gs = gs as usize;
    if n % gs != 0 {
        return Err(ConvRotError::Indivisible { n, gs: gs as u32 });
    }
    let n_groups = n / gs;
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        for g in 0..n_groups {
            let base = g * gs;
            let xgrp = &x[r * n + base..r * n + base + gs];
            for j in 0..gs {
                // H[k, j] == h[k*gs + j] — column j of H
                let col: Vec<f32> = (0..gs).map(|k| h[k * gs + j]).collect();
                out[r * n + base + j] = dot_kchunk128(xgrp, &col);
            }
        }
    }
    Ok(out)
}

/// Result of the full ConvRot INT8 row-wise quantize of one weight.
pub struct ConvRotQuantized {
    /// Rotated+quantized INT8 payload (row-major, m×n).
    pub qdata: Vec<i8>,
    /// Row-wise dequant scale `[m, 1]` (f32).
    pub scale: Vec<f32>,
    /// Shape the streaming emitter must write for `weight_scale`. Taken
    /// verbatim from the quantizer so the caller never re-derives (and risk
    /// disagreeing with) the kernel's own shape rule.
    pub scale_shape: Vec<u64>,
    /// Dequantized ROTATED weight (f32, m×n) — `q.to(f32) * scale` row
    /// broadcast, exactly `TensorWiseINT8Layout::dequantize`.
    pub w_dq_rot: Vec<f32>,
    /// The ROTATED (unquantized) weight (f32, m×n) — `W_rot = W_grouped @ H^T`.
    /// Needed by the ConvRot bias correction, which runs BOTH GEMMs on the
    /// rotated operands (`Y_ref = X_rot @ W_rot.T`). Returned here so the
    /// caller never rotates twice (the rotation is the expensive step).
    pub w_rot: Vec<f32>,
}

/// The full per-tensor ConvRot pipeline in ONE call (learned_rounding.py
/// `_convert_int8_tensorwise` simple mode, :863-940):
/// 1. rotate W by H (`W_rot = W_grouped @ H^T`);
/// 2. row-wise INT8 quantize (`TensorWiseINT8Layout.quantize`,
///    is_weight=true: `row_max = abs().amax(dim=1)`,
///    `quant_scale = 127/max(1e-12, row_max)`, clamp ±127, round-ties-even);
/// 3. dequant (`q.to(f32) * scale`) — the ROTATED dequantized weight.
///
/// Both rotated operands are returned (`w_rot`, `w_dq_rot`) because the
/// ConvRot bias correction needs `W_rot` for its reference GEMM — rotating
/// twice would double the dominant cost and risk two non-bit-identical
/// rotations drifting apart.
///
/// The quantize/dequant reuse `quant.rs`/`dequantize_int8` directly (the
/// INT8 row-wise kernel is already a byte-parity port of exactly these
/// lines), so only the rotation is new numerics here.
pub fn convrot_int8_weight(
    w: &[f32],
    m: usize,
    n: usize,
    gs: u32,
) -> Result<ConvRotQuantized, ConvRotError> {
    let h = build_hadamard(gs)?;
    let w_rot = rotate_weight(w, &h, m, n, gs)?;
    let r = quantize_int8_weight(&w_rot, m, n, ScalingMode::Row, 128);
    let w_dq_rot = dequantize_int8(&r.qdata, &r.scale, m, n, ScalingMode::Row, 128);
    Ok(ConvRotQuantized {
        qdata: r.qdata,
        scale: r.scale,
        scale_shape: r.scale_shape,
        w_dq_rot,
        w_rot,
    })
}

/// ConvRot bias correction (learned_rounding.py Phase 4, :930-939).
///
/// `b_new = b + mean(Y_ref - Y_qnt)` where `Y_ref = X_rot @ W_rot.T` and
/// `Y_qnt = X_rot @ W_dq.T` — TWO separate GEMMs on the ROTATED operands
/// (NOT the generic `b - mean(X @ err.T)`). The subtraction is plain f32;
/// the mean is the cascade `sum(dim=0)` then f32 `/S` (same emulation as
/// `bias_correction.rs`); the final update ADDS (the convrot path corrects
/// toward the reference outputs, the generic path subtracts the error).
///
/// Returns `None` on cooperative cancellation (mirrors `correct_bias`).
// Nine operand/dim parameters mirror `bias_correction::correct_bias`'s
// signature style; a parameter object would just move the same data.
#[allow(clippy::too_many_arguments)]
pub fn correct_bias_convrot(
    x_rot: &[f32],
    w_rot: &[f32],
    w_dq_rot: &[f32],
    bias: &[f32],
    m: usize,
    n: usize,
    s_count: usize,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Option<Vec<f32>> {
    use rayon::prelude::*;

    debug_assert_eq!(x_rot.len(), s_count * n);
    debug_assert_eq!(w_rot.len(), m * n);
    debug_assert_eq!(w_dq_rot.len(), m * n);
    debug_assert_eq!(bias.len(), m);

    const GEMM_K_CHUNK: usize = 128;

    // One pass per (s, i): the two GEMMs Y_ref and Y_qnt share the same
    // operands layout; each element's K-chain is independent, so fusing
    // them per (s, i) does not change any accumulated bit. Parallel over
    // calibration rows (bit-safe: each (s, i) owns its own chains).
    #[derive(Clone, Copy)]
    struct YPair {
        y_ref: f32,
        y_qnt: f32,
    }
    let y: Vec<YPair> = {
        let mut rows: Vec<Vec<YPair>> = vec![Vec::new(); s_count];
        let cancelled = rows
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(s, row)| {
                if let Some(flag) = cancel {
                    if flag.load(std::sync::atomic::Ordering::Relaxed) {
                        return Err(());
                    }
                }
                let xr = &x_rot[s * n..(s + 1) * n];
                row.reserve(m);
                for i in 0..m {
                    let wr = &w_rot[i * n..(i + 1) * n];
                    let wd = &w_dq_rot[i * n..(i + 1) * n];
                    let mut yr = 0.0f32;
                    let mut yq = 0.0f32;
                    let mut first = true;
                    let mut j0 = 0usize;
                    while j0 < n {
                        let j1 = (j0 + GEMM_K_CHUNK).min(n);
                        let mut pr = 0.0f32;
                        let mut pq = 0.0f32;
                        for j in j0..j1 {
                            let xj = xr[j] as f64;
                            pr = (xj * (wr[j] as f64) + (pr as f64)) as f32;
                            pq = (xj * (wd[j] as f64) + (pq as f64)) as f32;
                        }
                        if first {
                            yr = pr;
                            yq = pq;
                        } else {
                            yr = ((yr as f64) + (pr as f64)) as f32;
                            yq = ((yq as f64) + (pq as f64)) as f32;
                        }
                        first = false;
                        j0 = j1;
                    }
                    row.push(YPair {
                        y_ref: yr,
                        y_qnt: yq,
                    });
                }
                Ok(())
            })
            .is_err();
        if cancelled {
            return None;
        }
        rows.into_iter().flatten().collect()
    };

    // diff = Y_ref - Y_qnt elementwise f32, then mean(dim=0) via the SAME
    // cascade dispatch as bias_correction.rs::emulate_sum_dim0 (columns of
    // the (S, m) matrix indexed by i in 0..m; group 32 when m >= 8 else 4;
    // plain cascade for the first plain_count columns, ilp strided
    // partials for the tail), then mean = cascade / S, then b + adj.
    let mut sums = vec![0.0f32; m];
    let group = if m >= 8 { 32 } else { 4 };
    let plain_count = (m / group) * group;
    sums.par_iter_mut().enumerate().for_each(|(i, slot)| {
        let diff_col = |s: usize| (y[s * m + i].y_ref - y[s * m + i].y_qnt) as f32;
        *slot = if i < plain_count {
            crate::bias_correction::plain_col_sum(&(0..s_count).map(diff_col).collect::<Vec<_>>())
        } else {
            crate::bias_correction::ilp_col_sum(&(0..s_count).map(diff_col).collect::<Vec<_>>())
        };
    });

    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let adj = sums[i] / s_count as f32;
        out.push(bias[i] + adj);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hadamard_is_orthogonal_and_power_of_4_only() {
        for gs in [4u32, 16, 64, 256] {
            let h = build_hadamard(gs).unwrap();
            let n = gs as usize;
            // H·H^T = I (normalized): dot of row r with itself == 1, with
            // row r' != r == 0. All in f64 to avoid f32 accumulation noise.
            for r in 0..n {
                for r2 in 0..n {
                    let mut dot = 0.0f64;
                    for c in 0..n {
                        dot += (h[r * n + c] as f64) * (h[r2 * n + c] as f64);
                    }
                    let want = if r == r2 { 1.0 } else { 0.0 };
                    assert!(
                        (dot - want).abs() < 1e-5,
                        "gs={gs} H·Hᵀ[{r}][{r2}] = {dot}, want {want}"
                    );
                }
            }
        }
        // H4 base (Theorem 3.3 note: every row and column of the
        // UNNORMALIZED matrix sums to exactly 2; ours is normalized by
        // sqrt(4)=2 so rows sum to 1.0 — verify both).
        let h4 = build_hadamard(4).unwrap();
        for r in 0..4 {
            let s_norm: f32 = h4[r * 4..r * 4 + 4].iter().sum();
            assert_eq!(s_norm, 1.0f32, "normalized H4 row {r} must sum to 1");
            let s_raw: f32 = h4[r * 4..r * 4 + 4].iter().map(|v| v * 2.0).sum();
            assert_eq!(s_raw, 2.0f32, "raw H4 row {r} must sum to exactly 2");
        }
        // Non-power-of-4 and non-power-of-2 are rejected.
        assert!(matches!(
            build_hadamard(8),
            Err(ConvRotError::NotPowerOf4(8))
        ));
        assert!(matches!(
            build_hadamard(100),
            Err(ConvRotError::NotPowerOf4(100))
        ));
        assert!(matches!(
            build_hadamard(512),
            Err(ConvRotError::NotPowerOf4(512))
        ));
        assert!(matches!(
            build_hadamard(2),
            Err(ConvRotError::NotPowerOf4(2))
        ));
    }

    #[test]
    fn hadamard_matches_reference_values() {
        // The reference H4 (Theorem 3.3, Eq 9) — check the base block and
        // the 16×16 Kronecker corner against hand-computed values.
        let h4 = build_hadamard(4).unwrap();
        let want4 = [
            [1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0],
        ];
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(h4[r * 4 + c], want4[r][c] / 2.0, "H4[{r}][{c}]");
            }
        }
        // kron(H4, H4)[0][0] = H4[0][0]*H4[0][0] = 1 → /4
        let h16 = build_hadamard(16).unwrap();
        assert_eq!(h16[0], 1.0 / 4.0);
        // kron(H4, H4)[0][5]: block (0,1) → H4[0][1]*H4[0][1] = 1 → /4
        assert_eq!(h16[5], 1.0 / 4.0);
        // kron corner [0][15]: H4[0][3]*H4[3][3] = (-1)*(-1) = 1 → /4
        assert_eq!(h16[15], 1.0 / 4.0);
        // [0][3]: H4[0][3]*H4[3][3]... no: [0][3] = block(0,0) H4[0][?]:
        // row 0 col 3 inside block (0,0): H4[0][3] = -1 → -1/4
        assert_eq!(h16[3], -1.0 / 4.0);
    }

    #[test]
    fn rotation_is_involution_on_dyadic_data() {
        // Rotating twice by the same H gives back the original (H is
        // symmetric orthogonal, so H·H = I). For gs=4 the matrix entries
        // are exactly ±0.5 (dyadic): products of dyadic data with ±0.5
        // stay dyadic, and 4-element dyadic sums within f32's 24-bit
        // mantissa are EXACT — so with small-integer data the double
        // rotation is bit-exactly the identity. (For arbitrary floats the
        // mid-chain sums can round, so this test pins dyadic data only.)
        let gs = 4u32;
        let n = 8usize; // two groups
        let rows = 3usize;
        let w: Vec<f32> = (0..rows * n).map(|i| ((i % 7) as f32) - 3.0).collect();
        let h = build_hadamard(gs).unwrap();
        let once = rotate_weight(&w, &h, rows, n, gs).unwrap();
        let twice = rotate_weight(&once, &h, rows, n, gs).unwrap();
        for (a, b) in w.iter().zip(twice.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "gs=4 double rotation on dyadic data must be exact identity"
            );
        }
    }

    #[test]
    fn rotate_activation_equals_rotate_weight_transposed_semantics() {
        // H is symmetric → rotating activations (@ H) equals rotating with
        // H^T (the weight path). Same numbers must give identical bytes.
        let gs = 64u32;
        let n = 128usize;
        let rows = 2usize;
        let x: Vec<f32> = (0..rows * n).map(|i| (i as f32 * 0.11).cos()).collect();
        let h = build_hadamard(gs).unwrap();
        let a = rotate_activation(&x, &h, rows, n, gs).unwrap();
        let b = rotate_weight(&x, &h, rows, n, gs).unwrap();
        for (p, q) in a.iter().zip(b.iter()) {
            assert_eq!(p.to_bits(), q.to_bits());
        }
    }

    #[test]
    fn indivisible_in_features_errors() {
        let h = build_hadamard(64).unwrap();
        let w = vec![0.0f32; 4 * 100];
        assert!(matches!(
            rotate_weight(&w, &h, 4, 100, 64),
            Err(ConvRotError::Indivisible { n: 100, gs: 64 })
        ));
    }

    /// The pipeline returns BOTH rotated operands (`w_rot` and the rotated
    /// dequant) so the caller never rotates twice: `w_rot` must be bit-exactly
    /// `rotate_weight(w, H, ...)` and `w_dq_rot` the dequant of the payload.
    #[test]
    fn pipeline_returns_rotated_weight_and_rotated_dequant() {
        let gs = 64u32;
        let (m, n) = (3usize, 128usize);
        let w: Vec<f32> = (0..m * n)
            .map(|i| ((i % 37) as f32) * 0.017 - 0.3)
            .collect();
        let h = build_hadamard(gs).unwrap();
        let expected_rot = rotate_weight(&w, &h, m, n, gs).unwrap();

        let r = convrot_int8_weight(&w, m, n, gs).unwrap();
        assert_eq!(r.w_rot.len(), m * n);
        for (a, b) in r.w_rot.iter().zip(expected_rot.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "w_rot must be exactly W@H^T");
        }
        // Row-wise scale shape [m, 1] comes straight from the quantizer.
        assert_eq!(r.scale_shape, vec![m as u64, 1]);
        assert_eq!(r.scale.len(), m);
        // w_dq_rot == q * scale (row broadcast) — the rotated dequant.
        for i in 0..m {
            for j in 0..n {
                let want = (r.qdata[i * n + j] as f32) * r.scale[i];
                assert_eq!(
                    r.w_dq_rot[i * n + j].to_bits(),
                    want.to_bits(),
                    "w_dq_rot[{i}][{j}] must be q*scale"
                );
            }
        }
    }

    #[test]
    fn determinism_same_input_same_bytes() {
        let gs = 256u32;
        let m = 4usize;
        let n = 256usize;
        let w: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.031).sin()).collect();
        let a = convrot_int8_weight(&w, m, n, gs).unwrap();
        let b = convrot_int8_weight(&w, m, n, gs).unwrap();
        assert_eq!(a.qdata, b.qdata);
        assert!(a
            .scale
            .iter()
            .zip(b.scale.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}

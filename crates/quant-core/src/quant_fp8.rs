//! FP8 E4M3 (float8_e4m3fn) quantization kernels — byte-parity port of the
//! convert_to_quant *simple* (no learned rounding) FP8 paths.
//!
//! Sources ported (all in `convert_to_quant/converters/learned_rounding.py`,
//! `no_learned_rounding=True` branches):
//!
//! - `_convert_fp8` (tensor mode): global amax → `quant_scale = 448/amax.clamp_min(1e-12)`
//! - `_convert_fp8_rowwise`: per-row amax, same formula, dequant shape `(M,)`
//! - `_convert_fp8_block2d`: per `(bs×bs)` tile amax, dequant shape `(M/bs, N/bs)`;
//!   falls back to row-wise when a dim is not divisible by `bs`
//!
//! Parity-critical details (all f32, `COMPUTE_DTYPE = float32`):
//!
//! - The **stored** scale is the *dequant* scale `1.0 / quant_scale` — a real
//!   reciprocal division, NOT `amax/448`. Division order matters for bit-exactness.
//! - `quant_scale = FP8_MAX / amax.clamp_min(1e-12)` with `FP8_MAX = 448.0`.
//! - `W_scaled = W * quant_scale` (elementwise f32), then `.clamp(-448, 448)`,
//!   then `.to(float8_e4m3fn)` (the E4M3 cast in `dtype.rs`, bit-exact vs torch).
//!   The explicit clamp is redundant with the cast's saturation (any |x|>448
//!   already maps to ±448) but is kept for fidelity to the reference.
//! - All-zeros tensor early return: qdata zeros, dequant scale ones (per-mode shape).
//!
//! Max-reductions (`abs().amax`) are order-independent, so no accumulation hazard.

use rayon::prelude::*;

use crate::dtype::{f32_to_fp8_e4m3_bits, fp8_e4m3_bits_to_f32};

/// FP8 max finite value (torch.finfo(float8_e4m3fn).max).
pub const FP8_MAX: f32 = 448.0;
/// clamp_min floor used by the reference before dividing.
const CLAMP_MIN: f32 = 1e-12;

/// `quant_scale = FP8_MAX / amax.clamp_min(1e-12)` — but computed the way torch
/// does scalar/tensor division: `scalar * reciprocal(tensor)`. torch's
/// `reciprocal` is correctly-rounded (== IEEE `1/x`), so this is
/// `FP8_MAX * (1.0 / amax_clamped)`, NOT `FP8_MAX / amax_clamped` (the two can
/// differ by 1 ulp; the golden was produced with the reciprocal-multiply form).
#[inline]
fn quant_scale_from_amax(amax: f32) -> f32 {
    FP8_MAX * (1.0 / amax.max(CLAMP_MIN))
}

/// Output of one FP8 quantization: E4M3 bytes + dequant scales.
#[derive(Debug, Clone)]
pub struct Fp8QuantResult {
    /// E4M3 (float8_e4m3fn) bytes in row-major order, same element count as input.
    pub qdata: Vec<u8>,
    /// Dequant scales (= `1.0 / quant_scale`). Shape depends on mode:
    /// - Tensor: `[]` (single scalar)
    /// - Row: `[M]`
    /// - Block: `[M/bs, N/bs]`
    pub scale: Vec<f32>,
    pub scale_shape: Vec<u64>,
}

/// Quantize a 2D weight tensor (row-major f32) to FP8 E4M3.
///
/// `w` must contain exactly `m * n` elements. `mode` selects tensor/row/block
/// scaling; `block_size` is only used for `Block`. Mirrors the reference
/// simple-mode routing including the block→row fallback for indivisible dims
/// and the all-zeros early return.
pub fn quantize_fp8_weight(
    w: &[f32],
    m: usize,
    n: usize,
    mode: Fp8ScalingMode,
    block_size: usize,
) -> Fp8QuantResult {
    debug_assert_eq!(w.len(), m * n);

    // All-zeros early return (learned_rounding.py convert()): qdata zeros,
    // dequant scale ones with the per-mode shape. NOTE the reference quirk:
    // block mode with non-divisible dims falls through to the tensor-wise
    // `torch.ones(1)` branch here (NOT row-wise, unlike the nonzero path).
    if w.iter().all(|&v| v == 0.0) {
        let block_divisible = m % block_size == 0 && n % block_size == 0;
        let (scale_shape, scale_len) = match mode {
            Fp8ScalingMode::Tensor => (vec![], 1),
            Fp8ScalingMode::Row => (vec![m as u64], m),
            Fp8ScalingMode::Block if block_divisible => (
                vec![(m / block_size) as u64, (n / block_size) as u64],
                (m / block_size) * (n / block_size),
            ),
            Fp8ScalingMode::Block => (vec![1], 1), // tensor-wise ones(1) quirk
        };
        return Fp8QuantResult {
            qdata: vec![0u8; m * n],
            scale: vec![1.0; scale_len],
            scale_shape,
        };
    }

    match mode {
        Fp8ScalingMode::Tensor => {
            let w_max = w.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let quant_scale = quant_scale_from_amax(w_max);
            let qdata = scale_clamp_cast(w, quant_scale);
            Fp8QuantResult {
                qdata,
                scale: vec![1.0 / quant_scale],
                scale_shape: vec![],
            }
        }
        Fp8ScalingMode::Row => quantize_fp8_row(w, m, n),
        Fp8ScalingMode::Block => {
            if m % block_size == 0 && n % block_size == 0 {
                quantize_fp8_block2d(w, m, n, block_size)
            } else {
                // Reference falls back to row-wise when dims aren't divisible.
                quantize_fp8_row(w, m, n)
            }
        }
    }
}

/// FP8 scaling mode, mirroring `QuantConfig.scaling_mode` for the FP8 paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8ScalingMode {
    Tensor,
    Row,
    Block,
}

// --- shared helpers -------------------------------------------------------- //

/// `W_scaled = w * quant_scale; clamp(±FP8_MAX); to(e4m3)` — elementwise.
#[inline]
fn scale_clamp_cast(w: &[f32], quant_scale: f32) -> Vec<u8> {
    w.par_iter()
        .map(|&v| {
            let scaled = v * quant_scale;
            let clamped = scaled.clamp(-FP8_MAX, FP8_MAX);
            f32_to_fp8_e4m3_bits(clamped)
        })
        .collect()
}

/// Row-wise: one scale per row, dequant shape `(M,)`.
fn quantize_fp8_row(w: &[f32], m: usize, n: usize) -> Fp8QuantResult {
    let rows: Vec<&[f32]> = w.chunks_exact(n).collect();
    let out: Vec<(f32, Vec<u8>)> = rows
        .par_iter()
        .map(|row| {
            let row_max = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let quant_scale = quant_scale_from_amax(row_max);
            (1.0 / quant_scale, scale_clamp_cast(row, quant_scale))
        })
        .collect();
    let mut scale = Vec::with_capacity(m);
    let mut qdata = Vec::with_capacity(m * n);
    for (s, q) in out {
        scale.push(s);
        qdata.extend(q);
    }
    Fp8QuantResult {
        qdata,
        scale,
        scale_shape: vec![m as u64],
    }
}

/// 2D block-wise: one scale per `(bs×bs)` tile, dequant shape `(M/bs, N/bs)`.
fn quantize_fp8_block2d(w: &[f32], m: usize, n: usize, bs: usize) -> Fp8QuantResult {
    let bm = m / bs;
    let bn = n / bs;

    // Per-tile amax over the (bs, bs) block.
    let mut quant_scale = vec![0.0f32; bm * bn];
    for bi in 0..bm {
        for bj in 0..bn {
            let mut amax = 0.0f32;
            for i in bi * bs..(bi + 1) * bs {
                let row = &w[i * n..(i + 1) * n];
                for &v in &row[bj * bs..(bj + 1) * bs] {
                    amax = amax.max(v.abs());
                }
            }
            quant_scale[bi * bn + bj] = quant_scale_from_amax(amax);
        }
    }

    // Scale/clamp/cast per tile, writing into row-major output. Parallel over
    // block-rows (order-preserving).
    let q_chunks: Vec<Vec<u8>> = (0..bm)
        .into_par_iter()
        .map(|bi| {
            let mut out = Vec::with_capacity(bs * n);
            for i in bi * bs..(bi + 1) * bs {
                let row = &w[i * n..(i + 1) * n];
                for bj in 0..bn {
                    let qs = quant_scale[bi * bn + bj];
                    let tile = &row[bj * bs..(bj + 1) * bs];
                    out.extend(tile.iter().map(|&v| {
                        let scaled = v * qs;
                        f32_to_fp8_e4m3_bits(scaled.clamp(-FP8_MAX, FP8_MAX))
                    }));
                }
            }
            out
        })
        .collect();
    let mut qdata = Vec::with_capacity(m * n);
    for chunk in q_chunks {
        qdata.extend(chunk);
    }

    let scale: Vec<f32> = quant_scale.iter().map(|&qs| 1.0 / qs).collect();
    Fp8QuantResult {
        qdata,
        scale,
        scale_shape: vec![bm as u64, bn as u64],
    }
}

/// Dequantize an FP8 result back to f32 for bias correction — bit-exact
/// port of the reference dequantizations that each `_convert_fp8*` returns
/// alongside `(qdata, scale)` (`learned_rounding.py`).
///
/// Parity-critical asymmetry (verified line-by-line against the reference):
///
/// - Tensor (`learned_rounding.py:1938`) and Row (`:1998`):
///   `W_f8.to(f32) / quant_scale` — **TRUE IEEE division** by the ORIGINAL
///   quant scale (the same value that was used to quantize). NOT
///   `× (1/quant_scale)`: `1/(1/qs) != qs` for ~13.8% of f32 values (1 ulp),
///   so dividing by a recovered reciprocal differs from the reference at tie
///   points. `quant_scale` is therefore recomputed from `w_orig` with the
///   exact formula used at quantize time (`quant_scale_from_amax`), which is
///   bit-identical to the reference's in-scope `scale` tensor.
/// - Block (`:2069`): `W_f8.to(f32) * dequant_scale` — **multiply** by the
///   stored dequant scale per tile (the reference builds `dequant_broadcast`
///   from `1.0 / quant_scale` and multiplies).
///
/// `w_orig` must be the same `m * n` f32 slice that was quantized. `mode`
/// and `block_size` must match the quantize call. Block mode with
/// non-divisible dims fell back to row-wise at quantize time (mirrored here,
/// matching `_convert_fp8_block2d`'s fallback at `:2044-2046`).
pub fn dequantize_fp8(
    w_orig: &[f32],
    r: &Fp8QuantResult,
    m: usize,
    n: usize,
    mode: Fp8ScalingMode,
    block_size: usize,
) -> Vec<f32> {
    debug_assert_eq!(w_orig.len(), m * n);
    debug_assert_eq!(r.qdata.len(), m * n);

    match mode {
        Fp8ScalingMode::Tensor => {
            let w_max = w_orig.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let quant_scale = quant_scale_from_amax(w_max);
            r.qdata
                .iter()
                .map(|&b| fp8_e4m3_bits_to_f32(b) / quant_scale)
                .collect()
        }
        Fp8ScalingMode::Row => dequantize_fp8_row(w_orig, r, m, n),
        Fp8ScalingMode::Block => {
            if m % block_size == 0 && n % block_size == 0 {
                dequantize_fp8_block2d(r, m, n, block_size)
            } else {
                // Quantize fell back to row-wise; dequant must too.
                dequantize_fp8_row(w_orig, r, m, n)
            }
        }
    }
}

/// Row-wise dequant: TRUE IEEE division by the per-row quant scale
/// recomputed from `w_orig` (reference `:1998`).
fn dequantize_fp8_row(w_orig: &[f32], r: &Fp8QuantResult, m: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        let row = &w_orig[i * n..(i + 1) * n];
        let row_max = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let quant_scale = quant_scale_from_amax(row_max);
        for &b in &r.qdata[i * n..(i + 1) * n] {
            out.push(fp8_e4m3_bits_to_f32(b) / quant_scale);
        }
    }
    out
}

/// Block-wise dequant: MULTIPLY by the stored dequant scale per tile
/// (reference `:2069`). Position-independent elementwise multiply, so the
/// reference's blocked→permute→reshape round trip is a no-op here.
fn dequantize_fp8_block2d(r: &Fp8QuantResult, m: usize, n: usize, bs: usize) -> Vec<f32> {
    let bm = m / bs;
    let bn = n / bs;
    let mut out = vec![0.0f32; m * n];
    for bi in 0..bm {
        for bj in 0..bn {
            let ds = r.scale[bi * bn + bj];
            for i in bi * bs..(bi + 1) * bs {
                for j in bj * bs..(bj + 1) * bs {
                    let idx = i * n + j;
                    out[idx] = fp8_e4m3_bits_to_f32(r.qdata[idx]) * ds;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensorwise_hand_computed() {
        // [2,2]: 4, -2, 1, 0 → amax 4 → quant_scale = 448/4 = 112.
        let w = [4.0f32, -2.0, 1.0, 0.0];
        let r = quantize_fp8_weight(&w, 2, 2, Fp8ScalingMode::Tensor, 128);
        // 4*112 = 448 → 0x7E; -2*112 = -224 → E4M3(-224); 1*112 = 112.
        assert_eq!(r.qdata[0], 0x7E);
        assert_eq!(r.scale_shape, vec![] as Vec<u64>);
        assert_eq!(r.scale[0], 1.0 / 112.0);
    }

    #[test]
    fn tensorwise_zero_tensor_ones_scale() {
        let w = vec![0.0f32; 256 * 128];
        let r = quantize_fp8_weight(&w, 256, 128, Fp8ScalingMode::Tensor, 128);
        assert!(r.qdata.iter().all(|&v| v == 0));
        assert_eq!(r.scale, vec![1.0]);
        assert_eq!(r.scale_shape, vec![] as Vec<u64>);
    }

    #[test]
    fn rowwise_per_row_scales() {
        let w = [8.0f32, -4.0, 1.0, 2.0, 100.0, 0.5];
        let r = quantize_fp8_weight(&w, 3, 2, Fp8ScalingMode::Row, 128);
        let q0 = FP8_MAX / 8.0f32.max(CLAMP_MIN);
        assert_eq!(r.scale[0], 1.0 / q0);
        let q2 = FP8_MAX / 100.0f32.max(CLAMP_MIN);
        assert_eq!(r.scale[2], 1.0 / q2);
        assert_eq!(r.scale_shape, vec![3]);
        // 8 * (448/8) = 448 → 0x7E.
        assert_eq!(r.qdata[0], 0x7E);
    }

    #[test]
    fn block2d_scale_grid() {
        // [2,2] with bs=1: four independent tiles.
        let w = [2.0f32, 4.0, 8.0, 16.0];
        let r = quantize_fp8_weight(&w, 2, 2, Fp8ScalingMode::Block, 1);
        assert_eq!(r.scale_shape, vec![2, 2]);
        assert_eq!(r.scale[0], 1.0 / (FP8_MAX / 2.0));
        assert_eq!(r.scale[3], 1.0 / (FP8_MAX / 16.0));
        // Each tile's max element maps to 448 (0x7E).
        assert!(r.qdata.iter().all(|&b| b == 0x7E));
    }

    #[test]
    fn block_falls_back_to_row_when_indivisible() {
        // [3,2] with bs=2: 3 % 2 != 0 → row-wise fallback.
        let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = quantize_fp8_weight(&w, 3, 2, Fp8ScalingMode::Block, 2);
        assert_eq!(r.scale_shape, vec![3]); // row-wise shape
    }

    #[test]
    fn parallel_equals_sequential_block() {
        // Byte-for-byte equality between rayon-parallel block result and a
        // sequential recomputation.
        let (m, n, bs) = (256usize, 128usize, 128usize);
        let mut w = vec![0.0f32; m * n];
        let mut x: u64 = 0x243F6A8885A308D3;
        for v in &mut w {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *v = ((x % 20_000) as f32 / 10_000.0) - 1.0;
        }
        let par = quantize_fp8_weight(&w, m, n, Fp8ScalingMode::Block, bs);

        let bm = m / bs;
        let bn = n / bs;
        let mut seq_q = vec![0u8; m * n];
        let mut seq_s = vec![0.0f32; bm * bn];
        for bi in 0..bm {
            for bj in 0..bn {
                let mut amax = 0.0f32;
                for i in bi * bs..(bi + 1) * bs {
                    for j in bj * bs..(bj + 1) * bs {
                        amax = amax.max(w[i * n + j].abs());
                    }
                }
                let qs = quant_scale_from_amax(amax);
                seq_s[bi * bn + bj] = 1.0 / qs;
                for i in bi * bs..(bi + 1) * bs {
                    for j in bj * bs..(bj + 1) * bs {
                        let scaled = w[i * n + j] * qs;
                        seq_q[i * n + j] = f32_to_fp8_e4m3_bits(scaled.clamp(-FP8_MAX, FP8_MAX));
                    }
                }
            }
        }
        assert_eq!(par.qdata, seq_q);
        assert_eq!(par.scale, seq_s);
    }

    // --- dequantize_fp8 (Phase B.1) ---------------------------------------- //

    /// f32 bit pattern, for ulp-exact assertions.
    fn bits(x: f32) -> u32 {
        x.to_bits()
    }

    #[test]
    fn dequantize_fp8_tensor_row_is_division() {
        // Tie case where TRUE IEEE division differs from reciprocal-multiply
        // by 1 ulp (probe-verified: 1/(1/qs) != qs for ~13.8% of f32).
        //
        // amax = 0.015625 (2^-6) → quant_scale = 448 * (1/0.015625) = 28672.
        // w = [0.015625, 1.5/28672]:
        //   elem0: 0.015625 * 28672 = 448      → E4M3 0x7E
        //   elem1: (1.5/28672) * 28672 = 1.5   → E4M3 0x3C
        // Dequant (tensor): f32(code) / quant_scale (reference :1938):
        //   elem0: 448 / 28672 = 0.015625            (0x3C800000)
        //   elem1: 1.5 / 28672 = 5.231584873399697e-05 (0x385B6DB7)
        // Reciprocal-multiply would give elem1 = 0x385B6DB8 (1 ulp higher),
        // so asserting the exact bits proves the division path.
        let w = [0.015625f32, 1.5 / 28672.0];
        let r = quantize_fp8_weight(&w, 1, 2, Fp8ScalingMode::Tensor, 128);
        assert_eq!(r.qdata, [0x7E, 0x3C]);
        let dq = dequantize_fp8(&w, &r, 1, 2, Fp8ScalingMode::Tensor, 128);
        assert_eq!(bits(dq[0]), 0x3C80_0000, "elem0 = 448/28672 = 0.015625");
        assert_eq!(
            bits(dq[1]),
            0x385B_6DB7,
            "elem1 must be IEEE division, not ×(1/qs)"
        );
        assert_ne!(
            bits(dq[1]),
            0x385B_6DB8,
            "reciprocal-multiply would give this"
        );

        // Row mode (reference :1998): same division, per-row quant_scale.
        // Row 0 = tensor case above; row 1 amax = 0.03125 → qs = 14336.
        //   row1 elem0: 0.03125*14336 = 448 → 0x7E; dequant 448/14336 = 0.03125
        //   row1 elem1: (1.5/14336)*14336 = 1.5 → 0x3C;
        //               dequant 1.5/14336 = 1.0463169746799394e-4 (0x38DB6DB7),
        //               reciprocal-multiply would give 0x38DB6DB8.
        let w2 = [0.015625f32, 1.5 / 28672.0, 0.03125, 1.5 / 14336.0];
        let r2 = quantize_fp8_weight(&w2, 2, 2, Fp8ScalingMode::Row, 128);
        assert_eq!(r2.qdata, [0x7E, 0x3C, 0x7E, 0x3C]);
        let dq2 = dequantize_fp8(&w2, &r2, 2, 2, Fp8ScalingMode::Row, 128);
        assert_eq!(bits(dq2[0]), 0x3C80_0000);
        assert_eq!(bits(dq2[1]), 0x385B_6DB7);
        assert_eq!(bits(dq2[2]), 0x3D00_0000); // 0.03125
        assert_eq!(
            bits(dq2[3]),
            0x38DB_6DB7,
            "row1 elem1 must be IEEE division"
        );
    }

    #[test]
    fn dequantize_fp8_block_is_multiply() {
        // Block mode (reference :2069): f32(code) × stored dequant scale —
        // MULTIPLY, not division. One 2×2 tile (bs=2): amax = 0.015625 →
        // quant_scale = 28672, stored = 1/28672 = 3.487723370199092e-5.
        //
        //   elem0: 0.015625 × 28672 = 448 → 0x7E; dequant 448 × stored
        //          = 0.015625 exactly (0x3C800000; mul == div here).
        //   elem1: (1.5/28672) × 28672 = 1.5 → 0x3C; dequant
        //          1.5 × stored = 5.231585237197578e-5 (0x385B6DB8).
        //          TRUE IEEE division would give 0x385B6DB7 → asserting the
        //          multiply bits proves the block path is NOT division.
        //   elem2/3: zeros → 0x00 → +0.0.
        let w = [0.015625f32, 1.5 / 28672.0, 0.0, 0.0];
        let r = quantize_fp8_weight(&w, 2, 2, Fp8ScalingMode::Block, 2);
        assert_eq!(r.qdata, [0x7E, 0x3C, 0x00, 0x00]);
        assert_eq!(r.scale_shape, vec![1, 1]);
        let dq = dequantize_fp8(&w, &r, 2, 2, Fp8ScalingMode::Block, 2);
        assert_eq!(bits(dq[0]), 0x3C80_0000, "elem0 (tile max) = 0.015625");
        assert_eq!(
            bits(dq[1]),
            0x385B_6DB8,
            "elem1 must be multiply, not division"
        );
        assert_ne!(bits(dq[1]), 0x385B_6DB7, "division would give this");
        assert_eq!(dq[2], 0.0);
        assert_eq!(dq[3], 0.0);
    }

    #[test]
    fn dequantize_fp8_block_indivisible_falls_back_to_row() {
        // Block mode with non-divisible dims fell back to row-wise at
        // quantize time (scale_shape [M]); dequant must mirror the fallback.
        let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = quantize_fp8_weight(&w, 3, 2, Fp8ScalingMode::Block, 2);
        assert_eq!(r.scale_shape, vec![3]); // row-wise shape
        let dq = dequantize_fp8(&w, &r, 3, 2, Fp8ScalingMode::Block, 2);
        assert_eq!(dq.len(), 6);
        // Cross-check against an explicit row-mode dequant (same result).
        let dq_row = dequantize_fp8(&w, &r, 3, 2, Fp8ScalingMode::Row, 2);
        assert_eq!(dq, dq_row);
    }

    #[test]
    fn dequantize_fp8_all_zeros_ones_scale() {
        // All-zeros early return: qdata zeros, dequant scale ones. Dequant
        // divides 0 by the recomputed quant_scale (amax 0 → clamp_min floor)
        // → all +0.0.
        let w = vec![0.0f32; 4 * 4];
        let r = quantize_fp8_weight(&w, 4, 4, Fp8ScalingMode::Tensor, 128);
        assert!(r.qdata.iter().all(|&b| b == 0));
        assert_eq!(r.scale, vec![1.0]);
        let dq = dequantize_fp8(&w, &r, 4, 4, Fp8ScalingMode::Tensor, 128);
        assert!(dq.iter().all(|&v| v == 0.0 && v.is_sign_positive()));
    }

    #[test]
    fn dequant_roundtrip_bounded_fp8() {
        // Phase B.4 (FP8 arm): quantize a random matrix, dequantize, assert
        // finite + bounded (|dq| ≤ amax·(1+ε)) and that re-quantizing the
        // dequant reproduces the codes exactly (idempotent round trip).
        let (m, n) = (64usize, 48usize);
        let mut w = vec![0.0f32; m * n];
        let mut x: u64 = 0x243F6A8885A308D3;
        for v in &mut w {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *v = ((x % 20_000) as f32 / 10_000.0) - 1.0;
        }
        let amax = w.iter().fold(0.0f32, |a, &v| a.max(v.abs()));

        let cases: &[(Fp8ScalingMode, usize)] = &[
            (Fp8ScalingMode::Tensor, 128),
            (Fp8ScalingMode::Row, 128),
            (Fp8ScalingMode::Block, 16), // 64,48 divisible by 16
        ];
        for &(mode, bs) in cases {
            let r = quantize_fp8_weight(&w, m, n, mode, bs);
            let dq = dequantize_fp8(&w, &r, m, n, mode, bs);
            assert_eq!(dq.len(), m * n);
            for &v in &dq {
                assert!(v.is_finite(), "dequant produced non-finite value");
            }
            let dq_max = dq.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            assert!(
                dq_max <= amax * 1.01,
                "dequant amax {dq_max} exceeds bound {}",
                amax * 1.01
            );
            // Re-quantizing the dequant must reproduce the same codes.
            let r2 = quantize_fp8_weight(&dq, m, n, mode, bs);
            assert_eq!(r2.qdata, r.qdata, "round trip not idempotent for {mode:?}");
        }
    }
}

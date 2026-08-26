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

use crate::dtype::f32_to_fp8_e4m3_bits;

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
}

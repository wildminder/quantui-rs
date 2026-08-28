//! MXFP8 (Microscaling FP8) quantization kernel — byte-parity port of the
//! convert_to_quant *simple* (no learned rounding) MXFP8 path.
//!
//! Sources ported (simple path, matching `gen_golden_formats.py` `simple=True`):
//! `convert_to_quant/converters/mxfp8_converter.py` `MXFP8Converter.quantize`
//! (the F32→bf16 input cast), `comfy_kitchen/backends/eager/quantization.py`
//! `quantize_mxfp8` (the block-scale + quantize pipeline — the golden was
//! generated on CPU, so the eager backend is the authoritative reference), and
//! `convert_to_quant/utils/float_utils.py` (`e8m0_to_f32`, `to_blocked`).
//! The learned path (`learned_mxfp8.py`) adds an all-zeros early return
//! (`_quantize_zeros`), but the simple path has none, so this kernel handles
//! all-zeros through the normal pipeline.
//!
//! Pipeline (all f32 compute, `COMPUTE_DTYPE = float32`):
//!
//! 0. Round every input value through bf16 (`f32 → bf16 → f32`). The simple
//!    path casts the F32 weight to bf16 before the kernel ("CUDA kernel only
//!    supports FP16/BF16 input"), and the eager kernel upcasts bf16→f32 for
//!    compute; the net effect is quantization on bf16-grid values. Parity-
//!    critical: skipping it breaks byte-exactness for F32/F16 inputs
//!    (verified against all four golden cases).
//! 1. Pad to multiples of 32 (zero pad at the end of rows/cols).
//! 2. Per 32-element block (row-major within each row):
//!    - `block_max = amax(abs(block))` (max-reduction: order-independent)
//!    - `scale_needed = block_max / 448.0` — torch tensor/scalar division is
//!      TRUE IEEE division (verified 3M/3M in tools/probe_mxfp8_scale.py),
//!      unlike scalar/tensor which is reciprocal-multiply (see quant_fp8.rs)
//!    - clamp min `2^-127` (0x0040_0000, smallest f32 subnormal)
//!    - `exp_biased = ceil(log2(scale_needed)) + 127`, clamp [0, 254] → e8m0.
//!      `torch.log2` + `ceil` is emulated EXACTLY by
//!      `ceil(f32(f64::log2(x)))` (verified 2M/2M incl. ±32 ulp neighborhoods
//!      of every power of two in [-126, 20]; a bit-based
//!      "E + (mantissa != 0)" rule is WRONG in ~1.4% of cases because f32
//!      rounding of log2 just above 2^E can land exactly on E).
//!    - `scale_f32 = e8m0_to_f32(e8m0)` = `2^(e-127)` (0.0 if e == 0),
//!      replaced by 1.0 for zero blocks (`zero_mask = block_max == 0`)
//! 4. Quantize: `v / scale_f32` (tensor/tensor = TRUE IEEE division, verified
//!    1M/1M), zero blocks forced to +0.0 by the reference `torch.where`
//!    (matters: E4M3 cast of -0.0 is 0x80, of +0.0 is 0x00), clamp ±448,
//!    cast E4M3 (`dtype::f32_to_fp8_e4m3_bits`).
//! 5. Scales → cuBLAS 2D block scaling factors layout (`to_blocked`,
//!    flatten=False): shape `(roundup(M_pad,128), roundup(num_blocks,4))`,
//!    zero-padded. Closed-form index mapping verified against torch for
//!    multiple-of and non-multiple-of shapes (tools/probe_mxfp8_scale.py).
//!
//! Degenerate edge (not exercised by fixtures, kept faithful): a nonzero block
//! with `block_max <= 448*2^-127` clamps to e8m0 = 0 → scale_f32 = 0.0, so
//! elements divide by zero (±inf/NaN) → clamp → ±448/NaN code, exactly as
//! torch does (zero_mask is false there, so no rescue).

use rayon::prelude::*;

use crate::dtype::{
    bf16_bits_to_f32, f32_to_bf16_bits, f32_to_fp8_e4m3_bits, fp8_e4m3_bits_to_f32,
};
use crate::quant_fp8::FP8_MAX;

/// MXFP8 fixed block size.
pub const BLOCK_SIZE: usize = 32;
/// E8M0 exponent bias.
const E8M0_BIAS: i32 = 127;
/// Clamp floor for `scale_needed`: `2^-127`, the smallest f32 subnormal.
const SCALE_MIN: f32 = f32::from_bits(0x0040_0000);

/// Round `x` up to the nearest multiple of `mult` (reference `roundup`).
#[inline]
fn roundup(x: usize, mult: usize) -> usize {
    (x + mult - 1) / mult * mult
}

/// Output of one MXFP8 quantization.
#[derive(Debug, Clone)]
pub struct Mxfp8QuantResult {
    /// E4M3 bytes, row-major. Shape `qdata_shape` (padded to multiples of 32).
    pub qdata: Vec<u8>,
    pub qdata_shape: Vec<u64>,
    /// E8M0 block scales in cuBLAS tiled (to_blocked) layout, zero-padded.
    pub scale: Vec<u8>,
    /// `(roundup(qdata_rows, 128), roundup(num_blocks, 4))`.
    pub scale_shape: Vec<u64>,
}

/// E8M0 byte + f32 dequant scale for one block (reference
/// `_compute_block_scales` per-block math, zero-mask rescue included).
#[inline]
fn block_scale(block_max: f32) -> (u8, f32) {
    // torch tensor/scalar division = true IEEE division (probe-verified).
    let scale_needed = (block_max / FP8_MAX).max(SCALE_MIN);
    // Exact emulation of torch.ceil(torch.log2(x)) — see module docs.
    let log2_scale = (f64::from(scale_needed)).log2() as f32;
    // f32→i32 cast saturates (NaN→0), matching torch `.to(int32)` for NaN;
    // our inputs never produce inf here (bf16-derived maxima are finite).
    let exp_biased = (log2_scale.ceil() as i32 + E8M0_BIAS).clamp(0, 254);
    let e8m0 = exp_biased as u8;
    // e8m0_to_f32: 2^(e-127), or 0.0 for e == 0.
    let mut scale_f32 = if e8m0 == 0 {
        0.0
    } else {
        f32::from_bits(u32::from(e8m0) << 23)
    };
    // Zero blocks get scale 1.0 (torch.where(zero_mask, ones, scales)).
    if block_max == 0.0 {
        scale_f32 = 1.0;
    }
    (e8m0, scale_f32)
}

/// cuBLAS 2D block scaling factors layout (`to_blocked`, flatten=False).
///
/// Input `(rows, cols)` row-major → output `(roundup(rows,128), roundup(cols,4))`
/// zero-padded. Closed form (verified against torch, see module docs):
/// for input cell `(r, c)` with `rb = r/128`, `rrem = r%128`, `cb = c/4`,
/// `crem = c%4`, `b = rb*ncb + cb`, `r0 = rrem/32`, `r1 = rrem%32`:
/// `flat = b*512 + r1*16 + r0*4 + crem` (index into the flattened output).
///
/// Shared by MXFP8 (E8M0 scales) and NVFP4 (E4M3 block scales).
pub fn to_blocked_u8(src: &[u8], rows: usize, cols: usize) -> (Vec<u8>, Vec<u64>) {
    let nrb = roundup(rows, 128) / 128;
    let ncb = roundup(cols, 4) / 4;
    let padded_rows = nrb * 128;
    let padded_cols = ncb * 4;
    let mut out = vec![0u8; padded_rows * padded_cols];
    for (r, c, &v) in src_iter(src, rows, cols) {
        let rb = r / 128;
        let rrem = r % 128;
        let cb = c / 4;
        let crem = c % 4;
        let b = rb * ncb + cb;
        let r0 = rrem / 32;
        let r1 = rrem % 32;
        let flat = b * 512 + r1 * 16 + r0 * 4 + crem;
        out[flat] = v;
    }
    (out, vec![padded_rows as u64, padded_cols as u64])
}

/// Iterate `(row, col, &value)` over a row-major `(rows, cols)` buffer.
fn src_iter(src: &[u8], rows: usize, cols: usize) -> impl Iterator<Item = (usize, usize, &u8)> {
    debug_assert_eq!(src.len(), rows * cols);
    src.iter()
        .enumerate()
        .map(move |(i, v)| (i / cols, i % cols, v))
}

/// Quantize a 2D weight tensor (row-major f32) to MXFP8.
///
/// `w` must contain exactly `m * n` elements. Mirrors the reference
/// simple-mode path (`MXFP8Converter.quantize` → eager `quantize_mxfp8`):
/// round the input through bf16, pad to 32x, then block-scale + quantize.
/// The simple path has NO all-zeros early return (that belongs to the learned
/// path only), so all-zeros tensors flow through the normal pipeline.
pub fn quantize_mxfp8_weight(w: &[f32], m: usize, n: usize) -> Mxfp8QuantResult {
    debug_assert_eq!(w.len(), m * n);

    // Pad to multiples of 32 (torch.nn.functional.pad, zero fill). Round each
    // input value through bf16 first (simple-path F32→bf16 cast, then the
    // eager kernel's bf16→f32 upcast for compute).
    let m_pad = roundup(m, BLOCK_SIZE);
    let n_pad = roundup(n, BLOCK_SIZE);
    let num_blocks = n_pad / BLOCK_SIZE;
    let mut wp = vec![0.0f32; m_pad * n_pad];
    for r in 0..m {
        for c in 0..n {
            let v = w[r * n + c];
            wp[r * n_pad + c] = bf16_bits_to_f32(f32_to_bf16_bits(v));
        }
    }

    // Per row: block scales + quantized bytes (order-preserving parallel).
    let rows_out: Vec<(Vec<u8>, Vec<u8>)> = (0..m_pad)
        .into_par_iter()
        .map(|r| {
            let row = &wp[r * n_pad..(r + 1) * n_pad];
            let mut row_e8m0 = Vec::with_capacity(num_blocks);
            let mut row_q = Vec::with_capacity(n_pad);
            for b in 0..num_blocks {
                let block = &row[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                let block_max = block.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                let (e8m0, scale_f32) = block_scale(block_max);
                row_e8m0.push(e8m0);
                if block_max == 0.0 {
                    // torch.where(zero_mask, zeros_like, data_scaled) forces
                    // +0.0 (cast(-0.0) would be 0x80, not 0x00).
                    row_q.extend_from_slice(&[0u8; BLOCK_SIZE]);
                } else {
                    for &v in block {
                        // tensor/tensor division = true IEEE division.
                        let scaled = v / scale_f32;
                        let clamped = scaled.clamp(-FP8_MAX, FP8_MAX);
                        row_q.push(f32_to_fp8_e4m3_bits(clamped));
                    }
                }
            }
            (row_e8m0, row_q)
        })
        .collect();

    let mut e8m0 = Vec::with_capacity(m_pad * num_blocks);
    let mut qdata = Vec::with_capacity(m_pad * n_pad);
    for (row_e, row_q) in rows_out {
        e8m0.extend(row_e);
        qdata.extend(row_q);
    }

    let (scale, scale_shape) = to_blocked_u8(&e8m0, m_pad, num_blocks);

    Mxfp8QuantResult {
        qdata,
        qdata_shape: vec![m_pad as u64, n_pad as u64],
        scale,
        scale_shape,
    }
}

/// E8M0 byte → f32 dequant scale: `2^(e-127)`, or 0.0 for `e == 0`.
/// Verbatim port of `float_utils.e8m0_to_f32` (pure exponent format).
#[inline]
pub fn e8m0_to_f32(e: u8) -> f32 {
    if e == 0 {
        0.0
    } else {
        f32::from_bits(u32::from(e) << 23)
    }
}

/// Inverse of `to_blocked_u8`: cuBLAS tiled layout → row-major `(rows, cols)`,
/// cropping the zero padding. Port of `float_utils.from_blocked` (the reshape/
/// transpose chain is exactly the inverse of the `to_blocked` closed form).
///
/// `blocked` must be the `(roundup(rows,128), roundup(cols,4))` buffer.
pub fn from_blocked_u8(blocked: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let nrb = roundup(rows, 128) / 128;
    let ncb = roundup(cols, 4) / 4;
    let padded_cols = ncb * 4;
    debug_assert_eq!(blocked.len(), nrb * 128 * padded_cols);
    let mut out = vec![0u8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let rb = r / 128;
            let rrem = r % 128;
            let cb = c / 4;
            let crem = c % 4;
            let b = rb * ncb + cb;
            let r0 = rrem / 32;
            let r1 = rrem % 32;
            let flat = b * 512 + r1 * 16 + r0 * 4 + crem;
            out[r * cols + c] = blocked[flat];
        }
    }
    out
}

/// Dequantize an MXFP8 result back to f32 for bias correction — bit-exact
/// port of the reference eager `dequantize_mxfp8`
/// (`comfy_kitchen/backends/eager/quantization.py:330`), which the CPU goldens
/// exercise (the format module calls `converter.dequantize(...,
/// output_dtype=torch.float32)` → comfy_kitchen eager path).
///
/// Pipeline (reference lines 344-366):
/// 1. `from_blocked` the E8M0 scale bytes over `(m_pad, num_blocks)` —
///    `m_pad`/`num_blocks` are the PADDED quantize dims (the qdata shape),
///    exactly as the reference derives them from `qx.shape`.
/// 2. `e8m0_to_f32` per scale.
/// 3. `f32(E4M3_code) × scale` per 32-block (elementwise multiply — the
///    reference is `data_f32 * block_scales_f32.unsqueeze(-1)`).
/// 4. Crop the padded `(m_pad, n_pad)` result to the original `[m, n]`
///    (the format module slices `dequant_w[:m, :n]` after dequantize).
///
/// Zero blocks: the stored e8m0 byte is 0 → scale 0.0 → `0 × 0 = +0.0`,
/// matching the reference (qdata there is all 0x00 = +0.0 by construction).
pub fn dequantize_mxfp8(r: &Mxfp8QuantResult, m: usize, n: usize) -> Vec<f32> {
    let m_pad = r.qdata_shape[0] as usize;
    let n_pad = r.qdata_shape[1] as usize;
    debug_assert!(m <= m_pad && n <= n_pad);
    let num_blocks = n_pad / BLOCK_SIZE;

    // Unswizzle the E8M0 scales over the padded grid, then decode.
    let scales = from_blocked_u8(&r.scale, m_pad, num_blocks);

    let mut padded = vec![0.0f32; m_pad * n_pad];
    for row in 0..m_pad {
        for b in 0..num_blocks {
            let scale = e8m0_to_f32(scales[row * num_blocks + b]);
            let q = &r.qdata[row * n_pad + b * BLOCK_SIZE..row * n_pad + (b + 1) * BLOCK_SIZE];
            let out = &mut padded[row * n_pad + b * BLOCK_SIZE..row * n_pad + (b + 1) * BLOCK_SIZE];
            for (o, &byte) in out.iter_mut().zip(q.iter()) {
                *o = fp8_e4m3_bits_to_f32(byte) * scale;
            }
        }
    }

    // Crop to the original [m, n].
    let mut out = Vec::with_capacity(m * n);
    for row in 0..m {
        out.extend_from_slice(&padded[row * n_pad..row * n_pad + n]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_block_scale_and_codes() {
        // 1×32 row, values 1..=32 (all exact in bf16): block_max = 32.
        // scale_needed = 32/448 = 1/14 → log2 ≈ -3.807 → ceil -3 → e8m0 124
        // → scale_f32 = 2^-3 = 0.125. Max element 32/0.125 = 256 → 0x78.
        // Input is padded to 32×32 (rows 1..31 are zero blocks).
        let w: Vec<f32> = (1..=32).map(|v| v as f32).collect();
        let r = quantize_mxfp8_weight(&w, 1, 32);
        assert_eq!(r.qdata_shape, vec![32, 32]);
        assert_eq!(r.scale_shape, vec![128, 4]);
        // e8m0 for row 0 block 0 lands at flat index 0.
        assert_eq!(r.scale[0], 124);
        assert_eq!(r.qdata[31], 0x78); // 256.0 in E4M3
                                       // 16/0.125 = 128 → 2^7 → biased exp 14, mant 0 → 0x70.
        assert_eq!(r.qdata[15], 0x70);
    }

    #[test]
    fn power_of_two_block_maxima_hit_exact_e8m0() {
        // block_max = 448·2^k (exact in bf16) → scale_needed = 2^k → e8m0 127+k.
        for k in -8i32..=4 {
            let max = 448.0f32 * (2.0f32).powi(k);
            let mut w = vec![0.0f32; 32];
            w[0] = max;
            let r = quantize_mxfp8_weight(&w, 1, 32);
            assert_eq!(r.scale[0] as i32, 127 + k, "k={k}");
            // The max element must land exactly at ±448 → 0x7E.
            assert_eq!(r.qdata[0], 0x7E, "k={k}");
        }
    }

    #[test]
    fn zero_block_gets_zero_e8m0_and_zero_qdata() {
        // 1×64: first block nonzero, second block all zeros.
        let mut w = vec![0.0f32; 64];
        w[0] = 448.0;
        let r = quantize_mxfp8_weight(&w, 1, 64);
        assert_eq!(r.scale[0], 127); // first block: 448/448 = 1 → 2^0
        assert_eq!(r.scale[1], 0); // zero block → e8m0 0 (flat index 1)
        assert!(r.qdata[32..64].iter().all(|&b| b == 0x00));
        assert_eq!(r.scale_shape, vec![128, 4]);
    }

    #[test]
    fn negative_zero_in_zero_block_is_positive_zero_code() {
        // Reference where() forces +0.0 in zero blocks; cast(-0.0) = 0x80.
        let mut w = vec![0.0f32; 64];
        w[0] = 1.0;
        w[32] = -0.0;
        let r = quantize_mxfp8_weight(&w, 1, 64);
        assert_eq!(r.qdata[32], 0x00);
    }

    #[test]
    fn all_zeros_flows_through_normal_pipeline() {
        // The simple path has NO all-zeros early return: 48×40 all-zeros pads
        // to 64×64, every block is a zero block → e8m0 0, qdata all 0x00.
        let w = vec![0.0f32; 48 * 40];
        let r = quantize_mxfp8_weight(&w, 48, 40);
        assert_eq!(r.qdata_shape, vec![64, 64]);
        assert_eq!(r.qdata.len(), 64 * 64);
        assert!(r.qdata.iter().all(|&b| b == 0x00));
        assert_eq!(r.scale_shape, vec![128, 4]);
        assert!(r.scale.iter().all(|&b| b == 0));
    }

    #[test]
    fn padding_path_pads_qdata_to_32x() {
        // 3×5 → padded 32×32, num_blocks = 1. Rows 0..2 hold 1.0 (block_max 1.0
        // → e8m0 119); rows 3..31 are zero padding → e8m0 0.
        let w = vec![1.0f32; 3 * 5];
        let r = quantize_mxfp8_weight(&w, 3, 5);
        assert_eq!(r.qdata_shape, vec![32, 32]);
        assert_eq!(r.qdata.len(), 32 * 32);
        // Row r block 0 lands at flat r*16 (rows < 32, single block column).
        assert_eq!(r.scale[0], 119); // row 0: block_max 1.0
        assert_eq!(r.scale[16], 119); // row 1: block_max 1.0
        assert_eq!(r.scale[2 * 16], 119); // row 2: block_max 1.0
        assert_eq!(r.scale[3 * 16], 0); // row 3 (padding) → zero block
    }

    #[test]
    fn bf16_input_rounding_applied() {
        // A value not representable in bf16 must be rounded before quantization.
        // 1.0 + 2^-9 = 1.001953125 rounds to 1.0 in bf16 (8 mantissa bits).
        // With block_max = that value, bf16 rounding makes block_max exactly 1.0
        // → e8m0 119 (same as a clean 1.0 block).
        let mut w = vec![0.0f32; 32];
        w[0] = 1.0 + 2.0f32.powi(-9); // rounds to 1.0 in bf16
        let r = quantize_mxfp8_weight(&w, 1, 32);
        assert_eq!(r.scale[0], 119);
    }

    #[test]
    fn to_blocked_closed_form_small() {
        // rows=128, cols=4: flat = (r%32)*16 + (r/32)*4 + c.
        let src: Vec<u8> = (0..128 * 4).map(|i| (i % 251) as u8).collect();
        let (out, shape) = to_blocked_u8(&src, 128, 4);
        assert_eq!(shape, vec![128, 4]);
        for r in 0..128usize {
            for c in 0..4usize {
                let flat = (r % 32) * 16 + (r / 32) * 4 + c;
                assert_eq!(out[flat], src[r * 4 + c], "r={r} c={c}");
            }
        }
    }

    #[test]
    fn to_blocked_pads_with_zeros() {
        // rows=130, cols=5 → (256, 8); untouched cells must be zero.
        let src = vec![7u8; 130 * 5];
        let (out, shape) = to_blocked_u8(&src, 130, 5);
        assert_eq!(shape, vec![256, 8]);
        let sevens = out.iter().filter(|&&b| b == 7).count();
        assert_eq!(sevens, 130 * 5);
        assert_eq!(out.len(), 256 * 8);
    }

    #[test]
    fn parallel_equals_sequential() {
        let (m, n) = (256usize, 128usize);
        let mut w = vec![0.0f32; m * n];
        let mut x: u64 = 0x9E3779B97F4A7C15;
        for v in &mut w {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *v = ((x % 20_000) as f32 / 10_000.0) - 1.0;
        }
        let par = quantize_mxfp8_weight(&w, m, n);

        // Sequential recomputation (bf16 rounding + padded shape).
        let m_pad = roundup(m, BLOCK_SIZE);
        let n_pad = roundup(n, BLOCK_SIZE);
        let num_blocks = n_pad / BLOCK_SIZE;
        let mut seq_q = vec![0u8; m_pad * n_pad];
        let mut seq_e = vec![0u8; m_pad * num_blocks];
        for r in 0..m_pad {
            for b in 0..num_blocks {
                let mut bm = 0.0f32;
                for k in 0..BLOCK_SIZE {
                    let v = if r < m && b * BLOCK_SIZE + k < n {
                        bf16_bits_to_f32(f32_to_bf16_bits(w[r * n + b * BLOCK_SIZE + k]))
                    } else {
                        0.0
                    };
                    bm = bm.max(v.abs());
                }
                let (e, s) = block_scale(bm);
                seq_e[r * num_blocks + b] = e;
                for k in 0..BLOCK_SIZE {
                    let v = if r < m && b * BLOCK_SIZE + k < n {
                        bf16_bits_to_f32(f32_to_bf16_bits(w[r * n + b * BLOCK_SIZE + k]))
                    } else {
                        0.0
                    };
                    let code = if bm == 0.0 {
                        0x00
                    } else {
                        f32_to_fp8_e4m3_bits((v / s).clamp(-FP8_MAX, FP8_MAX))
                    };
                    seq_q[r * n_pad + b * BLOCK_SIZE + k] = code;
                }
            }
        }
        assert_eq!(par.qdata, seq_q);
        let (seq_blocked, seq_shape) = to_blocked_u8(&seq_e, m_pad, num_blocks);
        assert_eq!(par.scale, seq_blocked);
        assert_eq!(par.scale_shape, seq_shape);
    }

    // --- Phase B.2: e8m0_to_f32 / from_blocked_u8 / dequantize_mxfp8 ------- //

    #[test]
    fn e8m0_to_f32_values() {
        assert_eq!(e8m0_to_f32(0), 0.0);
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(124), 0.125); // 2^-3
        assert_eq!(e8m0_to_f32(254), 2.0f32.powi(127));
        assert_eq!(e8m0_to_f32(1), 2.0f32.powi(-126)); // smallest normal
        assert_eq!(e8m0_to_f32(119), 2.0f32.powi(-8));
    }

    #[test]
    fn from_blocked_inverts_to_blocked() {
        // Round-trip on multiple shapes incl. non-multiples of 128/4.
        for &(rows, cols) in &[
            (128usize, 4usize),
            (130, 5),
            (1, 1),
            (32, 8),
            (256, 12),
            (33, 3),
        ] {
            let src: Vec<u8> = (0..rows * cols)
                .map(|i| ((i * 7 + 3) % 251) as u8)
                .collect();
            let (blocked, shape) = to_blocked_u8(&src, rows, cols);
            assert_eq!(shape[0] as usize % 128, 0);
            assert_eq!(shape[1] as usize % 4, 0);
            let back = from_blocked_u8(&blocked, rows, cols);
            assert_eq!(back, src, "round trip failed for ({rows}, {cols})");
        }
    }

    #[test]
    fn dequantize_mxfp8_hand_computed() {
        // 1×32 row, values 1..=32: block_max = 32 → scale_needed = 32/448 →
        // e8m0 = 124 → scale_f32 = 0.125. Dequant = f32(code) × 0.125.
        //   v=1:  1/0.125 = 8   → E4M3 8   → dequant 8 × 0.125   = 1.0
        //   v=16: 16/0.125 = 128 → E4M3 128 → dequant 128 × 0.125 = 16.0
        //   v=32: 32/0.125 = 256 → E4M3 256 → dequant 256 × 0.125 = 32.0
        let w: Vec<f32> = (1..=32).map(|v| v as f32).collect();
        let r = quantize_mxfp8_weight(&w, 1, 32);
        let dq = dequantize_mxfp8(&r, 1, 32);
        assert_eq!(dq.len(), 32);
        assert_eq!(dq[0], 1.0);
        assert_eq!(dq[15], 16.0);
        assert_eq!(dq[31], 32.0);
        // Every dequantized value is a multiple of 0.125 (code × 2^-3) and
        // within one block quantum of the original.
        for (i, (&o, &v)) in dq.iter().zip(w.iter()).enumerate() {
            let rem = (o / 0.125).fract();
            assert_eq!(rem, 0.0, "dq[{i}] = {o} not on the 0.125 grid");
            assert!(
                (o - v).abs() <= 16.0 * 0.125 + 1e-6,
                "dq[{i}] too far from {v}"
            );
        }
    }

    #[test]
    fn dequantize_mxfp8_crops_padding() {
        // 3×5 all-ones → padded 32×32. Dequant must crop back to 3×5, all 1.0
        // (1/2^-8 = 256 → E4M3 256 → 256 × 2^-8 = 1.0).
        let w = vec![1.0f32; 3 * 5];
        let r = quantize_mxfp8_weight(&w, 3, 5);
        assert_eq!(r.qdata_shape, vec![32, 32]);
        let dq = dequantize_mxfp8(&r, 3, 5);
        assert_eq!(dq.len(), 15);
        assert!(dq.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn dequantize_mxfp8_zero_block_is_zero() {
        // 1×64: first block nonzero, second all zeros. Zero block → e8m0 0 →
        // scale 0.0 → dequant +0.0 (not NaN, not negative zero).
        let mut w = vec![0.0f32; 64];
        w[0] = 448.0;
        let r = quantize_mxfp8_weight(&w, 1, 64);
        let dq = dequantize_mxfp8(&r, 1, 64);
        assert_eq!(dq[0], 448.0); // 448/1 = 448 → code 0x7E → 448 × 1.0
        for &v in &dq[32..] {
            assert_eq!(v, 0.0);
            assert!(v.is_sign_positive(), "zero block must dequant to +0.0");
        }
    }

    #[test]
    fn dequant_roundtrip_bounded_mxfp8() {
        // Phase B.4 (MXFP8 arm): quantize a random matrix, dequantize, assert
        // finite + bounded, and that the dequantized values are an EXACT
        // fixed point of the quantize→dequantize cycle.
        //
        // NOTE: strict code idempotency does NOT hold for MXFP8 — e8m0 scales
        // are powers of two, and when a block's max code rounds to ≤ 224 the
        // re-quantization shrinks the scale by one exponent and doubles the
        // codes (e.g. 224×2^0 → 448×2^-1). The representation changes but the
        // dequantized values are bit-identical (code×2^e == 2·code×2^(e-1),
        // both exact dyadic products), which is what bias correction consumes.
        let (m, n) = (64usize, 48usize);
        let mut w = vec![0.0f32; m * n];
        let mut x: u64 = 0x9E3779B97F4A7C15;
        for v in &mut w {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *v = ((x % 20_000) as f32 / 10_000.0) - 1.0;
        }
        let amax = w.iter().fold(0.0f32, |a, &v| a.max(v.abs()));

        let r = quantize_mxfp8_weight(&w, m, n);
        let dq = dequantize_mxfp8(&r, m, n);
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
        // Fixed point: re-quantizing the dequant and dequantizing again must
        // reproduce the same f32 values bit-for-bit.
        let r2 = quantize_mxfp8_weight(&dq, m, n);
        let dq2 = dequantize_mxfp8(&r2, m, n);
        assert_eq!(
            dq2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            dq.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "dequant values are not a fixed point of quantize→dequantize"
        );
    }
}

//! NVFP4 (FP4 E2M1) quantization kernel — byte-parity port of the
//! convert_to_quant *simple* (no learned rounding) NVFP4 path.
//!
//! Sources ported (simple path, matching `gen_golden_formats.py` `simple=True`):
//! `convert_to_quant/formats/nvfp4_conversion.py` (F32 cast + converter call),
//! `convert_to_quant/converters/nvfp4_converter.py` `NVFP4Converter.quantize`
//! (per-tensor scale from the F32 tensor + bf16 input cast),
//! `comfy_kitchen/backends/eager/quantization.py` `quantize_nvfp4` (the
//! block-scale + quantize pipeline), and `convert_to_quant/utils/float_utils.py`
//! (`_f32_to_floatx_unpacked`, `pack_uint4`, `to_blocked`).
//!
//! Golden provenance: the goldens MUST be generated with CUDA hidden
//! (`CUDA_VISIBLE_DEVICES=-1`). `nvfp4_conversion.py:108` hardcodes
//! `device = "cuda" if torch.cuda.is_available() else "cpu"` and ignores the
//! `device="cpu"` the gen script passes; with CUDA visible the compiled CUDA
//! kernel runs instead of the eager backend and its data division is
//! reciprocal-multiply (`x * (1/total)`) rather than IEEE division, breaking
//! byte-parity at E2M1 tie points. The CPU/eager goldens are the plan's parity
//! target ("NVFP4: nvfp4_converter.py::_quantize_pytorch ... All bit-exact
//! portable"). The CUDA-vs-eager divergence was root-caused in
//! `tools/probe_nvfp4.py` (see its provenance note).
//!
//! Pipeline (all f32 compute):
//!
//! 0. `per_tensor_scale = amax(abs(w_f32)) / (448 * 6.0)` — computed on the
//!    F32 tensor BEFORE the bf16 cast; tensor/scalar division = true IEEE
//!    division (probe-verified, Phase 3.2/3.3).
//! 1. Round every input value through bf16 (`f32 → bf16 → f32`): the
//!    converter casts to bf16 before the kernel ("CUDA kernel only supports
//!    FP16/BF16 input"), and the eager kernel upcasts bf16→f32 for compute.
//! 2. Pad to multiples of 16 (zero pad at the end of rows/cols).
//! 3. Per 16-element block (row-major within each row):
//!    - `block_max = amax(abs(block))` (max-reduction: order-independent)
//!    - `block_scale = block_max / 6.0` (tensor/scalar = true IEEE division)
//!    - `scaled = block_scale / per_tensor_scale` (tensor/tensor = true IEEE)
//!    - clamp MAX only: `min(scaled, 448)` (reference has no min clamp)
//!    - `scale_byte = E4M3(scaled_clamped)`; `scaled_f32 = f32(scale_byte)`
//!      (the `_float8_round` round-trip)
//!    - `total = per_tensor_scale * scaled_f32`
//! 4. Quantize: zero blocks (`total == 0`) forced to +0.0 by the reference
//!    `torch.where` (all 16 codes 0 → packed bytes 0x00); otherwise
//!    `d = v / total` (tensor/tensor = true IEEE division), clamp ±6,
//!    E2M1-encode (`_f32_to_floatx_unpacked(x, 2, 1)` port), and pack pairs
//!    hi-first (even index → high nibble, `pack_uint4(hi_first=True)`).
//! 5. Scales → cuBLAS 2D block scaling factors layout (`to_blocked`,
//!    flatten=False), shared closed form with MXFP8 (`quant_mxfp8::to_blocked_u8`).
//!
//! E2M1 encoder notes (ebits=2, mbits=1, exp_bias=1): values
//! 0, 0.5, 1, 1.5, 2, 3, 4, 6 are codes 0..7 (sign bit = 8). The magic-adder
//! integer arithmetic is a verbatim port, so even degenerate inputs (NaN)
//! reproduce torch's deterministic bit behavior. Saturation: |x| ≥ 6 → code 7
//! (the kernel clamps to ±6 first, so only exactly ±6 saturates in practice).

use rayon::prelude::*;

use crate::dtype::{
    bf16_bits_to_f32, f32_to_bf16_bits, f32_to_fp8_e4m3_bits, fp8_e4m3_bits_to_f32,
};
use crate::quant_fp8::FP8_MAX;
use crate::quant_mxfp8::to_blocked_u8;

/// NVFP4 fixed block size (`FP4_BLOCK_SIZE`).
pub const BLOCK_SIZE: usize = 16;
/// E2M1 max finite value (`F4_E2M1_MAX`).
const FP4_MAX: f32 = 6.0;
/// Per-tensor-scale divisor: `F8_E4M3_MAX * F4_E2M1_MAX` = 448 * 6.
const PTS_DIVISOR: f32 = FP8_MAX * FP4_MAX;

/// Round `x` up to the nearest multiple of `mult` (reference `roundup`).
#[inline]
fn roundup(x: usize, mult: usize) -> usize {
    (x + mult - 1) / mult * mult
}

/// Max-reduction of absolute values, matching `torch.amax(torch.abs(t))`:
/// NaN-propagating (torch.amax returns NaN if any element is NaN).
fn amax_abs(vals: &[f32]) -> f32 {
    let mut m = 0.0f32;
    let mut any_nan = false;
    for &v in vals {
        let a = v.abs();
        if a.is_nan() {
            any_nan = true;
        } else if a > m {
            m = a;
        }
    }
    if any_nan {
        f32::NAN
    } else {
        m
    }
}

/// `torch.clamp(x, max=c)` semantics: NaN-preserving (unlike `f32::min`, which
/// returns the non-NaN operand). Matters for all-zeros tensors where
/// `scaled = 0/0 = NaN` must flow through to the E4M3 cast (→ 0x7F).
#[inline]
fn clamp_max(x: f32, c: f32) -> f32 {
    if x > c {
        c
    } else {
        x
    }
}

/// Output of one NVFP4 quantization.
#[derive(Debug, Clone)]
pub struct Nvfp4QuantResult {
    /// Packed E2M1 bytes (two codes per byte, even index in the high nibble),
    /// row-major. Shape `qdata_shape` = `(m_pad, n_pad / 2)`.
    pub qdata: Vec<u8>,
    pub qdata_shape: Vec<u64>,
    /// E4M3 block scales in cuBLAS tiled (to_blocked) layout, zero-padded.
    pub scale: Vec<u8>,
    /// `(roundup(m_pad, 128), roundup(num_blocks, 4))`.
    pub scale_shape: Vec<u64>,
    /// Per-tensor scale (`weight_scale_2`), f32.
    pub per_tensor_scale: f32,
}

/// f32 → FP4 E2M1 code (sign bit 0x8), verbatim port of
/// `_f32_to_floatx_unpacked(x, ebits=2, mbits=1)`.
///
/// Magic-adder round-to-nearest-even. Branches mirror the torch masks:
/// `|x| >= 6.0` → saturate (code 7); `|x| < 1.0` → denormal path (add 2^22,
/// subtract back — rounds to the 0.5 grid, may overflow into normal codes,
/// which is correct since the code space is contiguous); otherwise the normal
/// path (bias delta + half-ulp bias + mant_odd RNE trick). NaN comparisons are
/// false, so NaN falls into the normal path exactly like torch's
/// `normal_mask = ~(saturate | denormal)`.
#[inline]
fn f32_to_e2m1_bits(x: f32) -> u8 {
    const EXP_BIAS: i32 = 1; // _n_ones(ebits - 1), ebits = 2
    const MAX_INT: u8 = 0x07; // _n_ones(ebits + mbits)
    const SIGN_MASK: u8 = 0x08; // 1 << (ebits + mbits)
    const MAGIC_ADDER: i32 = 0x1F_FFFF; // _n_ones(MBITS_F32 - mbits - 1)
    const MAX_NORMAL: f32 = 6.0; // 2^(3-1) * (3/2)
    const MIN_NORMAL: f32 = 1.0; // 2^(1-exp_bias)
    const DENORM_EXP: i32 = (127 - 1) + (23 - 1) + 1; // 149
    const DENORM_MASK_INT: i32 = DENORM_EXP << 23;
    const DENORM_MASK_FLOAT: f32 = f32::from_bits(DENORM_MASK_INT as u32); // 2^22

    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let abs_bits = bits ^ sign;
    let ax = f32::from_bits(abs_bits);

    let code: u8 = if ax >= MAX_NORMAL {
        MAX_INT
    } else if ax < MIN_NORMAL {
        // Denormal path: f32 add of 2^22 performs RNE onto the 0.5 grid.
        let dx = ax + DENORM_MASK_FLOAT;
        (dx.to_bits() as i32).wrapping_sub(DENORM_MASK_INT) as u8
    } else {
        // Normal path: RNE via mant_odd + rounding-bias trick.
        let mant_odd = ((abs_bits >> 22) & 1) as i32;
        let val_to_add = ((EXP_BIAS - 127) << 23) + MAGIC_ADDER;
        let mut nx = abs_bits as i32;
        nx = nx.wrapping_add(val_to_add);
        nx = nx.wrapping_add(mant_odd);
        (nx >> 22) as u8
    };

    // sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits) = sign >> 28.
    let sign_lp = ((sign >> 28) as u8) & SIGN_MASK;
    code | sign_lp
}

/// Encode one element: IEEE-divide by the block's total scale, clamp ±6,
/// E2M1-encode. (Zero blocks never reach this — they are forced to +0.0.)
#[inline]
fn encode_element(v: f32, total_scale: f32) -> u8 {
    // tensor/tensor division = true IEEE division (probe-verified).
    let d = v / total_scale;
    // torch.clamp(NaN) = NaN; f32::clamp returns self for NaN — same.
    let d = d.clamp(-FP4_MAX, FP4_MAX);
    f32_to_e2m1_bits(d)
}

/// Quantize a 2D weight tensor (row-major f32) to NVFP4.
///
/// `w` must contain exactly `m * n` elements. Mirrors the reference
/// simple-mode path (`NVFP4Converter.quantize` → eager `quantize_nvfp4`):
/// per-tensor scale from the F32 amax, round the input through bf16, pad to
/// 16x, then block-scale + quantize + pack.
pub fn quantize_nvfp4_weight(w: &[f32], m: usize, n: usize) -> Nvfp4QuantResult {
    debug_assert_eq!(w.len(), m * n);

    // Per-tensor scale from the F32 tensor BEFORE the bf16 cast
    // (nvfp4_converter.py: amax over the f32 tensor; tensor/scalar division
    // = true IEEE division). amax is NaN-propagating like torch.amax.
    let amax = amax_abs(w);
    let per_tensor_scale = amax / PTS_DIVISOR;

    // Pad to multiples of 16 (torch.nn.functional.pad, zero fill), rounding
    // each input value through bf16 first (converter's `.to(torch.bfloat16)`
    // then the eager kernel's `.float()` upcast).
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

    // Per row: block scales + packed codes (order-preserving parallel).
    let rows_out: Vec<(Vec<u8>, Vec<u8>)> = (0..m_pad)
        .into_par_iter()
        .map(|r| {
            let row = &wp[r * n_pad..(r + 1) * n_pad];
            let mut row_scale = Vec::with_capacity(num_blocks);
            let mut row_q = Vec::with_capacity(n_pad / 2);
            for b in 0..num_blocks {
                let block = &row[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                let block_max = amax_abs(block);
                // tensor/scalar = true IEEE division.
                let block_scale = block_max / FP4_MAX;
                // tensor/tensor = true IEEE division.
                let scaled = block_scale / per_tensor_scale;
                // Reference clamps MAX only (no min clamp); NaN-preserving.
                let scaled_clamped = clamp_max(scaled, FP8_MAX);
                let scale_byte = f32_to_fp8_e4m3_bits(scaled_clamped);
                // _float8_round: E4M3 round-trip.
                let scaled_f32 = fp8_e4m3_bits_to_f32(scale_byte);
                let total_scale = per_tensor_scale * scaled_f32;
                row_scale.push(scale_byte);
                if total_scale == 0.0 {
                    // Zero block: torch.where forces +0.0 for all 16 elements
                    // → code 0 → packed bytes 0x00.
                    row_q.extend_from_slice(&[0u8; BLOCK_SIZE / 2]);
                } else {
                    for pair in block.chunks_exact(2) {
                        // pack_uint4(hi_first=True): even index → high nibble.
                        let hi = encode_element(pair[0], total_scale);
                        let lo = encode_element(pair[1], total_scale);
                        row_q.push(hi << 4 | lo);
                    }
                }
            }
            (row_scale, row_q)
        })
        .collect();

    let mut scale_rows = Vec::with_capacity(m_pad * num_blocks);
    let mut qdata = Vec::with_capacity(m_pad * (n_pad / 2));
    for (row_s, row_q) in rows_out {
        scale_rows.extend(row_s);
        qdata.extend(row_q);
    }

    let (scale, scale_shape) = to_blocked_u8(&scale_rows, m_pad, num_blocks);

    Nvfp4QuantResult {
        qdata,
        qdata_shape: vec![m_pad as u64, (n_pad / 2) as u64],
        scale,
        scale_shape,
        per_tensor_scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_lut_values() {
        // Positive LUT: 0, 0.5, 1, 1.5, 2, 3, 4, 6 → codes 0..7.
        let vals = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(f32_to_e2m1_bits(v), i as u8, "v={v}");
            assert_eq!(f32_to_e2m1_bits(-v), (i as u8) | 0x08, "v={v} (neg)");
        }
    }

    #[test]
    fn e2m1_rne_ties_pick_even() {
        // Midpoints round to the value with an even mantissa code.
        assert_eq!(f32_to_e2m1_bits(5.0), 6); // 4↔6 tie → 4 (code 6, even)
        assert_eq!(f32_to_e2m1_bits(2.5), 4); // 2↔3 tie → 2 (code 4, even)
        assert_eq!(f32_to_e2m1_bits(1.25), 2); // 1↔1.5 tie → 1 (code 2, even)
        assert_eq!(f32_to_e2m1_bits(0.75), 2); // 0.5↔1 tie → 1 (code 2, even)
        assert_eq!(f32_to_e2m1_bits(0.25), 0); // 0↔0.5 tie → 0 (code 0, even)
        // Just off the ties.
        assert_eq!(f32_to_e2m1_bits(5.1), 7);
        assert_eq!(f32_to_e2m1_bits(4.9), 6);
        assert_eq!(f32_to_e2m1_bits(0.26), 1);
        assert_eq!(f32_to_e2m1_bits(0.24), 0);
    }

    #[test]
    fn e2m1_saturation_and_subnormals() {
        assert_eq!(f32_to_e2m1_bits(6.0), 7);
        assert_eq!(f32_to_e2m1_bits(100.0), 7);
        assert_eq!(f32_to_e2m1_bits(-100.0), 15);
        assert_eq!(f32_to_e2m1_bits(0.49), 1); // rounds to 0.5
        assert_eq!(f32_to_e2m1_bits(1e-30), 0); // deep underflow → 0
        // Largest f32 below 1.0 still rounds up to 1.0 (code 2) via the
        // denormal path overflowing into normal codes.
        assert_eq!(f32_to_e2m1_bits(f32::from_bits(0x3F7F_FFFF)), 2);
    }

    #[test]
    fn single_block_known_codes() {
        // amax = 2688 → pts = 1.0; block_max = 2688 → block_scale = 448 →
        // scaled = 448 → E4M3 0x7E → total = 448. Elements are multiples of
        // 224 = 448*0.5, so v/448 lands exactly on the E2M1 LUT.
        let vals = [
            2688.0, 224.0, 448.0, 672.0, 896.0, 1344.0, 1792.0, 2688.0, -224.0, -448.0, -672.0,
            -896.0, -1344.0, -1792.0, -2688.0, 0.0,
        ];
        let r = quantize_nvfp4_weight(&vals, 1, 16);
        assert_eq!(r.qdata_shape, vec![16, 8]);
        assert_eq!(r.scale_shape, vec![128, 4]);
        assert_eq!(r.per_tensor_scale, 1.0);
        assert_eq!(r.scale[0], 0x7E); // 448 in E4M3, flat index 0
        // Codes: 7 1 2 3 4 5 6 7 | 9 10 11 12 13 14 15 0 (hi-first packing).
        let expect = [0x71u8, 0x23, 0x45, 0x67, 0x9A, 0xBC, 0xDE, 0xF0];
        assert_eq!(&r.qdata[..8], &expect);
        // Padding rows (1..16) are zero blocks → all-zero bytes.
        assert!(r.qdata[8..].iter().all(|&b| b == 0x00));
    }

    #[test]
    fn zero_block_forces_positive_zero_codes() {
        // 1×32: first block nonzero, second block all zeros.
        let mut w = vec![0.0f32; 32];
        w[0] = 448.0;
        let r = quantize_nvfp4_weight(&w, 1, 32);
        assert_eq!(r.qdata_shape, vec![16, 16]);
        // Zero block: scale byte 0x00 and packed bytes 0x00 (+0.0 forcing —
        // encode(-0.0) would be code 8, so this is observable).
        assert_eq!(r.scale[1], 0x00);
        assert!(r.qdata[8..16].iter().all(|&b| b == 0x00));
        // Nonzero block's single element: 448/total ≈ 6 → code 7.
        assert_eq!(r.qdata[0] >> 4, 7);
    }

    #[test]
    fn negative_zero_element_encodes_code_8() {
        // Outside zero blocks, -0.0 keeps its sign through divide/clamp and
        // encodes as code 8 (negative zero), matching torch's bit behavior.
        assert_eq!(f32_to_e2m1_bits(-0.0), 8);
    }

    #[test]
    fn pts_uses_pre_bf16_amax() {
        // 1.0 + 2^-9 rounds to 1.0 in bf16, but the per-tensor scale must be
        // computed from the ORIGINAL f32 amax (before the bf16 cast).
        let mut w = vec![0.0f32; 16];
        w[0] = 1.0 + 2.0f32.powi(-9);
        let r = quantize_nvfp4_weight(&w, 1, 16);
        assert_eq!(r.per_tensor_scale, (1.0 + 2.0f32.powi(-9)) / 2688.0);
        assert_ne!(r.per_tensor_scale, 1.0f32 / 2688.0);
    }

    #[test]
    fn padding_path_structure() {
        // 3×5 all-ones → padded 16×16; rows 0..2 have block_max 1.0, rows
        // 3..15 are zero padding.
        let w = vec![1.0f32; 3 * 5];
        let r = quantize_nvfp4_weight(&w, 3, 5);
        assert_eq!(r.qdata_shape, vec![16, 8]);
        assert_eq!(r.scale_shape, vec![128, 4]);
        // Rows 0..2: 1.0/(pts*448) ≈ 6 → code 7 → byte 0x77.
        for row in 0..3usize {
            assert_eq!(r.qdata[row * 8], 0x77, "row {row}");
        }
        // Padding rows → zero blocks → all-zero bytes.
        assert!(r.qdata[3 * 8..].iter().all(|&b| b == 0x00));
        // Scale bytes: rows 0..2 block 0 land at flat indices 0, 16, 32.
        assert_eq!(r.scale[0], 0x7E); // scaled = 448 → E4M3 max
        assert_eq!(r.scale[16], 0x7E);
        assert_eq!(r.scale[32], 0x7E);
        assert_eq!(r.scale[48], 0x00); // row 3 (padding) → zero block
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
        let par = quantize_nvfp4_weight(&w, m, n);

        // Sequential recomputation (same pipeline, no rayon).
        let amax = amax_abs(&w);
        let pts = amax / PTS_DIVISOR;
        let m_pad = roundup(m, BLOCK_SIZE);
        let n_pad = roundup(n, BLOCK_SIZE);
        let num_blocks = n_pad / BLOCK_SIZE;
        let mut seq_q = vec![0u8; m_pad * (n_pad / 2)];
        let mut seq_s = vec![0u8; m_pad * num_blocks];
        for r in 0..m_pad {
            for b in 0..num_blocks {
                let mut vals = [0.0f32; BLOCK_SIZE];
                for k in 0..BLOCK_SIZE {
                    vals[k] = if r < m && b * BLOCK_SIZE + k < n {
                        bf16_bits_to_f32(f32_to_bf16_bits(w[r * n + b * BLOCK_SIZE + k]))
                    } else {
                        0.0
                    };
                }
                let bm = amax_abs(&vals);
                let scaled = (bm / FP4_MAX) / pts;
                let sb = f32_to_fp8_e4m3_bits(clamp_max(scaled, FP8_MAX));
                seq_s[r * num_blocks + b] = sb;
                let total = pts * fp8_e4m3_bits_to_f32(sb);
                for p in 0..BLOCK_SIZE / 2 {
                    let byte = if total == 0.0 {
                        0x00
                    } else {
                        let hi = encode_element(vals[2 * p], total);
                        let lo = encode_element(vals[2 * p + 1], total);
                        hi << 4 | lo
                    };
                    seq_q[r * (n_pad / 2) + b * (BLOCK_SIZE / 2) + p] = byte;
                }
            }
        }
        assert_eq!(par.qdata, seq_q);
        assert_eq!(par.per_tensor_scale, pts);
        let (seq_blocked, seq_shape) = to_blocked_u8(&seq_s, m_pad, num_blocks);
        assert_eq!(par.scale, seq_blocked);
        assert_eq!(par.scale_shape, seq_shape);
    }
}

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
use crate::quant_mxfp8::{from_blocked_u8, to_blocked_u8};

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

/// E2M1 code → f32 LUT (reference `E2M1_LUT` in
/// `comfy_kitchen/backends/eager/quantization.py`): codes 0..7 →
/// 0, 0.5, 1, 1.5, 2, 3, 4, 6; codes 8..15 → their negatives (code 8 = -0.0).
/// All values are exact dyadics, so every product below is correctly rounded.
const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Dequantize an NVFP4 result back to f32 for bias correction — bit-exact
/// port of the reference eager `dequantize_nvfp4`
/// (`comfy_kitchen/backends/eager/quantization.py:174`), which the CPU
/// goldens exercise (the format module calls `converter.dequantize(...,
/// output_dtype=torch.float32)`).
///
/// Pipeline (reference lines 186-219):
/// 1. Unpack the uint4 pairs hi-first (even index = high nibble, matching
///    `pack_uint4(hi_first=True)` used at quantize time).
/// 2. `E2M1_LUT[code]` per element.
/// 3. `from_blocked` the E4M3 block-scale bytes over `(m_pad, num_blocks)` —
///    the PADDED quantize dims, exactly as the reference derives them from
///    the unpacked shape.
/// 4. `total = per_tensor_scale × f32(scale_byte)` (keep this association —
///    plan §3.5), then `value × total` per 16-block.
/// 5. Crop the padded `(m_pad, n_pad)` result to the original `[m, n]`
///    (the format module slices `dequant_w[:m, :n]` after dequantize).
///
/// Zero blocks: scale byte 0x00 → f32 0.0 → total 0.0 → `0.0 × 0.0 = +0.0`
/// (codes there are all 0 → LUT[0] = +0.0), matching the reference.
pub fn dequantize_nvfp4(r: &Nvfp4QuantResult, m: usize, n: usize) -> Vec<f32> {
    let m_pad = r.qdata_shape[0] as usize;
    let n_pad = r.qdata_shape[1] as usize * 2; // qdata stores n_pad/2 packed bytes
    debug_assert!(m <= m_pad && n <= n_pad);
    let num_blocks = n_pad / BLOCK_SIZE;

    // Unswizzle the E4M3 block scales over the padded grid, then decode.
    let scale_bytes = from_blocked_u8(&r.scale, m_pad, num_blocks);

    let mut padded = vec![0.0f32; m_pad * n_pad];
    for row in 0..m_pad {
        for b in 0..num_blocks {
            let scale_f32 = fp8_e4m3_bits_to_f32(scale_bytes[row * num_blocks + b]);
            // Reference association: total = pts * block_scale_f32.
            let total = r.per_tensor_scale * scale_f32;
            let packed = &r.qdata[row * (n_pad / 2) + b * (BLOCK_SIZE / 2)
                ..row * (n_pad / 2) + (b + 1) * (BLOCK_SIZE / 2)];
            let out = &mut padded[row * n_pad + b * BLOCK_SIZE..row * n_pad + (b + 1) * BLOCK_SIZE];
            for (pair, &byte) in out.chunks_exact_mut(2).zip(packed.iter()) {
                let hi = (byte >> 4) as usize;
                let lo = (byte & 0x0F) as usize;
                pair[0] = E2M1_LUT[hi] * total;
                pair[1] = E2M1_LUT[lo] * total;
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

// ---------------------------------------------------------------------------
// Tier 2 S03 — opt-in anchored alternating L2 scale search
// ---------------------------------------------------------------------------
//
// `quantize_nvfp4_weight` above is the parity path and is byte-frozen. Nothing
// in this section is reachable from it; the only entry point is
// `quantize_nvfp4_weight_quality`, which a caller must name explicitly.
//
// # Why the search must be anchored
//
// The reconstruction is `x̂ᵢ = cᵢ · (s_T · s_G)` — the per-tensor scale and the
// block scale appear ONLY as their product. So:
//
//   * rescaling `s_T → k·s_T`, `s_G → s_G/k` leaves the product invariant, and
//   * because the E2M1 grid is `{2ʲ, 1.5·2ʲ}`, the *codes* are invariant too
//     whenever `k = 2ʲ`.
//
// The L2 objective is therefore **exactly flat** along powers of two — not
// merely flat, but flat to the last bit, because nothing downstream observes
// the split. A search that only "minimises L2" is not unconstrained, it is
// *ambiguous*: `s_T` can inflate without limit while the error stands still.
// (A prototype observed +32939% before anything objected.)
//
// The flatness is not infinite, and where it ends is the interesting part.
// Measured on the 3-decade fixture in `tests/nvfp4_scale_search.rs`: the
// reconstruction is bit-identical for k = 0..=4 — `s_T` inflated 16×, every
// emitted `s_G` byte changed, error unchanged to the last bit — and then it
// stops being flat at k = 5, precisely where the smallest block's halved scale
// enters the E4M3 subnormal range and halving is no longer exact. By k = 12
// (4096×) 22 of 64 block scales have rounded to E4M3 **zero** and the error is
// worse than the reference. The objective cannot see any of that coming, so
// nothing in the search itself will stop it.
//
// Two guards, both structural rather than advisory:
//
//   1. **Fixed byte window.** `s_G` is only ever chosen from the contiguous
//      E4M3 byte-code window `[anchor ± 4]` around the reference block scale.
//      E4M3 positive bytes are numerically monotone, so that window is a
//      bounded multiplicative neighbourhood (≈ ×1.41 either way). Picking a
//      *byte* rather than a float means the search can never optimise a value
//      the hardware never sees.
//   2. **Anchored `s_T` clamp.** The refit per-tensor scale is confined to
//      `s_T⁰ / 4 … s_T⁰ · 4`.
//
// # What is actually left to optimise
//
// Since only the product matters and the powers of two are free, the entire
// remaining gain is in landing `s_G` on a **different E4M3 grid point** than
// the anchor. That is a real, non-trivial choice: the reference scale puts the
// block max at `d = block_max / total ≈ 6`, i.e. hard against the E2M1
// saturation boundary, where a half-ULP of E4M3 rounding decides whether the
// largest element clips. Moving one grid point trades a little clipping on one
// element for resolution on the other fifteen, and the least-squares refit of
// `s_T` then cancels the systematic bias that E4M3 block-scale rounding
// leaves across all blocks at once.
//
// # Determinism
//
// A genuinely sequential dependency exists: `s_T` depends on every `s_G`.
// Parallelising the per-block search is safe because each block's search reads
// only its own 16 values, `pts`, and its own window, and rayon fills the
// result `Vec` **by index**. The `s_T` reduction is then folded over that
// `Vec` in ascending block index in f64. No `HashMap`, no `par_iter().sum()`,
// no `reduce` — every one of those makes the float result depend on a
// non-specified order, which is exactly the bug this section is shaped to
// avoid.
//
// # The objective is measured on the PRE-bf16 tensor
//
// The single most important subtlety in this section, and the source of the
// one real bug found while writing it.
//
// The codes must be chosen from the bf16-rounded values — that rounding is a
// stage of the reference pipeline and changing it would break parity. But the
// error must be measured against the **original** values, because that is what
// the caller handed us and what the base path's own `per_tensor_scale` is
// derived from (`pts_uses_pre_bf16_amax`).
//
// Scoring the rounded values instead looks harmless and is not. For a 1×1
// tensor of `0.27318907` the rounded value is `0.2734375`; a search scoring
// the rounded value "improves" to a perfect 0.0 by reconstructing `0.2734375`,
// and emits it. Measured against the tensor the user actually passed in, that
// reconstruction is off by 2.5e-4 in the wrong direction — the base path
// reconstructs that input *exactly*. A search that optimises the wrong
// reference is not a quality mode; it is a bug with a good error bar. Hence
// `Nvfp4SearchCtx` carrying two views of the input, and every SSE in this
// section being computed against the raw one.

/// Number of alternating (block search → `s_T` refit) rounds.
///
/// A **fixed** count, not "iterate until convergence". A convergence test on
/// a float objective is not reproducible across machines, and the objective is
/// flat along powers of two anyway, so there is nothing to converge *to*.
const QUALITY_ALTERNATIONS: usize = 3;

/// Half-width of the E4M3 byte-code window searched around the anchor.
///
/// E4M3 has a 3-bit mantissa, so 8 codes per octave: ±4 codes is about
/// ±0.5 octave, i.e. a factor of ~1.41 either way. Wide enough to cross the
/// saturation boundary in either direction, narrow enough that "bounded
/// neighbourhood of the reference scale" is a true statement.
const QUALITY_CODE_WINDOW: u8 = 4;

/// Factor by which the refit per-tensor scale may leave the anchor `s_T⁰`.
const QUALITY_PTS_CLAMP: f32 = 4.0;

/// Highest finite E4M3 code (`0x7F` is the NaN pattern and is never a scale).
const FP8_E4M3_MAX_CODE: u8 = 0x7E;

/// Padded-grid geometry plus the two views of the input the search needs.
///
/// Every round of the search takes `&Nvfp4SearchCtx` rather than a handful of
/// loose numbers, which keeps each helper's arity under clippy's threshold and
/// makes it impossible for two helpers to disagree about the block layout.
///
/// # Why two views of the input
///
/// `wp` is bf16-rounded: that is what the E2M1 codes must be chosen from, since
/// rounding an input through bf16 is a stage of the reference pipeline and
/// changing it would break parity.
///
/// `w_raw` is the **pre-bf16** tensor, zero-padded but otherwise untouched, and
/// it is what the objective is measured against. This is not a refinement, it
/// is a correctness requirement, and the reference already insists on it:
/// `quantize_nvfp4_weight` derives `per_tensor_scale` from the F32 amax
/// *before* the cast (see `pts_uses_pre_bf16_amax`). Scoring against `wp`
/// instead would let the search chase the bf16 rounding error — for a 1×1
/// tensor of `0.27318907` it "improves" the score to a perfect 0 by
/// reconstructing the rounded `0.2734375`, while the reconstruction the user
/// actually receives is off by 2.5e-4 in the wrong direction. A search that
/// optimises the wrong reference is not a quality mode, it is a bug with a
/// good error bar.
struct Nvfp4SearchCtx<'a> {
    /// Row-major `(m_pad, n_pad)`, bf16-rounded, zero-padded. The encode input.
    wp: &'a [f32],
    /// Row-major `(m_pad, n_pad)`, zero-padded, NOT bf16-rounded. The objective.
    w_raw: &'a [f32],
    /// Padded row count.
    m_pad: usize,
    /// Blocks per padded row (`n_pad / BLOCK_SIZE`).
    num_blocks: usize,
    /// Padded row stride in elements.
    n_pad: usize,
}

impl<'a> Nvfp4SearchCtx<'a> {
    /// Number of 16-element blocks in the padded grid.
    #[inline]
    fn total_blocks(&self) -> usize {
        self.m_pad * self.num_blocks
    }

    /// Start index of the `bi`-th block of the padded grid, in flat row-major
    /// block order.
    ///
    /// Flat order is this crate's determinism contract for float reductions:
    /// the parallel rounds fill their result `Vec` by index, and the `s_T`
    /// refit folds that `Vec` in ascending `bi`.
    #[inline]
    fn block_start(&self, bi: usize) -> usize {
        let row = bi / self.num_blocks;
        let col = (bi % self.num_blocks) * BLOCK_SIZE;
        row * self.n_pad + col
    }

    /// The `bi`-th block of the bf16-rounded grid — the encode input.
    #[inline]
    fn block(&self, bi: usize) -> &'a [f32] {
        let s = self.block_start(bi);
        &self.wp[s..s + BLOCK_SIZE]
    }

    /// The `bi`-th block of the pre-bf16 grid — the objective input.
    #[inline]
    fn block_raw(&self, bi: usize) -> &'a [f32] {
        let s = self.block_start(bi);
        &self.w_raw[s..s + BLOCK_SIZE]
    }
}

/// One block's chosen E4M3 scale byte, its codes, and the SSE they imply.
#[derive(Debug, Clone, Copy)]
struct Nvfp4QualityBlock {
    /// E4M3 block-scale byte to emit.
    scale_byte: u8,
    /// E2M1 code per element, in block order (not yet nibble-packed).
    codes: [u8; BLOCK_SIZE],
    /// Reconstruction SSE of `codes`, measured in f64 against the exact f32
    /// value `dequantize_nvfp4` emits. Recomputed whenever `s_T` moves, so it
    /// is always the error of the *current* pair, never a stale one.
    sse: f64,
    /// Inclusive low end of the anchored search window. Fixed at setup from
    /// the reference scale and carried unchanged, so the window never slides.
    lo: u8,
    /// Inclusive high end of the anchored search window.
    hi: u8,
    /// Blocks excluded from the search (reference total is zero or
    /// non-finite). Their codes are still re-encoded when `s_T` moves.
    frozen: bool,
}

/// Encode one 16-element block at product scale `total`, returning its
/// reconstruction SSE in f64.
///
/// `enc` is the bf16-rounded block (the codes are chosen from it) and `obj` is
/// the pre-bf16 block (the error is measured against it) — see
/// [`Nvfp4SearchCtx`] for why those are two different arrays.
///
/// The reconstructed value is formed as `E2M1_LUT[code] * total` in f32, the
/// same expression `dequantize_nvfp4` evaluates, and only then widened to f64
/// for the error sum. Scoring in f64 while emitting f32 means the ranking is
/// done on the values that will actually be written, not on idealised ones.
///
/// `total == 0.0` takes the reference's zero-block path: all 16 codes become
/// +0.0 rather than `encode_element(0/0)`'s NaN, so the block reconstructs to
/// +0.0 and the SSE is `Σ x²`.
fn nvfp4_encode_block(enc: &[f32], obj: &[f32], total: f32, codes: &mut [u8; BLOCK_SIZE]) -> f64 {
    debug_assert_eq!(enc.len(), obj.len());
    if total == 0.0 {
        codes.fill(0);
        return obj
            .iter()
            .map(|&v| {
                let d = v as f64;
                d * d
            })
            .sum();
    }
    let mut sse = 0.0f64;
    for i in 0..enc.len() {
        let code = encode_element(enc[i], total);
        codes[i] = code;
        let rec = (E2M1_LUT[code as usize] * total) as f64;
        let d = obj[i] as f64 - rec;
        sse += d * d;
    }
    sse
}

/// Sum a per-block SSE table in ascending block index.
///
/// f64 addition is not associative, so the fold order is part of the contract.
/// `slice::iter().sum()` walks the slice front to back, which is the required
/// order; `par_iter().sum()` and any `HashMap` walk do not guarantee it.
#[inline]
fn nvfp4_total_sse(blocks: &[Nvfp4QualityBlock]) -> f64 {
    blocks.iter().map(|b| b.sse).sum()
}

/// The anchored search window around a reference scale byte.
///
/// Positive E4M3 bytes are numerically monotone in the byte value, so a
/// contiguous code range is a contiguous range of representable scales. The
/// high end is capped at `0x7E` because `0x7F` is the NaN pattern, which is not
/// a scale the hardware could ever store.
#[inline]
fn nvfp4_scale_window(anchor: u8) -> (u8, u8) {
    (
        anchor.saturating_sub(QUALITY_CODE_WINDOW),
        anchor
            .saturating_add(QUALITY_CODE_WINDOW)
            .min(FP8_E4M3_MAX_CODE),
    )
}

/// The reference (parity) state of every block, computed exactly as
/// `quantize_nvfp4_weight` computes it — same expressions, same order, same
/// IEEE divisions — so that this table *is* the base path and can be emitted
/// verbatim as the fallback.
fn nvfp4_reference_blocks(ctx: &Nvfp4SearchCtx, pts: f32) -> Vec<Nvfp4QualityBlock> {
    let mut codes = [0u8; BLOCK_SIZE];
    let mut out: Vec<Nvfp4QualityBlock> = Vec::with_capacity(ctx.total_blocks());
    for bi in 0..ctx.total_blocks() {
        let block = ctx.block(bi);
        let block_max = amax_abs(block);
        let block_scale = block_max / FP4_MAX;
        let scaled = block_scale / pts;
        let scaled_clamped = clamp_max(scaled, FP8_MAX);
        let scale_byte = f32_to_fp8_e4m3_bits(scaled_clamped);
        let total = pts * fp8_e4m3_bits_to_f32(scale_byte);
        // Codes and `block_max` come from the bf16-rounded block (parity); only
        // the error is scored against the pre-bf16 one.
        let sse = nvfp4_encode_block(block, ctx.block_raw(bi), total, &mut codes);
        let (lo, hi) = nvfp4_scale_window(scale_byte);
        out.push(Nvfp4QualityBlock {
            scale_byte,
            codes,
            sse,
            lo,
            hi,
            // A zero or non-finite reference total is the zero-block /
            // NaN-input case. Searching it could only turn a zero block into a
            // nonzero one, so it stays frozen.
            frozen: total == 0.0 || !total.is_finite(),
        });
    }
    out
}

/// One round of the per-block search at a fixed per-tensor scale `pts`.
///
/// Every block is independent here: it reads its own 16 values, the shared
/// `pts`, and its own fixed window. Rayon collects by index, so the returned
/// `Vec` is in ascending block order regardless of thread scheduling.
///
/// The incumbent for each block is `cur[bi]` **re-scored at `pts`**, and the
/// window is scanned in ascending byte order with a strict `<` comparison. A
/// candidate therefore has to be strictly better to displace anything, and a
/// block whose window contains no improvement keeps the byte it came in with.
fn nvfp4_quality_round(
    ctx: &Nvfp4SearchCtx,
    pts: f32,
    cur: &[Nvfp4QualityBlock],
) -> Vec<Nvfp4QualityBlock> {
    (0..ctx.total_blocks())
        .into_par_iter()
        .map(|bi| {
            // Scratch lives inside the closure: rayon calls this `Fn`, so a
            // shared `&mut` buffer would not compile — and a shared one would
            // be a data race if it did.
            let mut codes = [0u8; BLOCK_SIZE];
            let block = ctx.block(bi);
            let block_raw = ctx.block_raw(bi);
            let prev = cur[bi];

            let incumbent_total = pts * fp8_e4m3_bits_to_f32(prev.scale_byte);
            let incumbent_sse = nvfp4_encode_block(block, block_raw, incumbent_total, &mut codes);
            let mut best = Nvfp4QualityBlock {
                scale_byte: prev.scale_byte,
                codes,
                sse: incumbent_sse,
                ..prev
            };
            if prev.frozen {
                return best;
            }

            for byte in prev.lo..=prev.hi {
                if byte == prev.scale_byte {
                    continue;
                }
                let total = pts * fp8_e4m3_bits_to_f32(byte);
                let sse = nvfp4_encode_block(block, block_raw, total, &mut codes);
                if sse < best.sse {
                    best = Nvfp4QualityBlock {
                        scale_byte: byte,
                        codes,
                        sse,
                        ..prev
                    };
                }
            }
            best
        })
        .collect()
}

/// Re-encode every block with its scale byte held fixed and `s_T` moved to
/// `pts`. This is what makes the alternation real: the `s_T` refit is optimal
/// for the *codes it was derived from*, and moving `s_T` re-rounds every
/// element, so the codes it actually produces have to be measured rather than
/// assumed.
fn nvfp4_rencode_at(
    ctx: &Nvfp4SearchCtx,
    pts: f32,
    blocks: &[Nvfp4QualityBlock],
) -> Vec<Nvfp4QualityBlock> {
    blocks
        .par_iter()
        .enumerate()
        .map(|(bi, blk)| {
            let mut codes = [0u8; BLOCK_SIZE];
            let total = pts * fp8_e4m3_bits_to_f32(blk.scale_byte);
            let sse = nvfp4_encode_block(ctx.block(bi), ctx.block_raw(bi), total, &mut codes);
            Nvfp4QualityBlock { codes, sse, ..*blk }
        })
        .collect()
}

/// Closed-form least-squares refit of the per-tensor scale.
///
/// With `g_b` the block's E4M3 scale and `c` the E2M1 code values, the
/// reconstruction is `x̂ = c · g_b · s_T`, so minimising `Σ (x − c·g_b·s_T)²`
/// over `s_T` gives
///
/// ```text
/// s_T* = Σ_b Σ_i (cᵢ·g_b·xᵢ) / Σ_b Σ_i (cᵢ·g_b)²
/// ```
///
/// which reduces to the familiar `Σ(cᵢxᵢ) / Σ(cᵢ²)` when the block scale is
/// factored out (single block, or `g ≡ 1`). Accumulated in f64 in ascending
/// block index.
///
/// Returns `None` when there is nothing to fit — an all-zero tensor, or one
/// whose codes are all zero — so the caller keeps `s_T` rather than dividing by
/// a zero denominator.
fn nvfp4_closed_form_pts(ctx: &Nvfp4SearchCtx, blocks: &[Nvfp4QualityBlock]) -> Option<f32> {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (bi, blk) in blocks.iter().enumerate() {
        let g = fp8_e4m3_bits_to_f32(blk.scale_byte) as f64;
        // The regression is against the pre-bf16 values, matching the SSE the
        // candidate states are scored by. Fitting the rounded values here while
        // scoring the raw ones would make the refit optimise a different
        // objective than the one that judges it.
        let block = ctx.block_raw(bi);
        for (i, &code) in blk.codes.iter().enumerate() {
            let c = E2M1_LUT[code as usize] as f64 * g;
            num += c * block[i] as f64;
            den += c * c;
        }
    }
    if den <= 0.0 || !num.is_finite() || !den.is_finite() {
        return None;
    }
    let pts = (num / den) as f32;
    if pts.is_finite() && pts > 0.0 {
        Some(pts)
    } else {
        None
    }
}

/// Confine a refit per-tensor scale to the anchored neighbourhood of `s_T⁰`.
///
/// The second of the two anchor guards. The first bounds the `s_G` choice to a
/// fixed byte window; this one bounds the unidentifiable `s_T ↔ s_G` direction
/// that the flat objective leaves open. The bounds are exact powers of two, so
/// the clamp introduces no rounding of its own.
///
/// A non-positive or non-finite anchor (an all-zero or NaN tensor) is returned
/// unchanged: there is no neighbourhood to stay inside.
#[inline]
fn nvfp4_clamp_pts(pts: f32, anchor: f32) -> f32 {
    if !(anchor > 0.0) || !anchor.is_finite() {
        return anchor;
    }
    pts.clamp(anchor / QUALITY_PTS_CLAMP, anchor * QUALITY_PTS_CLAMP)
}

/// Quantize a 2D weight tensor (row-major f32) to NVFP4 with the anchored
/// alternating L2 scale search — the `Quality::Nvfp4L2ScaleSearch` kernel.
///
/// **Not byte-exact with the reference, by design.** This is the opt-in
/// counterpart of [`quantize_nvfp4_weight`], which is unchanged and remains the
/// parity path. Every stage other than scale *selection* is identical: the same
/// pre-bf16 `per_tensor_scale` computation, the same bf16 input rounding, the
/// same zero padding, the same E2M1 encoder, the same hi-first packing and the
/// same cuBLAS `to_blocked` scale layout.
///
/// # Monotonicity
///
/// The emitted state is the best of {reference state} ∪ {every state visited},
/// compared on summed f64 SSE, and the reference state is always a candidate.
/// A search that finds nothing therefore reproduces [`quantize_nvfp4_weight`]
/// **byte for byte**, and the result is never worse. This is a floor, not a
/// hope: it is what makes `scale_search_never_increases_rel_l2` a property
/// rather than an observation.
///
/// # Determinism
///
/// A fixed `QUALITY_ALTERNATIONS` rounds, a fixed reduction order, and a
/// byte-indexed parallel map. Two calls in one process, or in two processes on
/// two machines, emit the same bytes.
pub fn quantize_nvfp4_weight_quality(w: &[f32], m: usize, n: usize) -> Nvfp4QuantResult {
    debug_assert_eq!(w.len(), m * n);

    // Stage 0, identical to the parity path: per-tensor scale from the F32 amax
    // BEFORE the bf16 cast. This value is the anchor for the whole search.
    let amax = amax_abs(w);
    let pts_anchor = amax / PTS_DIVISOR;

    // Stages 1-2, identical: pad to multiples of 16, rounding through bf16.
    let m_pad = roundup(m, BLOCK_SIZE);
    let n_pad = roundup(n, BLOCK_SIZE);
    let num_blocks = n_pad / BLOCK_SIZE;
    let mut wp = vec![0.0f32; m_pad * n_pad];
    // Same padding, but WITHOUT the bf16 cast: this is the tensor the search
    // measures its error against, for the reasons given on `Nvfp4SearchCtx`.
    let mut w_raw = vec![0.0f32; m_pad * n_pad];
    for r in 0..m {
        for c in 0..n {
            let v = w[r * n + c];
            wp[r * n_pad + c] = bf16_bits_to_f32(f32_to_bf16_bits(v));
            w_raw[r * n_pad + c] = v;
        }
    }

    let ctx = Nvfp4SearchCtx {
        wp: &wp,
        w_raw: &w_raw,
        m_pad,
        num_blocks,
        n_pad,
    };

    // The reference state, which is also the fallback and the incumbent to beat.
    let reference = nvfp4_reference_blocks(&ctx, pts_anchor);
    let reference_sse = nvfp4_total_sse(&reference);

    let mut best_pts = pts_anchor;
    let mut best_blocks = reference.clone();
    let mut best_sse = reference_sse;

    // A non-finite reference SSE means NaN/inf inputs; there is no meaningful
    // ordering to optimise against, so the reference state is emitted as-is.
    if reference_sse.is_finite() {
        let mut cur_pts = pts_anchor;
        let mut cur_blocks = reference;
        for _ in 0..QUALITY_ALTERNATIONS {
            // (A) per-block byte search at the current s_T
            let searched = nvfp4_quality_round(&ctx, cur_pts, &cur_blocks);
            let searched_sse = nvfp4_total_sse(&searched);
            if searched_sse < best_sse {
                best_sse = searched_sse;
                best_pts = cur_pts;
                best_blocks.clone_from(&searched);
            }

            // (B) closed-form least-squares refit of s_T for the chosen codes
            let Some(refit) = nvfp4_closed_form_pts(&ctx, &searched) else {
                break;
            };
            let next_pts = nvfp4_clamp_pts(refit, pts_anchor);
            if next_pts.to_bits() == cur_pts.to_bits() {
                // Refit is a no-op; further rounds would repeat this one.
                break;
            }

            // (C) moving s_T re-rounds every element, so re-encode and measure
            // the state that would actually be emitted.
            let reencoded = nvfp4_rencode_at(&ctx, next_pts, &searched);
            let reencoded_sse = nvfp4_total_sse(&reencoded);
            if reencoded_sse < best_sse {
                best_sse = reencoded_sse;
                best_pts = next_pts;
                best_blocks.clone_from(&reencoded);
            }

            cur_pts = next_pts;
            cur_blocks = reencoded;
        }
    }

    // Stages 3-5, identical to the parity path, driven by the searched table.
    // `best_blocks`' codes were encoded at `best_pts` by construction, so the
    // emitted bytes and the scored error describe the same state.
    let mut scale_rows = Vec::with_capacity(m_pad * num_blocks);
    let mut qdata = Vec::with_capacity(m_pad * (n_pad / 2));
    for bi in 0..m_pad * num_blocks {
        let blk = &best_blocks[bi];
        scale_rows.push(blk.scale_byte);
        let total = best_pts * fp8_e4m3_bits_to_f32(blk.scale_byte);
        if total == 0.0 {
            // Zero block: torch.where forces +0.0 for all 16 elements.
            qdata.extend_from_slice(&[0u8; BLOCK_SIZE / 2]);
        } else {
            for p in 0..BLOCK_SIZE / 2 {
                qdata.push(blk.codes[2 * p] << 4 | blk.codes[2 * p + 1]);
            }
        }
    }

    let (scale, scale_shape) = to_blocked_u8(&scale_rows, m_pad, num_blocks);

    Nvfp4QuantResult {
        qdata,
        qdata_shape: vec![m_pad as u64, (n_pad / 2) as u64],
        scale,
        scale_shape,
        per_tensor_scale: best_pts,
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

    // --- Phase B.3: dequantize_nvfp4 --------------------------------------- //

    #[test]
    fn dequantize_nvfp4_hand_computed() {
        // Reuse the single_block_known_codes fixture: pts = 1.0, scale byte
        // 0x7E (448) → total = 1.0 × 448 = 448. Codes 7 1 2 3 4 5 6 7 |
        // 9 10 11 12 13 14 15 0 → LUT values × 448.
        let vals = [
            2688.0, 224.0, 448.0, 672.0, 896.0, 1344.0, 1792.0, 2688.0, -224.0, -448.0, -672.0,
            -896.0, -1344.0, -1792.0, -2688.0, 0.0,
        ];
        let r = quantize_nvfp4_weight(&vals, 1, 16);
        let dq = dequantize_nvfp4(&r, 1, 16);
        let expect: [f32; 16] = [
            2688.0, 224.0, 448.0, 672.0, 896.0, 1344.0, 1792.0, 2688.0, -224.0, -448.0, -672.0,
            -896.0, -1344.0, -1792.0, -2688.0, 0.0,
        ];
        assert_eq!(dq.len(), 16);
        for (i, (&o, &e)) in dq.iter().zip(expect.iter()).enumerate() {
            assert_eq!(o.to_bits(), e.to_bits(), "dq[{i}] = {o}, expected {e}");
        }
    }

    #[test]
    fn dequantize_nvfp4_total_scale_order() {
        // total = per_tensor_scale × block_scale_f32 (reference association,
        // plan §3.5), then value × total. Non-dyadic pts exercises the product:
        // w = [3.0, 0×15] → amax 3.0 → pts = 3/2688 = 1/896 (non-dyadic).
        // block_scale = 3/6 = 0.5; scaled = 0.5/(1/896) = 448 → byte 0x7E →
        // scaled_f32 = 448; total = (1/896) × 448 = 0.5. Element 3.0/0.5 = 6
        // → code 7 → LUT 6.0; dequant = 6.0 × 0.5 = 3.0 exactly.
        let mut w = vec![0.0f32; 16];
        w[0] = 3.0;
        let r = quantize_nvfp4_weight(&w, 1, 16);
        assert_eq!(r.per_tensor_scale, 3.0 / 2688.0);
        let dq = dequantize_nvfp4(&r, 1, 16);
        assert_eq!(
            dq[0].to_bits(),
            3.0f32.to_bits(),
            "value × (pts × scale) = 3.0"
        );
        // Remaining elements are +0.0 (code 0 → LUT[0] = +0.0 × total).
        for &v in &dq[1..] {
            assert_eq!(v, 0.0);
            assert!(v.is_sign_positive());
        }
    }

    #[test]
    fn dequantize_nvfp4_crops_padding() {
        // 3×5 all-ones → padded 16×16. Dequant must crop back to 3×5, all 1.0
        // (1.0/(pts×448) ≈ 6 → code 7 → LUT 6.0; total = pts×448; 6×total/6
        // reconstructs 1.0 within the E2M1 grid — here exactly, since the
        // block max lands on the LUT).
        let w = vec![1.0f32; 3 * 5];
        let r = quantize_nvfp4_weight(&w, 3, 5);
        assert_eq!(r.qdata_shape, vec![16, 8]);
        let dq = dequantize_nvfp4(&r, 3, 5);
        assert_eq!(dq.len(), 15);
        // All-ones block: every element is the block max → code 7 → LUT 6.0;
        // dequant = 6.0 × total where total = pts × 448. Verify bounded and
        // close to 1.0 (E2M1 grid spacing at this magnitude).
        for &v in &dq {
            assert!(v.is_finite());
            assert!((v - 1.0).abs() < 0.25, "dq {v} not near 1.0");
        }
    }

    #[test]
    fn dequantize_nvfp4_zero_block_is_zero() {
        // 1×32: first block nonzero, second all zeros. Zero block → scale
        // byte 0x00 → total 0.0 → dequant +0.0 (codes all 0 → LUT[0] = +0.0).
        let mut w = vec![0.0f32; 32];
        w[0] = 448.0;
        let r = quantize_nvfp4_weight(&w, 1, 32);
        let dq = dequantize_nvfp4(&r, 1, 32);
        assert!(dq[0] > 0.0, "nonzero block element must be positive");
        for &v in &dq[16..] {
            assert_eq!(v, 0.0);
            assert!(v.is_sign_positive(), "zero block must dequant to +0.0");
        }
    }

    #[test]
    fn dequant_roundtrip_bounded_nvfp4() {
        // Phase B.4 (NVFP4 arm): quantize a random matrix, dequantize, assert
        // finite + bounded, and that the reconstruction error is bounded by
        // the E2M1 grid.
        //
        // NOTE: unlike MXFP8, NVFP4 dequant values are NOT an exact fixed
        // point of quantize→dequantize — the per-tensor scale is non-dyadic,
        // so dequant values are not bf16-representable, and re-quantization's
        // bf16 input rounding perturbs them. The meaningful B.4 property here
        // is bounded reconstruction error: the E2M1 grid {0,.5,1,1.5,2,3,4,6}
        // has max spacing 2 over max value 6, so the worst-case relative error
        // is ~1/6 ≈ 0.167 (measured 0.1674 on this fixture). We assert a
        // generous 0.25·amax bound.
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

        let r = quantize_nvfp4_weight(&w, m, n);
        let dq = dequantize_nvfp4(&r, m, n);
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
        // Bounded reconstruction error (E2M1 grid).
        let mut max_err = 0.0f32;
        for (&a, &b) in w.iter().zip(dq.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err <= amax * 0.25,
            "reconstruction error {max_err} exceeds 0.25·amax = {}",
            amax * 0.25
        );
    }
}

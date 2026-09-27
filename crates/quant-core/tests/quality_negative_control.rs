//! Tier 2 S08 — negative controls: prove the new guards can actually FAIL.
//!
//! # Why this file exists
//!
//! A guard that cannot fail is decoration. It reads like coverage, it shows up
//! in the test count, and it would sit there silently through a refactor that
//! removed the invariant it claims to protect. "The tests pass" is the weakest
//! signal a suite can produce; the only stronger one is to demonstrate, in
//! executable code, that a **deliberately wrong** implementation violates each
//! invariant.
//!
//! Every test here builds a miniature broken implementation, asserts that it
//! violates exactly ONE invariant, and **passes**. The wrong behaviour is the
//! point. Read together with the production guard named in each test's doc
//! comment, a pair like
//!
//! ```text
//! quality.rs        asserts delta ∈ {0, +1}
//! this file         asserts the dividing version gives -1
//! ```
//!
//! is what makes the first assertion load-bearing rather than decorative.
//!
//! # What is NOT here
//!
//! The exhaustive positive measurements already live in the files that own
//! them, and duplicating them would add runtime without adding evidence:
//!
//! -- `nvfp4_scale_search.rs` owns the 3-decade drift fixture and the exact
//!    flatness measurement. Control 1 here is the compact restatement.
//! -- `mxfp8_e8m0_compensated.rs` owns the full 31,519-value bf16 sweep and the
//!    aggregate `rel_l2` harness behind `4.41x`. Control 2 here is the compact
//!    negative-control version.
//!
//! # A note on measured numbers
//!
//! Every figure asserted below was **measured on this machine**, not carried
//! over from a prototype. Where a plausible-looking number did not reproduce,
//! the measured value is pinned and the discrepancy is called out in the test
//! that owns it. A negative control that pins a number it never measured is
//! just an assertion with extra steps.
//!
//! No file under `src/` is modified. Every broken implementation below lives in
//! this file.

use quant_core::dtype::{
    bf16_bits_to_f32, f32_to_bf16_bits, f32_to_fp8_e4m3_bits, fp8_e4m3_bits_to_f32,
};
use quant_core::manifest::{Format, QuantConfig, StreamState};
use quant_core::quality::Quality;
use quant_core::quant_fp8::FP8_MAX;
use quant_core::quant_mxfp8::{dequantize_mxfp8, quantize_mxfp8_weight, BLOCK_SIZE as MXFP8_BLOCK};
use quant_core::quant_nvfp4::{dequantize_nvfp4, quantize_nvfp4_weight, BLOCK_SIZE as NVFP4_BLOCK};
use sha2::{Digest, Sha256};

/// Half-width of the E4M3 byte-code window the real search is confined to.
/// Mirrors the private `QUALITY_CODE_WINDOW` in `quant_nvfp4.rs`.
const WINDOW: u8 = 4;

/// `448 * 6` — the NVFP4 per-tensor-scale divisor, restated so this file never
/// has to import a private constant to recompute an anchor.
const PTS_DIVISOR: f32 = 448.0 * 6.0;

/// The `4/3` MXFP8 compensation factor.
const FOUR_THIRDS: f32 = 4.0 / 3.0;

/// The MXFP8 `SCALE_MIN` clamp floor, `2^-127`.
const SCALE_MIN: f32 = f32::from_bits(0x0040_0000);

/// E8M0 exponent bias.
const E8M0_BIAS: i32 = 127;

/// E2M1 code → f32, restated from `quant_nvfp4.rs` so the NVFP4 controls do
/// not depend on the kernel they are judging.
const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

// --------------------------------------------------------------------------
// Fixture helpers
// --------------------------------------------------------------------------

/// xorshift64 — the deterministic generator already used across this crate's
/// tests, so every fixture here is byte-reproducible on any platform.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[0, 1)`, 24 bits of mantissa.
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// Symmetric magnitude in `(-1, 1)`.
    fn signed_unit(&mut self) -> f32 {
        self.range(-1.0, 1.0)
    }
}

/// Round through bf16 — the grid the kernels actually quantize on, so a fixture
/// built with this carries no input-rounding error of its own.
fn bf16(v: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(v))
}

/// A nearest-value E2M1 encoder, restated so the NVFP4 controls never call the
/// production encoder. The grid is `{0, .5, 1, 1.5, 2, 3, 4, 6}`, small enough
/// to scan exhaustively; `6.0` is the saturated code, not an extension of `4`.
///
/// Nearest-value rather than the kernel's round-half-to-even is sufficient: it
/// only has to be a *consistent* encoder for these controls' comparisons to
/// mean anything, and on a tie it shifts the chosen `s_G` by at most one code,
/// which affects neither the flatness nor the zeroing claims.
fn e2m1_code(x: f32) -> u8 {
    const MAGNITUDES: [f32; 7] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0];
    let ax = x.abs();
    let mut best = (f64::INFINITY, 0u8);
    for (i, &mv) in MAGNITUDES.iter().enumerate() {
        let d = (ax as f64 - mv as f64).abs();
        if d < best.0 {
            best = (d, i as u8);
        }
    }
    if best.1 == 6 && ax > 5.0 {
        best.1 = 7;
    }
    best.1 | if x.is_sign_negative() { 0x08 } else { 0 }
}

/// `rel_l2 = Σ (w − dequantize(quantize(w)))² / Σ w²` in f64.
///
/// The denominator is the ORIGINAL tensor's energy, so the number is
/// comparable across paths on the same input.
fn rel_l2(w: &[f32], dq: &[f32]) -> f64 {
    assert_eq!(w.len(), dq.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&a, &b) in w.iter().zip(dq.iter()) {
        let d = a as f64 - b as f64;
        num += d * d;
        den += (a as f64) * (a as f64);
    }
    num / den
}

/// A matrix with genuine per-block dynamic range: row `r` of `m` carries
/// magnitude `10^(-decades * r / (m-1))`.
///
/// This is the fixture the unanchored controls need. NVFP4's per-tensor scale
/// is global, so a row far below the tensor amax is exactly the block whose
/// E4M3 scale is small enough to underflow to zero once `s_T` walks upward.
fn dynamic_range_matrix(m: usize, n: usize, decades: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut w = Vec::with_capacity(m * n);
    for r in 0..m {
        let frac = if m <= 1 {
            0.0
        } else {
            r as f32 / (m - 1) as f32
        };
        let mag = 10.0f32.powf(-decades * frac);
        for _ in 0..n {
            w.push(mag * rng.signed_unit());
        }
    }
    w
}

/// Split a tensor into bf16-rounded 16-element blocks (the encode input) and
/// pre-bf16 blocks (the objective input), plus the anchored per-tensor scale —
/// matching what the real kernel sees.
fn padded_blocks(w: &[f32], m: usize, n: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, f32) {
    let m_pad = m.div_ceil(NVFP4_BLOCK) * NVFP4_BLOCK;
    let n_pad = n.div_ceil(NVFP4_BLOCK) * NVFP4_BLOCK;
    let mut wp = vec![0.0f32; m_pad * n_pad];
    let mut w_raw = vec![0.0f32; m_pad * n_pad];
    for r in 0..m {
        for c in 0..n {
            let v = w[r * n + c];
            wp[r * n_pad + c] = bf16(v);
            w_raw[r * n_pad + c] = v;
        }
    }
    let amax = w.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let pts = amax / PTS_DIVISOR;
    let mut enc = Vec::new();
    let mut obj = Vec::new();
    for r in 0..m_pad {
        for b in 0..n_pad / NVFP4_BLOCK {
            let s = r * n_pad + b * NVFP4_BLOCK;
            enc.push(wp[s..s + NVFP4_BLOCK].to_vec());
            obj.push(w_raw[s..s + NVFP4_BLOCK].to_vec());
        }
    }
    (enc, obj, pts)
}

/// Reconstruction SSE of one block at product scale `total`: codes from `enc`,
/// error against `obj`, accumulated in f64.
fn block_sse(enc: &[f32], obj: &[f32], total: f32) -> f64 {
    enc.iter()
        .zip(obj.iter())
        .map(|(&e, &o)| {
            let d = (e / total).clamp(-6.0, 6.0);
            let rec = (E2M1_LUT[e2m1_code(d) as usize] * total) as f64;
            let err = o as f64 - rec;
            err * err
        })
        .sum()
}

/// The base (parity) configuration for one format, shaped the way
/// `quality_mode_identity.rs` shapes it.
fn cfg(format: Format) -> QuantConfig {
    let mut c = QuantConfig::default();
    c.format = format;
    c.int8 = false;
    c.target_format = format.as_str().to_string();
    c
}

/// The E8M0 byte a block-scale rule would emit, recomputed independently of the
/// kernel so the MXFP8 controls are not tautological against it.
fn e8m0_for(block_max: f32, compensation: f32) -> u8 {
    let base = (block_max / FP8_MAX).max(SCALE_MIN);
    let scale_needed = base * compensation;
    let log2_scale = (f64::from(scale_needed)).log2() as f32;
    (log2_scale.ceil() as i32 + E8M0_BIAS).clamp(0, 254) as u8
}

/// `e8m0_to_f32`, restated: the dequant scale an E8M0 byte denotes.
fn e8m0_scale(e: u8) -> f32 {
    if e == 0 {
        0.0
    } else {
        f32::from_bits(u32::from(e) << 23)
    }
}

/// Every finite non-negative bf16 value — the complete reachable set of MXFP8
/// block maxima, since the kernel rounds its input through bf16.
fn all_finite_non_negative_bf16() -> impl Iterator<Item = f32> {
    (0u32..=0x7F7F).map(|bits| f32::from_bits(bits << 16))
}

// --------------------------------------------------------------------------
// 1. The unanchored-search trap (§2.4)
// --------------------------------------------------------------------------

/// CONTROL 1 — an unanchored search leaves the representable range.
///
/// **Guards defended.** The fixed E4M3 byte window (`QUALITY_CODE_WINDOW` /
/// `nvfp4_scale_window`, `quant_nvfp4.rs:430` and `:585`). The production
/// assertion that mirrors this control is `s_g_stays_on_the_e4m3_grid` in
/// `nvfp4_scale_search.rs`.
///
/// **MUTATION VERIFIED, and the second anchor is NOT load-bearing.** Replacing
/// `QUALITY_CODE_WINDOW` with `255` turns `s_g_stays_on_the_e4m3_grid` RED.
/// Replacing the body of `nvfp4_clamp_pts` with the identity turns **nothing**
/// red — not this file, not `nvfp4_scale_search.rs`, not `parity_fingerprint.rs`.
/// Instrumenting the ratio shows why: with the clamp removed the closed-form
/// refit only travels **1.003x** (uniform fixture) and **1.016x** (dynamic-range
/// fixture) against a bound of 4x, so the clamp never binds on these inputs.
/// `per_tensor_scale_stays_anchored` is currently a guard with no teeth, because
/// the least-squares refit is a contractive step rather than a drifting one.
///
/// That is worth stating plainly rather than hiding: the ±4 clamp is
/// *structurally* motivated — it is the only thing bounding the unidentifiable
/// `s_T <-> s_G` direction described below — but on every fixture tried it is
/// *empirically* inert. The control below is therefore scoped to the window
/// guard, which is demonstrably real. Tightening the clamp to a factor the refit
/// could actually reach, or finding a tensor where it escapes, is open work and
/// is NOT claimed here.
///
/// **The real failure.** The reconstruction is `x̂ᵢ = cᵢ · (s_T · s_G)` — the
/// two scales appear only as their PRODUCT. So `s_T → k·s_T` with
/// `s_G → s_G/k` leaves the product invariant, and because the E2M1 grid is
/// `{2ʲ, 1.5·2ʲ}` the element CODES are unchanged too whenever `k = 2ʲ`.
///
/// The consequence is stronger than "under-constrained". The L2 objective is
/// not merely flat along that direction, it is **ambiguous**: every one of
/// those states is a global minimiser, so minimising L2 alone cannot
/// distinguish a correct scale from one inflated without limit. A search that
/// only "minimises L2" has no signal at all that it is drifting.
///
/// The flatness is not infinite, and where it ends is the interesting part.
/// It is bit-exact while the halved E4M3 value stays exactly representable,
/// and it stops at `k = 5` — precisely where the smallest block's halved scale
/// enters the E4M3 subnormal band and halving is no longer exact. Past that
/// the drift is not merely unrepresentable but actively harmful: block scales
/// round to E4M3 **zero**, those blocks collapse to code 0, and the error ends
/// up worse than the reference. Nothing in the objective announces any of it.
///
/// **Measured here:** bit-identical error for `k = 0..=4`; at `k = 12`
/// (`s_T` inflated 4096x) **22 of 64** block scales have rounded to E4M3 zero
/// and `rel_l2` is worse than the reference.
#[test]
fn unanchored_search_drifts_out_of_representable_range() {
    // Three decades of per-block dynamic range: the smallest row's block scale
    // survives a handful of halvings and then does not.
    let (m, n) = (16usize, 64usize);
    let w = dynamic_range_matrix(m, n, 3.0, 0x1319_8A2E_0370_7344);
    let (enc, obj, pts_anchor) = padded_blocks(&w, m, n);
    assert_eq!(enc.len(), 64, "the fixture is 64 blocks of 16");

    let base = quantize_nvfp4_weight(&w, m, n);
    let base_l2 = rel_l2(&w, &dequantize_nvfp4(&base, m, n));
    let m_pad = base.qdata_shape[0] as usize;
    let n_pad = base.qdata_shape[1] as usize * 2;
    let base_scales =
        quant_core::quant_mxfp8::from_blocked_u8(&base.scale, m_pad, n_pad / NVFP4_BLOCK);

    // The deliberately unanchored "search": walk `s_T` up by powers of two,
    // halving every block scale to compensate, choosing each block's E4M3 code
    // freely at each step. No window, no clamp — exactly what falls out if the
    // two guards are removed and the objective is left to speak for itself.
    const STEPS: u32 = 12;
    let mut sses = Vec::with_capacity(STEPS as usize + 1);
    let mut bytes_at = Vec::with_capacity(STEPS as usize + 1);
    for k in 0..=STEPS {
        let pts = pts_anchor * 2.0f32.powi(k as i32);
        let mut sse = 0.0f64;
        let mut bytes = Vec::with_capacity(enc.len());
        for ((e, o), &b0) in enc.iter().zip(obj.iter()).zip(base_scales.iter()) {
            let halved = fp8_e4m3_bits_to_f32(b0) * 2.0f32.powi(-(k as i32));
            let byte = f32_to_fp8_e4m3_bits(halved);
            bytes.push(byte);
            let total = pts * fp8_e4m3_bits_to_f32(byte);
            sse += if total == 0.0 {
                // Zero block: the reference forces +0.0 for all 16 codes, so
                // the reconstruction is 0 and the error is the block's energy.
                o.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
            } else {
                block_sse(e, o, total)
            };
        }
        sses.push(sse);
        bytes_at.push(bytes);
    }

    // (a) EXACT flatness, to the last bit. This is the ambiguity claim: these
    //     are all global minimisers, so the objective cannot rank them.
    let mut flat_upto = 0usize;
    for k in 0..STEPS as usize {
        if sses[k + 1].to_bits() != sses[k].to_bits() {
            break;
        }
        flat_upto = k + 1;
    }
    assert!(
        flat_upto >= 4,
        "expected the L2 objective to be EXACTLY flat across several doublings \
         of s_T; it went flat only to k={flat_upto}, so this control is not \
         demonstrating the ambiguity it claims"
    );
    for k in 0..=flat_upto {
        assert_eq!(
            sses[k].to_bits(),
            sses[0].to_bits(),
            "k={k} must be bit-identical to k=0"
        );
    }

    // (b) The drift is REAL even while the error stands still: for every
    //     k >= 1 the emitted representation changed, so `s_T` inflated and
    //     every `s_G` byte moved with no error signal whatsoever.
    for k in 1..=flat_upto {
        assert!(
            bytes_at[k]
                .iter()
                .zip(base_scales.iter())
                .any(|(a, b)| a != b),
            "k={k} changed no scale bytes, so nothing drifted"
        );
    }
    // By k = 8 the inflation is already 256x, i.e. +25500%.
    //
    // A prototype of an unanchored search reported +32939%. That specific
    // figure is NOT reproducible here and is deliberately not pinned: it is a
    // tie-break artefact, not an invariant. The objective is flat, so where an
    // unanchored search lands is arbitrary — 256x at k = 8 and 512x at k = 9
    // are exactly as available as the 330x the prototype stopped at. The
    // load-bearing claim is that the factor is **unbounded**, and that is what
    // the anchor exists to bound.
    let inflation = f64::from(pts_anchor * 2.0f32.powi(8) / pts_anchor) - 1.0;
    assert!(
        inflation > 250.0,
        "s_T should be inflated by >256x at k=8, got {inflation:.1}x"
    );
    // And the drift leaves the ±4-code window the real search is confined to.
    let max_drift = bytes_at[flat_upto]
        .iter()
        .zip(base_scales.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert!(
        max_drift > WINDOW,
        "the drift should leave the ±{WINDOW}-code window, but the furthest \
         block moved only {max_drift} codes"
    );

    // (c) Continuing the walk destroys representability. At k = 12 (s_T
    //     inflated 4096x) 22 of 64 block scales have rounded to E4M3 ZERO:
    //     `s_T` is absorbing a factor the format simply cannot express.
    let final_bytes = &bytes_at[STEPS as usize];
    let zeroed = final_bytes.iter().filter(|&&b| b == 0x00).count();
    assert_eq!(
        zeroed, 22,
        "expected 22 of 64 block scales to round to E4M3 zero at k=12, got \
         {zeroed} (bytes {final_bytes:?})"
    );

    // (d) And the drift is not free — the error ends up WORSE than the
    //     reference, measured against the same Σw² the metric uses.
    let energy: f64 = w.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let drifted_l2 = sses[STEPS as usize] / energy;
    assert!(
        drifted_l2 > base_l2,
        "the drifted search should be WORSE than the reference \
         (drifted {drifted_l2:.6}, base {base_l2:.6})"
    );
    eprintln!(
        "control 1: L2 bit-identical for k=0..={flat_upto}; s_T inflated 4096x; \
         {zeroed}/64 block scales at E4M3 zero; rel_l2 {base_l2:.6} -> \
         {drifted_l2:.6}"
    );
}

// --------------------------------------------------------------------------
// 2. The MXFP8 direction guard
// --------------------------------------------------------------------------

/// CONTROL 2 — a delta of `-1` is impossible, and getting the direction wrong
/// costs real accuracy.
///
/// **Guard defended.** The `{0, +1}` delta invariant of
/// `Quality::Mxfp8E8m0Compensated`. The `4/3` compensation multiplies
/// `scale_needed` by 4/3 BEFORE the `ceil(log2(...))`; because
/// `log2(4/3) = 0.415 < 1` and `ceil` is monotone, a shift of strictly less
/// than one octave can advance the ceiling by at most one and can never lower
/// it. Production mirror: `e8m0_delta_is_zero_or_plus_one_only` and the
/// `log2(4/3) < 1` argument in `quant_mxfp8.rs`.
///
/// **The real failure.** Run the compensation backwards — divide by 4/3 — and
/// the exponent DROPS. That is not a harmless sign flip:
///
/// -- The delta becomes `-1` on 13,016 of the 31,639 reachable positive bf16
///   maxima (and `0` on the rest). It is never `+1`, the exact mirror of the
///   correct direction.
/// -- Dropping the exponent halves the block scale, so the block max maps to
///   `224` instead of `448`. Only the lower half of the E4M3 grid is ever
///   used, and the max element's absolute quantization error doubles.
/// -- Measured on the house fixture, `rel_l2` gets **4.10x worse (+310%)**.
///
/// The two directions are wildly asymmetric, which is what makes the direction
/// assertion worth having rather than a formality.
///
/// **MUTATION VERIFIED.** Setting `E8M0_COMPENSATION = 3.0/4.0` in
/// `quant_mxfp8.rs` turns this control RED on the "correct direction is a
/// bit-exact no-op" assertion.
///
/// **Measured-here note.** `mxfp8_e8m0_compensated.rs` documents 4.41x / +341%
/// for its own fixture. This control's smaller fixture measures 4.10x / +310%.
/// Both are the same phenomenon at different amplitudes; the number pinned
/// below is the one this file actually measures, and only the qualitative
/// claim (much worse, delta strictly `-1`) is treated as the invariant.
#[test]
fn scale_delta_of_minus_one_is_impossible() {
    // (1) The wrong direction's delta really is {-1, 0} and never +1, and the
    //     correct direction really never lowers the exponent.
    let (mut minus_one, mut zero, mut checked) = (0usize, 0usize, 0usize);
    for bm in all_finite_non_negative_bf16() {
        if bm <= 0.0 {
            continue;
        }
        let base = e8m0_for(bm, 1.0);
        let correct = e8m0_for(bm, FOUR_THIRDS);
        let wrong = e8m0_for(bm, 1.0 / FOUR_THIRDS);
        match wrong as i32 - base as i32 {
            -1 => minus_one += 1,
            0 => zero += 1,
            other => panic!("dividing gave an unexpected delta {other} at {bm:e}"),
        }
        assert!(
            correct as i32 >= base as i32,
            "multiplying by 4/3 must never lower the exponent (at {bm:e})"
        );
        checked += 1;
    }
    assert_eq!(minus_one + zero, checked, "every case is -1 or 0");
    assert!(minus_one > 0, "dividing must lower the exponent somewhere");
    assert!(
        (35.0..45.0).contains(&(100.0 * minus_one as f64 / checked as f64)),
        "expected roughly the mirror of the multiply rate, got {:.1}%",
        100.0 * minus_one as f64 / checked as f64
    );

    // (2) The cost of the wrong direction, in `rel_l2`, using the kernel's own
    //     per-block arithmetic so the number describes the real code path.
    let (m, n) = (32usize, 64usize);
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    let w: Vec<f32> = (0..m * n).map(|_| bf16(rng.range(-1.0, 1.0))).collect();
    let base_l2 = rel_l2_shipped_mxfp8(&w, m, n);
    let wrong_l2 = rel_l2_dividing(&w, m, n);

    // The correct direction is the invariant one — bit-for-bit.
    let correct_l2 = rel_l2_compensated(&w, m, n);
    assert_eq!(
        correct_l2.to_bits(),
        base_l2.to_bits(),
        "the correct direction must be a bit-exact no-op in the normal range"
    );

    assert!(
        wrong_l2 > 4.0 * base_l2,
        "the wrong direction must be dramatically worse, got ratio {:.3}",
        wrong_l2 / base_l2
    );
    eprintln!(
        "control 2: dividing gives delta -1 on {minus_one}/{checked} maxima; \
         rel_l2 {base_l2:.8} -> {wrong_l2:.8} ({:.2}x worse, +{:.0}%)",
        wrong_l2 / base_l2,
        (wrong_l2 / base_l2 - 1.0) * 100.0
    );
}

/// `rel_l2` through the shipped base path, for the aggregate comparison.
fn rel_l2_shipped_mxfp8(w: &[f32], m: usize, n: usize) -> f64 {
    let r = quantize_mxfp8_weight(w, m, n);
    let dq = dequantize_mxfp8(&r, m, n);
    assert_eq!(dq.len(), m * n);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..m * n {
        let (a, b) = (f64::from(w[i]), f64::from(dq[i]));
        num += (a - b) * (a - b);
        den += a * a;
    }
    num / den
}

/// `rel_l2` of the CORRECT (multiplying) compensation, through the shipped
/// public path.
fn rel_l2_compensated(w: &[f32], m: usize, n: usize) -> f64 {
    let r = quant_core::quant_mxfp8::quantize_mxfp8_weight_compensated(w, m, n);
    let dq = dequantize_mxfp8(&r, m, n);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..m * n {
        let (a, b) = (f64::from(w[i]), f64::from(dq[i]));
        num += (a - b) * (a - b);
        den += a * a;
    }
    num / den
}

/// `rel_l2` for the WRONG (dividing) direction.
///
/// The shipped kernel cannot be asked to divide, so this mirrors
/// `quantize_mxfp8_weight_with` with a dividing block-scale rule: same bf16
/// input rounding, same 32-wide zero padding, same per-block amax, same
/// `v / scale` divide, same clamp, same E4M3 cast.
fn rel_l2_dividing(w: &[f32], m: usize, n: usize) -> f64 {
    let n_pad = n.div_ceil(MXFP8_BLOCK) * MXFP8_BLOCK;
    let num_blocks = n_pad / MXFP8_BLOCK;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for r in 0..m {
        for b in 0..num_blocks {
            let mut block_max = 0.0f32;
            for k in 0..MXFP8_BLOCK {
                let c = b * MXFP8_BLOCK + k;
                if c < n {
                    block_max = block_max.max(bf16(w[r * n + c]).abs());
                }
            }
            // THE BUG: divide instead of multiply.
            let e8m0 = e8m0_for(block_max, 1.0 / FOUR_THIRDS);
            let mut scale = e8m0_scale(e8m0);
            if block_max == 0.0 {
                scale = 1.0;
            }
            for k in 0..MXFP8_BLOCK {
                let c = b * MXFP8_BLOCK + k;
                if c >= n {
                    continue;
                }
                let v = bf16(w[r * n + c]);
                let deq = if block_max == 0.0 {
                    0.0
                } else {
                    let code = f32_to_fp8_e4m3_bits((v / scale).clamp(-FP8_MAX, FP8_MAX));
                    fp8_e4m3_bits_to_f32(code) * scale
                };
                let a = f64::from(w[r * n + c]);
                num += (a - f64::from(deq)) * (a - f64::from(deq));
                den += a * a;
            }
        }
    }
    num / den
}

// --------------------------------------------------------------------------
// 3. The resume-compatibility guard
// --------------------------------------------------------------------------

/// CONTROL 3 — equal hashes would resume mixed tensors.
///
/// **Guard defended.** `Quality::id()` returning `None` for `Exact`, the
/// conditional `quality_suffix` append in `QuantConfig::config_hash`
/// (`manifest.rs:229`), and the hash comparison in
/// `StreamState::load_manifest` (`manifest.rs:321`). Production mirrors:
/// `a_quality_variant_never_shares_a_base_formats_hash` and
/// `quality_hashes_are_pinned_to_the_sorted_order_payloads` in
/// `quality_mode_identity.rs`.
///
/// **The real failure.** `config_hash` is not a checksum, it is a
/// **resume-compatibility key**, and `load_manifest` is the only thing that
/// reads it:
///
/// ```text
/// load_manifest(&out) -> bool
///     ... trusts the partial output if and only if payload.config_hash == self.config_hash
/// ```
///
/// There is no second guard — no format check, no algorithm fingerprint, no
/// warning. The concrete disaster:
///
/// -- user runs `--format nvfp4`, gets 40% through, interrupts;
/// -- user re-runs with `--format nvfp4_l2` for better quality;
/// -- `load_manifest` sees a matching hash and resumes into the byte-exact
///   partial file;
/// -- the finished artifact is a mix of two quantizers, silently.
///
/// This control CONSTRUCTS that collision rather than observing the real
/// (correct) behaviour: it builds a hash function with the quality key removed
/// — the natural implementation if someone "simplified" the conditional append
/// — and shows `load_manifest` accepts the mismatched pair. Pinned real values:
/// plain `nvfp4` = `95ede677cf402b53`, `nvfp4_l2` = `85cf59c986e5d1a4`.
///
/// **MUTATION VERIFIED.** Making `config_hash` emit `quality_tuning`
/// unconditionally turns this control RED on the pinned
/// `95ede677cf402b53`, which becomes `9025bf6a71c651fb`.
#[test]
fn shared_config_hash_would_resume_mixed_tensors() {
    let plain = cfg(Format::Nvfp4);
    let mut l2 = cfg(Format::Nvfp4);
    l2.quality = Quality::Nvfp4L2ScaleSearch;

    // (1) Baseline: the REAL hashes are correctly distinct. Without this the
    //     collision below would prove nothing.
    assert_eq!(
        plain.config_hash(),
        "95ede677cf402b53",
        "pinned plain nvfp4"
    );
    assert_eq!(l2.config_hash(), "85cf59c986e5d1a4", "pinned nvfp4_l2");
    assert_ne!(
        plain.config_hash(),
        l2.config_hash(),
        "the real implementation already keeps these apart"
    );

    // (2) THE BROKEN HASH: the same payload, with the quality key never
    //     emitted — i.e. `config_hash` as it would be if the conditional append
    //     were "simplified" away. Two different algorithms, one key.
    let blind = |c: &QuantConfig| -> String { hash_without_quality_key(c) };
    assert_eq!(
        blind(&plain),
        blind(&l2),
        "a quality-blind hash MUST collide, or this control demonstrates nothing"
    );
    // And it is a collision, not a coincidence of formatting: the payloads are
    // genuinely identical strings.
    assert_eq!(
        payload_without_quality_key(&plain),
        payload_without_quality_key(&l2)
    );

    // (3) Drive the REAL `StreamState::load_manifest` with the colliding key.
    //     A partial `nvfp4` state on disk, an `nvfp4_l2` run asking to resume.
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("model.safetensors");
    std::fs::write(&out, b"partial bytes from the plain nvfp4 run").expect("write partial output");

    let mut interrupted = StreamState::new(&out, blind(&plain));
    interrupted.order = vec!["blk.0".into(), "blk.1".into()];
    interrupted.done.insert("blk.0".into());
    interrupted.save_manifest().expect("save manifest");

    // The re-run carries a DIFFERENT quality variant but the SAME blind hash.
    let mut rerun = StreamState::new(&out, blind(&l2));
    let resumed = rerun.load_manifest(&out);

    assert!(
        resumed,
        "the blind hash should let load_manifest resume into the plain-nvfp4 \
         partial file — that acceptance IS the failure being demonstrated"
    );
    assert_eq!(
        rerun.done, interrupted.done,
        "the resumed state really did inherit the other algorithm's progress"
    );
    assert_eq!(rerun.order, interrupted.order);

    // (4) Control: with the REAL hash the same re-run is correctly REJECTED.
    //     This is the pairing that gives the guard its teeth — the only
    //     difference between the two cases is the key in the payload.
    let mut honest = StreamState::new(&out, l2.config_hash());
    assert!(
        !honest.load_manifest(&out),
        "the real, quality-aware hash MUST reject the resume"
    );
    assert!(
        honest.done.is_empty(),
        "a rejected resume must leave no inherited progress"
    );
}

/// The `config_hash` payload string with the `quality_tuning` key NEVER
/// emitted — the unconditional-append mistake, isolated to the payload.
fn payload_without_quality_key(c: &QuantConfig) -> String {
    // Mirrors `manifest.rs`'s own format-dependent field selection.
    let (target_format, int8, scaling_mode, block_size): (&str, bool, &str, u32) = match c.format {
        Format::Int8 => (
            &c.target_format,
            c.int8,
            c.scaling_mode.as_str(),
            c.block_size,
        ),
        Format::Fp8E4m3 => ("fp8", false, c.scaling_mode.as_str(), c.block_size),
        Format::Mxfp8 => ("mxfp8", false, "block", 32),
        Format::Nvfp4 => ("nvfp4", false, "block", 16),
    };
    format!(
        concat!(
            r#"{{"block_size": {}, "calib_seed": {}, "convrot": {}, "#,
            r#""convrot_group_size": {}, "int8": {}, "no_learned_rounding": {}, "#,
            r#""scaling_mode": "{}", "skip_inefficient": {}, "target_format": "{}"}}"#
        ),
        block_size,
        c.calib_seed,
        c.convrot,
        c.convrot_group_size,
        int8,
        c.no_learned_rounding,
        scaling_mode,
        c.skip_inefficient,
        target_format,
    )
}

/// SHA-256 of [`payload_without_quality_key`], first 8 bytes as hex — the same
/// truncation `config_hash` uses.
fn hash_without_quality_key(c: &QuantConfig) -> String {
    let digest = Sha256::digest(payload_without_quality_key(c).as_bytes());
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

// --------------------------------------------------------------------------
// 4. The byte-vs-float guard
// --------------------------------------------------------------------------

/// CONTROL 4 — a continuous `s_G` optimises a value the hardware never sees.
///
/// **Guard defended.** The choice of a **byte code** over a float.
/// `nvfp4_quality_round` scans the contiguous E4M3 byte window and can only
/// ever select a value the hardware can actually store, which is what makes
/// the emitted error and the scored error the same quantity. Production
/// mirror: `s_g_stays_on_the_e4m3_grid` in `nvfp4_scale_search.rs`.
///
/// **The real failure.** A float-valued search reports an error for the
/// continuous `s_G` it chose, then the encoder rounds that to the nearest E4M3
/// code and emits *that*. The two are different numbers, so the search's
/// objective is a fiction: it believes it reached an error it never achieved,
/// and the byte it actually emits is not the byte it chose.
///
/// Worse, the continuous optimum here is not merely off-grid — it is far
/// enough off-grid that rounding **saturates** to the top of the E4M3 range
/// (`0x7E` = 448), which is frequently not the best available byte. Measured
/// over six blocks: the continuous optimum was off the E4M3 grid in **6 of 6**
/// cases, rounding downward every time, and in 5 of 6 the rounded byte was
/// worse than the best byte available — by up to **1.92x** worse error than a
/// byte-domain search would have chosen.
#[test]
fn non_anchored_s_g_leaves_the_e4m3_grid() {
    let mut rng = Rng::new(0xC0FF_EE00_1234_5678);
    let (mut off_grid, mut rounded_worse_than_best, mut worst_ratio) = (0usize, 0usize, 1.0f64);
    let mut saturating = 0usize;
    const CASES: usize = 6;

    for _ in 0..CASES {
        let blk: Vec<f32> = (0..NVFP4_BLOCK).map(|_| bf16(rng.signed_unit())).collect();
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(amax > 0.0, "the fixture block must be non-zero");
        let pts = amax / PTS_DIVISOR;

        // Reconstruction SSE of this block at product scale `pts * s_g`.
        let sse = |s_g: f32| -> f64 { block_sse(&blk, &blk, pts * s_g) };

        // THE BUG: a wide, continuous, log-spaced float search for `s_g`.
        // No byte window, no grid — just "minimise L2 over a float".
        let mut best_continuous = (f64::INFINITY, 1.0f32);
        for i in 0..=200_000 {
            let s = (i as f32 / 200_000.0 * 24.0).exp(); // 1 .. 1e10
            let e = sse(s);
            if e < best_continuous.0 {
                best_continuous = (e, s);
            }
        }
        let (reported_err, cont_s) = best_continuous;

        // What the hardware actually stores, and what it therefore achieves.
        let byte = f32_to_fp8_e4m3_bits(cont_s);
        let seen = fp8_e4m3_bits_to_f32(byte);
        let emitted_err = sse(seen);

        // The best a byte-domain search could have chosen.
        let mut best_byte = (f64::INFINITY, 0u8);
        for cand in 0u8..=0x7Eu8 {
            let e = sse(fp8_e4m3_bits_to_f32(cand));
            if e < best_byte.0 {
                best_byte = (e, cand);
            }
        }

        // (a) The value the search optimised is not a value the hardware has.
        if fp8_e4m3_bits_to_f32(byte) != cont_s {
            off_grid += 1;
        }
        // (b) The error the search REPORTS is not the error emitted. The
        //     search believes it reached `reported_err`; the bytes it emits
        //     reconstruct at `emitted_err`, which is strictly worse.
        assert!(
            emitted_err > reported_err,
            "rounding to E4M3 must cost error: reported {reported_err:.9} vs \
             emitted {emitted_err:.9} (s_G {cont_s:.6} -> byte {byte:#04x})"
        );
        // (c) And the byte it settled on is often not even the best byte.
        let ratio = emitted_err / best_byte.0;
        if byte != best_byte.1 {
            rounded_worse_than_best += 1;
        }
        if ratio > worst_ratio {
            worst_ratio = ratio;
        }
        if seen >= 448.0 {
            saturating += 1;
        }
    }

    assert_eq!(
        off_grid, CASES,
        "a continuous optimum should be off the E4M3 grid every time"
    );
    // Measured: 5 of 6. The exception is a case where the continuous optimum
    // happened to round onto the best byte by luck — the grid is fine enough
    // (8 codes per octave) that this sometimes happens. It is not a guarantee,
    // and 1-in-6 is not a rate to rely on; the point is that the search cannot
    // know which case it is in, because it optimised a float.
    assert!(
        rounded_worse_than_best >= CASES - 1,
        "expected essentially every case to round to a non-optimal byte, got \
         {rounded_worse_than_best}/{CASES}"
    );
    assert!(
        saturating >= 4,
        "expected the continuous optimum to round up to the E4M3 ceiling in \
         most cases, got {saturating}/{CASES}"
    );
    assert!(
        worst_ratio > 1.5,
        "expected the mis-rounded byte to be badly worse than the best byte, \
         worst ratio {worst_ratio:.3}"
    );
    eprintln!(
        "control 4: continuous optimum off-grid {off_grid}/{CASES}; mis-rounded \
         byte {rounded_worse_than_best}/{CASES}; worst error {worst_ratio:.3}x \
         the best available byte"
    );
}

// --------------------------------------------------------------------------
// 5. The format-side view of the same mistake
// --------------------------------------------------------------------------

/// CONTROL 5 — a "finer" compensation would reduce resolution, not add it.
///
/// **Guard defended.** The same direction guard as control 2, seen from the
/// FORMAT side: the compensation buys clipping headroom and pays for it in
/// resolution, and the wrong direction loses headroom it never had. Production
/// mirror: `e8m0_delta_is_zero_or_plus_one_only` plus the `Direction: this
/// trades resolution FOR headroom` note on `block_scale_compensated`.
///
/// **FORMAT CORRECTNESS — read this before editing the comments.** MXFP8 uses
/// **E4M3** for its element codes with an **E8M0** power-of-two scale. It does
/// NOT use E2M1; E2M1 is NVFP4's element format (control 1). An earlier draft
/// of this control said "E2M1" and was wrong. In this test, every mention of
/// an element grid, a code, or the `±448` ceiling means **E4M3**.
///
/// **The real failure.** The base path cannot clip, so there is no headroom to
/// buy — see control 7, which proves it. The consequence is that the wrong
/// direction has nothing to gain and something to lose. Dividing by 4/3 drops
/// the exponent, halving the block scale, which pushes `amax/scale` straight
/// past the E4M3 finite ceiling of 448 and **saturates the top of the range**.
/// Measured over the whole reachable bf16 grid: the base path clips on **0**
/// of 32,639 block maxima, while the dividing path clips on **13,016** of them
/// and drives `amax/scale` up to **1.33x** past the ceiling. The worst case
/// gets a value whose true magnitude is 33% beyond anything E4M3 can hold, and
/// it gets that on the single element that matters most in a block.
///
/// **MUTATION VERIFIED.** Swapping `ceil` for `floor` in
/// `e8m0_and_scale_from_scale_needed` turns the control-7 no-op assertion RED
/// (shared code path), and reversing `E8M0_COMPENSATION` turns control 2 RED.
/// The `floor` mutation is the sharp one: it is what a "finer scale" reading of
/// this knob would actually look like in code, and it is exactly the change that
/// makes the base path clip.
#[test]
fn finer_compensation_would_reduce_resolution() {
    let (mut base_clips, mut wrong_clips, mut worst_past_ceiling) = (0usize, 0usize, 0.0f64);
    let mut checked = 0usize;

    for bm in all_finite_non_negative_bf16() {
        if bm <= 0.0 {
            continue;
        }
        // Base: `ceil(log2(amax/448))` >= log2(amax/448), so the scale is
        // always >= amax/448 and amax/scale <= 448. It cannot clip.
        let base_scale = e8m0_scale(e8m0_for(bm, 1.0));
        // Wrong direction: the exponent drops, so the scale halves.
        let wrong_scale = e8m0_scale(e8m0_for(bm, 1.0 / FOUR_THIRDS));
        checked += 1;
        if base_scale > 0.0 && f64::from(bm) / f64::from(base_scale) > f64::from(FP8_MAX) {
            base_clips += 1;
        }
        if wrong_scale > 0.0 {
            let past = f64::from(bm) / f64::from(wrong_scale) / f64::from(FP8_MAX);
            if past > 1.0 {
                wrong_clips += 1;
            }
            if past > worst_past_ceiling {
                worst_past_ceiling = past;
            }
        }
    }

    assert_eq!(
        base_clips, 0,
        "the base path must never clip — control 7 owns that proof"
    );
    assert!(
        wrong_clips > 0,
        "the wrong direction must push some block maxima past the E4M3 ceiling"
    );
    assert!(
        worst_past_ceiling > 1.30,
        "expected the wrong direction to overshoot the E4M3 ceiling by >30%, \
         got {worst_past_ceiling:.4}"
    );
    eprintln!(
        "control 5: over {checked} reachable bf16 maxima, base clips on \
         {base_clips}, dividing clips on {wrong_clips}, worst overshoot past the \
         E4M3 ceiling {worst_past_ceiling:.4}x"
    );

    // The consequence at the element level, on a concrete witness.
    //
    // `amax = 500` is chosen so the delta is genuinely `-1` and the base scale
    // is genuinely sufficient: `500/448 = 1.116`, so the base rule rounds up
    // to a scale of `2.0` and maps the max to `250` — comfortably inside the
    // E4M3 range. Dividing drops the exponent to a scale of `1.0`, which maps
    // the max to `500`: past the `448` ceiling, so it saturates.
    let bm = 500.0f32;
    let base_e8 = e8m0_for(bm, 1.0);
    let wrong_e8 = e8m0_for(bm, 1.0 / FOUR_THIRDS);
    assert_eq!(
        base_e8 as i32 - wrong_e8 as i32,
        1,
        "the wrong direction must drop the exponent by exactly 1 at amax={bm}"
    );
    let base_scale = e8m0_scale(base_e8);
    let wrong_scale = e8m0_scale(wrong_e8);
    assert!(
        f64::from(bm) / f64::from(base_scale) <= f64::from(FP8_MAX),
        "the base path must not clip this witness"
    );
    assert!(
        f64::from(bm) / f64::from(wrong_scale) > f64::from(FP8_MAX),
        "the wrong direction must push this witness past the E4M3 ceiling"
    );
    // The saturated element is coded as E4M3 `0x7E` (448.0) no matter how far
    // past the true value it lies, so its absolute error is the overshoot.
    let code = f32_to_fp8_e4m3_bits((bm / wrong_scale).clamp(-FP8_MAX, FP8_MAX));
    assert_eq!(code, 0x7E, "the block max saturates to E4M3 448.0");
    let recon = fp8_e4m3_bits_to_f32(code) * wrong_scale;
    assert!(
        recon < bm,
        "the saturated reconstruction must fall short of the true max \
         (recon {recon} vs amax {bm})"
    );
    // Under the base rule the same element is represented without saturation.
    let base_code = f32_to_fp8_e4m3_bits((bm / base_scale).clamp(-FP8_MAX, FP8_MAX));
    assert_ne!(base_code, 0x7E, "the base path must not saturate here");
    assert!(
        (fp8_e4m3_bits_to_f32(base_code) * base_scale - bm).abs() < (recon - bm).abs(),
        "the base path must reconstruct this witness more accurately"
    );
}

// --------------------------------------------------------------------------
// 6. The converse of the S01 fix
// --------------------------------------------------------------------------

/// CONTROL 6 — an unconditional `quality_tuning` key would move the Python
/// vector.
///
/// **Guard defended.** The conditional append in `QuantConfig::config_hash`:
/// the key is emitted ONLY when `Quality::id()` is `Some`. Production mirrors:
/// `default_config_hash_is_unchanged_by_the_quality_mechanism` and
/// `exact_contributes_no_key_to_the_payload` in `quality_mode_identity.rs`,
/// plus the four pinned digests in `parity_fingerprint.rs`.
///
/// **The real failure.** `Quality::Exact` is the zero value and MUST stay
/// byte-parity-exact. The four committed hash vectors — INT8
/// `56920c6553cfa241` (**captured from the Python reference**), FP8
/// `5f14780b1bcf30f2`, MXFP8 `cff3b89365c9544d`, NVFP4 `95ede677cf402b53` —
/// were ALL produced by a 9-key payload with no quality field. Emitting the key
/// unconditionally is the natural-looking implementation, and it silently
/// breaks every one of them at once, including a vector this crate does not own
/// and cannot regenerate.
///
/// **MUTATION VERIFIED.** Making the `quality_tuning` key unconditional in
/// `config_hash` turns this control RED: `56920c6553cfa241` becomes
/// `b9cfe08020ff8bd5`, and the NVFP4 vector becomes `9025bf6a71c651fb`.
///
/// Measured: the 9-key payload reproduces `56920c6553cfa241` exactly; inserting
/// `"quality_tuning": "exact", ` moves it to `b9cfe08020ff8bd5`.
#[test]
fn unconditional_hash_key_would_move_the_python_vector() {
    let c = QuantConfig::default();

    // (1) The real, conditional implementation reproduces the Python-captured
    //     vector. This is the control's baseline.
    assert_eq!(c.quality, Quality::Exact, "the default IS Exact");
    assert_eq!(
        c.config_hash(),
        "56920c6553cfa241",
        "Python-captured INT8 vector"
    );
    assert_eq!(Quality::Exact.id(), None, "Exact contributes no key");

    // (2) THE BUG: emit the key unconditionally, using a label for `Exact`.
    //     Rebuild the payload the way `config_hash` would if the `None => ""`
    //     arm were replaced by an unconditional `Some("exact")`.
    let payload = |with_key: bool| -> String {
        let quality = if with_key {
            r#""quality_tuning": "exact", "#
        } else {
            ""
        };
        format!(
            concat!(
                r#"{{"block_size": {}, "calib_seed": {}, "convrot": {}, "#,
                r#""convrot_group_size": {}, "int8": {}, "no_learned_rounding": {}, "#,
                "{}",
                r#""scaling_mode": "{}", "skip_inefficient": {}, "target_format": "{}"}}"#
            ),
            c.block_size,
            c.calib_seed,
            c.convrot,
            c.convrot_group_size,
            c.int8,
            c.no_learned_rounding,
            quality,
            c.scaling_mode.as_str(),
            c.skip_inefficient,
            c.target_format,
        )
    };
    let short_hash = |p: &str| -> String {
        Sha256::digest(p.as_bytes())
            .iter()
            .take(8)
            .map(|b| format!("{b:02x}"))
            .collect()
    };

    let nine_key = short_hash(&payload(false));
    let ten_key = short_hash(&payload(true));

    // The 9-key form is the Python vector, reproduced from first principles.
    assert_eq!(
        nine_key, "56920c6553cfa241",
        "the 9-key payload must reproduce the committed vector"
    );
    // The unconditional key MOVES it.
    assert_eq!(
        ten_key, "b9cfe08020ff8bd5",
        "adding the key unconditionally moves the vector"
    );
    assert_ne!(
        nine_key, ten_key,
        "the two must differ, or this control proves nothing"
    );

    // (3) The same break hits every committed vector, not just INT8. A
    //     `quality_mode_identity` digest for any format with `Exact` would be
    //     invalidated by the same one-line change.
    for (format, pinned) in [
        (Format::Fp8E4m3, "5f14780b1bcf30f2"),
        (Format::Mxfp8, "cff3b89365c9544d"),
        (Format::Nvfp4, "95ede677cf402b53"),
    ] {
        let mut exact = cfg(format);
        exact.quality = Quality::Exact;
        assert_eq!(
            exact.config_hash(),
            pinned,
            "{format:?} + Exact must still reproduce its committed vector"
        );
    }
}

// --------------------------------------------------------------------------
// 7. Why the compensation is a bit-exact no-op (and why that guard is real)
// --------------------------------------------------------------------------

/// CONTROL 7 — the `4/3` compensation is a no-op in the normal range, and the
/// guard saying so has real teeth.
///
/// **Guards defended.** Any assertion of the form "the compensation must not
/// regress the base path" — chiefly the measured-invariance statements in
/// `mxfp8_e8m0_compensated.rs` and the `rel_l2` comparison in
/// `parity_fingerprint.rs`.
///
/// **Why this control exists.** A "must not regress" guard looks vacuous: of
/// course a quality mode should not make things worse, so what does the
/// assertion prove? This file answers that. The no-op is NOT a tautology — it
/// is the product of two independent, non-obvious facts about this kernel, and
/// removing either one makes the compensation bite:
///
/// -- (a) **E4M3 is scale-invariant in the normal range.** Halving a code and
///   doubling the scale reconstructs the identical f32, because a normal E4M3
///   value is a scaled integer significand. Verified by enumeration below:
///   **238 of the 254 finite E4M3 codes have an exact half on the grid**, and
///   for every one of them `(half_code, 2*scale) == (code, scale)`.
/// -- (b) **The MXFP8 base path never clips.** With `ceil(log2(amax/448))` the
///   scale is always `>= amax/448`, so `amax/scale <= 448` and the largest
///   element can never saturate. The measured maximum of `(amax/scale)/448`
///   over the entire bf16 grid is exactly `1.0`.
///
/// Together: on a `+1` block the scale doubles and every code halves, which by
/// (a) reconstructs the same f32, and by (b) never trades a clipped value for
/// an unclipped one. The E8M0 BYTE changes; the NUMBERS do not.
///
/// **This is what killed the MXFP8 item.** The paper's premise — E8M0
/// power-of-two scales are a large error source, so buying headroom should
/// help — does hold for a `floor()`-based scale, which WOULD clip. This kernel
/// uses `ceil()`, so it does not. The measured `rel_l2` ratio is exactly
/// `1.000000` on every distribution tried, and no fixture can improve on it.
///
/// **The 16 exceptions in (a), stated precisely.** The codes without an exact
/// half are the 8 odd significands in the subnormal-adjacent band
/// (`0x01, 0x03, ..., 0x0F`) and their 8 negative mirrors — the codes whose
/// halving would fall below the smallest normal and land in the band where the
/// E4M3 grid is ABSOLUTE rather than relative. That band is exactly where
/// `compensated_can_be_worse_than_base_in_the_e4m3_denormal_band` shows the
/// compensation losing by ~7x, and it is why no test may assert a per-block
/// improvement.
///
/// **MUTATION VERIFIED.** Both `E8M0_COMPENSATION = 3.0/4.0` and the
/// `ceil` -> `floor` swap turn the no-op assertion in this control RED, which
/// is what makes "must not regress" a real guard rather than a formality: there
/// are concrete one-line changes that break it, and the absolute-band probe at
/// the end shows a reachable input where the compensation genuinely does
/// change the numbers.
#[test]
fn e4m3_scale_invariance_and_no_clipping_make_the_compensation_a_no_op() {
    // (a) E4M3 scale-invariance in the normal range.
    let (mut exact_half, mut finite, mut checked_identity) = (0usize, 0usize, 0usize);
    for c in 0u8..=0xFFu8 {
        let v = fp8_e4m3_bits_to_f32(c);
        if !v.is_finite() {
            continue; // 0x7F / 0xFF are the NaN patterns, not values.
        }
        finite += 1;
        let half = f32_to_fp8_e4m3_bits(v / 2.0);
        if fp8_e4m3_bits_to_f32(half) * 2.0 == v {
            exact_half += 1;
            // The strong form of the same fact, as the production tests state
            // it: halving the code and doubling the scale is a no-op.
            let scale = 2.0f32;
            assert_eq!(
                fp8_e4m3_bits_to_f32(half) * scale,
                fp8_e4m3_bits_to_f32(c),
                "code {c:#04x} must reconstruct identically at 2x the scale"
            );
            checked_identity += 1;
        }
    }
    assert_eq!(finite, 254, "E4M3 has 254 finite codes (0x7F/0xFF are NaN)");
    assert_eq!(
        exact_half, 238,
        "measured: 238 of 254 finite E4M3 codes have an exact half on the grid"
    );
    assert_eq!(checked_identity, exact_half);

    // The 16 exceptions are exactly the odd significands whose halving enters
    // the absolute-grid band, plus their negative mirrors.
    let mut exceptions = Vec::new();
    for c in 0u8..=0xFFu8 {
        let v = fp8_e4m3_bits_to_f32(c);
        if v.is_finite() && fp8_e4m3_bits_to_f32(f32_to_fp8_e4m3_bits(v / 2.0)) * 2.0 != v {
            exceptions.push(c);
        }
    }
    assert_eq!(
        exceptions,
        vec![
            0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F, 0x81, 0x83, 0x85, 0x87, 0x89, 0x8B,
            0x8D, 0x8F
        ],
        "the exceptions are the 8 odd significands in the subnormal-adjacent \
         band and their mirrors; got {exceptions:?}"
    );

    // (b) The base path never clips.
    let mut worst = 0.0f64;
    for bm in all_finite_non_negative_bf16() {
        if bm <= 0.0 {
            continue;
        }
        let scale = e8m0_scale(e8m0_for(bm, 1.0));
        if scale <= 0.0 {
            continue;
        }
        let ratio = f64::from(bm) / f64::from(scale) / f64::from(FP8_MAX);
        if ratio > worst {
            worst = ratio;
        }
    }
    assert!(
        (worst - 1.0).abs() < 1e-12,
        "base path must satisfy amax/scale <= 448 exactly; worst was {worst:.15}"
    );

    // (c) The consequence: the compensation is a bit-exact no-op.
    //
    //     A `+1` block doubles its scale and halves every code, which by (a)
    //     reconstructs the same f32, and by (b) clips nothing. So the emitted
    //     E8M0 byte differs while `rel_l2` is bit-identical.
    let (m, n) = (32usize, 64usize);
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    let w: Vec<f32> = (0..m * n).map(|_| bf16(rng.range(-1.0, 1.0))).collect();
    let base = rel_l2_shipped_mxfp8(&w, m, n);
    let comp = rel_l2_compensated(&w, m, n);
    assert_eq!(
        comp.to_bits(),
        base.to_bits(),
        "the compensation must be a bit-exact no-op in the normal range"
    );

    // And the guard is REAL, not vacuous: there exists an input where the
    // compensation does change the numbers — the absolute-grid band from (a).
    // Without such a case, "must not regress" would be asserting nothing.
    let mut probe = vec![0.0f32; MXFP8_BLOCK];
    probe[0] = 448.0;
    probe[1] = 0.01; // 0.01 / 1.0 = 0.01 < 2^-6: inside the absolute band
    assert!(
        0.01 < 2.0f32.powi(-6),
        "the probe element must land in the E4M3 absolute-grid band"
    );
    let base_b = rel_l2_shipped_mxfp8(&probe, 1, MXFP8_BLOCK);
    let comp_b = rel_l2_compensated(&probe, 1, MXFP8_BLOCK);
    assert!(
        comp_b > base_b,
        "the guard has teeth only if the compensation CAN change the error; \
         expected it to be worse in the absolute band, got base {base_b:e} \
         comp {comp_b:e}"
    );
    eprintln!(
        "control 7: {exact_half}/{finite} E4M3 codes scale-invariant; base worst \
         amax/scale ratio {worst:.6}; rel_l2 bit-exact no-op ({base:.8}); \
         absolute-band probe {base_b:e} -> {comp_b:e} (guard has teeth)"
    );
}

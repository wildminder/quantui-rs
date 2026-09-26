//! Tier 2 S03 — the anchored NVFP4 L2 scale search.
//!
//! # What is being tested
//!
//! [`quantize_nvfp4_weight_quality`] is the first real quality algorithm in this
//! crate: it deliberately breaks byte-parity in exchange for lower L2
//! reconstruction error. The test surface is therefore not "does it produce the
//! reference bytes" (it must NOT) but three things that a quality mode has to
//! earn before it is allowed to emit anything at all:
//!
//!  1. **It actually helps.** `nvfp4_l2_reduces_rel_l2_on_fixed_fixture` is the
//!     exit gate. Measured through the shipped [`dequantize_nvfp4`], never a
//!     private reimplementation — otherwise the metric would describe the
//!     search's own idea of the output rather than the bytes it emits.
//!  2. **It is anchored.** `unanchored_search_would_drift_unrepresentably` is a
//!     negative control that deliberately runs the *unanchored* version of the
//!     same idea and demonstrates the failure the anchor exists to prevent. It
//!     passes by showing the wrong behaviour is real and reachable.
//!  3. **It is deterministic.** Alternating searches have a sequential
//!     dependency, which is exactly where a rayon reduction or a `HashMap`
//!     iteration order would leak nondeterminism into the output bytes.
//!
//! # Why the anchor is not optional
//!
//! `X̂ = s_T · s_G · Q(X / (s_T·s_G))` observes only the **product** `s_T·s_G`.
//! Rescaling `s_T → 2ʲ·s_T` and `s_G → s_G/2ʲ` changes neither the product nor,
//! because the E2M1 grid is `{2ʲ, 1.5·2ʲ}`, the element codes. The objective is
//! not just "unconstrained" along that direction — it is **exactly flat** there,
//! so a search that minimises L2 alone cannot tell a correct scale from one
//! inflated by orders of magnitude. `unanchored_search_would_drift_unrepresentably`
//! measures that flatness directly instead of taking it on trust.
//!
//! Case count: proptest's default of 256, matching `recipe_property.rs`.

use proptest::prelude::*;

use quant_core::dtype::{
    bf16_bits_to_f32, f32_to_bf16_bits, f32_to_fp8_e4m3_bits, fp8_e4m3_bits_to_f32,
};
use quant_core::quant_mxfp8::from_blocked_u8;
use quant_core::quant_nvfp4::{
    dequantize_nvfp4, quantize_nvfp4_weight, quantize_nvfp4_weight_quality, BLOCK_SIZE,
};

/// Half-width of the E4M3 byte-code window the kernel searches. Mirrors
/// `QUALITY_CODE_WINDOW`; duplicated here because the point of
/// `s_g_stays_on_the_e4m3_grid` is to check the emitted bytes against an
/// independently-stated bound.
const WINDOW: u8 = 4;

/// `448 * 6` — the per-tensor-scale divisor, restated so this file does not
/// have to import a private constant to recompute an anchor.
const PTS_DIVISOR: f32 = 448.0 * 6.0;

// --------------------------------------------------------------------------
// Fixtures
// --------------------------------------------------------------------------

/// xorshift64, the deterministic generator already used across this crate's
/// tests. A fixed seed is what makes "fixed fixture" mean anything.
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

    /// Uniform in `[0, 1)`.
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// Symmetric magnitude in `(0, 1]`.
    fn signed_unit(&mut self) -> f32 {
        self.range(-1.0, 1.0)
    }
}

/// A dense, full-range matrix: every 16-element block has comparable energy,
/// so the per-tensor scale and the block scales are close to proportional.
fn uniform_matrix(m: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..m * n).map(|_| rng.signed_unit()).collect()
}

/// A matrix with genuine per-block dynamic range: row `r` of `m` carries
/// magnitude `10^(-decades * r / (m - 1))`, so the smallest row's block scale
/// sits `decades` below the largest.
///
/// This is the fixture the negative control needs. NVFP4's per-tensor scale is
/// global, so a row far below the tensor amax is exactly the block whose E4M3
/// scale is small enough to underflow to zero if the search is allowed to walk
/// `s_T` upward.
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

// --------------------------------------------------------------------------
// Metric
// --------------------------------------------------------------------------

/// `rel_l2 = Σ (w − dequantize(quantize(w)))² / Σ w²`, accumulated in f64.
///
/// The denominator is the energy of the ORIGINAL tensor, so the number is
/// comparable across the two paths on the same input. f64 throughout: the
/// numerator is a sum of thousands of small terms and an f32 accumulator would
/// lose enough precision to make the "never worse" comparison noise-sensitive.
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

/// Round-trip a tensor through both paths and return `(base, quality)`.
///
/// Both reconstructions come from the shipped [`dequantize_nvfp4`], so the
/// comparison is between the bytes the crate actually emits — not between two
/// models of what it might emit.
fn measure_both(w: &[f32], m: usize, n: usize) -> (f64, f64) {
    let base = quantize_nvfp4_weight(w, m, n);
    let quality = quantize_nvfp4_weight_quality(w, m, n);
    let base_l2 = rel_l2(w, &dequantize_nvfp4(&base, m, n));
    let quality_l2 = rel_l2(w, &dequantize_nvfp4(&quality, m, n));
    (base_l2, quality_l2)
}

/// Unswizzle a result's block scales back to row-major `(m_pad, num_blocks)`.
///
/// The emitted `scale` is in cuBLAS `to_blocked` layout, which is fine for
/// hashing and useless for reasoning about "which byte belongs to which
/// block". `from_blocked_u8` is the crate's own inverse, so using it here
/// cannot introduce a second opinion about the layout.
fn row_major_scales(r: &quant_core::quant_nvfp4::Nvfp4QuantResult) -> (Vec<u8>, usize, usize) {
    let m_pad = r.qdata_shape[0] as usize;
    let n_pad = r.qdata_shape[1] as usize * 2;
    let num_blocks = n_pad / BLOCK_SIZE;
    let scales = from_blocked_u8(&r.scale, m_pad, num_blocks);
    (scales, m_pad, num_blocks)
}

// --------------------------------------------------------------------------
// 1. The exit gate
// --------------------------------------------------------------------------

/// The search must measurably reduce relative L2 against the base path.
///
/// The threshold is 5%, a deliberately loose floor: the point of the test is
/// that the direction of the change is right and is worth having, not that a
/// particular fixture hits a particular number. A prototype measured 44.7%, so
/// anything near 5% means something is subtly broken rather than merely
/// unlucky.
///
/// Fixtures span the two regimes that behave differently: uniform blocks
/// (block scale nearly proportional to the tensor scale, so the per-tensor
/// refit is what does the work) and wide-dynamic-range blocks (small rows
/// sitting near the E2M1 underflow boundary, where the byte search does the
/// work).
#[test]
fn nvfp4_l2_reduces_rel_l2_on_fixed_fixture() {
    let cases: [(&str, Vec<f32>, usize, usize); 4] = [
        (
            "uniform 64x64",
            uniform_matrix(64, 64, 0x243F_6A88_85A3_08D3),
            64,
            64,
        ),
        (
            "uniform 32x48",
            uniform_matrix(32, 48, 0x9E37_79B9_7F4A_7C15),
            32,
            48,
        ),
        (
            "3 decades 16x64",
            dynamic_range_matrix(16, 64, 3.0, 0x1319_8A2E_0370_7344),
            16,
            64,
        ),
        (
            "1.5 decades 16x32",
            dynamic_range_matrix(16, 32, 1.5, 0xA409_3822_299F_31D0),
            16,
            32,
        ),
    ];

    for (label, w, m, n) in cases {
        let (base_l2, quality_l2) = measure_both(&w, m, n);
        let reduction = 1.0 - quality_l2 / base_l2;
        eprintln!(
            "{label:>20}: base rel_l2 = {base_l2:.6}, quality rel_l2 = {quality_l2:.6}, \
             reduction = {:.2}%",
            reduction * 100.0
        );
        assert!(
            reduction >= 0.05,
            "{label}: relative L2 reduction was {:.2}%, below the 5% floor \
             (base {base_l2:.6}, quality {quality_l2:.6})",
            reduction * 100.0
        );
    }
}

/// The search must be a strict improvement, not a re-derivation of the same
/// answer — if `qdata` and `scale` were identical to the base path then
/// `rel_l2` would be identical and the gate above would be measuring nothing.
#[test]
fn quality_path_actually_changes_the_emitted_bytes() {
    let w = uniform_matrix(64, 64, 0x243F_6A88_85A3_08D3);
    let base = quantize_nvfp4_weight(&w, 64, 64);
    let quality = quantize_nvfp4_weight_quality(&w, 64, 64);

    assert_ne!(
        base.qdata, quality.qdata,
        "quality path emitted the base codes unchanged — the search found nothing"
    );
    assert_ne!(
        base.scale, quality.scale,
        "quality path emitted the base block scales unchanged"
    );
    assert_ne!(
        base.per_tensor_scale.to_bits(),
        quality.per_tensor_scale.to_bits(),
        "the per-tensor scale never moved, so the closed-form refit never fired"
    );
}

// --------------------------------------------------------------------------
// 2. The negative control
// --------------------------------------------------------------------------

/// E2M1 code → f32, the reference `E2M1_LUT`. Restated here so the negative
/// control does not depend on the kernel under test.
const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// The unanchored walk: rescale the **given** reference pair by `2ᵏ` and
/// re-measure.
///
/// This models the flat direction exactly. `s_T → s_T·2ᵏ` and
/// `s_G → s_G/2ᵏ` leave the product `s_T·s_G` invariant, so the reconstruction —
/// and therefore the error — must be unchanged. The only thing that moves is
/// the emitted representation: `s_T` inflates without bound while `s_G` walks
/// down the E4M3 grid toward zero, and nothing in the objective objects.
///
/// `k > 0` is where representability starts to matter: halving an E4M3 value is
/// exact only while it stays in the normal range. Once the smallest block's
/// scale enters subnormal territory the halving rounds, the product is no
/// longer exactly preserved, and the error starts to move — which is precisely
/// the cliff an unanchored search falls off with no signal that it has.
///
/// Returns `(total SSE, the block-scale byte chosen at each step)`.
fn rescale_probe(
    enc: &[Vec<f32>],
    obj: &[Vec<f32>],
    pts0: f32,
    base_bytes: &[u8],
    k: i32,
) -> (f64, Vec<u8>) {
    let pts = pts0 * 2.0f32.powi(k);
    let mut sse = 0.0f64;
    let mut bytes = Vec::with_capacity(enc.len());
    for ((e, o), &b0) in enc.iter().zip(obj.iter()).zip(base_bytes.iter()) {
        let g = fp8_e4m3_bits_to_f32(b0) * 2.0f32.powi(-k);
        let byte = f32_to_fp8_e4m3_bits(g);
        bytes.push(byte);
        let total = pts * fp8_e4m3_bits_to_f32(byte);
        sse += if total == 0.0 {
            // Zero block: the reference forces +0.0 for all 16 codes, so the
            // reconstruction is 0 and the error is the block's own energy.
            obj_block_energy(o)
        } else {
            block_sse(e, o, total)
        };
    }
    (sse, bytes)
}

/// `Σ x²` over a block, in f64.
fn obj_block_energy(obj: &[f32]) -> f64 {
    obj.iter()
        .map(|&v| {
            let d = v as f64;
            d * d
        })
        .sum()
}

/// Reconstruction SSE of one block: codes from `enc`, error against `obj`.
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

/// The reference E2M1 encoder, restated as a nearest-value rule so the control
/// does not depend on the kernel under test. The grid is
/// `{0, .5, 1, 1.5, 2, 3, 4, 6}` — small enough to scan exhaustively.
///
/// Nearest-value (rather than the kernel's round-half-to-even) is sufficient
/// here because this function only has to be a *consistent* encoder for the
/// control's comparison to mean anything; it is not asserting parity. Where
/// the two disagree on a tie, the control's chosen `s_G` shifts by at most one
/// code, which does not affect the flatness or zeroing claims.
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
    // 6.0 is the saturated code (7), not an extension of the 4.0 entry.
    if best.1 == 6 && ax > 5.0 {
        best.1 = 7;
    }
    best.1 | if x.is_sign_negative() { 0x08 } else { 0 }
}

/// Split a tensor into bf16-rounded, zero-padded 16-element blocks (the encode
/// input) alongside pre-bf16 zero-padded blocks (the objective input) and the
/// anchored per-tensor scale — matching what the kernel actually sees.
fn padded_blocks(w: &[f32], m: usize, n: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, f32) {
    let m_pad = m.div_ceil(BLOCK_SIZE) * BLOCK_SIZE;
    let n_pad = n.div_ceil(BLOCK_SIZE) * BLOCK_SIZE;
    let mut wp = vec![0.0f32; m_pad * n_pad];
    let mut w_raw = vec![0.0f32; m_pad * n_pad];
    for r in 0..m {
        for c in 0..n {
            let v = w[r * n + c];
            wp[r * n_pad + c] = bf16_bits_to_f32(f32_to_bf16_bits(v));
            w_raw[r * n_pad + c] = v;
        }
    }
    let amax = w.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let pts = amax / PTS_DIVISOR;
    let mut enc = Vec::new();
    let mut obj = Vec::new();
    for r in 0..m_pad {
        for b in 0..n_pad / BLOCK_SIZE {
            let s = r * n_pad + b * BLOCK_SIZE;
            enc.push(wp[s..s + BLOCK_SIZE].to_vec());
            obj.push(w_raw[s..s + BLOCK_SIZE].to_vec());
        }
    }
    (enc, obj, pts)
}

/// The negative control.
///
/// Walks `s_T` up by `2ʲ` with no anchor and no neighbourhood bound, choosing
/// the per-block E4M3 code freely at each step, and asserts the two things the
/// anchor exists to prevent:
///
///  * **The objective cannot see the drift.** Between the steps where every
///    block scale is still representable, the L2 error is *bit-identical* —
///    not approximately, exactly. Every one of those `k` is a global
///    minimiser, so "minimise L2" is genuinely ambiguous rather than merely
///    under-constrained.
///  * **The drift is not free.** Once `s_T` has climbed far enough, the
///    smallest blocks' E4M3 scales round to **zero** — `s_G` is no longer
///    E4M3-representable as a scale at all — those blocks collapse to code 0,
///    and the total error ends up *worse* than the reference.
///
/// The test PASSES by demonstrating that this happens.
#[test]
fn unanchored_search_would_drift_unrepresentably() {
    // Three decades of per-block dynamic range: the smallest row's block scale
    // is ~0.448, which survives a handful of doublings and then does not.
    let (m, n) = (16usize, 64usize);
    let w = dynamic_range_matrix(m, n, 3.0, 0x1319_8A2E_0370_7344);
    let (enc, obj, pts_anchor) = padded_blocks(&w, m, n);

    // The reference error, through the shipped dequantizer.
    let base = quantize_nvfp4_weight(&w, m, n);
    let base_l2 = rel_l2(&w, &dequantize_nvfp4(&base, m, n));

    // The reference scale bytes, row-major, straight out of the base path.
    let (base_scales, m_pad, num_blocks) = row_major_scales(&base);

    // (a) EXACT flatness of the rescaling direction.
    //
    //     Rescale the *given* reference pair — `s_T → s_T·2ᵏ`,
    //     `s_G → s_G/2ᵏ` — and re-measure. The product is invariant, so the
    //     reconstruction must be bit-identical, and the measurement confirms it
    //     is: the SSE is equal to the last bit for k = 0..=4, i.e. while the
    //     halved E4M3 value is still exactly representable. It stops being flat
    //     at k = 5 precisely when the smallest block's scale reaches the E4M3
    //     subnormal range and halving stops being exact — which is the whole
    //     point: the flat direction is not infinitely long, it ends where
    //     representability ends, and nothing in the objective announces it.
    const STEPS: u32 = 12;
    let mut rescaled_sse = Vec::with_capacity(STEPS as usize + 1);
    let mut rescaled_bytes: Vec<Vec<u8>> = Vec::with_capacity(STEPS as usize + 1);
    for k in 0..=STEPS {
        let (sse, bytes) = rescale_probe(&enc, &obj, pts_anchor, &base_scales, k as i32);
        rescaled_sse.push(sse);
        rescaled_bytes.push(bytes);
    }

    let mut flat_upto = 0usize;
    'outer: for k in 0..STEPS as usize {
        if rescaled_sse[k + 1].to_bits() != rescaled_sse[k].to_bits() {
            break 'outer;
        }
        flat_upto = k + 1;
    }
    assert!(
        flat_upto >= 4,
        "expected the L2 objective to be EXACTLY flat across several doublings \
         of s_T (flat through k={flat_upto}); it changed after k={flat_upto}, \
         so the negative control is not demonstrating what it claims"
    );
    // Every one of those states is a global minimiser with a wildly different
    // s_T, so "just minimise L2" cannot distinguish them.
    for k in 0..=flat_upto {
        assert_eq!(
            rescaled_sse[k], rescaled_sse[0],
            "k={k} should be bit-identical to k=0"
        );
    }
    // For every k >= 1 the artifact really did change: `s_G` moved off the bytes
    // the parity path would have emitted, while the error stayed identical.
    // (k = 0 is the identity rescale, so it is excluded.)
    for k in 1..=flat_upto {
        assert!(
            rescaled_bytes[k]
                .iter()
                .zip(base_scales.iter())
                .any(|(a, b)| a != b),
            "k={k} changed no scale bytes, so nothing drifted"
        );
    }
    eprintln!(
        "unanchored: L2 BIT-IDENTICAL for k = 0..={flat_upto} with s_T inflated \
         up to {}× and every s_G byte changed — the objective cannot see it",
        1u64 << flat_upto
    );

    // (b) The drift leaves the anchored neighbourhood. At k = flat_upto the
    //     block scales have moved by `flat_upto` octaves, which is 8 codes each
    //     — twice the ±4 window the real search is confined to.
    let drifted = &rescaled_bytes[flat_upto];
    let max_drift = drifted
        .iter()
        .zip(base_scales.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert!(
        max_drift > WINDOW,
        "the unanchored drift should leave the ±{WINDOW}-code window, but the \
         furthest block moved only {max_drift} codes"
    );

    // (c) Continuing the walk destroys representability: block scales round to
    //     E4M3 zero, so `s_T` is absorbing a factor the format cannot express,
    //     and those blocks collapse to code 0.
    let final_bytes = &rescaled_bytes[STEPS as usize];
    let zeroed = final_bytes.iter().filter(|&&b| b == 0x00).count();
    assert!(
        zeroed > 0,
        "expected the drift to push some block scales to E4M3 zero \
         (s_T inflated {}×); final scale bytes were {final_bytes:?}",
        1u64 << STEPS
    );

    // (d) And the error is strictly WORSE than the reference, normalised by the
    //     same Σw² the metric uses.
    let energy: f64 = w.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let drifted_l2 = rescaled_sse[STEPS as usize] / energy;
    assert!(
        drifted_l2 > base_l2,
        "the drifted search should be WORSE than the reference path \
         (drifted {drifted_l2:.6}, base {base_l2:.6})"
    );
    eprintln!(
        "unanchored: s_T inflated {}×, {zeroed}/{} block scale(s) rounded to E4M3 \
         zero, rel_l2 {base_l2:.6} -> {drifted_l2:.6} ({:.2}× worse)",
        1u64 << STEPS,
        final_bytes.len(),
        drifted_l2 / base_l2.max(f64::MIN_POSITIVE)
    );

    let _ = (m_pad, num_blocks);
}

// --------------------------------------------------------------------------
// 3. The anchor guard, on the bytes actually emitted
// --------------------------------------------------------------------------

/// Every emitted block-scale byte must be a real E4M3 value and must sit in the
/// anchored neighbourhood of the reference byte for that same block.
///
/// The anchor is taken from the BASE path's own output rather than recomputed
/// here. That is deliberate: it makes the assertion independent of any anchor
/// formula written in this file, and it directly tests the property that
/// matters — the quality path stays near what the parity path would have said.
#[test]
fn s_g_stays_on_the_e4m3_grid() {
    for (label, w, m, n) in [
        (
            "uniform",
            uniform_matrix(64, 64, 0x243F_6A88_85A3_08D3),
            64usize,
            64usize,
        ),
        (
            "dynamic range",
            dynamic_range_matrix(16, 64, 3.0, 0x1319_8A2E_0370_7344),
            16,
            64,
        ),
    ] {
        let base = quantize_nvfp4_weight(&w, m, n);
        let quality = quantize_nvfp4_weight_quality(&w, m, n);
        let (base_scales, _, num_blocks) = row_major_scales(&base);
        let (quality_scales, m_pad, _) = row_major_scales(&quality);

        assert_eq!(
            base_scales.len(),
            m_pad * num_blocks,
            "{label}: unexpected block count"
        );

        let mut moved = 0usize;
        for (i, (&q, &b)) in quality_scales.iter().zip(base_scales.iter()).enumerate() {
            // Round-trips: the byte decodes to a value that re-encodes to
            // itself, so the search chose a representable scale and not a
            // continuous value the hardware never sees.
            let decoded = fp8_e4m3_bits_to_f32(q);
            assert!(
                decoded.is_finite(),
                "{label}: block {i} emitted non-finite scale (byte {q:#04x})"
            );
            assert_ne!(
                q, 0x7F,
                "{label}: block {i} emitted the E4M3 NaN pattern as a scale"
            );
            assert_eq!(
                f32_to_fp8_e4m3_bits(decoded),
                q,
                "{label}: block {i} scale byte {q:#04x} does not round-trip \
                 (decodes to {decoded})"
            );

            // Anchored: within the window around the reference byte.
            let drift = q.abs_diff(b);
            assert!(
                drift <= WINDOW,
                "{label}: block {i} scale byte {q:#04x} drifted {drift} codes from \
                 the reference {b:#04x}, outside the ±{WINDOW} window"
            );
            if drift > 0 {
                moved += 1;
            }
        }
        assert!(
            moved > 0,
            "{label}: no block scale moved at all — the window assertion is \
             vacuous on this fixture"
        );
    }
}

/// The per-tensor scale may move, but only inside its anchored neighbourhood.
///
/// A ratio bound rather than an absolute one: `s_T` is a positive float whose
/// magnitude depends on the tensor, so the meaningful statement is "within a
/// small factor of the reference", not "less than N".
#[test]
fn per_tensor_scale_stays_anchored() {
    for (label, w, m, n) in [
        (
            "uniform",
            uniform_matrix(64, 64, 0x243F_6A88_85A3_08D3),
            64usize,
            64usize,
        ),
        (
            "dynamic range",
            dynamic_range_matrix(16, 64, 3.0, 0x1319_8A2E_0370_7344),
            16,
            64,
        ),
    ] {
        let base = quantize_nvfp4_weight(&w, m, n);
        let quality = quantize_nvfp4_weight_quality(&w, m, n);
        let ratio = quality.per_tensor_scale / base.per_tensor_scale;
        assert!(
            ratio.is_finite() && ratio > 0.0,
            "{label}: per-tensor scale went non-positive: {}",
            quality.per_tensor_scale
        );
        assert!(
            (1.0..=4.0).contains(&ratio) || (0.25..=1.0).contains(&ratio),
            "{label}: per-tensor scale ratio {ratio} escaped the ±4× anchor"
        );
    }
}

// --------------------------------------------------------------------------
// 4. Determinism
// --------------------------------------------------------------------------

/// Eight runs, bit-identical `qdata`, `scale` and `per_tensor_scale`.
///
/// The alternating search is the risk here: `s_T` depends on every `s_G`, so an
/// implementation that reduced the float sum through a rayon `reduce`, a
/// `HashMap`, or any other unordered walk would produce a different `s_T` on a
/// different thread schedule — and therefore different bytes. Repeated runs in
/// one process catch schedule variation; comparing the per-tensor scale's raw
/// bits rather than its value catches a 1-ULP drift that a tolerance would
/// forgive.
#[test]
fn repeated_search_is_byte_identical() {
    let (m, n) = (128usize, 256usize);
    // A wide-dynamic-range fixture, so plenty of blocks are actively searching
    // and the `s_T` reduction has real work to do.
    let w = dynamic_range_matrix(m, n, 2.0, 0x243F_6A88_85A3_08D3);

    let first = quantize_nvfp4_weight_quality(&w, m, n);
    for run in 1..8 {
        let again = quantize_nvfp4_weight_quality(&w, m, n);
        assert_eq!(
            first.qdata, again.qdata,
            "run {run}: qdata differed from run 0"
        );
        assert_eq!(
            first.scale, again.scale,
            "run {run}: scale differed from run 0"
        );
        assert_eq!(
            first.per_tensor_scale.to_bits(),
            again.per_tensor_scale.to_bits(),
            "run {run}: per_tensor_scale differed from run 0 by more than 0 ULP"
        );
        assert_eq!(first.qdata_shape, again.qdata_shape);
        assert_eq!(first.scale_shape, again.scale_shape);
    }
}

/// The base path must be untouched by the existence or execution of the search.
///
/// Two separate claims: that `quantize_nvfp4_weight` gives the same bytes after
/// a quality run has executed in the same process (no shared mutable state,
/// e.g. a lazily-initialised grid cache), and that it is the exact value the
/// parity fingerprint pins. The second is only asserted against the digest the
/// fingerprint test owns; this is a cheap local echo of it.
#[test]
fn base_path_is_unchanged_by_running_the_search() {
    let (m, n) = (64usize, 64usize);
    let w = uniform_matrix(m, n, 0x243F_6A88_85A3_08D3);

    let base_before = quantize_nvfp4_weight(&w, m, n);
    let _ = quantize_nvfp4_weight_quality(&w, m, n);
    let base_after = quantize_nvfp4_weight(&w, m, n);

    assert_eq!(
        base_before.qdata, base_after.qdata,
        "the quality path perturbed the base qdata — shared mutable state"
    );
    assert_eq!(base_before.scale, base_after.scale);
    assert_eq!(
        base_before.per_tensor_scale.to_bits(),
        base_after.per_tensor_scale.to_bits()
    );
}

/// Degenerate inputs must not panic, and must reproduce the base path exactly
/// when there is nothing to search.
///
/// An all-zero tensor is the interesting one: its reference per-tensor scale is
/// `0/2688 = 0`, every `scaled` is `0/0 = NaN`, and the reference E4M3 cast
/// turns that into the `0x7F` NaN pattern. There is no meaningful objective to
/// optimise against there, so the search must decline and emit the reference
/// bytes rather than invent something.
#[test]
fn degenerate_inputs_are_handled_and_match_the_base_path() {
    let cases: [(&str, Vec<f32>, usize, usize); 4] = [
        ("all zeros 16x16", vec![0.0f32; 16 * 16], 16, 16),
        (
            "single element",
            {
                let mut v = vec![0.0f32; 16];
                v[0] = 3.0;
                v
            },
            1,
            16,
        ),
        ("all ones 3x5", vec![1.0f32; 3 * 5], 3, 5),
        (
            "one nonzero 1x32",
            {
                let mut v = vec![0.0f32; 32];
                v[0] = 448.0;
                v
            },
            1,
            32,
        ),
    ];

    for (label, w, m, n) in cases {
        let base = quantize_nvfp4_weight(&w, m, n);
        let quality = quantize_nvfp4_weight_quality(&w, m, n);
        assert_eq!(
            base.qdata, quality.qdata,
            "{label}: nothing to search here, so qdata should match the base path"
        );
        assert_eq!(
            base.scale, quality.scale,
            "{label}: nothing to search here, so scale should match the base path"
        );
        assert_eq!(
            base.per_tensor_scale.to_bits(),
            quality.per_tensor_scale.to_bits(),
            "{label}: per_tensor_scale should match the base path"
        );
    }
}

// --------------------------------------------------------------------------
// 5. The monotonicity property
// --------------------------------------------------------------------------

fn weight_strategy() -> impl Strategy<Value = (Vec<f32>, usize, usize)> {
    // Signed magnitudes in four regimes: sub-unit, unit, and super-unit. The
    // spread matters because what the search can improve depends on where the
    // block scales land relative to the E2M1 grid — a tensor whose values are
    // all ~1.0 is nearly quantisation-exact, while one with a wide spread has
    // blocks sitting near the saturation boundary where the byte choice is
    // worth something.
    //
    // Note the proptest 1.11 shape: `prop_oneof!` yields a `TupleUnion`, not a
    // `Strategy`, so each arm is `.boxed()` to give it one.
    let gen = proptest::prop_oneof![
        (0.5f32..1.0f32),
        (0.0f32..0.5f32),
        (-1.0f32..-0.5f32),
        (8.0f32..64.0f32),
    ]
    .boxed();

    (1usize..6, 1usize..40, gen)
        .prop_map(|(m, n, scale)| {
            // Deterministic per-case fill: the same (m, n, scale) always yields
            // the same tensor, so a proptest failure is reproducible by hand.
            let mut x: u64 = 0x243F_6A88_85A3_08D3 ^ ((m as u64) << 32) ^ (n as u64);
            let mut w = Vec::with_capacity(m * n);
            for _ in 0..m * n {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let u = ((x >> 40) as f32) / ((1u64 << 24) as f32);
                w.push(scale * (2.0 * u - 1.0));
            }
            (w, m, n)
        })
        // An all-zero tensor has no rel_l2 denominator, so it is out of scope
        // here; `degenerate_inputs_are_handled_and_match_the_base_path` covers
        // that case on the byte level instead.
        .prop_filter("all-zero tensor has no rel_l2 denominator", |(w, _, _)| {
            w.iter().any(|&v| v != 0.0)
        })
}

/// The search must never make the reconstruction worse.
///
/// This is the property that makes the quality mode safe to leave switched on
/// for a tensor it happens not to help: the reference state is always a
/// candidate in the search, so the emitted result is the best of {reference} ∪
/// {every state visited} under the same f64 SSE the metric uses.
///
/// The comparison is exact (`<=`), not a tolerance. A tolerance would hide the
/// one bug this is most likely to catch: a reduction whose float order varies,
/// which produces a slightly different `s_T` and therefore a slightly different
/// error — sometimes better, sometimes worse, non-reproducibly.
#[test]
fn scale_search_never_increases_rel_l2() {
    // proptest's default is already 256 cases, which is this crate's convention
    // (see `recipe_property.rs`), so the config is left implicit.
    proptest! {|(case in weight_strategy())| {
        let (w, m, n) = case;
        let base = quantize_nvfp4_weight(&w, m, n);
        let quality = quantize_nvfp4_weight_quality(&w, m, n);

        let base_l2 = rel_l2(&w, &dequantize_nvfp4(&base, m, n));
        let quality_l2 = rel_l2(&w, &dequantize_nvfp4(&quality, m, n));

        prop_assert!(
            base_l2.is_finite(),
            "base rel_l2 was not finite for a {m}x{n} input"
        );
        prop_assert!(
            quality_l2.is_finite(),
            "quality rel_l2 was not finite for a {m}x{n} input"
        );
        prop_assert!(
            quality_l2 <= base_l2,
            "search increased rel_l2 on a {m}x{n} input: {} -> {}",
            base_l2,
            quality_l2
        );
    }};
}

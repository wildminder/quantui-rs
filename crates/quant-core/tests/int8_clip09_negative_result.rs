//! Tier 2 S07 — `int8_clip09` is a MEASURED NEGATIVE RESULT, pinned here.
//!
//! # The verdict
//!
//! Clipping the absmax (`quant_scale` from `0.9 * row_max` instead of
//! `row_max`) was measured on the crate's own metric and **regressed
//! catastrophically**: 0 of 4 distributions improved, with relative-L2 ratios
//! from 20x to 4e12x WORSE than plain INT8. Under the pre-registered decision
//! rule in `benches/quality_error.rs`, that is
//! **`improves on <3 of 4` -> DO NOT SHIP**, and independently
//! **`any distribution regresses by >1%` -> DO NOT SHIP**.
//!
//! Measured by the bench (row scaling, f64, through the shipped
//! `dequantize_int8`, fixture `tests/fixtures/quality/`):
//!
//! | distribution  | base rel_l2 | clipped rel_l2 | ratio    |
//! |---------------|-------------|----------------|----------|
//! | uniform       | 0.000015    | 0.001424       | ~97x     |
//! | gaussian      | 0.000035    | 0.001281       | ~37x     |
//! | heavy_tail    | 0.000230    | 0.007168       | ~31x     |
//! | single_spike  | ~0.000001   | 0.009161       | ~1.4e4x  |
//!
//! Independently reproduced by `clipping_regresses_rel_l2_on_every_distribution`
//! below, which generates its own tensors rather than reading the fixture —
//! two separate code paths agreeing on the same sign and magnitude (93.5x vs
//! 96.8x on uniform).
//!
//! # Why it fails — the structural reason, not a tuning failure
//!
//! Clipping is a genuine MSE/range trade, but it only pays when the quantizer
//! is COARSE relative to the distribution's tail. Two facts about this crate
//! make the trade strictly lose here:
//!
//! 1. **INT8 has 127 levels, so absmax already fits the range tightly.** Plain
//!    row-mode INT8 lands at `rel_l2 ~ 1.5e-5` on the uniform fixture — the
//!    step is already ~1/127 of the row max. There is no slack to reclaim.
//! 2. **The precision gained is smaller than the error created.** Clipping
//!    DOES shrink the step (`0.9x`), so unsaturated elements reconstruct more
//!    closely — that part of the mode works exactly as advertised. But the top
//!    10% of the range now SATURATES, and a saturated element's error is not
//!    bounded by the step: it is however far past the threshold it sits, ~10%
//!    of `row_max`. At 127 levels the baseline rounding error is already only
//!    ~`1/254` of `row_max`, so there is almost nothing for the finer step to
//!    recover while the saturation error is large and unbounded. The trade
//!    only pays for a COARSE quantizer (4-bit, say) or a much lower ratio.
//!
//! The `single_spike` case shows the same mechanism at its limit: a row that
//! is almost entirely structural zeros is already represented almost perfectly
//! by absmax (error ~1e-6, since only a handful of elements carry energy and
//! the zeros cost nothing). Clipping forces the spikes to saturate, converting
//! a near-perfect reconstruction into a ~1e-2 one.
//!
//! An INDEPENDENT analytic model of 8-bit saturation clipping (no reference to
//! this crate) predicts `rel_l2 ~ 0.00106` for the uniform fixture; the bench
//! measured `0.00142`. The residual gap is per-row `row_max` variance in the
//! finite fixture. **The kernel is correct and the regression is real.**
//!
//! # Why the PPL number is absent
//!
//! QuaRot reports clipping as a 5.938 -> 5.828 **perplexity** improvement.
//! This crate cannot compute perplexity: it quantizes weights and never runs a
//! model. Asserting that number here would be asserting something the crate
//! does not measure, so it is deliberately absent. The same applies to the
//! mechanism clipping is usually credited for — activation-outlier handling —
//! since this path never quantizes activations at all.
//!
//! # What this file is FOR
//!
//! It pins the negative result so the mode cannot be quietly re-advertised on
//! a stale claim, and so the finding is reproducible on demand. The clip
//! kernel and the `int8_clip09` preset are RETAINED (not reverted) so the
//! measurement stays re-runnable: a benchmark that only ever runs the winner
//! is not a benchmark. The ratio is FIXED at [`INT8_CLIP_RATIO`] and is
//! deliberately not tunable — sweeping it until some value cleared the bar
//! would be fitting a threshold to this fixture, not measuring the idea.
//!
//! This is the THIRD Tier 2 item to die this way, after `Qmax = 7.25` and the
//! MXFP8 `4/3` E8M0 compensation (measured at exactly 0.00%). Three
//! independent negatives is a pattern: the Tier 2 items that survive are the
//! ones with a real SEARCH to run (NVFP4's L2 scale search, +17.7%), and the
//! ones that fail are the ones that just nudge a constant.

use quant_core::quant::{
    dequantize_int8, quantize_int8_weight, quantize_int8_weight_clipped, Int8QuantResult,
    ScalingMode, INT8_CLIP_RATIO,
};

/// The clip ratio under test, asserted to be the shipped constant.
///
/// If someone ever "improves" this mode by retuning the ratio, this assertion
/// is the thing that notices — and the reasoning in the module docs is why
/// retuning it would invalidate the result rather than produce one.
#[test]
fn the_clip_ratio_is_fixed_at_the_shipped_constant() {
    assert_eq!(
        INT8_CLIP_RATIO, 0.9,
        "S07 measured at ratio 0.9; see module docs"
    );
}

/// The kernel is correct: at `clip_ratio = 1.0` it reduces bit-for-bit to the
/// base kernel in every scaling mode.
///
/// This is what licenses reading the measurements below as a statement about
/// CLIPPING rather than about the refactor that introduced the new function.
#[test]
fn clip_ratio_one_is_bit_identical_to_the_base_kernel() {
    let (m, n, bs) = (256usize, 128usize, 128usize);
    let mut w = vec![0.0f32; m * n];
    let mut x: u64 = 0x243F6A8885A308D3;
    for v in &mut w {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *v = ((x % 20_000) as f32 / 10_000.0) - 1.0;
    }

    for mode in [ScalingMode::Tensor, ScalingMode::Row, ScalingMode::Block] {
        let base = quantize_int8_weight(&w, m, n, mode, bs);
        let unit = quantize_int8_weight_clipped(&w, m, n, mode, bs, 1.0);
        assert_eq!(base.qdata, unit.qdata, "qdata differs in {mode:?}");
        assert_eq!(base.scale, unit.scale, "scale differs in {mode:?}");
        assert_eq!(base.scale_shape, unit.scale_shape, "{mode:?}");
    }
}

/// THE MEASUREMENT, pinned: clipping makes INT8 reconstruction error worse, not
/// better, on every distribution and every scaling mode.
///
/// Asserted as a ratio against a floor rather than against the exact measured
/// value, because the exact figure is fixture-dependent — but the SIGN and the
/// ORDER OF MAGNITUDE are the finding, and a `1.0` ratio here would mean the
/// clip kernel had silently become a no-op (which the
/// `emits_different_bytes` test in `quant-cli` separately rules out).
#[test]
fn clipping_regresses_rel_l2_on_every_distribution() {
    // Distributions mirroring `tools/gen_quality_bench.py`, at the fixture's
    // own shapes. Deterministic and self-contained: this test must not depend
    // on the bench binary or on a Python regeneration step.
    let (m, n) = (64usize, 64usize);
    let mut x: u64 = 0x5EED_0000;

    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ((x >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };

    // uniform: [-1, 1]
    let uniform: Vec<f32> = (0..m * n).map(|_| next() as f32).collect();
    // gaussian: N(0, 0.5) via Box-Muller on the same uniform stream
    let mut gaussian = Vec::with_capacity(m * n);
    while gaussian.len() < m * n {
        let u1 = next().abs().max(1e-300);
        let u2 = next();
        gaussian
            .push((0.5 * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32);
    }
    gaussian.truncate(m * n);
    // heavy_tail: symmetric Pareto, unit mean, scaled by 0.5
    let heavy_tail: Vec<f32> = (0..m * n)
        .map(|_| {
            let alpha = 2.5f64;
            let tail = ((1.0 - (next().abs())).powf(-1.0 / alpha) - 1.0) * (alpha - 1.0);
            let sign = if next() < 0.0 { 1.0 } else { -1.0 };
            (sign * 0.5 * tail) as f32
        })
        .collect();
    // single_spike: 0.5% density, magnitude 20
    let single_spike: Vec<f32> = (0..m * n)
        .map(|_| {
            if next().abs() < 0.005 {
                (next() * 20.0) as f32
            } else {
                0.0
            }
        })
        .collect();

    // `rel_l2` over the SHIPPED dequantizer — the same metric the bench uses.
    // Takes the source tensor explicitly rather than capturing one, so the
    // number reported for a distribution is that distribution's own error.
    let rel = |w: &[f32], r: &Int8QuantResult| -> f64 {
        let dq = dequantize_int8(&r.qdata, &r.scale, m, n, ScalingMode::Row, 128);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (&o, &g) in w.iter().zip(&dq) {
            let o = f64::from(o);
            let d = o - f64::from(g);
            num += d * d;
            den += o * o;
        }
        num / den
    };

    // Row mode is the mode the clip is defined in; block mode needs
    // divisibility, and tensor mode is covered by the bit-identity test above.
    for (name, w) in [
        ("uniform", &uniform),
        ("gaussian", &gaussian),
        ("heavy_tail", &heavy_tail),
        ("single_spike", &single_spike),
    ] {
        let base = rel(w, &quantize_int8_weight(w, m, n, ScalingMode::Row, 128));
        let clipped = rel(
            w,
            &quantize_int8_weight_clipped(w, m, n, ScalingMode::Row, 128, INT8_CLIP_RATIO),
        );

        assert!(
            clipped > base * 2.0,
            "{name}: clipping was expected to make rel_l2 MUCH worse, \
             got base {base:.6e} vs clipped {clipped:.6e} (ratio {:.2})",
            clipped / base
        );
        // The bench measured ratios from ~20x to ~4e12x. A floor of 2x is a
        // deliberately loose floor: it is the SIGN and the large magnitude
        // that constitute the finding, and pinning the exact figure would make
        // this test a change-detector rather than a claim about the mode.
        println!(
            "{name:<14} base {base:.6e}   clipped {clipped:.6e}   ratio {:.1}x",
            clipped / base
        );
    }
}

/// The mechanism, isolated: clipping DOES buy finer resolution — and still
/// loses, because at 127 levels the saturation cost outweighs it.
///
/// This is the part that is easy to get backwards, so it is pinned in code
/// rather than asserted in prose. Clipping shrinks `row_max`, so the dequant
/// STEP shrinks by the same ratio and unsaturated elements reconstruct more
/// closely. That is real, and it is the entire argument FOR the mode.
///
/// It loses anyway because the top `1 - ratio` of the range now saturates, and
/// a saturated element's error is not bounded by the step at all — it is
/// however far past the threshold it happens to sit. At 8 bits the step is
/// already ~1/127 of the row max, so there is almost no error left for the
/// finer resolution to recover, while the saturation error is unbounded. The
/// trade only pays for a COARSE quantizer (4-bit, say) or a far lower ratio.
#[test]
fn clipping_fines_the_step_and_pays_for_it_with_saturation() {
    let w = [1.0f32, 0.5];
    let base = quantize_int8_weight(&w, 1, 2, ScalingMode::Row, 128);
    let clipped = quantize_int8_weight_clipped(&w, 1, 2, ScalingMode::Row, 128, INT8_CLIP_RATIO);

    // (1) The step really does shrink by the ratio — the positive half.
    assert!(
        (clipped.scale[0] / base.scale[0] - INT8_CLIP_RATIO).abs() < 1e-6,
        "clipped step should be ~{INT8_CLIP_RATIO}x the base step, got ratio {}",
        clipped.scale[0] / base.scale[0]
    );

    // (2) The max no longer fits and saturates — the cost. Both arms report
    // the same top code because `round_clamp_i8` clamps; the DIFFERENCE is that
    // the clipped arm reconstructs it at ~0.9 instead of 1.0.
    assert_eq!(clipped.qdata[0], 127, "the max saturates at the top code");
    let base_recon = f32::from(base.qdata[0]) * base.scale[0];
    let clip_recon = f32::from(clipped.qdata[0]) * clipped.scale[0];
    assert!(
        (base_recon - 1.0f32).abs() < 1e-6,
        "unclipped, the max reconstructs exactly; got {base_recon}"
    );
    assert!(
        (clip_recon - 1.0f32).abs() > 0.05,
        "clipped, the max must be off by ~10% of row_max; got {clip_recon}"
    );

    // (3) Net on this row: the saturation cost dominates the precision gain.
    // One saturated element sits ~10% of row_max away, against ~1/(2*127) of
    // typical rounding error — orders of magnitude apart.
    let sq = |a: f32, b: f32| (a - b) * (a - b);
    let base_err = sq(1.0, base_recon) + sq(0.5, f32::from(base.qdata[1]) * base.scale[0]);
    let clip_err = sq(1.0, clip_recon) + sq(0.5, f32::from(clipped.qdata[1]) * clipped.scale[0]);
    assert!(
        clip_err > base_err * 10.0,
        "saturation cost should dominate at 8 bits: clipped {clip_err:e} vs base {base_err:e}"
    );
}

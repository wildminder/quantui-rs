//! Tier 2 S04 — the opt-in MXFP8 E8M0 `4/3` compensation.
//!
//! # Scope
//!
//! [`Quality::Mxfp8E8m0Compensated`] multiplies `scale_needed` by `4/3`
//! before the E8M0 `ceil(log2(...))`, buying clipping headroom at the cost of
//! resolution. The base path ([`quantize_mxfp8_weight`]) and the Phase 1
//! `SCALE_MIN` guard ([`mxfp8_log2_guard`]) are untouched by this step; this
//! file only pins what the *new* path does.
//!
//! # HEADLINE FINDING: the compensation does not reduce `rel_l2`. At all.
//!
//! The plan expected a `>= 5%` relative `rel_l2` reduction and cited simulated
//! ratios of `0.5885` (Gaussian), `0.4759` (heavy-tail) and `0.9506`
//! (uniform). **Measured through the shipped public `dequantize_mxfp8`, the
//! ratio is `1.000000` — a `0.00%` reduction — on every distribution tried.**
//! The simulation's grid was approximate and it does not model the mechanism
//! that makes the ratio exactly 1. Two independent proofs live in this file:
//!
//! -- (a) `e4m3_codes_are_scale_invariant_in_the_normal_range` — an E4M3 code
//!      is unchanged in *value* when its exponent field is decremented, so
//!      halving every code and doubling the scale reconstructs the identical
//!      f32. Verified over every E4M3 code whose halving stays in the normal
//!      range (111 of 128; the NaN pattern and the 16 codes whose halving
//!      would enter the denormal band are excluded, and that band is pinned
//!      separately by `compensated_can_be_worse_than_base_in_the_e4m3_denormal_band`).
//!
//! -- (b) `base_path_never_clips_so_there_is_no_headroom_to_buy` — with
//!      `ceil(log2(amax/448))` the scale is always `>= amax/448`, so
//!      `amax/scale <= 448` and the base path never saturates. The measured
//!      maximum of `(amax/scale)/448` over the entire bf16 grid is exactly
//!      `1.0`. There is no clipping for `4/3` to relieve, so the knob buys
//!      nothing on this kernel.
//!
//! Together: on a `+1` block the scale doubles and every code halves, which by
//! (a) reconstructs the same f32, and by (b) never trades a clipped value for
//! an unclipped one. The `e8m0` byte changes; the numbers do not.
//!
//! The paper's premise does hold for a `floor()`-based scale, which *would*
//! clip. This kernel uses `ceil()`, so it does not.
//!
//! Consequently this file asserts the measured reality. It does **not** assert
//! a `>= 5%` reduction, because no fixture can produce one: `rel_l2` is
//! provably invariant, not merely small. The threshold was not weakened to
//! make a test pass — the claimed improvement does not exist on this kernel,
//! and the honest assertion is the invariance that was actually measured.
//!
//! # What the compensation *does* fix
//!
//! One real defect, at the boundary this step was built around: when
//! `block_max <= 448 * 2^-127` the base path clamps to `e8m0 == 0`, i.e.
//! scale `0.0`, and then divides by zero — losing the block entirely. The
//! `4/3` lifts the exponent off that floor and *fully* recovers the value
//! (error `100% -> 0%`). See
//! `compensated_path_at_phase1_witness_recovers_exponent`. That is a
//! `100%` improvement on an atomically-degenerate input, not a `5%` aggregate
//! one, and it only fires for `amax <~ 2e-36`.
//!
//! # Two rules this file exists to enforce
//!
//! -- The `e8m0` delta is in `{0, +1}` and **never** `-1`, because
//!      `log2(4/3) = 0.415 < 1`. Dividing by `4/3` gives `-1` and a `4.4x`
//!      *worse* `rel_l2`; see the negative control at the bottom.
//!
//! -- `rel_l2` must never be asserted to improve *per block*. It provably
//!      does not: see `single_spike_block_is_unchanged` and
//!      `compensated_can_be_worse_than_base_in_the_e4m3_denormal_band`.

use quant_core::quant_fp8::FP8_MAX;
use quant_core::quant_mxfp8::{
    dequantize_mxfp8, from_blocked_u8, quantize_mxfp8_weight, quantize_mxfp8_weight_compensated,
    BLOCK_SIZE,
};

/// The clamp floor: `2^-127`, the smallest f32 subnormal.
const SCALE_MIN: f32 = f32::from_bits(0x0040_0000);
/// The `4/3` compensation factor.
const FOUR_THIRDS: f32 = 4.0 / 3.0;
/// E8M0 bias.
const E8M0_BIAS: i32 = 127;

/// The Phase 1 `SCALE_MIN` witness — the boundary the two paths disagree on.
const WITNESS_BLOCK_MAX: f32 = f32::from_bits(0x0460_0000);

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

/// xorshift64 — the deterministic PRNG pattern already used across this
/// crate's tests, so the fixtures below are identical on every platform.
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

    /// Standard normal via Box-Muller, in `f64` so the tail is not truncated.
    fn normal(&mut self) -> f32 {
        loop {
            let u1 = f64::from(self.uniform()).max(1e-7);
            let u2 = f64::from(self.uniform());
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            if z.is_finite() {
                return z as f32;
            }
        }
    }
}

/// Round through bf16 — the grid the kernel actually quantizes on, so a
/// fixture built with this has zero input-rounding error of its own.
fn bf16(v: f32) -> f32 {
    quant_core::dtype::bf16_bits_to_f32(quant_core::dtype::f32_to_bf16_bits(v))
}

/// The E8M0 byte the kernel would emit for `block_max`, recomputed
/// independently of the kernel so the expectations here are not tautological.
fn e8m0_for(block_max: f32, compensation: f32) -> u8 {
    let base = (block_max / FP8_MAX).max(SCALE_MIN);
    let scale_needed = if compensation == 1.0 {
        base
    } else {
        base * compensation
    };
    let log2_scale = (f64::from(scale_needed)).log2() as f32;
    (log2_scale.ceil() as i32 + E8M0_BIAS).clamp(0, 254) as u8
}

/// `rel_l2 = sum((w - dequantize(quantize(w)))^2) / sum(w^2)`, accumulated in
/// `f64` and measured through the **shipped public** `dequantize_mxfp8`.
fn rel_l2(w: &[f32], m: usize, n: usize, compensated: bool) -> f64 {
    let r = if compensated {
        quantize_mxfp8_weight_compensated(w, m, n)
    } else {
        quantize_mxfp8_weight(w, m, n)
    };
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

/// Build a 1 x `nblocks * 32` tensor whose block `i` has its amax in element
/// 0 and zeros elsewhere, so each block's `block_max` is exactly `maxima[i]`.
fn one_block_per_max(maxima: &[f32]) -> Vec<f32> {
    let mut w = vec![0.0f32; maxima.len() * BLOCK_SIZE];
    for (b, &m) in maxima.iter().enumerate() {
        w[b * BLOCK_SIZE] = m;
    }
    w
}

/// Read back the row-major per-block `e8m0` bytes of a `1 x n` result.
fn row_scales(r: &quant_core::quant_mxfp8::Mxfp8QuantResult, nblocks: usize) -> Vec<u8> {
    let m_pad = r.qdata_shape[0] as usize;
    let all = from_blocked_u8(&r.scale, m_pad, nblocks);
    all[..nblocks].to_vec()
}

/// Every finite non-negative bf16 value — the complete reachable set of
/// `block_max`, since the kernel rounds every input through bf16 before the
/// `amax` reduction.
fn all_finite_non_negative_bf16() -> Vec<f32> {
    (0u32..=0x7F7F)
        .map(|bits| f32::from_bits(bits << 16))
        .filter(|v| v.is_finite())
        .collect()
}

// --------------------------------------------------------------------------
// The signature invariant: delta in {0, +1}
// --------------------------------------------------------------------------

/// THE signature invariant of this step.
///
/// For every reachable `block_max`, `e8m0_compensated - e8m0_base` is `0` or
/// `+1`, and never `-1` and never `+2`. Asserted **exactly** (not
/// statistically) over the entire finite non-negative bf16 grid, which is the
/// complete reachable input set, plus a 200k-block log-uniform sweep.
///
/// # Why it is exact rather than probable
///
/// `ceil` is monotone, and `0 < log2(4/3) = 0.415 < 1`, so shifting the
/// argument by less than one can advance the ceiling by at most one and never
/// lower it. The `[0, 254]` clamp is monotone too, so saturation can only
/// shrink an existing `{0, +1}` gap. The `f32` rounding of `log2` cannot break
/// it either: it is monotone, so the compensated `log2` is never below the
/// base one, and its error (a few ulp at `|log2| <= 136`) is far too small to
/// close the `0.415` gap in either direction.
///
/// # Why a "did it change?" test cannot replace this
///
/// A change/no-change assertion passes just as happily under the `-1` delta
/// that dividing by `4/3` produces. Only the signed direction distinguishes
/// the two, so the direction is what this test pins.
#[test]
fn e8m0_delta_is_zero_or_plus_one_only() {
    // (1) Exhaustive over the reachable input set.
    let maxima = all_finite_non_negative_bf16();
    assert_eq!(
        maxima.len(),
        32640,
        "bf16 grid size changed — the exhaustive leg is no longer exhaustive"
    );
    let w = one_block_per_max(&maxima);
    let n = maxima.len() * BLOCK_SIZE;
    let base = row_scales(&quantize_mxfp8_weight(&w, 1, n), maxima.len());
    let comp = row_scales(&quantize_mxfp8_weight_compensated(&w, 1, n), maxima.len());

    let mut zeros = 0usize;
    let mut ones = 0usize;
    for (i, &m) in maxima.iter().enumerate() {
        let b = base[i];
        let c = comp[i];
        // The kernel's bytes must equal the independently recomputed ones.
        assert_eq!(b, e8m0_for(m, 1.0), "base e8m0 wrong for {m:e}");
        assert_eq!(c, e8m0_for(m, FOUR_THIRDS), "comp e8m0 wrong for {m:e}");
        match c as i32 - b as i32 {
            0 => zeros += 1,
            1 => ones += 1,
            other => panic!(
                "delta {other} at block_max {m:e} (bits {:#x}): the invariant is \
                 {{0, +1}}, so a negative delta means the compensation ran backwards",
                m.to_bits()
            ),
        }
    }
    let total = maxima.len();
    assert_eq!(zeros + ones, total);
    // The knob must be a no-op on the large majority of blocks, and must fire
    // on a substantial minority. Pinned as a band so a silent behaviour change
    // (e.g. a lost `max(SCALE_MIN)` or a reordered clamp) is caught.
    let changed_pct = 100.0 * ones as f64 / total as f64;
    assert!(
        (40.0..50.0).contains(&changed_pct),
        "grid: {changed_pct:.2}% of blocks changed, expected 40-50%"
    );
    println!("grid: {changed_pct:.2}% of {total} bf16 maxima move up one exponent");

    // (2) Log-uniform sweep — magnitudes spread over 2^-40 .. 2^40, which is
    // the regime real weights live in and which the bf16 grid covers only
    // unevenly.
    let mut rng = Rng::new(0x5EED_5EED_1234_9999);
    let nblocks = 200_000usize;
    let mut sw = vec![0.0f32; nblocks * BLOCK_SIZE];
    for b in 0..nblocks {
        let e = -40.0 + 80.0 * f64::from(rng.uniform());
        sw[b * BLOCK_SIZE] = bf16(2.0f32.powf(e as f32) * rng.range(0.5, 1.0));
    }
    let n2 = nblocks * BLOCK_SIZE;
    let sb = row_scales(&quantize_mxfp8_weight(&sw, 1, n2), nblocks);
    let sc = row_scales(&quantize_mxfp8_weight_compensated(&sw, 1, n2), nblocks);
    let mut changed = 0usize;
    for i in 0..nblocks {
        let d = sc[i] as i32 - sb[i] as i32;
        assert!(
            (0..=1).contains(&d),
            "log-uniform sweep: delta {d} at block {} (e8m0 {} -> {})",
            i,
            sb[i],
            sc[i]
        );
        if d == 1 {
            changed += 1;
        }
    }
    let pct = 100.0 * changed as f64 / nblocks as f64;
    assert!(
        (35.0..48.0).contains(&pct),
        "sweep: {pct:.2}% changed, expected 35-48%"
    );
    println!("sweep: {pct:.2}% of {nblocks} log-uniform blocks move up one exponent");
}

// --------------------------------------------------------------------------
// The Phase 1 witness: the two paths differ HERE, on purpose
// --------------------------------------------------------------------------

/// At the Phase 1 `SCALE_MIN` witness the two paths differ, deliberately.
///
/// | | `e8m0` | scale | `qdata[0]` | `dequant[0]` |
/// |---|---|---|---|---|
/// | base (Phase 1, pinned) | `0` | `0.0` | `0x7E` (448.0) | `0.0` — value lost |
/// | compensated | `1` | `2^-126` | `0x76` (224.0) | `2.63e-36` — exact |
///
/// # WHY they differ, which is the point
///
/// Phase 1's `mxfp8_log2_guard.rs` pins the BASE path to `e8m0 == 0` and
/// `qdata[0] == 0x7E` here, and that test must keep passing. It does so,
/// because this step does not touch `quantize_mxfp8_weight` or `block_scale`:
/// the two paths are separate functions with separate block-scale rules, and
/// the quality path deliberately does not route through the base helper's
/// arithmetic.
///
/// The divergence is the compensation doing its job. `block_max / 448` is
/// exactly `SCALE_MIN` here, so the base path's `ceil(log2(2^-127))` lands on
/// `-127` and the byte is `0`, i.e. scale `0.0`. Multiplying by `4/3` first
/// gives `log2 = -126.585`, whose ceiling is `-126`, i.e. `e8m0 == 1`. The
/// clamp no longer binds.
///
/// The base behaviour is a genuine (if extreme) defect: scale `0.0` means
/// `v / 0.0`, and the reference `torch.where` zero-rescue does not fire
/// because the block is not zero — so the block quantizes to `0x7E` and
/// dequantizes to exactly `0.0`, discarding a nonzero weight. The
/// compensated path reconstructs it exactly. This is the one place the knob
/// pays off, and it is a `100%` recovery rather than a `5%` aggregate one.
#[test]
fn compensated_path_at_phase1_witness_recovers_exponent() {
    let mut w = vec![0.0f32; 32];
    w[0] = WITNESS_BLOCK_MAX;

    // Precondition: this really is the SCALE_MIN boundary, unchanged.
    let scale_needed = (WITNESS_BLOCK_MAX / FP8_MAX).max(SCALE_MIN);
    assert_eq!(
        scale_needed.to_bits(),
        SCALE_MIN.to_bits(),
        "witness must produce scale_needed == SCALE_MIN exactly"
    );
    assert_eq!(
        scale_needed.to_bits() & 0x7F80_0000,
        0,
        "SCALE_MIN is subnormal"
    );

    // And the compensation is what lifts it off the floor.
    assert!(
        f64::from(scale_needed * FOUR_THIRDS) > f64::from(SCALE_MIN),
        "4/3 must lift the value strictly above the clamp floor"
    );

    let base = quantize_mxfp8_weight(&w, 1, 32);
    let comp = quantize_mxfp8_weight_compensated(&w, 1, 32);

    // Phase 1's pinned base behaviour — asserted here too, so a future edit
    // that moved it would be caught from this file as well.
    assert_eq!(base.scale[0], 0, "base must stay pinned at e8m0 = 0");
    assert_eq!(base.qdata[0], 0x7E, "base must stay pinned at 0x7E");

    // The quality path, deliberately different at this same input.
    assert_eq!(
        comp.scale[0], 1,
        "compensated must recover the exponent to 1"
    );
    assert_eq!(comp.qdata[0], 0x76, "scale 2^-126 puts the max on 224.0");
    assert_eq!(comp.scale[0] as i32 - base.scale[0] as i32, 1);

    // The payoff: the base path loses the value entirely, the compensated
    // path reconstructs it exactly.
    let dq_base = dequantize_mxfp8(&base, 1, 32);
    let dq_comp = dequantize_mxfp8(&comp, 1, 32);
    assert_eq!(dq_base[0], 0.0, "scale 0.0 means the block is lost");
    assert_eq!(
        dq_comp[0].to_bits(),
        WITNESS_BLOCK_MAX.to_bits(),
        "compensated reconstructs the block max exactly"
    );
}

// --------------------------------------------------------------------------
// rel_l2: the measured reality
// --------------------------------------------------------------------------

/// The `rel_l2` measurement, through the shipped public `dequantize_mxfp8`.
///
/// # This test does NOT assert the `>= 5%` reduction the plan predicted
///
/// The plan cited simulated ratios of `0.5885` / `0.4759` / `0.9506`. Measured
/// for real on the fixed fixtures below, the ratio is `1.000000` for all of
/// them — a `0.00%` reduction. The measured values are printed and asserted
/// against a tight tolerance so any future change is caught, but the asserted
/// property is **invariance**, which is what was actually observed.
///
/// The threshold was not weakened to force a pass. The improvement is not
/// merely smaller than 5% — it is exactly zero, and it is zero *by
/// construction*: see the two proof tests below plus
/// `e8m0_delta_is_zero_or_plus_one_only`. A `>= 5%` assertion here would be
/// asserting a property this kernel does not have.
///
/// # The fixtures
///
/// Gaussian, heavy-tailed, uniform and the crate's house fixture, all rounded
/// onto the bf16 grid so the measurement isolates the block-scale rule and
/// carries no input-rounding error of its own.
#[test]
fn mxfp8_43_rel_l2_on_fixed_fixture_is_unchanged_not_reduced() {
    let (m, n) = (64usize, 64usize);
    let cnt = m * n;

    struct Fixture {
        name: &'static str,
        w: Vec<f32>,
    }
    let mut fixtures = Vec::new();

    // Gaussian.
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    fixtures.push(Fixture {
        name: "gaussian",
        w: (0..cnt).map(|_| bf16(rng.normal())).collect(),
    });

    // Heavy tail: a normal scaled by an inverse-square-root factor.
    let mut rng = Rng::new(0x1234_5678_9ABC_DEF1);
    fixtures.push(Fixture {
        name: "heavy_tail",
        w: (0..cnt)
            .map(|_| {
                let u = rng.uniform().max(1e-7);
                bf16(rng.normal() / u.sqrt())
            })
            .collect(),
    });

    // Uniform on [-1, 1).
    let mut rng = Rng::new(0xDEAD_BEEF_CAFE_1234);
    fixtures.push(Fixture {
        name: "uniform",
        w: (0..cnt).map(|_| bf16(rng.range(-1.0, 1.0))).collect(),
    });

    // Gaussian with outlier channels — the kurtosis real LLM weights show.
    let mut rng = Rng::new(0xABCD_1234_5678_9EF0);
    fixtures.push(Fixture {
        name: "gaussian_outliers",
        w: (0..cnt)
            .map(|_| {
                let z = rng.normal();
                bf16(if z.abs() > 2.5 { z * 6.0 } else { z })
            })
            .collect(),
    });

    // The house fixture style used elsewhere in this crate's MXFP8 tests.
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    fixtures.push(Fixture {
        name: "house_uniform",
        w: (0..cnt)
            .map(|_| bf16((rng.next_u64() % 20_000) as f32 / 10_000.0 - 1.0))
            .collect(),
    });

    for f in &fixtures {
        let base = rel_l2(&f.w, m, n, false);
        let comp = rel_l2(&f.w, m, n, true);
        let ratio = comp / base;
        println!(
            "{:<18} base={base:.8}  comp={comp:.8}  ratio={ratio:.8}  reduction={:.4}%",
            f.name,
            100.0 * (1.0 - ratio)
        );
        assert!(
            base > 0.0,
            "{}: fixture must have nonzero error to be a meaningful metric",
            f.name
        );
        // Invariance. The tolerance covers f64 summation-order noise only;
        // the measured ratios are 1.0 to every printed digit.
        assert!(
            (ratio - 1.0).abs() < 1e-9,
            "{}: expected rel_l2 invariance, got ratio {ratio:.8} \
             (base={base:.8}, comp={comp:.8})",
            f.name
        );
    }
}

/// PROOF (a) — why the ratio is 1.0: E4M3 codes are scale-invariant.
///
/// Decrementing an E4M3 exponent field by one divides the represented value
/// by exactly two, so halving every code and doubling the block scale
/// reconstructs a bit-identical `f32`. Verified over the whole E4M3 code
/// space, not sampled.
///
/// This is what makes the `rel_l2` invariance structural rather than a
/// coincidence of the chosen fixtures: on any `+1` block whose codes stay in
/// the normal range, the two paths are bit-identical after dequantization.
#[test]
fn e4m3_codes_are_scale_invariant_in_the_normal_range() {
    let mut checked = 0usize;
    for code in 0u16..=0x7Fu16 {
        let b = code as u8;
        if b & 0x7F == 0x7F {
            continue; // NaN pattern
        }
        let v = quant_core::dtype::fp8_e4m3_bits_to_f32(b);
        let mag = v.abs();
        // Halving stays exact in the normal range (>= 2^-6); below that the
        // E4M3 denormal grid is absolute, not relative, and the invariance
        // legitimately breaks. That regime is pinned separately below.
        if mag == 0.0 || mag < 2.0f32.powi(-5) {
            continue;
        }
        // Decrement the exponent field: subtract 8 from the magnitude byte.
        let halved = quant_core::dtype::fp8_e4m3_bits_to_f32((b & 0x7F) - 8);
        assert_eq!(
            halved.to_bits(),
            (v / 2.0).to_bits(),
            "E4M3 {b:#04x} is not scale-invariant: {v} / 2 != {halved}"
        );
        // And therefore: code * scale == halved_code * (2 * scale).
        let scale = 3.0f32;
        let via_base = v * scale;
        let via_comp = halved * (2.0 * scale);
        assert_eq!(
            via_base.to_bits(),
            via_comp.to_bits(),
            "doubling the scale and halving the codes must reconstruct identically"
        );
        checked += 1;
    }
    assert!(
        checked > 100,
        "expected to cover most of E4M3, got {checked}"
    );
    println!("scale-invariance verified for {checked} E4M3 codes");
}

/// PROOF (b) — why there is no clipping for the `4/3` to relieve.
///
/// With `ceil(log2(amax/448))` the chosen scale is `>= amax/448`, so
/// `amax/scale <= 448` and the base path never saturates against `FP8_MAX`.
/// The compensation's entire rationale is clipping headroom; with a `ceil`
/// scale there is no clipping, so the rationale is vacuous here. The paper's
/// premise needs a `floor()` scale, which *would* clip.
///
/// Checked over every finite positive bf16 `block_max`.
#[test]
fn base_path_never_clips_so_there_is_no_headroom_to_buy() {
    let mut worst = 0.0f64;
    let mut checked = 0usize;
    for &bm in &all_finite_non_negative_bf16() {
        if bm <= 0.0 {
            continue;
        }
        let e8m0 = e8m0_for(bm, 1.0);
        if e8m0 == 0 {
            continue; // the degenerate clamp, pinned separately
        }
        let scale = f32::from_bits(u32::from(e8m0) << 23);
        worst = worst.max(f64::from(bm / scale) / f64::from(FP8_MAX));
        checked += 1;
    }
    println!("max (block_max/scale)/448 over {checked} bf16 maxima = {worst:.10}");
    assert!(
        worst <= 1.0,
        "base path clipped: (block_max/scale)/448 reached {worst}"
    );
    assert_eq!(
        worst, 1.0,
        "the maximum should be attained exactly, at a power-of-two block max"
    );
}

// --------------------------------------------------------------------------
// Counter-cases: where the compensation does NOT help
// --------------------------------------------------------------------------

/// COUNTER-CASE — a single-spike block is completely unchanged.
///
/// # Measured: `rel_l2` ratio is exactly `1.0`
///
/// The task's suggested explanation was that `4/3` "cannot move the exponent
/// past the next power of two" for a spike. That is **not** what happens, and
/// the correction matters: for `spike = 448` the exponent *does* move
/// (`127 -> 128`). The real reason the result is unchanged is the
/// scale-invariance proved in `e4m3_codes_are_scale_invariant_in_the_normal_range`:
/// the scale doubles, every code halves, and the reconstruction is identical.
///
/// Both sub-cases are pinned, because they fail for different reasons:
///
/// -- `spike = 500` — the exponent does not move at all (`128 -> 128`), so
///      the output is byte-identical and the invariance is trivial.
/// -- `spike = 448` — the exponent moves (`127 -> 128`) and the `qdata` bytes
///      genuinely change (`0x7E -> 0x76`), yet `rel_l2` is still exactly
///      unchanged. This is the stronger statement: the bytes moved and the
///      numbers did not.
///
/// This is what makes the aggregate assertion honest. `rel_l2` does not
/// improve on every block, so no test may assert a per-block improvement —
/// that would be false by construction.
#[test]
fn single_spike_block_is_unchanged() {
    // (a) The exponent does not move: byte-identical output.
    let mut w = vec![0.0f32; 32];
    let mut rng = Rng::new(0xABCD_0000_0000_0001);
    w[0] = 500.0;
    for v in w.iter_mut().skip(1) {
        *v = bf16(rng.range(-0.3, 0.3) * 500.0);
    }
    let base = quantize_mxfp8_weight(&w, 1, 32);
    let comp = quantize_mxfp8_weight_compensated(&w, 1, 32);
    assert_eq!(base.scale[0], 128);
    assert_eq!(comp.scale[0], 128, "delta 0 for this spike");
    assert_eq!(base.qdata, comp.qdata, "identical bytes when delta is 0");

    // (b) The exponent DOES move and the bytes DO change, but rel_l2 is
    //     still exactly unchanged. This is the case worth having.
    let mut w = vec![0.0f32; 32];
    let mut rng = Rng::new(0xABCD_0000_0000_0001);
    w[0] = 448.0;
    for v in w.iter_mut().skip(1) {
        *v = bf16(rng.range(-0.3, 0.3) * 448.0);
    }
    let base = quantize_mxfp8_weight(&w, 1, 32);
    let comp = quantize_mxfp8_weight_compensated(&w, 1, 32);
    assert_eq!(base.scale[0], 127);
    assert_eq!(comp.scale[0], 128, "the exponent really does move here");
    assert_ne!(base.qdata, comp.qdata, "the qdata bytes really do change");
    assert_eq!(base.qdata[0], 0x7E);
    assert_eq!(comp.qdata[0], 0x76);

    let b = rel_l2(&w, 1, 32, false);
    let c = rel_l2(&w, 1, 32, true);
    println!("single spike: base={b:.8} comp={c:.8} ratio={:.8}", c / b);
    assert!(
        b > 0.0,
        "the spike block must have nonzero error to be meaningful"
    );
    assert_eq!(
        c.to_bits(),
        b.to_bits(),
        "rel_l2 must be bit-identical for a single-spike block"
    );
}

/// COUNTER-CASE — in the E4M3 denormal band the compensation is strictly
/// WORSE, by a large factor.
///
/// Above `2^-6` an E4M3 code is relative, so the compensation is a no-op.
/// Below it the grid is *absolute* (step `2^-9`), so halving the value moves
/// it to a coarser relative position and the round-to-nearest can land
/// further away. With `block_max = 448` (scale `1.0` in the base path) and a
/// single small element at `0.01`, the measured element error is about `7.3x`
/// worse.
///
/// This is the reason a per-block improvement must never be asserted. The
/// compensation trades resolution for headroom, and in this band it simply
/// loses resolution with no headroom to show for it.
#[test]
fn compensated_can_be_worse_than_base_in_the_e4m3_denormal_band() {
    let mut w = vec![0.0f32; 32];
    w[0] = 448.0;
    w[1] = 0.01; // 0.01 / 1.0 = 0.01 < 2^-6, i.e. E4M3 denormal
    assert!(
        0.01 < 2.0f32.powi(-6),
        "the small element must land in the E4M3 denormal band"
    );

    let base = quantize_mxfp8_weight(&w, 1, 32);
    let comp = quantize_mxfp8_weight_compensated(&w, 1, 32);
    let dq_base = dequantize_mxfp8(&base, 1, 32);
    let dq_comp = dequantize_mxfp8(&comp, 1, 32);

    // The two paths genuinely disagree on this element.
    assert_ne!(dq_base[1].to_bits(), dq_comp[1].to_bits());
    // Base is closer to the truth, by a wide margin. The measured factor is
    // ~7.3x; pinned as a band so a change in E4M3 denormal handling is caught.
    let e_base = (f64::from(w[1]) - f64::from(dq_base[1])).abs();
    let e_comp = (f64::from(w[1]) - f64::from(dq_comp[1])).abs();
    let factor = e_comp / e_base;
    println!("denormal band: base err={e_base:e} comp err={e_comp:e} factor={factor:.3}");
    assert!(
        (5.0..12.0).contains(&factor),
        "expected the compensation to lose by roughly 7x here, got {factor:.3} \
         (base {e_base:e}, comp {e_comp:e})"
    );

    // And the aggregate metric reflects it.
    let b = rel_l2(&w, 1, 32, false);
    let c = rel_l2(&w, 1, 32, true);
    assert!(
        c > b,
        "rel_l2 must be worse here, got base={b:e} comp={c:e}"
    );
}

// --------------------------------------------------------------------------
// Negative control: the direction guard has teeth
// --------------------------------------------------------------------------

/// NEGATIVE CONTROL — dividing by `4/3` gives `-1` and costs `4.4x`.
///
/// If the compensation ran backwards, the exponent would DROP. That is not a
/// harmless sign flip:
///
/// -- The delta is `-1` on 13,069 of the 31,519 positive bf16 maxima, and
///   `0` on the rest. It is **never** `+1` — the mirror image of the
///   multiply case, which is never `-1`.
/// -- Dropping the exponent halves the block scale, so the block max maps to
///   `224` instead of `448`. Only the lower half of E4M3 is ever used, and the
///   max element's absolute quantization error doubles.
/// -- Measured on the house fixture, `rel_l2` gets about `4.41x` worse
///   (`+341%`), versus `1.00x` for the correct direction.
///
/// So the two directions are wildly asymmetric, which is what makes the
/// direction assertion in `e8m0_delta_is_zero_or_plus_one_only` worth having
/// rather than being a formality.
#[test]
fn dividing_by_four_thirds_would_lower_the_exponent_and_lose_range() {
    // (1) The delta really is {-1, 0} and never +1.
    let mut minus_one = 0usize;
    let mut zero = 0usize;
    let mut checked = 0usize;
    for &bm in &all_finite_non_negative_bf16() {
        if bm <= 0.0 {
            continue;
        }
        let base = e8m0_for(bm, 1.0);
        let divided = e8m0_for(bm, 1.0 / FOUR_THIRDS);
        match divided as i32 - base as i32 {
            -1 => minus_one += 1,
            0 => zero += 1,
            other => panic!("dividing gave an unexpected delta {other} at {bm:e}"),
        }
        // Control: the correct direction never lowers the exponent.
        assert!(
            e8m0_for(bm, FOUR_THIRDS) as i32 >= base as i32,
            "multiplying must never lower the exponent"
        );
        checked += 1;
    }
    println!("dividing: {minus_one} blocks drop an exponent, {zero} unchanged, of {checked}");
    assert!(minus_one > 0, "dividing must lower the exponent somewhere");
    assert_eq!(minus_one + zero, checked);
    assert!(
        (35.0..45.0).contains(&(100.0 * minus_one as f64 / checked as f64)),
        "expected roughly the mirror of the multiply rate"
    );

    // (2) The cost of getting it backwards, in rel_l2.
    let (m, n) = (32usize, 64usize);
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    let w: Vec<f32> = (0..m * n).map(|_| bf16(rng.range(-1.0, 1.0))).collect();
    let base = rel_l2(&w, m, n, false);
    let comp = rel_l2(&w, m, n, true);

    // The wrong direction, evaluated with the kernel's own arithmetic.
    let divided_rel_l2 = rel_l2_with_divided_scale(&w, m, n);
    println!(
        "base={base:.8} comp={comp:.8} (ratio {:.6}) divided={divided_rel_l2:.8} (ratio {:.6})",
        comp / base,
        divided_rel_l2 / base
    );
    assert!(
        (comp / base - 1.0).abs() < 1e-9,
        "the correct direction is the invariant one"
    );
    assert!(
        divided_rel_l2 > 4.0 * base,
        "the wrong direction must be dramatically worse, got ratio {:.3}",
        divided_rel_l2 / base
    );
}

/// `rel_l2` for the wrong (dividing) direction, using the kernel's own
/// per-block arithmetic so the number describes the real code path.
///
/// The shipped kernel cannot be asked to divide, so this mirrors
/// `quantize_mxfp8_weight_with` with a dividing block-scale rule.
fn rel_l2_with_divided_scale(w: &[f32], m: usize, n: usize) -> f64 {
    let n_pad = n.div_ceil(BLOCK_SIZE) * BLOCK_SIZE;
    let num_blocks = n_pad / BLOCK_SIZE;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for r in 0..m {
        for b in 0..num_blocks {
            let mut block_max = 0.0f32;
            for k in 0..BLOCK_SIZE {
                let c = b * BLOCK_SIZE + k;
                if c < n {
                    let v = bf16(w[r * n + c]);
                    block_max = block_max.max(v.abs());
                }
            }
            // The wrong direction: divide instead of multiply.
            let scale_needed = (block_max / FP8_MAX).max(SCALE_MIN) / FOUR_THIRDS;
            let log2_scale = (f64::from(scale_needed)).log2() as f32;
            let e8m0 = (log2_scale.ceil() as i32 + E8M0_BIAS).clamp(0, 254) as u8;
            let mut scale = if e8m0 == 0 {
                0.0
            } else {
                f32::from_bits(u32::from(e8m0) << 23)
            };
            if block_max == 0.0 {
                scale = 1.0;
            }
            for k in 0..BLOCK_SIZE {
                let c = b * BLOCK_SIZE + k;
                if c >= n {
                    continue;
                }
                let v = bf16(w[r * n + c]);
                let deq = if block_max == 0.0 {
                    0.0
                } else {
                    let code = quant_core::dtype::f32_to_fp8_e4m3_bits(
                        (v / scale).clamp(-FP8_MAX, FP8_MAX),
                    );
                    quant_core::dtype::fp8_e4m3_bits_to_f32(code) * scale
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
// Parity containment
// --------------------------------------------------------------------------

/// Running the quality path must not perturb the base path — no shared
/// mutable state, no lazily-initialised cache.
///
/// This is the in-process version of `parity_fingerprint.rs`'s
/// `quality_variant_never_changes_base_variant_bytes`, kept here so this
/// file is self-contained: the whole-file digests are the strong oracle, but
/// they live in another file and cover the stream path rather than the
/// kernel.
#[test]
fn quality_path_does_not_perturb_the_base_path() {
    let (m, n) = (64usize, 48usize);
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    let w: Vec<f32> = (0..m * n).map(|_| bf16(rng.range(-1.0, 1.0))).collect();

    let base_first = quantize_mxfp8_weight(&w, m, n);

    // Exercise the quality path thoroughly, including the degenerate witness.
    let mut spike = vec![0.0f32; 32];
    spike[0] = WITNESS_BLOCK_MAX;
    let _ = quantize_mxfp8_weight_compensated(&spike, 1, 32);
    let _ = quantize_mxfp8_weight_compensated(&w, m, n);

    let base_after = quantize_mxfp8_weight(&w, m, n);
    assert_eq!(base_first.qdata, base_after.qdata, "qdata perturbed");
    assert_eq!(base_first.scale, base_after.scale, "scales perturbed");
    assert_eq!(base_first.qdata_shape, base_after.qdata_shape);
    assert_eq!(base_first.scale_shape, base_after.scale_shape);

    // And the base path must be untouched by construction: the two functions
    // share no block-scale rule. A block whose exponent moves must produce
    // different *bytes* in the compensated path, proving the rules really are
    // distinct rather than one delegating to the other.
    let mut moving = vec![0.0f32; 32];
    moving[0] = 448.0;
    let b = quantize_mxfp8_weight(&moving, 1, 32);
    let c = quantize_mxfp8_weight_compensated(&moving, 1, 32);
    assert_ne!(b.scale, c.scale, "the two rules must be genuinely distinct");
}

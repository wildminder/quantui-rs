//! S04 — regression guard for the MXFP8 block-scale rule.
//!
//! # Why this file exists
//!
//! `quant_mxfp8.rs` computes the E8M0 block scale as
//! `ceil(f32(f64::log2(scale_needed)))`, NOT as the "obvious" bit-trick
//! `(exponent_field - 127) + (mantissa != 0)`. The bit-trick looks equivalent
//! and is not. This file turns that decision from a code comment into an
//! enforced invariant.
//!
//! # The exact divergence: ONE reachable input, not a rate
//!
//! The module doc's historical "~1.4% of cases" figure is a rate over
//! GENERIC f32 `scale_needed` values, which are unreachable here. `block_scale`
//! receives `block_max`, which the kernel has already rounded through **bf16**,
//! so `scale_needed = block_max / 448.0` only ever lands on a small set.
//! Enumerating that reachable set shows the two rules differ at exactly one
//! unclamped point, and it IS observable in the output bytes.
//!
//! The witness is `block_max = f32::from_bits(0x0460_0000)` = `2.6331e-36`,
//! the value whose `scale_needed` is exactly `SCALE_MIN = 2^-127`. `SCALE_MIN`
//! is an f32 **subnormal** (bits `0x0040_0000`, exponent field 0). The
//! bit-trick reads exponent field 0 as `0 - 127 = -127`, then adds 1 because
//! the mantissa is non-zero, giving `-126`. Its `+ (mantissa != 0)` term is
//! **wrong for subnormals**.
//!
//! | | e8m0 | scale | `qdata[0]` |
//! |---|---|---|---|
//! | `log2` (current, correct) | 0 | `0.0` | `0x7E` (448.0) |
//! | bit-trick (wrong)       | 1 | `2^-126` | `0x76` (224.0) |
//!
//! # No parity source is modified
//!
//! Every test here reads the PUBLIC API and asserts existing behaviour. The
//! kernel is untouched, so this file cannot perturb any golden.

use quant_core::quant_mxfp8::quantize_mxfp8_weight;

/// The reachable input at which the `log2` rule and the exponent-field
/// bit-trick disagree. `scale_needed` for this value is exactly `SCALE_MIN`.
///
/// bf16-exact: the low 16 bits are zero, so the kernel's bf16 round-trip
/// (`quantize_mxfp8_weight` step 0) leaves it unchanged.
const WITNESS_BLOCK_MAX: f32 = f32::from_bits(0x0460_0000);

/// The E4M3 code for 448.0 — what the correct rule produces.
const CODE_448: u8 = 0x7E;
/// The E4M3 code for 224.0 — what the bit-trick would produce instead.
const CODE_224: u8 = 0x76;

/// Reimplementation of the exponent-field bit-trick: `E + (mantissa != 0)`
/// where `E` is the f32 **biased** exponent field. This is the rule we must NOT
/// use.
///
/// The bit-trick folds the whole computation into the raw field, because the
/// E8M0 bias (127) and the f32 bias (127) are the same: `e8m0 = E_b + (mant !=
/// 0)`. For a NORMAL f32 that is correct — `1.0` has `E_b = 127` and
/// `mantissa = 0`, giving e8m0 127, i.e. `ceil(log2(1.0)) + 127 = 127`.
///
/// Returns the e8m0 byte the bit-trick computes.
fn bit_trick_exp_biased(scale_needed: f32) -> u8 {
    let bits = scale_needed.to_bits();
    let exponent_field = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;
    // E_b - 127 gives ceil(log2(x)); re-adding the E8M0 bias 127 cancels.
    let e = exponent_field - 127 + i32::from(mantissa != 0) + 127;
    e.clamp(0, 254) as u8
}

/// The `SCALE_MIN` clamp floor: `2^-127`, the smallest f32 subnormal.
const SCALE_MIN: f32 = f32::from_bits(0x0040_0000);

/// BOUNDARY — the single most important test in this file.
///
/// If anyone "simplifies" `block_scale` to the exponent-field bit-trick, this
/// goes red with `scale 0 != 1`. The wrong value is the point.
#[test]
fn scale_min_clamp_boundary_block_max_gives_e8m0_zero() {
    let mut w = vec![0.0f32; 32];
    w[0] = WITNESS_BLOCK_MAX;

    let r = quantize_mxfp8_weight(&w, 1, 32);

    // Precondition: this block's scale_needed really is exactly SCALE_MIN,
    // i.e. the divergence point is the one we think it is.
    let scale_needed = (WITNESS_BLOCK_MAX / 448.0f32).max(SCALE_MIN);
    assert_eq!(
        scale_needed.to_bits(),
        SCALE_MIN.to_bits(),
        "witness must produce scale_needed == SCALE_MIN exactly"
    );
    assert_eq!(
        scale_needed, SCALE_MIN,
        "SCALE_MIN is an f32 SUBNORMAL — that is what breaks the bit-trick"
    );
    assert_eq!(
        scale_needed.to_bits() & 0x7F80_0000,
        0,
        "exponent field must be 0 (subnormal) for the bit-trick to misfire"
    );

    // The assertion the guard exists for.
    assert_eq!(
        r.scale[0], 0,
        "e8m0 must be 0 (log2 form); the bit-trick form would give 1"
    );
    assert_eq!(
        r.qdata[0], CODE_448,
        "qdata[0] must be 0x7E (448.0); the bit-trick form would give 0x76"
    );
}

/// NEGATIVE / must-fail-by-construction — documents WHY the guard above is red
/// under the wrong rule.
///
/// This test PASSES. The wrong value is the assertion. If the kernel's rule
/// ever changes, this test is what explains the failure in the same file.
#[test]
fn bit_trick_form_would_change_the_scale_byte_and_the_code() {
    let scale_needed = (WITNESS_BLOCK_MAX / 448.0f32).max(SCALE_MIN);

    // The correct rule, spelled out independently of the kernel.
    let log2_e8m0 = ((f64::from(scale_needed)).log2() as f32).ceil() as i32 + 127;
    let log2_e8m0 = log2_e8m0.clamp(0, 254) as u8;

    // The wrong rule.
    let bt_e8m0 = bit_trick_exp_biased(scale_needed);

    // Control: the two rules AGREE on normal inputs. Without this, a broken
    // bit-trick helper would "pass" this test for the wrong reason. The
    // bit-trick is not wrong in general — it is wrong for SUBNORMAL
    // `scale_needed`, which is exactly what the witness is.
    for &normal in &[1.0f32, 2.0, 0.5, 0.25, 448.0, 1.0 / 448.0] {
        let lg = (((f64::from(normal)).log2() as f32).ceil() as i32 + 127).clamp(0, 254) as u8;
        assert_eq!(
            bit_trick_exp_biased(normal),
            lg,
            "bit-trick must agree with log2 on normal {normal}"
        );
    }

    // They disagree — and the disagreement is one full exponent step.
    assert_eq!(log2_e8m0, 0, "log2 form gives e8m0 = 0");
    assert_eq!(bt_e8m0, 1, "bit-trick form gives e8m0 = 1");
    assert_ne!(
        log2_e8m0, bt_e8m0,
        "the two rules must differ at this witness"
    );

    // And the divergence reaches the OUTPUT BYTES: e8m0 0 means scale 0.0, so
    // the block max saturates to +448.0 -> 0x7E. e8m0 1 means scale 2^-126, so
    // the block max lands on 224.0 -> 0x76.
    let scale_for = |e: u8| {
        if e == 0 {
            0.0f32
        } else {
            f32::from_bits(u32::from(e) << 23)
        }
    };
    let code_for = |e: u8| {
        let s = scale_for(e);
        // The kernel's own step 4: v / scale, clamp, cast.
        let scaled = if s == 0.0 {
            f32::INFINITY
        } else {
            WITNESS_BLOCK_MAX / s
        };
        quant_core::dtype::f32_to_fp8_e4m3_bits(scaled.clamp(-448.0, 448.0))
    };

    assert_eq!(scale_for(log2_e8m0), 0.0);
    assert_eq!(scale_for(bt_e8m0), 2.0f32.powi(-126));
    assert_eq!(code_for(log2_e8m0), CODE_448, "log2 form -> 0x7E");
    assert_eq!(code_for(bt_e8m0), CODE_224, "bit-trick form -> 0x76");

    // And the kernel agrees with the log2 form, not the bit-trick.
    let mut w = vec![0.0f32; 32];
    w[0] = WITNESS_BLOCK_MAX;
    let r = quantize_mxfp8_weight(&w, 1, 32);
    assert_eq!(r.scale[0], log2_e8m0);
    assert_eq!(r.qdata[0], code_for(log2_e8m0));
    assert_ne!(
        r.qdata[0], CODE_224,
        "kernel must not take the bit-trick path"
    );
}

/// DETERMINISM — the same input must give bit-identical output every time.
///
/// Defends against non-determinism from the `rayon` parallel path
/// (`quant_mxfp8.rs` `into_par_iter`). A max-reduction is order-independent so
/// this *should* hold, but "should" is exactly what a concurrency bug
/// violates, and a non-deterministic weight load is extremely hard to diagnose
/// downstream. Eight repeats also give a race a chance to surface under
/// thread-pool contention.
///
/// Uses an inline `xorshift64` PRNG rather than `proptest` so it runs
/// identically on every platform and in every seed mode — mirroring the
/// pattern already at `quant_mxfp8.rs:426-432` and `:576-582`.
#[test]
fn repeated_quantization_is_byte_identical() {
    let (m, n) = (64usize, 48usize);
    let mut w = vec![0.0f32; m * n];
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    for v in &mut w {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *v = ((x % 20_000) as f32 / 10_000.0) - 1.0;
    }

    let first = quantize_mxfp8_weight(&w, m, n);
    for rep in 1..8 {
        let again = quantize_mxfp8_weight(&w, m, n);
        assert_eq!(
            again.qdata, first.qdata,
            "qdata differed on repetition {rep}"
        );
        assert_eq!(
            again.scale, first.scale,
            "scale differed on repetition {rep}"
        );
        assert_eq!(again.qdata_shape, first.qdata_shape);
        assert_eq!(again.scale_shape, first.scale_shape);
    }
}

/// A zero block must yield a POSITIVE zero code, not `0x80`.
///
/// `cast(-0.0)` is `0x80`; the reference `torch.where` forces `+0.0`, which is
/// `0x00`. This test and `scale_min_clamp_boundary_...` together pin BOTH sides
/// of the `zero_mask` branch: that test has `scale == 0` with `qdata[0] ==
/// 0x7E`, this one has `scale == 0` with `qdata == 0x00`. The scale byte alone
/// cannot distinguish "scale byte is zero because the block is zero" from
/// "scale byte is zero because of the SCALE_MIN clamp" — the element codes
/// can.
#[test]
fn zero_block_still_yields_positive_zero_code() {
    let w = vec![0.0f32; 32];
    let r = quantize_mxfp8_weight(&w, 1, 32);
    assert_eq!(r.scale[0], 0, "zero block -> e8m0 0");
    assert!(
        r.qdata[..32].iter().all(|&b| b == 0x00),
        "zero block must be 0x00 (+0.0), not 0x80 (cast of -0.0): {:?}",
        &r.qdata[..32]
    );
    // Same for a block that is zero only after the sign is stripped.
    let mut w2 = vec![0.0f32; 64];
    w2[0] = 1.0;
    w2[32] = -0.0; // second block: value is -0.0, block_max is 0.0
    let r2 = quantize_mxfp8_weight(&w2, 1, 64);
    assert_eq!(r2.scale[1], 0);
    assert_eq!(r2.qdata[32], 0x00, "explicit -0.0 in a zero block -> 0x00");
}

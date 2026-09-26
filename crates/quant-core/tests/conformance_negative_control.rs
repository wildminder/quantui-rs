//! S08 — negative controls: prove the new guards can actually FAIL.
//!
//! # Why this file exists
//!
//! A guard test that cannot fail is decoration. It reads like coverage, it
//! appears in the test count, and it would not notice if the invariant it
//! claims to protect were broken. The only defence is to demonstrate, in
//! executable code, that a *deliberately wrong* implementation violates each
//! invariant.
//!
//! Every test here builds a **miniature broken implementation**, asserts that
//! it violates exactly one invariant, and passes. The wrong value is the
//! point. Deleting any guard mirrored here would make that guard red, and the
//! pairing is stated in each test's doc comment so a reader can check it.
//!
//! No parity source is touched. Each broken implementation is a few lines of
//! deliberately-wrong arithmetic living in this file only.

use std::collections::HashSet;

use quant_core::dtype::f32_to_fp8_e4m3_bits;
use quant_core::quant_mxfp8::{from_blocked_u8, to_blocked_u8};

/// Control for S01 (`golden_ruler_mxfp8_shared_codes_exhaustive`) and S02.
///
/// Mirrored guard: `golden_ruler_fp8_e4m3_vectors` +
/// `golden_ruler_mxfp8_shared_codes_exhaustive` assert `SatMax` overflow, and
/// `golden_ruler_mxfp8_variant_codes_are_the_documented_exception` pins the
/// 2-code variant split.
///
/// The `OvfNaN` policy used by `ml_dtypes`/JAX is legal under OCP MX v1.0 and
/// encodes overflow as `0x7F` (NaN) where we encode `0x7E` (448.0). If our
/// encoder were switched to it, the shared-code table would fail on the
/// overflow inputs.
#[test]
fn wrong_overflow_policy_would_produce_a_different_code() {
    /// The `OvfNaN` policy: values strictly beyond the range become the NaN
    /// pattern. Note `>` not `>=`: `448.0` IS representable, so it must stay
    /// `0x7E` under both policies. Only genuinely out-of-range values differ.
    fn ovf_nan_encode(f: f32) -> u8 {
        if f.abs() > 448.0 {
            0x7F
        } else {
            f32_to_fp8_e4m3_bits(f)
        }
    }

    // The real encoder saturates.
    assert_eq!(
        f32_to_fp8_e4m3_bits(500.0),
        0x7E,
        "our SatMax policy gives 0x7E (448.0)"
    );
    // The wrong policy disagrees — on both signs.
    assert_eq!(ovf_nan_encode(500.0), 0x7F, "OvfNaN gives 0x7F (NaN)");
    assert_eq!(
        ovf_nan_encode(-500.0) & 0x7F,
        0x7F,
        "OvfNaN gives 0xFF (|NaN|) for the negative side"
    );

    // And the wrong encoder fails the exact assertion the guard makes. This is
    // the line that would go red if `dtype.rs` were ever "fixed" to OvfNaN.
    assert_ne!(
        ovf_nan_encode(500.0),
        f32_to_fp8_e4m3_bits(500.0),
        "the two policies must disagree, or this control proves nothing"
    );

    // IN-RANGE values are policy-independent — including 448.0 itself, which
    // is exactly representable and therefore not an overflow case. This is
    // why the shared-code table has 254 passing entries and only the two
    // variant codes fail.
    for v in [1.0f32, 0.5, -0.25, 448.0, -448.0, 1e-4, 240.0, -240.0] {
        assert_eq!(
            ovf_nan_encode(v),
            f32_to_fp8_e4m3_bits(v),
            "in-range value {v} must be policy-independent"
        );
    }
    // And 480.0 — the OCP MX variant's max finite, and the value the two
    // variants disagree about — IS out of range for us.
    assert_eq!(f32_to_fp8_e4m3_bits(480.0), 0x7E, "ours saturates 480.0");
    assert_eq!(ovf_nan_encode(480.0), 0x7F, "OvfNaN would make it NaN");
}

/// Control for S04 (`scale_min_clamp_boundary_block_max_gives_e8m0_zero`).
///
/// Mirrored guard: S04 asserts `scale == 0 && qdata[0] == 0x7E` for
/// `block_max = f32::from_bits(0x0460_0000)`. The exponent-field bit-trick
/// gives `e8m0 = 1` there, and the resulting element code is `0x76`.
#[test]
fn wrong_log2_rule_would_produce_a_different_scale() {
    const WITNESS: f32 = f32::from_bits(0x0460_0000);
    const SCALE_MIN: f32 = f32::from_bits(0x0040_0000);

    /// The rule we must NOT use: read the exponent field and add 1 if the
    /// mantissa is non-zero. Correct for NORMAL f32, wrong for subnormals.
    fn bit_trick(scale_needed: f32) -> u8 {
        let bits = scale_needed.to_bits();
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x007F_FFFF;
        (exp - 127 + i32::from(mant != 0) + 127).clamp(0, 254) as u8
    }

    let scale_needed = (WITNESS / 448.0f32).max(SCALE_MIN);
    assert_eq!(scale_needed, SCALE_MIN, "the witness hits the clamp floor");
    assert_eq!(
        scale_needed.to_bits() & 0x7F80_0000,
        0,
        "SCALE_MIN is a SUBNORMAL — that is what breaks the bit-trick"
    );

    // Correct rule (what the kernel does) vs the wrong rule.
    let log2_rule = ((f64::from(scale_needed)).log2() as f32).ceil() as i32 + 127;
    let log2_rule = log2_rule.clamp(0, 254) as u8;
    assert_eq!(log2_rule, 0, "the log2 form gives e8m0 = 0");
    assert_eq!(bit_trick(scale_needed), 1, "the bit-trick gives e8m0 = 1");
    assert_ne!(
        log2_rule,
        bit_trick(scale_needed),
        "the rules must differ here, or this control proves nothing"
    );

    // And the divergence reaches the output bytes: 448.0 (0x7E) vs 224.0
    // (0x76). This is the exact failure message S04's guard would produce.
    let code = |e: u8| {
        let s = if e == 0 {
            0.0f32
        } else {
            f32::from_bits(u32::from(e) << 23)
        };
        f32_to_fp8_e4m3_bits((WITNESS / s).clamp(-448.0, 448.0))
    };
    assert_eq!(code(log2_rule), 0x7E);
    assert_eq!(code(bit_trick(scale_needed)), 0x76);
    assert_ne!(code(log2_rule), code(bit_trick(scale_needed)));
}

/// Control for S03 (`padding_cells_are_zero_and_shape_is_roundup`).
///
/// Mirrored guard: S03 asserts that on a `(130, 5)` all-`7` source exactly
/// 650 bytes are non-zero. A `to_blocked` stand-in that forgets to zero the
/// output buffer leaves stale data in the padding.
#[test]
fn nonzero_padding_would_break_the_padding_assertion() {
    /// A `to_blocked` with the zero-initialisation REMOVED — the classic
    /// "allocate with `vec![]` + push, or forget `vec![0u8; n]`" bug.
    fn to_blocked_no_zero_init(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
        let roundup = |x: usize, m: usize| (x + m - 1) / m * m;
        let ncb = roundup(cols, 4) / 4;
        let padded = roundup(rows, 128) * ncb * 4;
        // BUG: buffer starts as 0xFF "stale memory" rather than zeros.
        let mut out = vec![0xFFu8; padded];
        for (i, &v) in src.iter().enumerate() {
            let (r, c) = (i / cols, i % cols);
            let b = (r / 128) * ncb + c / 4;
            let flat = b * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + c % 4;
            out[flat] = v;
        }
        out
    }

    let (rows, cols) = (130usize, 5usize);
    let src = vec![7u8; rows * cols];

    // The real implementation satisfies S03's assertion.
    let (real, _) = to_blocked_u8(&src, rows, cols);
    let sevens = real.iter().filter(|&&b| b == 7).count();
    assert_eq!(sevens, rows * cols, "real impl: only source cells are 7");
    assert!(
        real.iter().all(|&b| b == 0 || b == 7),
        "real impl: padding is zero"
    );

    // The broken one does NOT — stale 0xFF survives in the padding, which is
    // exactly what S03 exists to catch and what a round-trip test cannot see
    // (from_blocked crops the padding away).
    let broken = to_blocked_no_zero_init(&src, rows, cols);
    let sevens = broken.iter().filter(|&&b| b == 7).count();
    assert_eq!(sevens, rows * cols, "source cells still land correctly");
    assert!(
        broken.iter().any(|&b| b != 0 && b != 7),
        "the broken impl leaves stale non-zero data in the padding"
    );
    // And S03's actual assertion, restated, fails against it:
    assert_ne!(
        broken.iter().filter(|&&b| b == 0).count(),
        256 * 8 - 650,
        "zero-padding count differs, so S03's assertion would go red"
    );
    // A round trip would NOT notice — that is the whole point of S03.
    assert_eq!(
        from_blocked_u8(&broken, rows, cols),
        src,
        "a round trip passes even with garbage padding — S03 is the only net"
    );
}

/// Control for S06 (`swizzle_is_injective_property`).
///
/// Mirrored guard: S06 asserts all `rows * cols` computed flat indices are
/// distinct and in bounds. An index map that drops the `r0` term collides.
#[test]
fn aliased_swizzle_would_break_injectivity() {
    let (rows, cols) = (130usize, 5usize);
    let ncb = (cols + 3) / 4;

    // The real index map, and the deliberately-colliding variant that drops
    // the `r0` term from `r1 * 16 + r0 * 4 + crem`.
    let real_index = |r: usize, c: usize| {
        let b = (r / 128) * ncb + c / 4;
        b * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + c % 4
    };
    let aliased_index = |r: usize, c: usize| {
        let b = (r / 128) * ncb + c / 4;
        b * 512 + (r % 32) * 16 + c % 4 // r0 term DROPPED -> aliases
    };

    let mut real_slots: HashSet<usize> = HashSet::new();
    let mut broken_slots: HashSet<usize> = HashSet::new();
    for r in 0..rows {
        for c in 0..cols {
            real_slots.insert(real_index(r, c));
            broken_slots.insert(aliased_index(r, c));
        }
    }

    // The real map is injective over all 650 cells.
    assert_eq!(real_slots.len(), rows * cols, "real map is injective");
    // The broken one is not: distinct cells collide onto shared slots.
    assert!(
        broken_slots.len() < rows * cols,
        "dropping r0 must alias cells ({} distinct vs {} cells)",
        broken_slots.len(),
        rows * cols
    );
    // A concrete collision, so the failure is legible rather than a count.
    let a = aliased_index(0, 0);
    let b = aliased_index(32, 0); // r0 differs, r1 is the same
    assert_eq!(a, b, "cells (0,0) and (32,0) collide in the broken map");

    // Both are in bounds — aliasing is the dangerous half, not OOB.
    let (_, shape) = to_blocked_u8(&vec![0u8; rows * cols], rows, cols);
    let len = (shape[0] * shape[1]) as usize;
    assert!(real_index(rows - 1, cols - 1) < len);
    assert!(a < len);
}

/// Control for S05 (`is_nvfp4_e4m3_scale_policy_only`).
///
/// Mirrored guard: the classifier must only downgrade a difference that is
/// ENTIRELY NaN-vs-saturated scale bytes. A predicate that forgets to check
/// the byte POSITION would excuse data-byte differences too — the over-capture
/// failure the plan names as this step's main risk.
#[test]
fn position_blind_policy_check_would_hide_data_bugs() {
    /// The correct predicate: only scale positions (offset 0 of each 9-byte
    /// NVFP4 block) may differ, and only as a NaN/saturated pair.
    fn correct(ob: &[u8], rb: &[u8]) -> bool {
        let mut saw = false;
        for (i, (&a, &b)) in ob.iter().zip(rb.iter()).enumerate() {
            if a == b {
                continue;
            }
            if i % 9 != 0 {
                return false;
            }
            if !matches!(
                (a, b),
                (0x7F, 0x7E) | (0x7E, 0x7F) | (0xFF, 0xFE) | (0xFE, 0xFF)
            ) {
                return false;
            }
            saw = true;
        }
        saw
    }

    /// The over-broad predicate: forgets the position check, so a NaN-looking
    /// byte ANYWHERE is excused as "policy".
    fn position_blind(ob: &[u8], rb: &[u8]) -> bool {
        let mut saw = false;
        for (&a, &b) in ob.iter().zip(rb.iter()) {
            if a == b {
                continue;
            }
            if !matches!(
                (a, b),
                (0x7F, 0x7E) | (0x7E, 0x7F) | (0xFF, 0xFE) | (0xFE, 0xFF)
            ) {
                return false;
            }
            saw = true;
        }
        saw
    }

    // A real data-byte difference that happens to be a NaN/saturated PAIR.
    // Byte 1 is inside the packed E2M1 nibbles — it is DATA, not scale.
    // Both pairs are listed in BOTH directions, exactly as the real predicate
    // does, so the only difference between the two functions is the position
    // check. That isolates the behaviour under test.
    let ours = [0x7Fu8, 0x7E, 0x11];
    let theirs = [0x7Eu8, 0x7F, 0x11];
    let is_policy_pair = |a: u8, b: u8| {
        matches!(
            (a, b),
            (0x7F, 0x7E) | (0x7E, 0x7F) | (0xFF, 0xFE) | (0xFE, 0xFF)
        )
    };

    // The correct predicate refuses to downgrade it.
    assert!(
        !correct(&ours, &theirs),
        "a data-byte difference must NOT be classified as policy"
    );
    // The position-blind predicate wrongly excuses it — the exact bug the
    // position check exists to prevent.
    assert!(
        position_blind(&ours, &theirs),
        "the over-broad predicate DOES hide the data bug — that is the point"
    );

    // Sanity: the differing bytes really ARE policy pairs, so the ONLY reason
    // the correct predicate refuses is the position.
    assert!(is_policy_pair(ours[0], theirs[0]));
    assert!(is_policy_pair(ours[1], theirs[1]));

    // Sanity: a genuine scale-byte policy pair IS accepted by both.
    let scale_ours = [0x7Fu8, 0x11];
    let scale_theirs = [0x7Eu8, 0x11];
    assert!(correct(&scale_ours, &scale_theirs));
    assert!(position_blind(&scale_ours, &scale_theirs));
}

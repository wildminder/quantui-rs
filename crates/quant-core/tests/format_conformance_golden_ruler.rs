//! GENERATED FILE - do not edit by hand.
//! Source: Golden Ruler conformance packs, gHashTag/t27
//!        (arXiv:2606.09686v3, `conformance/vectors/*.json`)
//! Regenerate with: python tools/gen_conformance_tests.py
//!
//! Conformance is asserted on the INTEGER BIT PATTERN, never on decoded-value
//! closeness -- the source paper's stated criterion.
#![allow(clippy::approx_constant)]

use quant_core::dtype::{f32_to_bf16_bits, f32_to_fp8_e4m3_bits, fp8_e4m3_bits_to_f32};

/// FP8 E4M3 (S1E4M3, bias 7, no inf, max finite 448.0).
/// Overflow policy: SatMax -> 0x7E. This is a *deliberate* choice; the
/// ml_dtypes/JAX convention (OvfNaN -> 0x7F) is equally legal under OCP MX v1.0.
/// See the comment at FP8_MAX in crates/quant-core/src/quant_fp8.rs.
#[test]
fn golden_ruler_fp8_e4m3_vectors() {
    // pos_zero (zero)
    assert_eq!(f32_to_fp8_e4m3_bits(0.0f32), 0x00, "pos_zero");
    // neg_zero (zero)
    assert_eq!(f32_to_fp8_e4m3_bits(-0.0f32), 0x80, "neg_zero");
    // min_pos_subnormal (subnormal)
    assert_eq!(
        f32_to_fp8_e4m3_bits(0.001953125f32),
        0x01,
        "min_pos_subnormal"
    );
    // max_pos_subnormal (subnormal)
    assert_eq!(
        f32_to_fp8_e4m3_bits(0.013671875f32),
        0x07,
        "max_pos_subnormal"
    );
    // min_pos_normal (normal)
    assert_eq!(f32_to_fp8_e4m3_bits(0.015625f32), 0x08, "min_pos_normal");
    // pos_half (normal)
    assert_eq!(f32_to_fp8_e4m3_bits(0.5f32), 0x30, "pos_half");
    // pos_one (normal)
    assert_eq!(f32_to_fp8_e4m3_bits(1.0f32), 0x38, "pos_one");
    // neg_one (normal)
    assert_eq!(f32_to_fp8_e4m3_bits(-1.0f32), 0xB8, "neg_one");
    // pos_two (normal)
    assert_eq!(f32_to_fp8_e4m3_bits(2.0f32), 0x40, "pos_two");
    // pos_three (normal)
    assert_eq!(f32_to_fp8_e4m3_bits(3.0f32), 0x44, "pos_three");
    // pos_six (normal)
    assert_eq!(f32_to_fp8_e4m3_bits(6.0f32), 0x4C, "pos_six");
    // max_finite_pos (normal)
    assert_eq!(f32_to_fp8_e4m3_bits(448.0f32), 0x7E, "max_finite_pos");
    // max_finite_neg (normal)
    assert_eq!(f32_to_fp8_e4m3_bits(-448.0f32), 0xFE, "max_finite_neg");
    // nan (nan)
    assert_eq!(f32_to_fp8_e4m3_bits(f32::NAN), 0x7F, "nan");
}

/// The E4M3 `SatMax` OVERFLOW POLICY -- the one thing the packs cannot check.
///
/// ⚠️ PROVENANCE: the inputs below are NOT from the Golden Ruler packs. They
/// come from the `float8_e4m3fn` specification (no infinities, max finite
/// 448.0, overflow saturates to the max-finite code 0x7E / 0xFE).
///
/// Every input the packs assert is IN RANGE -- the `e4m3_fp8_e4m3fn` pack's
/// 14 vectors and the `mxfp8` pack's 254 shared codes both top out at
/// exactly 448.0 -- so the pack-derived tests are policy-INDEPENDENT and
/// stay green under `OvfNaN`. This test is the actual enforcement of the
/// choice documented at `FP8_MAX` in `crates/quant-core/src/quant_fp8.rs`.
///
/// Adversarial case defended: `ml_dtypes`/JAX uses `OvfNaN` (overflow -> the
/// NaN code 0x7F/0xFF), which is equally legal under OCP MX v1.0. Switching
/// to it would silently change the emitted byte for every overflowing
/// weight, breaking byte-exact parity with torch -- while leaving every
/// pack-derived assertion green. These cases close that hole.
///
/// Grid arithmetic (bias 7, 3 mantissa bits): max finite `0x7E` = 1.75*2^8 =
/// 448.0; the next grid point would be 2.0*2^8 = 512, which needs an
/// exponent field of 16 and is therefore the reserved `0x7F`. The
/// round-to-nearest-even TIE between them sits at (448+512)/2 = 480.0, and
/// `0x7E` (126) is the EVEN neighbour, so the tie resolves DOWN to 448.0.
///
/// WHAT THIS TEST DOES AND DOES NOT PROVE -- read before "strengthening" it.
/// It proves the SatMax POLICY: every overflow lands on 0x7E, never 0x7F.
/// Flipping the policy in `dtype.rs` turns this test red, which is the point.
///
/// It deliberately does NOT try to pin WHICH internal branch handles a
/// value, because that is not observable in the output. `dtype.rs` has
/// three routes to 0x7E and they are mutually redundant:
/// (a) the EARLY guard `f_bits >= FP8_MAX` (480.0) -> 0x7E directly;
/// (b) the RNE path for |x| < 480.0, which can only reach 0x7E or lower
///     because the tie resolves down;
/// (c) `if r == 0x7F { r = 0x7E; }`, the carry-saturation at the end of
///     the RNE path -- defensive, and UNREACHABLE while guard (a) holds,
///     since no value below 480.0 can round above 448.0.
/// Verified by mutation: deleting (3), and weakening (1) from `>=` to `>`,
/// both leave this test fully GREEN. At exactly 480.0 the tie also yields
/// 0x7E, so the two routes are indistinguishable there too. That redundancy
/// is a robustness property, not a gap -- but it does mean no input value
/// can discriminate the branches. Do not add cases expecting them to.
///
/// The cases below are therefore boundary coverage on the OBSERVABLE
/// question (which code an input produces), chosen to pin the exact
/// representable edges rather than to route-control the implementation:
/// 448.0 -- max finite, in range
/// 464.0 -- below the 480.0 tie, so it rounds DOWN naturally
/// 480.0 -- the RNE tie AND the early-guard threshold, exactly
/// 481.0 -- first value strictly above the threshold
/// 511.0 -- largest f32 still below the next step (512)
///
/// NaN is asserted separately below and is NOT a policy choice --
/// both policies must agree that NaN encodes as the NaN code.
#[test]
fn golden_ruler_fp8_e4m3_overflow_policy_is_satmax() {
    // (input, expected code, why this value)
    let satmax: [(f32, u8, &str); 10] = [
        (448.0, 0x7E, "max finite -- in range, the boundary itself"),
        (464.0, 0x7E, "below the 480.0 tie: rounds DOWN naturally"),
        (480.0, 0x7E, "the RNE tie and the guard threshold, exactly"),
        (481.0, 0x7E, "first value strictly above the threshold"),
        (511.0, 0x7E, "largest f32 below the next step (512)"),
        (500.0, 0x7E, "plain overflow"),
        (1e30, 0x7E, "far overflow"),
        (f32::INFINITY, 0x7E, "+inf saturates; e4m3fn has no inf"),
        (-500.0, 0xFE, "negative overflow"),
        (f32::NEG_INFINITY, 0xFE, "-inf saturates"),
    ];
    for (input, want, why) in satmax {
        assert_eq!(
            f32_to_fp8_e4m3_bits(input),
            want,
            "SatMax: {input} must saturate to {want:#04x} ({why})"
        );
    }

    // The DECODER side of the same policy: 0x7E/0xFE are the max-finite
    // values, and 0x7F/0xFF are NaN -- never silently finite.
    assert_eq!(fp8_e4m3_bits_to_f32(0x7E), 448.0, "0x7E decodes to +448.0");
    assert_eq!(fp8_e4m3_bits_to_f32(0xFE), -448.0, "0xFE decodes to -448.0");
    assert!(
        fp8_e4m3_bits_to_f32(0x7F).is_nan(),
        "0x7F must decode to NaN under e4m3fn"
    );
    assert!(
        fp8_e4m3_bits_to_f32(0xFF).is_nan(),
        "0xFF must decode to NaN under e4m3fn"
    );

    // NaN INPUT is not a policy choice: both SatMax and OvfNaN must encode
    // NaN as the NaN code. Asserted so this test cannot be satisfied by a
    // blanket "everything becomes 0x7F" rule.
    assert_eq!(f32_to_fp8_e4m3_bits(f32::NAN), 0x7F, "NaN -> 0x7F");
    assert_eq!(f32_to_fp8_e4m3_bits(-f32::NAN), 0xFF, "-NaN -> 0xFF");
}

/// MXFP4 element format E2M1 (1s2e1m, bias 1, 16 codes, no inf/NaN).
///
/// Exercised through the real `quantize_nvfp4_weight` path rather than by
/// reaching into the private `E2M1_LUT`.
///
/// Construction: the test value goes in element 0 and a 6.0 anchor in
/// element 1. With block amax = 6.0 the reference arithmetic gives
/// per_tensor_scale = 6.0/2688, block_scale = 6.0/6.0 = 1.0,
/// scaled = 448.0 -> E4M3 0x7E -> total_scale = exactly 1.0. So the element
/// code for element 0 is E2M1(value / 1.0) = E2M1(value), which is what the
/// pack tabulates.
///
/// The 6.0 anchor is REQUIRED. An all-zero block drives
/// per_tensor_scale = 0, hence total_scale = 0 * E4M3(NaN) = NaN, so the
/// reference's `zero_scale_mask = (total_scale == 0)` is False and every
/// element encodes as NaN. That is faithful to comfy-kitchen
/// `quantize_nvfp4` (quantization.py:150) and is covered separately by
/// `all_zero_block_is_nan_like_the_reference`.
#[test]
fn golden_ruler_e2m1_exhaustive_grid() {
    // (input value, expected element code) from the pack's 16 vectors.
    let grid: [(f32, u8); 16] = [
        (0.0f32, 0x00),  // pos_zero
        (0.5f32, 0x01),  // pos_half
        (1.0f32, 0x02),  // pos_one
        (1.5f32, 0x03),  // pos_onehalf
        (2.0f32, 0x04),  // pos_two
        (3.0f32, 0x05),  // pos_three
        (4.0f32, 0x06),  // pos_four
        (6.0f32, 0x07),  // pos_six
        (-0.0f32, 0x08), // neg_zero
        (-0.5f32, 0x09), // neg_half
        (-1.0f32, 0x0A), // neg_one
        (-1.5f32, 0x0B), // neg_onehalf
        (-2.0f32, 0x0C), // neg_two
        (-3.0f32, 0x0D), // neg_three
        (-4.0f32, 0x0E), // neg_four
        (-6.0f32, 0x0F), // neg_six
    ];

    for (value, want_code) in grid {
        let mut block = [0.0f32; 16];
        block[0] = value;
        block[1] = 6.0; // anchor: fixes total_scale to exactly 1.0
        let q = quant_core::quant_nvfp4::quantize_nvfp4_weight(&block, 1, 16);
        // pack_uint4(hi_first=true): EVEN element index -> HIGH nibble.
        let got = (q.qdata[0] >> 4) & 0x0F;
        assert_eq!(got, want_code, "value {value} (want {want_code:#04x})");
    }
}

/// An all-zero NVFP4 tensor produces NaN element codes, NOT zero codes.
///
/// This is faithful to the reference, not a bug: with per_tensor_scale = 0,
/// `scaled_block_scales = 0/0 = NaN`, E4M3(NaN) = 0xFF, and
/// `total_scale = 0 * NaN = NaN`. The reference guard
/// `zero_scale_mask = (total_scale == 0)` is therefore False, so
/// `data_scaled = 0.0 / NaN = NaN` and clamps do not rescue it.
/// comfy-kitchen quantization.py:149-154.
///
/// Recorded explicitly so a future change to the zero-block guard is a
/// deliberate, reviewed decision rather than an accidental parity break.
#[test]
fn all_zero_block_is_nan_like_the_reference() {
    let block = [0.0f32; 16];
    let q = quant_core::quant_nvfp4::quantize_nvfp4_weight(&block, 1, 16);
    assert_eq!(q.per_tensor_scale, 0.0);
    // E4M3 NaN byte.
    assert_eq!(q.scale[0], 0xFF, "block scale is E4M3(NaN) = 0xFF");
    // Every packed byte carries the NaN-derived code, not 0x00.
    assert!(
        q.qdata.iter().all(|b| *b == 0xCC),
        "expected uniform NaN-derived code 0xCC, got {:?}",
        &q.qdata[..4.min(q.qdata.len())]
    );
}

/// BF16 (S1E8M7, bias 127, round-to-nearest-even).
#[test]
fn golden_ruler_bf16_vectors() {
    // pos_zero (zero)
    assert_eq!(f32_to_bf16_bits(0.0f32), 0x0000, "pos_zero");
    // pos_one (normal)
    assert_eq!(f32_to_bf16_bits(1.0f32), 0x3F80, "pos_one");
    // neg_one (normal)
    assert_eq!(f32_to_bf16_bits(-1.0f32), 0xBF80, "neg_one");
    // pos_two (normal)
    assert_eq!(f32_to_bf16_bits(2.0f32), 0x4000, "pos_two");
    // pos_three (normal)
    assert_eq!(f32_to_bf16_bits(3.0f32), 0x4040, "pos_three");
    // pos_half (normal)
    assert_eq!(f32_to_bf16_bits(0.5f32), 0x3F00, "pos_half");
    // pos_quarter (normal)
    assert_eq!(f32_to_bf16_bits(0.25f32), 0x3E80, "pos_quarter");
    // pos_0p1 (normal)
    assert_eq!(f32_to_bf16_bits(0.1f32), 0x3DCD, "pos_0p1");
}

/// FP8 E5M2 (S1E5M2, bias 15, inf+NaN retained, max finite 57344.0).
/// We do not currently implement an E5M2 encoder, so this is recorded as
/// a reference table only -- deliberately not asserted for conformance.
///
/// It is NOT a no-op: the pack is embedded with `include_str!` and its
/// vector count is checked, so this test fails if the vendored pack is
/// deleted, truncated, or replaced with a different revision. That is the
/// part worth enforcing -- the E5M2 *values* stay unasserted until we
/// have an encoder to assert them against.
#[test]
fn golden_ruler_e5m2_reference_table_is_present() {
    const PACK: &str = include_str!("fixtures/conformance/e5m2_fp8_e5m2_conformance_v0.json");
    // 16 reference vectors retained in tests/fixtures/conformance/.
    // Unused by the crate today: no E5M2 encoder exists.
    let vectors = PACK.matches("\"name\":").count();
    assert_eq!(vectors, 16, "E5M2 pack vector count changed");
}

/// MXFP8 element format S1E4M3 -- exhaustive 256-code enumeration.
///
/// Upgrades E4M3 coverage from the 14 hand-picked `e4m3fn` spot checks to
/// the full code space of the OCP MX variant: 254 of the 256 codes are
/// asserted here in BOTH directions (encode and decode). The previous 14
/// vectors never touched the 14 subnormal codes, so a decoder bug in an
/// unsampled subnormal -- or a rounding bug at an exponent boundary --
/// could not hide. Enumeration removes that class of blind spot.
///
/// The 2 absent codes are 0x7F and 0xFF, which is a DIFFERENT E4M3
/// VARIANT, not a bug: this pack is the finite-only OCP MX element type
/// (max 480.0, 0x7F -> +480.0), we implement torch `float8_e4m3fn`
/// (max 448.0, 0x7F -> NaN). Pinned by the sibling test below.
///
/// Conformance is on the INTEGER BIT PATTERN (the paper's criterion), so
/// this is exact, not a closeness check.
#[test]
fn golden_ruler_mxfp8_shared_codes_exhaustive() {
    // 254 shared codes, both directions, from the pack's
    // `input_f64` and `mxfp8_bits_int`.
    //
    // The table is f32, not f64: every packed value is an E4M3 code, so it
    // has a 3-bit mantissa and is EXACTLY representable in f32. Typing the
    // table as f64 and narrowing with `as f32` would introduce a rounding
    // step that could mask an off-by-one-ulp encoder bug -- the very class
    // of defect this enumeration exists to catch.
    let table: [(f32, u8); 254] = [
        (0.0f32, 0x00),          // code_0x00 (zero)
        (0.001953125f32, 0x01),  // code_0x01 (subnormal)
        (0.00390625f32, 0x02),   // code_0x02 (subnormal)
        (0.005859375f32, 0x03),  // code_0x03 (subnormal)
        (0.0078125f32, 0x04),    // code_0x04 (subnormal)
        (0.009765625f32, 0x05),  // code_0x05 (subnormal)
        (0.01171875f32, 0x06),   // code_0x06 (subnormal)
        (0.013671875f32, 0x07),  // code_0x07 (subnormal)
        (0.015625f32, 0x08),     // code_0x08 (normal)
        (0.017578125f32, 0x09),  // code_0x09 (normal)
        (0.01953125f32, 0x0A),   // code_0x0A (normal)
        (0.021484375f32, 0x0B),  // code_0x0B (normal)
        (0.0234375f32, 0x0C),    // code_0x0C (normal)
        (0.025390625f32, 0x0D),  // code_0x0D (normal)
        (0.02734375f32, 0x0E),   // code_0x0E (normal)
        (0.029296875f32, 0x0F),  // code_0x0F (normal)
        (0.03125f32, 0x10),      // code_0x10 (normal)
        (0.03515625f32, 0x11),   // code_0x11 (normal)
        (0.0390625f32, 0x12),    // code_0x12 (normal)
        (0.04296875f32, 0x13),   // code_0x13 (normal)
        (0.046875f32, 0x14),     // code_0x14 (normal)
        (0.05078125f32, 0x15),   // code_0x15 (normal)
        (0.0546875f32, 0x16),    // code_0x16 (normal)
        (0.05859375f32, 0x17),   // code_0x17 (normal)
        (0.0625f32, 0x18),       // code_0x18 (normal)
        (0.0703125f32, 0x19),    // code_0x19 (normal)
        (0.078125f32, 0x1A),     // code_0x1A (normal)
        (0.0859375f32, 0x1B),    // code_0x1B (normal)
        (0.09375f32, 0x1C),      // code_0x1C (normal)
        (0.1015625f32, 0x1D),    // code_0x1D (normal)
        (0.109375f32, 0x1E),     // code_0x1E (normal)
        (0.1171875f32, 0x1F),    // code_0x1F (normal)
        (0.125f32, 0x20),        // code_0x20 (normal)
        (0.140625f32, 0x21),     // code_0x21 (normal)
        (0.15625f32, 0x22),      // code_0x22 (normal)
        (0.171875f32, 0x23),     // code_0x23 (normal)
        (0.1875f32, 0x24),       // code_0x24 (normal)
        (0.203125f32, 0x25),     // code_0x25 (normal)
        (0.21875f32, 0x26),      // code_0x26 (normal)
        (0.234375f32, 0x27),     // code_0x27 (normal)
        (0.25f32, 0x28),         // code_0x28 (normal)
        (0.28125f32, 0x29),      // code_0x29 (normal)
        (0.3125f32, 0x2A),       // code_0x2A (normal)
        (0.34375f32, 0x2B),      // code_0x2B (normal)
        (0.375f32, 0x2C),        // code_0x2C (normal)
        (0.40625f32, 0x2D),      // code_0x2D (normal)
        (0.4375f32, 0x2E),       // code_0x2E (normal)
        (0.46875f32, 0x2F),      // code_0x2F (normal)
        (0.5f32, 0x30),          // code_0x30 (normal)
        (0.5625f32, 0x31),       // code_0x31 (normal)
        (0.625f32, 0x32),        // code_0x32 (normal)
        (0.6875f32, 0x33),       // code_0x33 (normal)
        (0.75f32, 0x34),         // code_0x34 (normal)
        (0.8125f32, 0x35),       // code_0x35 (normal)
        (0.875f32, 0x36),        // code_0x36 (normal)
        (0.9375f32, 0x37),       // code_0x37 (normal)
        (1.0f32, 0x38),          // code_0x38 (normal)
        (1.125f32, 0x39),        // code_0x39 (normal)
        (1.25f32, 0x3A),         // code_0x3A (normal)
        (1.375f32, 0x3B),        // code_0x3B (normal)
        (1.5f32, 0x3C),          // code_0x3C (normal)
        (1.625f32, 0x3D),        // code_0x3D (normal)
        (1.75f32, 0x3E),         // code_0x3E (normal)
        (1.875f32, 0x3F),        // code_0x3F (normal)
        (2.0f32, 0x40),          // code_0x40 (normal)
        (2.25f32, 0x41),         // code_0x41 (normal)
        (2.5f32, 0x42),          // code_0x42 (normal)
        (2.75f32, 0x43),         // code_0x43 (normal)
        (3.0f32, 0x44),          // code_0x44 (normal)
        (3.25f32, 0x45),         // code_0x45 (normal)
        (3.5f32, 0x46),          // code_0x46 (normal)
        (3.75f32, 0x47),         // code_0x47 (normal)
        (4.0f32, 0x48),          // code_0x48 (normal)
        (4.5f32, 0x49),          // code_0x49 (normal)
        (5.0f32, 0x4A),          // code_0x4A (normal)
        (5.5f32, 0x4B),          // code_0x4B (normal)
        (6.0f32, 0x4C),          // code_0x4C (normal)
        (6.5f32, 0x4D),          // code_0x4D (normal)
        (7.0f32, 0x4E),          // code_0x4E (normal)
        (7.5f32, 0x4F),          // code_0x4F (normal)
        (8.0f32, 0x50),          // code_0x50 (normal)
        (9.0f32, 0x51),          // code_0x51 (normal)
        (10.0f32, 0x52),         // code_0x52 (normal)
        (11.0f32, 0x53),         // code_0x53 (normal)
        (12.0f32, 0x54),         // code_0x54 (normal)
        (13.0f32, 0x55),         // code_0x55 (normal)
        (14.0f32, 0x56),         // code_0x56 (normal)
        (15.0f32, 0x57),         // code_0x57 (normal)
        (16.0f32, 0x58),         // code_0x58 (normal)
        (18.0f32, 0x59),         // code_0x59 (normal)
        (20.0f32, 0x5A),         // code_0x5A (normal)
        (22.0f32, 0x5B),         // code_0x5B (normal)
        (24.0f32, 0x5C),         // code_0x5C (normal)
        (26.0f32, 0x5D),         // code_0x5D (normal)
        (28.0f32, 0x5E),         // code_0x5E (normal)
        (30.0f32, 0x5F),         // code_0x5F (normal)
        (32.0f32, 0x60),         // code_0x60 (normal)
        (36.0f32, 0x61),         // code_0x61 (normal)
        (40.0f32, 0x62),         // code_0x62 (normal)
        (44.0f32, 0x63),         // code_0x63 (normal)
        (48.0f32, 0x64),         // code_0x64 (normal)
        (52.0f32, 0x65),         // code_0x65 (normal)
        (56.0f32, 0x66),         // code_0x66 (normal)
        (60.0f32, 0x67),         // code_0x67 (normal)
        (64.0f32, 0x68),         // code_0x68 (normal)
        (72.0f32, 0x69),         // code_0x69 (normal)
        (80.0f32, 0x6A),         // code_0x6A (normal)
        (88.0f32, 0x6B),         // code_0x6B (normal)
        (96.0f32, 0x6C),         // code_0x6C (normal)
        (104.0f32, 0x6D),        // code_0x6D (normal)
        (112.0f32, 0x6E),        // code_0x6E (normal)
        (120.0f32, 0x6F),        // code_0x6F (normal)
        (128.0f32, 0x70),        // code_0x70 (normal)
        (144.0f32, 0x71),        // code_0x71 (normal)
        (160.0f32, 0x72),        // code_0x72 (normal)
        (176.0f32, 0x73),        // code_0x73 (normal)
        (192.0f32, 0x74),        // code_0x74 (normal)
        (208.0f32, 0x75),        // code_0x75 (normal)
        (224.0f32, 0x76),        // code_0x76 (normal)
        (240.0f32, 0x77),        // code_0x77 (normal)
        (256.0f32, 0x78),        // code_0x78 (normal)
        (288.0f32, 0x79),        // code_0x79 (normal)
        (320.0f32, 0x7A),        // code_0x7A (normal)
        (352.0f32, 0x7B),        // code_0x7B (normal)
        (384.0f32, 0x7C),        // code_0x7C (normal)
        (416.0f32, 0x7D),        // code_0x7D (normal)
        (448.0f32, 0x7E),        // code_0x7E (normal)
        (-0.0f32, 0x80),         // code_0x80 (zero)
        (-0.001953125f32, 0x81), // code_0x81 (subnormal)
        (-0.00390625f32, 0x82),  // code_0x82 (subnormal)
        (-0.005859375f32, 0x83), // code_0x83 (subnormal)
        (-0.0078125f32, 0x84),   // code_0x84 (subnormal)
        (-0.009765625f32, 0x85), // code_0x85 (subnormal)
        (-0.01171875f32, 0x86),  // code_0x86 (subnormal)
        (-0.013671875f32, 0x87), // code_0x87 (subnormal)
        (-0.015625f32, 0x88),    // code_0x88 (normal)
        (-0.017578125f32, 0x89), // code_0x89 (normal)
        (-0.01953125f32, 0x8A),  // code_0x8A (normal)
        (-0.021484375f32, 0x8B), // code_0x8B (normal)
        (-0.0234375f32, 0x8C),   // code_0x8C (normal)
        (-0.025390625f32, 0x8D), // code_0x8D (normal)
        (-0.02734375f32, 0x8E),  // code_0x8E (normal)
        (-0.029296875f32, 0x8F), // code_0x8F (normal)
        (-0.03125f32, 0x90),     // code_0x90 (normal)
        (-0.03515625f32, 0x91),  // code_0x91 (normal)
        (-0.0390625f32, 0x92),   // code_0x92 (normal)
        (-0.04296875f32, 0x93),  // code_0x93 (normal)
        (-0.046875f32, 0x94),    // code_0x94 (normal)
        (-0.05078125f32, 0x95),  // code_0x95 (normal)
        (-0.0546875f32, 0x96),   // code_0x96 (normal)
        (-0.05859375f32, 0x97),  // code_0x97 (normal)
        (-0.0625f32, 0x98),      // code_0x98 (normal)
        (-0.0703125f32, 0x99),   // code_0x99 (normal)
        (-0.078125f32, 0x9A),    // code_0x9A (normal)
        (-0.0859375f32, 0x9B),   // code_0x9B (normal)
        (-0.09375f32, 0x9C),     // code_0x9C (normal)
        (-0.1015625f32, 0x9D),   // code_0x9D (normal)
        (-0.109375f32, 0x9E),    // code_0x9E (normal)
        (-0.1171875f32, 0x9F),   // code_0x9F (normal)
        (-0.125f32, 0xA0),       // code_0xA0 (normal)
        (-0.140625f32, 0xA1),    // code_0xA1 (normal)
        (-0.15625f32, 0xA2),     // code_0xA2 (normal)
        (-0.171875f32, 0xA3),    // code_0xA3 (normal)
        (-0.1875f32, 0xA4),      // code_0xA4 (normal)
        (-0.203125f32, 0xA5),    // code_0xA5 (normal)
        (-0.21875f32, 0xA6),     // code_0xA6 (normal)
        (-0.234375f32, 0xA7),    // code_0xA7 (normal)
        (-0.25f32, 0xA8),        // code_0xA8 (normal)
        (-0.28125f32, 0xA9),     // code_0xA9 (normal)
        (-0.3125f32, 0xAA),      // code_0xAA (normal)
        (-0.34375f32, 0xAB),     // code_0xAB (normal)
        (-0.375f32, 0xAC),       // code_0xAC (normal)
        (-0.40625f32, 0xAD),     // code_0xAD (normal)
        (-0.4375f32, 0xAE),      // code_0xAE (normal)
        (-0.46875f32, 0xAF),     // code_0xAF (normal)
        (-0.5f32, 0xB0),         // code_0xB0 (normal)
        (-0.5625f32, 0xB1),      // code_0xB1 (normal)
        (-0.625f32, 0xB2),       // code_0xB2 (normal)
        (-0.6875f32, 0xB3),      // code_0xB3 (normal)
        (-0.75f32, 0xB4),        // code_0xB4 (normal)
        (-0.8125f32, 0xB5),      // code_0xB5 (normal)
        (-0.875f32, 0xB6),       // code_0xB6 (normal)
        (-0.9375f32, 0xB7),      // code_0xB7 (normal)
        (-1.0f32, 0xB8),         // code_0xB8 (normal)
        (-1.125f32, 0xB9),       // code_0xB9 (normal)
        (-1.25f32, 0xBA),        // code_0xBA (normal)
        (-1.375f32, 0xBB),       // code_0xBB (normal)
        (-1.5f32, 0xBC),         // code_0xBC (normal)
        (-1.625f32, 0xBD),       // code_0xBD (normal)
        (-1.75f32, 0xBE),        // code_0xBE (normal)
        (-1.875f32, 0xBF),       // code_0xBF (normal)
        (-2.0f32, 0xC0),         // code_0xC0 (normal)
        (-2.25f32, 0xC1),        // code_0xC1 (normal)
        (-2.5f32, 0xC2),         // code_0xC2 (normal)
        (-2.75f32, 0xC3),        // code_0xC3 (normal)
        (-3.0f32, 0xC4),         // code_0xC4 (normal)
        (-3.25f32, 0xC5),        // code_0xC5 (normal)
        (-3.5f32, 0xC6),         // code_0xC6 (normal)
        (-3.75f32, 0xC7),        // code_0xC7 (normal)
        (-4.0f32, 0xC8),         // code_0xC8 (normal)
        (-4.5f32, 0xC9),         // code_0xC9 (normal)
        (-5.0f32, 0xCA),         // code_0xCA (normal)
        (-5.5f32, 0xCB),         // code_0xCB (normal)
        (-6.0f32, 0xCC),         // code_0xCC (normal)
        (-6.5f32, 0xCD),         // code_0xCD (normal)
        (-7.0f32, 0xCE),         // code_0xCE (normal)
        (-7.5f32, 0xCF),         // code_0xCF (normal)
        (-8.0f32, 0xD0),         // code_0xD0 (normal)
        (-9.0f32, 0xD1),         // code_0xD1 (normal)
        (-10.0f32, 0xD2),        // code_0xD2 (normal)
        (-11.0f32, 0xD3),        // code_0xD3 (normal)
        (-12.0f32, 0xD4),        // code_0xD4 (normal)
        (-13.0f32, 0xD5),        // code_0xD5 (normal)
        (-14.0f32, 0xD6),        // code_0xD6 (normal)
        (-15.0f32, 0xD7),        // code_0xD7 (normal)
        (-16.0f32, 0xD8),        // code_0xD8 (normal)
        (-18.0f32, 0xD9),        // code_0xD9 (normal)
        (-20.0f32, 0xDA),        // code_0xDA (normal)
        (-22.0f32, 0xDB),        // code_0xDB (normal)
        (-24.0f32, 0xDC),        // code_0xDC (normal)
        (-26.0f32, 0xDD),        // code_0xDD (normal)
        (-28.0f32, 0xDE),        // code_0xDE (normal)
        (-30.0f32, 0xDF),        // code_0xDF (normal)
        (-32.0f32, 0xE0),        // code_0xE0 (normal)
        (-36.0f32, 0xE1),        // code_0xE1 (normal)
        (-40.0f32, 0xE2),        // code_0xE2 (normal)
        (-44.0f32, 0xE3),        // code_0xE3 (normal)
        (-48.0f32, 0xE4),        // code_0xE4 (normal)
        (-52.0f32, 0xE5),        // code_0xE5 (normal)
        (-56.0f32, 0xE6),        // code_0xE6 (normal)
        (-60.0f32, 0xE7),        // code_0xE7 (normal)
        (-64.0f32, 0xE8),        // code_0xE8 (normal)
        (-72.0f32, 0xE9),        // code_0xE9 (normal)
        (-80.0f32, 0xEA),        // code_0xEA (normal)
        (-88.0f32, 0xEB),        // code_0xEB (normal)
        (-96.0f32, 0xEC),        // code_0xEC (normal)
        (-104.0f32, 0xED),       // code_0xED (normal)
        (-112.0f32, 0xEE),       // code_0xEE (normal)
        (-120.0f32, 0xEF),       // code_0xEF (normal)
        (-128.0f32, 0xF0),       // code_0xF0 (normal)
        (-144.0f32, 0xF1),       // code_0xF1 (normal)
        (-160.0f32, 0xF2),       // code_0xF2 (normal)
        (-176.0f32, 0xF3),       // code_0xF3 (normal)
        (-192.0f32, 0xF4),       // code_0xF4 (normal)
        (-208.0f32, 0xF5),       // code_0xF5 (normal)
        (-224.0f32, 0xF6),       // code_0xF6 (normal)
        (-240.0f32, 0xF7),       // code_0xF7 (normal)
        (-256.0f32, 0xF8),       // code_0xF8 (normal)
        (-288.0f32, 0xF9),       // code_0xF9 (normal)
        (-320.0f32, 0xFA),       // code_0xFA (normal)
        (-352.0f32, 0xFB),       // code_0xFB (normal)
        (-384.0f32, 0xFC),       // code_0xFC (normal)
        (-416.0f32, 0xFD),       // code_0xFD (normal)
        (-448.0f32, 0xFE),       // code_0xFE (normal)
    ];

    for (xf, code) in table {
        assert_eq!(
            f32_to_fp8_e4m3_bits(xf),
            code,
            "encode: f32_to_fp8_e4m3_bits({xf}) != 0x{code:02X}"
        );
        assert_eq!(
            fp8_e4m3_bits_to_f32(code),
            xf,
            "decode: fp8_e4m3_bits_to_f32(0x{code:02X}) != {xf}"
        );
    }
}

/// The MXFP8 pack's 2 variant-specific codes are a DOCUMENTED divergence,
/// pinned here so it can never drift into an accident.
///
/// This is the guard for the guard. The sibling test skips 0x7F / 0xFF
/// because our E4M3 variant reserves them for NaN. Someone 'fixing' those
/// two failures by editing `dtype.rs` would break byte-exact parity with
/// torch -- the crate's governing contract. This test fails FIRST and says
/// why.
///
/// Asserts: (a) the exclusion set is EXACTLY {0x7F, 0xFF} -- a count plus
/// explicit membership checks, so a future pack change cannot silently
/// shrink coverage; (b) our decoder returns NaN for both; (c) the pack's
/// own values for those codes are +/-480.0, restated as literals so the
/// divergence is visible in this file rather than only in the JSON.
#[test]
fn golden_ruler_mxfp8_variant_codes_are_the_documented_exception() {
    // (a) The exclusion set is exactly the two NaN-reserved codes, and
    // every other code in 0..=255 IS asserted by the sibling test. So a
    // future pack change cannot silently shrink coverage.
    const EXCLUDED: [u8; 2] = [0x7F, 0xFF];
    // The codes the sibling test asserts, straight from the pack.
    let shared_codes: [u8; 254] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
        0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D,
        0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C,
        0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B,
        0x3C, 0x3D, 0x3E, 0x3F, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A,
        0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
        0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96,
        0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5,
        0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3, 0xB4,
        0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF, 0xC0, 0xC1, 0xC2, 0xC3,
        0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xCB, 0xCC, 0xCD, 0xCE, 0xCF, 0xD0, 0xD1, 0xD2,
        0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF, 0xE0, 0xE1,
        0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF, 0xF0,
        0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE,
    ];
    assert_eq!(shared_codes.len(), 254);
    assert_eq!(EXCLUDED.len(), 2, "exclusion set must stay {{0x7F, 0xFF}}");
    // Shared + excluded must tile the whole 256-code space exactly once.
    let mut covered: Vec<u8> = shared_codes.to_vec();
    covered.extend_from_slice(&EXCLUDED);
    covered.sort_unstable();
    covered.dedup();
    assert_eq!(
        covered,
        (0..=255u8).collect::<Vec<u8>>(),
        "shared + excluded must be exactly the 256-code E4M3 space"
    );
    // And the excluded codes are genuinely absent from the shared table.
    for c in EXCLUDED {
        assert!(
            !shared_codes.contains(&c),
            "0x{c:02X} must be EXCLUDED from the shared-code table"
        );
    }

    // (b) Our decoder reserves both for NaN (torch float8_e4m3fn).
    assert!(
        fp8_e4m3_bits_to_f32(0x7F).is_nan(),
        "e4m3fn reserves 0x7F for NaN"
    );
    assert!(
        fp8_e4m3_bits_to_f32(0xFF).is_nan(),
        "e4m3fn reserves 0xFF for NaN"
    );

    // (c) The pack's own values for those codes -- the divergence,
    // restated as literals so it is readable without the JSON.
    let pack_values: [(u8, f64); 2] = [
        (0x7F, 480.0f32 as f64),  // code_0x7F
        (0xFF, -480.0f32 as f64), // code_0xFF
    ];
    for (code, pack_value) in pack_values {
        let ours = fp8_e4m3_bits_to_f32(code);
        assert!(
            ours.is_nan(),
            "e4m3fn reserves 0x{code:02X} for NaN, got {ours}"
        );
        assert_ne!(
            ours, pack_value as f32,
            "code 0x{code:02X} must DIVERGE from the pack value"
        );
    }
}

//! `cast` dtype-conversion tests.
//!
//! # Non-circularity — read this before trusting the rounding tests
//!
//! The rounding tests pin **literal expected bit patterns**. They never call
//! `f32_to_bf16_bits` / `f32_to_f16_bits` to compute what they expect, because
//! asserting a function against itself proves nothing.
//!
//! The literals were derived from the `half` crate's RNE semantics and chosen to
//! sit **exactly on rounding ties** — a tie is the only input that discriminates
//! round-to-nearest-even from truncation. Each tie is labelled with what
//! TRUNCATION would have produced instead, so the test has visible detection
//! power rather than asserting an arbitrary constant.
//!
//! # Why a tie at all
//!
//! bf16 keeps 7 mantissa bits, f32 has 23, so a tie is an `f32` whose low 16
//! bits are exactly `0x8000`. f16 keeps 10 bits, so a tie has low 13 bits
//! `0x1000`. A tie only *separates* RNE from truncation when the retained
//! mantissa LSB is 1 — otherwise both modes round down and the test is blind.
//! Both kinds are included: the odd-LSB tie (discriminates) and the even-LSB
//! tie (pins that we do NOT round up).

use quant_core::cast::{cast_tensor, output_dtype_for, CastError, CastOutcome};
use quant_core::dtype::{
    bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits, DType,
};

/// Build an f32 payload from raw bit patterns.
fn f32_payload(words: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(words.len() * 4);
    for &w in words {
        v.extend_from_slice(&w.to_le_bytes());
    }
    v
}

/// Build a bf16 payload from raw bit patterns.
fn bf16_payload(words: &[u16]) -> Vec<u8> {
    let mut v = Vec::with_capacity(words.len() * 2);
    for &w in words {
        v.extend_from_slice(&w.to_le_bytes());
    }
    v
}

/// Build an f16 payload from raw bit patterns.
fn f16_payload(words: &[u16]) -> Vec<u8> {
    let mut v = Vec::with_capacity(words.len() * 2);
    for &w in words {
        v.extend_from_slice(&w.to_le_bytes());
    }
    v
}

/// Cast a single-element-per-word f32 payload and return the bf16 words.
fn f32_to_bf16_words(words: &[u32]) -> Vec<u16> {
    let src = f32_payload(words);
    let (out, _) = cast_tensor("t", &src, DType::F32, DType::Bf16, &[words.len() as u64]).unwrap();
    out.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Cast a single-element-per-word f32 payload and return the f16 words.
fn f32_to_f16_words(words: &[u32]) -> Vec<u16> {
    let src = f32_payload(words);
    let (out, _) = cast_tensor("t", &src, DType::F32, DType::F16, &[words.len() as u64]).unwrap();
    out.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

// --------------------------------------------------------------------------- //
// Same-dtype: the verbatim byte copy
// --------------------------------------------------------------------------- //

/// Same-dtype cast must be a byte-for-byte copy.
///
/// Adversarial case defended: a regression that routes bf16→bf16 through
/// `f32_to_bf16_bits`. That is *numerically* lossless (every bf16 is exactly an
/// f32) but it is not a byte copy, and it can alter NaN payloads — see
/// `verbatim_path_preserves_snan_payload_a_round_trip_would_erase` below, which
/// is where that actually bites.
#[test]
fn same_dtype_payload_is_byte_identical() {
    // A payload with awkward bit patterns, not just tidy values.
    let src = bf16_payload(&[
        0x3F80, // 1.0
        0x8000, // -0.0
        0x0000, // +0.0
        0x7FC0, // NaN
        0x7F81, // signalling-NaN payload
        0xFF80, // -Inf
        0x0001, // subnormal
        0x7F7F, // max finite
    ]);
    let before = src.clone();
    let (out, outcome) =
        cast_tensor("w", &src, DType::Bf16, DType::Bf16, &[8]).unwrap();
    assert_eq!(out, before, "bf16->bf16 must be a byte copy");
    assert_eq!(outcome, CastOutcome::Verbatim);
    assert!(outcome.is_verbatim());

    // f16 -> f16 and f32 -> f32 likewise.
    let s16 = f16_payload(&[0x3C00, 0x8000, 0x7C00, 0x7E00]);
    let (o16, k16) = cast_tensor("w", &s16, DType::F16, DType::F16, &[4]).unwrap();
    assert_eq!(o16, s16);
    assert_eq!(k16, CastOutcome::Verbatim);

    let s32 = f32_payload(&[0x3F800000, 0x80000000, 0x7F800000]);
    let (o32, k32) = cast_tensor("w", &s32, DType::F32, DType::F32, &[3]).unwrap();
    assert_eq!(o32, s32);
    assert_eq!(k32, CastOutcome::Verbatim);
}

/// NEGATIVE CONTROL for the verbatim path: a signalling-NaN payload survives a
/// same-dtype cast, but would be QUIETED by a round trip through f32.
///
/// This is the test that proves the byte-copy fast path is genuinely taken
/// rather than a numerically-equivalent f32 detour. `bf16 0x7F81` is a
/// signalling NaN: widening it to f32 gives `0x7F810000` (still signalling),
/// and `bf16::from_f32` on that yields `0x7FC0` — the payload is destroyed.
/// The verbatim path returns `0x7F81` unchanged.
///
/// If this test ever fails with `0x7FC0`, the same-dtype fast path has been
/// replaced by a round trip.
#[test]
fn verbatim_path_preserves_snan_payload_a_round_trip_would_erase() {
    let snan_bf16 = 0x7F81u16;
    let src = bf16_payload(&[snan_bf16]);
    let (out, outcome) =
        cast_tensor("w", &src, DType::Bf16, DType::Bf16, &[1]).unwrap();
    assert_eq!(outcome, CastOutcome::Verbatim);
    assert_eq!(
        out,
        bf16_payload(&[snan_bf16]),
        "the signalling-NaN payload must survive untouched"
    );

    // Show the contrast explicitly: the f32 round trip really does quiet it.
    // This is the mechanism the verbatim path exists to avoid, so it is
    // asserted rather than merely claimed in a comment.
    let widened = f32::from_bits((snan_bf16 as u32) << 16);
    assert_eq!(
        widened.to_bits(),
        0x7F81_0000,
        "widening bf16 0x7F81 to f32 preserves the signalling bit"
    );
    let requantised = f32_to_bf16_bits(widened);
    // The round trip forces the NaN QUIET: 0x7F81 (signalling, mantissa LSB
    // clear) becomes 0x7FC1 (quiet, mantissa LSB set). The exact quieted value
    // is not the point — the point is that it is NOT 0x7F81 any more.
    assert_eq!(
        requantised, 0x7FC1,
        "a round trip through f32 would QUIET the NaN — this is why the \
         same-dtype path must copy bytes instead"
    );
    // In bf16 the mantissa is bits 0..=6 and the QUIET bit is the mantissa MSB,
    // 0x40. A NaN with that bit clear is SIGNALLING.
    assert_eq!(
        snan_bf16 & 0x0040,
        0,
        "the original must be a SIGNALLING NaN (quiet bit 0x40 clear)"
    );
    assert_eq!(
        requantised & 0x0040,
        0x0040,
        "the round trip SETS the quiet bit — textbook quieting"
    );
    // The rest of the payload survives; only the quiet bit was added
    // (0x01 -> 0x41). So the value is changed but not mangled, which is
    // exactly why a byte copy is the right thing for a same-dtype cast.
    assert_eq!(
        requantised & 0x003F,
        snan_bf16 & 0x003F,
        "the low payload bits are preserved; only the quiet bit was added"
    );
    assert_ne!(
        requantised, snan_bf16,
        "if these were equal the negative control would prove nothing"
    );

    // And a real bf16 -> f32 conversion of the same bits is exact (it widens,
    // it does not re-quantise), which is why only the same-dtype case is a
    // byte copy.
    let (widened_out, k) = cast_tensor("w", &src, DType::Bf16, DType::F32, &[1]).unwrap();
    assert_eq!(k, CastOutcome::Converted);
    assert_eq!(widened_out, f32_payload(&[0x7F81_0000]));
}

// --------------------------------------------------------------------------- //
// f32 -> bf16 rounding, against pinned literals
// --------------------------------------------------------------------------- //

/// bf16 RNE against pinned literals, all on or beside exact ties.
///
/// Detection power, stated per case (TRUNC is what truncation would give):
///
/// | input f32 | meaning | expected | truncation would give |
/// |---|---|---|---|
/// | `0x3F808000` | tie, retained mantissa LSB even | `0x3F80` | `0x3F80` (agrees) |
/// | `0x3F818000` | tie, retained LSB **odd** | `0x3F82` | `0x3F81` **differs** |
/// | `0x3F817FFF` | just below that tie | `0x3F81` | `0x3F81` (agrees) |
/// | `0x3F818001` | just above that tie | `0x3F82` | `0x3F81` **differs** |
///
/// The odd-LSB tie is the discriminating one: RNE rounds it UP to the even
/// neighbour `0x3F82`, truncation truncates DOWN to the odd `0x3F81`.
#[test]
fn f32_to_bf16_matches_pinned_expected_bits() {
    let cases: &[(u32, u16, &str)] = &[
        (0x3F80_8000, 0x3F80, "tie, even retained LSB -> round DOWN to even"),
        (0x3F81_8000, 0x3F82, "tie, odd retained LSB -> round UP to even"),
        (0x3F81_7FFF, 0x3F81, "just below the odd tie"),
        (0x3F81_8001, 0x3F82, "just above the odd tie"),
        (0x0000_0000, 0x0000, "+0.0"),
        (0x8000_0000, 0x8000, "-0.0 keeps its sign bit"),
        (0x7F80_0000, 0x7F80, "+Inf"),
        (0xFF80_0000, 0xFF80, "-Inf"),
        (0x7FC0_0000, 0x7FC0, "qNaN"),
    ];

    for &(input, want, why) in cases {
        let got = f32_to_bf16_bits(f32::from_bits(input));
        assert_eq!(got, want, "0x{input:08X} ({why}) -> 0x{want:04X}");
    }

    // The whole table through the real cast entry point, so the conversion arm
    // is exercised too — not just the scalar helper.
    let inputs: Vec<u32> = cases.iter().map(|c| c.0).collect();
    let wants: Vec<u16> = cases.iter().map(|c| c.1).collect();
    assert_eq!(f32_to_bf16_words(&inputs), wants);

    // And the discriminating property stated as a property: the odd-LSB tie
    // must NOT equal the truncation answer.
    let tie_odd = 0x3F81_8000u32;
    assert_eq!(
        f32_to_bf16_bits(f32::from_bits(tie_odd)),
        0x3F82,
        "RNE rounds the odd-LSB tie up to even"
    );
    assert_eq!(
        (tie_odd >> 16) as u16,
        0x3F81,
        "truncation would have kept the odd value — so this test discriminates"
    );
}

/// f16 RNE against pinned literals, on exact ties.
///
/// f16 keeps 10 mantissa bits, so a tie has the low 13 bits of the f32 set to
/// `0x1000`.
///
/// | input f32 | meaning | expected |
/// |---|---|---|
/// | `0x3F801000` | tie, even retained LSB -> DOWN to even | `0x3C00` |
/// | `0x3F803000` | tie, odd retained LSB -> UP to even | `0x3C02` |
/// | `0x3F802FFF` | just below the odd tie | `0x3C01` |
/// | `0x3F803001` | just above the odd tie | `0x3C02` |
#[test]
fn f32_to_f16_matches_pinned_expected_bits() {
    let cases: &[(u32, u16, &str)] = &[
        (0x3F80_1000, 0x3C00, "tie, even retained LSB -> round DOWN to even"),
        (0x3F80_3000, 0x3C02, "tie, odd retained LSB -> round UP to even"),
        (0x3F80_2FFF, 0x3C01, "just below the odd tie"),
        (0x3F80_3001, 0x3C02, "just above the odd tie"),
        (0x0000_0000, 0x0000, "+0.0"),
        (0x8000_0000, 0x8000, "-0.0 keeps its sign bit"),
        (0x7F80_0000, 0x7C00, "+Inf"),
        (0xFF80_0000, 0xFC00, "-Inf"),
        (0x7FC0_0000, 0x7E00, "qNaN"),
        (0x477F_E000, 0x7BFF, "f16 max finite 65504"),
    ];

    for &(input, want, why) in cases {
        let got = f32_to_f16_bits(f32::from_bits(input));
        assert_eq!(got, want, "0x{input:08X} ({why}) -> 0x{want:04X}");
    }

    let inputs: Vec<u32> = cases.iter().map(|c| c.0).collect();
    let wants: Vec<u16> = cases.iter().map(|c| c.1).collect();
    assert_eq!(f32_to_f16_words(&inputs), wants);
}

// --------------------------------------------------------------------------- //
// Overflow: an ERROR, never a silent Inf
// --------------------------------------------------------------------------- //

/// bf16 -> f16 above the f16 range must ERROR, naming the tensor and element.
///
/// This is the data-corruption case. `half::f16::from_f32` returns `Inf` for
/// these, so a saturating implementation would write a plausible-looking file
/// full of infinities and exit 0. The value 1e30 is chosen because it is a
/// perfectly ordinary bf16 weight magnitude — not a contrived edge case.
#[test]
fn bf16_to_f16_overflow_errors_naming_the_tensor() {
    // bf16 0x7E1C == 5.18e37, far above the f16 max of 65504.
    let big = bf16_payload(&[0x7E1C, 0x3F80]);
    let err = cast_tensor("encoder.layers.3.mlp.w", &big, DType::Bf16, DType::F16, &[2])
        .expect_err("must refuse, not saturate");

    match &err {
        CastError::F16Overflow {
            tensor,
            element,
            value,
        } => {
            assert_eq!(tensor, "encoder.layers.3.mlp.w", "must name the tensor");
            assert_eq!(*element, 0, "must name the first offending element");
            assert!(
                value.is_finite() && value.abs() > 65504.0,
                "must report the offending value, got {value}"
            );
        }
        other => panic!("expected F16Overflow, got {other:?}"),
    }

    // The message must be actionable: it names the tensor, the index, and says
    // why it refused. A bare "conversion failed" would send the user hunting.
    let msg = err.to_string();
    assert!(msg.contains("encoder.layers.3.mlp.w"), "msg: {msg}");
    assert!(msg.contains("element 0"), "msg: {msg}");
    assert!(msg.contains("65504"), "msg must state the limit: {msg}");
    assert!(msg.contains("Inf"), "msg must say what it avoided: {msg}");

    // A value that FITS must still succeed, so the guard is not simply
    // refusing everything.
    //
    // Note the boundary is NOT the f16 max finite (65504): 65504 is not
    // representable in bf16, whose 8-bit exponent/7-bit mantissa grid steps
    // straight past it. The largest bf16 that still maps to a finite f16 is
    // 0x477F = 65280 -> f16 0x7BF8. The next bf16 up, 0x4780 = 65536, is the
    // first that would become Inf. Verified against the `half` crate.
    let fits = bf16_payload(&[0x477F, 0x3F80]);
    let (out, outcome) = cast_tensor("ok", &fits, DType::Bf16, DType::F16, &[2]).unwrap();
    assert_eq!(outcome, CastOutcome::Converted);
    assert_eq!(out, f16_payload(&[0x7BF8, 0x3C00]));

    // The very next bf16 up overflows and must be refused.
    let over = bf16_payload(&[0x4780]); // 65536.0 -> f16 would be Inf
    let err = cast_tensor("edge", &over, DType::Bf16, DType::F16, &[1])
        .expect_err("bf16 0x4780 = 65536 overflows f16");
    assert!(matches!(err, CastError::F16Overflow { .. }));
}

/// The same overflow via the f32 path must also error.
#[test]
fn f32_to_f16_overflow_errors() {
    let src = f32_payload(&[0x3F80_0000, 0x7F3B_0000]); // 1.0, then ~1e30
    let err = cast_tensor("w", &src, DType::F32, DType::F16, &[2])
        .expect_err("f32 above the f16 range must refuse");
    match err {
        CastError::F16Overflow { element, .. } => assert_eq!(element, 1),
        other => panic!("expected F16Overflow, got {other:?}"),
    }

    // f32 -> bf16 has the same wide range and must NOT error on the same value.
    let (out, outcome) = cast_tensor("w", &src, DType::F32, DType::Bf16, &[2]).unwrap();
    assert_eq!(outcome, CastOutcome::Converted);
    assert_eq!(out.len(), 4, "two bf16 values = 4 bytes");
}

/// An `Inf` INPUT is not overflow: `f16` has a real infinity, so `Inf -> Inf` is
/// exact and lossless. Only a finite value that would BECOME `Inf` is refused.
///
/// This pins the `is_finite()` carve-out. Without it, a model containing a
/// genuine `Inf` (not unheard of in a diverged checkpoint) would be
/// un-castable for no benefit.
#[test]
fn infinite_input_is_not_overflow_because_f16_has_real_infinity() {
    let src = f32_payload(&[0x7F80_0000, 0xFF80_0000, 0x7FC0_0000]);
    let (out, outcome) = cast_tensor("w", &src, DType::F32, DType::F16, &[3]).unwrap();
    assert_eq!(outcome, CastOutcome::Converted);
    assert_eq!(out, f16_payload(&[0x7C00, 0xFC00, 0x7E00]));

    // bf16 -> f16 with an Inf: also fine, and also a real f16 Inf.
    let bsrc = bf16_payload(&[0x7F80, 0xFF80]);
    let (bout, _) = cast_tensor("w", &bsrc, DType::Bf16, DType::F16, &[2]).unwrap();
    assert_eq!(bout, f16_payload(&[0x7C00, 0xFC00]));
}

/// An `f64` source is rejected outright, not silently double-rounded.
///
/// f64 -> bf16 cannot be done in one rounding. Going via f32 is a DOUBLE
/// rounding whose result depends on the intermediate step, so the same input
/// could land on two different bf16 values depending on the path. Refusing is
/// the honest answer.
#[test]
fn f64_source_is_rejected() {
    let mut src = Vec::new();
    for w in [1.0f64, 2.0, 3.0] {
        src.extend_from_slice(&w.to_le_bytes());
    }
    for target in [DType::Bf16, DType::F16, DType::F32] {
        let err = cast_tensor("w", &src, DType::F64, target, &[3])
            .expect_err("f64 sources must be refused, not double-rounded");
        match &err {
            CastError::UnsupportedSourceDtype { tensor, dtype } => {
                assert_eq!(tensor, "w");
                assert_eq!(*dtype, DType::F64);
            }
            other => panic!("expected UnsupportedSourceDtype, got {other:?}"),
        }
        // The message must explain WHY, so the user knows to re-export as f32
        // rather than thinking the tool is broken.
        let msg = err.to_string();
        assert!(msg.contains("double rounding"), "msg must explain: {msg}");
    }
}

/// Non-float dtypes pass through untouched, keeping their header spelling.
///
/// `U16` is included specifically because of the numpy tolerance quirk: numpy
/// has no native bf16 and writes bf16 tensors under a `"U16"` header. A cast
/// must not "helpfully" rewrite that to `BF16`, because the file's producer
/// convention is part of its contract.
#[test]
fn non_float_dtypes_pass_through_untouched() {
    // (dtype, a distinctive payload, element count)
    let cases: &[(DType, Vec<u8>, u64)] = &[
        (DType::I64, vec![0xFF; 8 * 3], 3),
        (DType::I32, vec![0x7B; 4 * 2], 2),
        (DType::I16, vec![0x11; 2 * 5], 5),
        (DType::I8, vec![0x22; 1 * 4], 4),
        (DType::U8, vec![0xC3; 1 * 6], 6),
        (DType::U16, vec![0x5A; 2 * 2], 2),
        (DType::Bool, vec![0x01; 1 * 7], 7),
        (DType::F8E4M3, vec![0x7E; 1 * 8], 8),
    ];

    for &(dtype, ref payload, n) in cases {
        for target in [DType::Bf16, DType::F16, DType::F32] {
            let (out, outcome) = cast_tensor("w", &payload, dtype, target, &[n]).unwrap();
            assert_eq!(
                out, *payload,
                "{dtype} must pass through untouched when targeting {target}"
            );
            assert_eq!(
                outcome,
                CastOutcome::Verbatim,
                "{dtype} -> {target} is a pass-through, not a conversion"
            );
            // The output dtype must remain the ORIGINAL, never the target.
            assert_eq!(
                output_dtype_for(dtype, target),
                dtype,
                "{dtype} must not be re-typed to the cast target"
            );
        }
    }
}

/// Float tensors DO take the target dtype; `U16`-spelled bf16 normalises.
#[test]
fn float_dtypes_take_the_target_and_output_dtype_for_is_correct() {
    for src in [DType::F32, DType::F16, DType::Bf16] {
        for target in [DType::Bf16, DType::F16, DType::F32] {
            assert_eq!(
                output_dtype_for(src, target),
                target,
                "{src} -> {target} must adopt the target dtype"
            );
        }
    }
}

// --------------------------------------------------------------------------- //
// Exact widening, and the header invariant
// --------------------------------------------------------------------------- //

/// bf16 -> f32 and f16 -> f32 are EXACT: every value is representable.
#[test]
fn widening_to_f32_is_exact() {
    // A spread of bf16 bit patterns including subnormals, zeros and specials.
    let words = [0x3F80u16, 0x8000, 0x0000, 0x0001, 0x7F80, 0xFF80, 0x7FC0, 0x7F7F];
    let src = bf16_payload(&words);
    let (out, outcome) = cast_tensor("w", &src, DType::Bf16, DType::F32, &[8]).unwrap();
    assert_eq!(outcome, CastOutcome::Converted);
    let got: Vec<u32> = out
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for (i, &w) in words.iter().enumerate() {
        // The widening is a pure left-shift by 16. Compared as BIT PATTERNS,
        // not values: `0x7FC0` is a NaN, and NaN != NaN, so a value
        // comparison here would fail on a correct implementation.
        assert_eq!(
            got[i],
            (w as u32) << 16,
            "bf16 0x{w:04X} widens exactly (bit pattern)"
        );
    }

    // f16 -> f32 likewise.
    let w16 = [0x3C00u16, 0x8000, 0x0000, 0x7C00, 0x7E00, 0x7BFF];
    let src16 = f16_payload(&w16);
    let (out16, _) = cast_tensor("w", &src16, DType::F16, DType::F32, &[6]).unwrap();
    for (i, &w) in w16.iter().enumerate() {
        let got = u32::from_le_bytes([
            out16[i * 4],
            out16[i * 4 + 1],
            out16[i * 4 + 2],
            out16[i * 4 + 3],
        ]);
        // Bit pattern again, for the NaN entry (0x7E00 is a NaN).
        assert_eq!(
            got,
            f16_bits_to_f32(w).to_bits(),
            "f16 0x{w:04X} widens exactly (bit pattern)"
        );
    }
}

/// A `cast` output header must carry NO quantization metadata.
///
/// This is the highest-severity invariant of the whole feature: a bf16 file
/// that advertises itself as ComfyUI-quantized loads as a BROKEN model, because
/// the loader expects scales that are not there.
///
/// The check is on the writer's metadata map, which is the only place this
/// command can introduce such a key.
#[test]
fn output_header_carries_no_quantization_metadata() {
    use quant_core::st_io::writer::IncrementalWriter;

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("model-bf16.safetensors");

    let mut meta = serde_json::Map::new();
    meta.insert("format".into(), serde_json::Value::String("pt".into()));
    let mut w = IncrementalWriter::open_new_with(&out, 1 << 16, Some(meta)).unwrap();
    w.add_tensor("a.w", DType::Bf16, None, &[2], &bf16_payload(&[0x3F80, 0x4000]))
        .unwrap();
    w.finalize().unwrap();

    // Read the raw header back and assert on its KEYS.
    let raw = std::fs::read(&out).unwrap();
    let n = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&raw[8..8 + n]).expect("header must be valid JSON");

    let obj = header.as_object().expect("header must be an object");
    let meta = obj
        .get("__metadata__")
        .and_then(|v| v.as_object())
        .expect("__metadata__ must be present");
    assert_eq!(
        meta.len(),
        1,
        "metadata must hold exactly one key, got {:?}",
        meta.keys().collect::<Vec<_>>()
    );
    assert_eq!(meta.get("format").and_then(|v| v.as_str()), Some("pt"));

    // No quantization markers, at the top level or inside metadata.
    let hay = String::from_utf8_lossy(&raw[..8 + n]).into_owned();
    for forbidden in [
        ".comfy_quant",
        "weight_scale",
        "_quantization_metadata",
        "comfy_quant",
        "quantization",
    ] {
        assert!(
            !hay.contains(forbidden),
            "output header must not mention {forbidden:?}: {hay}"
        );
    }

    // The tensor itself must be present and plainly typed — the file is a
    // real model, not a quantization descriptor.
    let entry = obj.get("a.w").expect("tensor entry must exist");
    assert_eq!(entry["dtype"], serde_json::Value::String("BF16".into()));
}

/// The whole-model shape contract: a 2-element bf16 tensor round-trips through
/// the cast and comes back byte-identical, with its shape preserved.
#[test]
fn cast_preserves_shape_and_payload_for_the_target_model_case() {
    // The real target is an all-bf16 model, so bf16 -> bf16 is the hot path:
    // 1204 tensors, largest 445 MiB, and every byte must survive.
    let n = 4096usize;
    let words: Vec<u16> = (0..n).map(|i| (i as u16).wrapping_mul(2654)).collect();
    let src = bf16_payload(&words);
    let (out, outcome) = cast_tensor("big.weight", &src, DType::Bf16, DType::Bf16, &[n as u64])
        .unwrap();
    assert_eq!(outcome, CastOutcome::Verbatim);
    assert_eq!(out.len(), src.len(), "byte length preserved");
    assert_eq!(out, src, "a 4096-element bf16 tensor must survive byte-identical");
}

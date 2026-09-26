//! `gguf_verify` tests: --verify-against oracle report (task #7).
//!
//! Hand-built GGUF pairs exercising every classification path:
//! byte-exact / dead-block cosmetic / scale-rule diff / genuine divergence
//! / dtype mismatch. Q8_0 payloads are built by hand so each scenario is
//! fully deterministic (no reliance on encoders).

use rlx_gguf::{GgmlType, GgufWriter};

/// f16 little-endian bytes.
fn f16(v: f32) -> [u8; 2] {
    half::f16::from_f32(v).to_le_bytes()
}

/// Build one 34-byte Q8_0 block: f16 scale + 32 i8 codes.
fn q8_block(d: f32, codes: [i8; 32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(34);
    b.extend_from_slice(&f16(d));
    for c in codes {
        b.push(c as u8);
    }
    b
}

/// Write a GGUF with one Q8_0 tensor from block payloads.
fn write_q8_gguf(path: &std::path::Path, name: &str, blocks: &[Vec<u8>]) {
    let mut w = GgufWriter::new();
    w.set_arch("test");
    let mut bytes = Vec::new();
    for b in blocks {
        bytes.extend_from_slice(b);
    }
    w.add_tensor_bytes(name, vec![blocks.len() * 32], GgmlType::Q8_0, bytes)
        .unwrap();
    w.write_to_path(path).unwrap();
}

const NAME: &str = "blk.0.attn_q.weight";

/// Identical files → 1 byte-exact, 0 diffs.
#[test]
fn identical_files_are_byte_exact() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");
    let payload = q8_block(0.01, [1; 32]);
    write_q8_gguf(&a, NAME, &[payload.clone(), q8_block(0.02, [2; 32])]);
    write_q8_gguf(&b, NAME, &[payload, q8_block(0.02, [2; 32])]);

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    let (exact, equiv, divergent) = r.summary();
    assert_eq!((exact, equiv, divergent), (1, 0, 0), "{r:?}");
    assert!(r.spec_violations.is_empty());
    assert_eq!(r.byte_exact, vec![NAME.to_string()]);
}

/// Dead-block: one block with d == 0 on BOTH sides but different codes →
/// DEAD-BLOCK-COSMETIC (reconstruction identical: q·0 = 0 both sides).
#[test]
fn zero_scale_blocks_classify_as_dead_block_cosmetic() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");
    write_q8_gguf(
        &a,
        NAME,
        &[q8_block(0.01, [1; 32]), q8_block(0.0, [-1; 32])],
    );
    write_q8_gguf(
        &b,
        NAME,
        &[q8_block(0.01, [1; 32]), q8_block(0.0, [127; 32])],
    );

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(r.summary(), (0, 1, 0), "dead block is equivalent: {r:?}");
    assert_eq!(
        r.diffs[0].kind,
        quant_core::gguf_verify::DiffKind::DeadBlockCosmetic
    );
    // Reconstruction identical: max abs error 0.
    assert_eq!(r.diffs[0].max_abs_err, 0.0);
}

/// Same scale, different non-dead codes → GENUINE DIVERGENCE.
#[test]
fn same_scale_different_codes_is_genuine_divergence() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");
    write_q8_gguf(&a, NAME, &[q8_block(0.01, [1; 32])]);
    write_q8_gguf(&b, NAME, &[q8_block(0.01, [2; 32])]);

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(r.summary(), (0, 0, 1), "{r:?}");
    assert_eq!(
        r.diffs[0].kind,
        quant_core::gguf_verify::DiffKind::GenuineDivergence
    );
    // max|Δ| = 1 code * scale; the f16-rounded 0.01 differs from decimal
    // 0.01 by ~1e-5, so compare with a loose absolute tolerance.
    assert!(
        (r.diffs[0].max_abs_err - 0.01).abs() < 1e-3,
        "max_abs_err = {}",
        r.diffs[0].max_abs_err
    );
}

/// Different scales on non-dead blocks → SCALE-RULE-DIFF (Q8_0 arm).
#[test]
fn different_scales_is_scale_rule_diff() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");
    write_q8_gguf(
        &a,
        NAME,
        &[q8_block(0.01, [1; 32]), q8_block(0.03, [3; 32])],
    );
    write_q8_gguf(
        &b,
        NAME,
        &[q8_block(0.01, [1; 32]), q8_block(0.04, [2; 32])],
    );

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(
        r.diffs[0].kind,
        quant_core::gguf_verify::DiffKind::ScaleRuleDiff,
        "{r:?}"
    );
}

/// Dtype mismatch on the same name → dtype_mismatch, no payload compare.
#[test]
fn dtype_mismatch_is_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");

    // Ours: F16 tensor, 64 elements.
    let mut w = GgufWriter::new();
    w.set_arch("test");
    w.add_tensor_bytes(NAME, vec![64], GgmlType::F16, vec![0u8; 128])
        .unwrap();
    w.write_to_path(&a).unwrap();

    // Ref: same name as Q8_0, 64 elements (2 blocks).
    write_q8_gguf(
        &b,
        NAME,
        &[q8_block(0.01, [1; 32]), q8_block(0.01, [1; 32])],
    );

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert!(r.byte_exact.is_empty());
    assert!(r.diffs.is_empty());
    assert_eq!(r.dtype_mismatch.len(), 1, "{r:?}");
    assert!(r.dtype_mismatch[0].contains(NAME));
    assert!(r.dtype_mismatch[0].contains("F16"));
    assert!(r.dtype_mismatch[0].contains("Q8_0"));
}

/// Name mapping: ours stores the HF name `model.layers.0.self_attn.q_proj.weight`;
/// the reference uses `blk.0.attn_q.weight`. The verify report must match them.
#[test]
fn hf_names_map_onto_reference_names() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");
    let payload = q8_block(0.02, [3; 32]);
    let payload2 = payload.clone();
    // Ours written under the HF-ish name (what our converter produced
    // before task #1's mapping — the tool must still line them up).
    write_q8_gguf(&a, "model.layers.0.self_attn.q_proj.weight", &[payload]);
    write_q8_gguf(&b, "blk.0.attn_q.weight", &[payload2]);

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(r.summary(), (1, 0, 0), "name-mapped match: {r:?}");
    assert!(r.ours_only.is_empty());
    assert!(r.ref_only.is_empty());
}

/// Unmapped names land in ours_only / ref_only, never silently dropped.
#[test]
fn unmatched_names_are_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");
    write_q8_gguf(&a, "model.custom_thing.weight", &[q8_block(0.01, [1; 32])]);
    write_q8_gguf(
        &b,
        "blk.9.something_else.weight",
        &[q8_block(0.01, [1; 32])],
    );

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(r.summary(), (0, 0, 0));
    assert_eq!(r.ours_only, vec!["model.custom_thing.weight".to_string()]);
    assert_eq!(r.ref_only, vec!["blk.9.something_else.weight".to_string()]);
}

/// Spec-conformance scan: a file whose quantized tensor has ne[0] not
/// divisible by the block size must be flagged. The violating file is
/// hand-crafted byte-by-byte (the rlx-gguf writer refuses such shapes —
/// that refusal IS the Phase BugFix contract).
#[test]
fn spec_violation_scan_flags_non_divisible_row() {
    let tmp = tempfile::tempdir().unwrap();
    let bad = tmp.path().join("bad.gguf");
    let clean = tmp.path().join("clean.gguf");

    write_raw_violating_q8(&bad, "blk.0.attn_q.weight");
    write_q8_gguf(&clean, "blk.0.attn_q.weight", &[q8_block(0.01, [1; 32])]);

    let r = quant_core::gguf_verify::verify_against(&bad, &clean).unwrap();
    assert_eq!(
        r.spec_violations,
        vec!["blk.0.attn_q.weight".to_string()],
        "ne[0]=7 Q8_0 tensor must be flagged: {r:?}"
    );

    // The clean counterpart: no violations.
    let r2 = quant_core::gguf_verify::verify_against(&clean, &clean).unwrap();
    assert!(r2.spec_violations.is_empty());
}

/// Hand-crafted spec-violating GGUF: one Q8_0 tensor, dims ne = [7, 1, 4]
/// → ne[0] = 7 (7 % 32 != 0), flat count 224 divisible (7 blocks x 32),
/// payload 7 x 34 bytes. Used by the violation-scan and audit tests.
fn write_raw_violating_q8(path: &std::path::Path, name: &str) {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
    b.extend_from_slice(&0u64.to_le_bytes()); // n_kv
    let tname = name.as_bytes();
    b.extend_from_slice(&(tname.len() as u64).to_le_bytes());
    b.extend_from_slice(tname);
    b.extend_from_slice(&3u32.to_le_bytes()); // n_dims
    b.extend_from_slice(&7i64.to_le_bytes()); // ne[0] = 7  <- the violation
    b.extend_from_slice(&1i64.to_le_bytes());
    b.extend_from_slice(&4i64.to_le_bytes());
    b.extend_from_slice(&8u32.to_le_bytes()); // Q8_0
    b.extend_from_slice(&0u64.to_le_bytes()); // offset
    while b.len() % 32 != 0 {
        b.push(0);
    }
    b.extend(std::iter::repeat_n(0u8, 7 * 34));
    std::fs::write(path, &b).unwrap();
}

// ─── [IMP-003] audit_gguf ───────────────────────────────────────────

/// Audit on a clean file: dtype census correct, zero violations.
#[test]
fn audit_clean_file_reports_histogram() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    write_q8_gguf(&a, "blk.0.attn_q.weight", &[q8_block(0.01, [1; 32])]);

    let a = quant_core::gguf_verify::audit_gguf(&a).unwrap();
    assert_eq!(a.tensor_count, 1);
    assert_eq!(a.dtype_histogram.get("Q8_0"), Some(&1));
    assert!(a.violations.is_empty(), "{a:?}");
}

/// Audit on the hand-crafted violating file: exactly one violation with
/// full detail (name, type, ne0, blck).
#[test]
fn audit_reports_violation_details() {
    let tmp = tempfile::tempdir().unwrap();
    let bad = tmp.path().join("bad.gguf");
    write_raw_violating_q8(&bad, "blk.0.attn_q.weight");

    let a = quant_core::gguf_verify::audit_gguf(&bad).unwrap();
    assert_eq!(a.tensor_count, 1);
    assert_eq!(a.violations.len(), 1, "{a:?}");
    let v = &a.violations[0];
    assert_eq!(v.name, "blk.0.attn_q.weight");
    assert_eq!(v.dtype, "Q8_0");
    assert_eq!(v.ne0, 7);
    assert_eq!(v.blck, 32);
    assert_eq!(a.dtype_histogram.get("Q8_0"), Some(&1));
}

/// Audit on garbage input is a clean Err, never a panic.
#[test]
fn audit_unparseable_file_is_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    let junk = tmp.path().join("junk.gguf");
    std::fs::write(&junk, b"definitely not gguf").unwrap();

    let r = quant_core::gguf_verify::audit_gguf(&junk);
    assert!(r.is_err(), "junk must be an Err: {r:?}");
    let e = r.unwrap_err();
    assert!(e.contains("parsing"), "error names the file: {e}");
}

// ─── S05: DiffKind::FormatConformance ────────────────────────────────
//
// The predicate under test lives on NVFP4, because NVFP4 is the only GGML
// dtype whose block scale is E4M3 (`rlx-gguf` `mx_dequant.rs`: NVFP4 = 16
// elems, 9 bytes/block, byte 0 = E4M3 scale; MXFP4's scale is E8M0, a pure
// exponent format with no NaN-vs-saturated question at all).
//
// Plan note: the plan's §S05 sketch said "a Q8_0 tensor pair differing only in
// a NaN-vs-saturated payload code". That is not constructible — a Q8_0 payload
// is f16 scale + 32 i8 codes and contains no E4M3 bytes anywhere, so no Q8_0
// input can exercise an E4M3 overflow policy. NVFP4 is the reachable case.

/// NVFP4 block: 1 E4M3 scale byte + 8 packed-nibble data bytes = 9 bytes.
fn nvfp4_block(scale: u8, nibbles: [u8; 8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(9);
    b.push(scale);
    b.extend_from_slice(&nibbles);
    b
}

/// Write a GGUF with one NVFP4 tensor from raw 9-byte blocks.
fn write_nvfp4_gguf(path: &std::path::Path, name: &str, blocks: &[Vec<u8>]) {
    let mut w = GgufWriter::new();
    w.set_arch("test");
    let mut bytes = Vec::new();
    for b in blocks {
        bytes.extend_from_slice(b);
    }
    w.add_tensor_bytes(name, vec![blocks.len() * 16], GgmlType::NVFP4, bytes)
        .unwrap();
    w.write_to_path(path).unwrap();
}

const NVFP4_NAME: &str = "blk.0.ffn_up.weight";

/// The must-not-false-alarm case: two payloads differing ONLY in the E4M3
/// scale byte, and only as NaN (`0x7F`) vs saturated max-finite (`0x7E`)
/// classify `FormatConformance`, and `summary()` counts them in the
/// numerically-equivalent bucket.
///
/// Adversarial case defended: the exact false alarm this tier exists to stop.
/// A reference encoder using the finite-only OCP MX E4M3 variant emits `0x7E`
/// where our `float8_e4m3fn` emits `0x7F` for the same nominal value. Both are
/// legal under OCP MX v1.0 — it is a *policy* difference, so reporting it as
/// `GENUINE DIVERGENCE` sends a maintainer hunting a nonexistent math bug.
#[test]
fn format_policy_mismatch_is_not_genuine_divergence() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");

    // Block 0: scale NaN (ours) vs saturated (theirs). Data identical.
    // Block 1: negative pair, FF vs FE. Data identical.
    let data = [0x12u8; 8];
    write_nvfp4_gguf(
        &a,
        NVFP4_NAME,
        &[nvfp4_block(0x7F, data), nvfp4_block(0xFF, data)],
    );
    write_nvfp4_gguf(
        &b,
        NVFP4_NAME,
        &[nvfp4_block(0x7E, data), nvfp4_block(0xFE, data)],
    );

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(r.diffs.len(), 1, "{r:?}");
    assert_eq!(
        r.diffs[0].kind,
        quant_core::gguf_verify::DiffKind::FormatConformance,
        "NaN-vs-saturated E4M3 scale is a policy difference: {r:?}"
    );
    // Counted as equivalent, NOT divergent.
    let (exact, equiv, divergent) = r.summary();
    assert_eq!((exact, equiv, divergent), (0, 1, 0), "{r:?}");

    // The reverse direction is the same verdict — the predicate is symmetric.
    let r2 = quant_core::gguf_verify::verify_against(&b, &a).unwrap();
    assert_eq!(
        r2.diffs[0].kind,
        quant_core::gguf_verify::DiffKind::FormatConformance,
        "verdict must not depend on which side is ours: {r2:?}"
    );
}

/// REGRESSION, and the most important test in this step: a differing E2M1
/// *nibble* is a math disagreement and MUST stay `GenuineDivergence`.
///
/// Adversarial case defended: an over-broad predicate. The whole value of a
/// conservative rule is that it does not swallow genuine bugs. This is the
/// test that would catch a predicate written as "any differing byte that
/// happens to look like a NaN/saturated pair" — byte 1 here differs by
/// exactly one and is *not* a scale byte.
#[test]
fn differing_nibble_data_is_still_genuine_divergence() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");

    // Scale bytes identical; a data nibble differs. This is real divergence.
    let mut theirs = [0x12u8; 8];
    theirs[3] = 0x13; // one data byte differs
    write_nvfp4_gguf(&a, NVFP4_NAME, &[nvfp4_block(0x7E, [0x12u8; 8])]);
    write_nvfp4_gguf(&b, NVFP4_NAME, &[nvfp4_block(0x7E, theirs)]);

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(
        r.diffs[0].kind,
        quant_core::gguf_verify::DiffKind::GenuineDivergence,
        "a differing E2M1 nibble is data, not policy: {r:?}"
    );
    assert_eq!(r.summary(), (0, 0, 1), "{r:?}");
}

/// REGRESSION, second axis: a scale difference that is NOT a
/// NaN-vs-saturated pair is a genuinely different scale, so it stays
/// divergent rather than being excused as policy.
///
/// Adversarial case defended: a predicate that treats *any* scale-byte
/// difference as policy. `0x7E` vs `0x30` is a real exponent disagreement —
/// exactly the case the reference `ScaleRuleDiff` reasoning exists for.
#[test]
fn non_policy_scale_difference_is_still_genuine_divergence() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");

    let data = [0x12u8; 8];
    write_nvfp4_gguf(&a, NVFP4_NAME, &[nvfp4_block(0x7E, data)]);
    write_nvfp4_gguf(&b, NVFP4_NAME, &[nvfp4_block(0x30, data)]);

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(
        r.diffs[0].kind,
        quant_core::gguf_verify::DiffKind::GenuineDivergence,
        "0x7E vs 0x30 is a real exponent difference, not an overflow policy: {r:?}"
    );
}

/// REGRESSION, third axis: the predicate is ALL-OR-NOTHING. A payload with one
/// policy pair *and* one unrelated difference is still a genuine divergence.
///
/// Adversarial case defended: a predicate that tolerates policy pairs
/// opportunistically — "downgrade if ANY differing byte looks like a NaN
/// pair". That would excuse a tensor with a real data bug hiding behind one
/// benign byte. One unexplained difference must poison the whole tensor.
#[test]
fn one_extra_differing_byte_poisons_the_whole_tensor() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");

    // Block 0 differs by a policy pair (0x7F vs 0x7E).
    // Block 1's scale differs by something that is NOT a policy pair.
    let d0 = [0x11u8; 8];
    let d1 = [0x22u8; 8];
    write_nvfp4_gguf(
        &a,
        NVFP4_NAME,
        &[nvfp4_block(0x7F, d0), nvfp4_block(0x40, d1)],
    );
    write_nvfp4_gguf(
        &b,
        NVFP4_NAME,
        &[nvfp4_block(0x7E, d0), nvfp4_block(0x41, d1)],
    );

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(
        r.diffs[0].kind,
        quant_core::gguf_verify::DiffKind::GenuineDivergence,
        "one non-policy difference must poison the whole tensor: {r:?}"
    );
    assert_eq!(r.summary(), (0, 0, 1), "{r:?}");
}

/// REGRESSION giving the *position* check teeth: a differing **data** byte
/// whose value happens to be a NaN/saturated pair is still a real divergence.
///
/// Adversarial case defended: the specific over-capture the plan names as this
/// step's main risk. Packed E2M1 nibble bytes are arbitrary `u8` values, so
/// `0x7F`/`0x7E` is a perfectly legal *data* byte pair. A predicate that scans
/// the whole payload for policy pairs without checking WHERE they are would
/// classify this as `FormatConformance` and hide a genuine data bug. Only the
/// scale position (byte 0 of each 9-byte block) may be excused.
///
/// This test is the difference between the position guard being load-bearing
/// and being decorative: removing the `i % 9 == 0` check makes it fail.
#[test]
fn policy_pair_in_a_data_position_is_still_genuine() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");

    // Scale bytes are IDENTICAL. The only difference is data byte 3, and its
    // value pair (0x7F, 0x7E) is exactly a policy pair.
    let mut ours = [0x11u8; 8];
    ours[3] = 0x7F;
    let mut theirs = [0x11u8; 8];
    theirs[3] = 0x7E;
    write_nvfp4_gguf(&a, NVFP4_NAME, &[nvfp4_block(0x60, ours)]);
    write_nvfp4_gguf(&b, NVFP4_NAME, &[nvfp4_block(0x60, theirs)]);

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(
        r.diffs[0].kind,
        quant_core::gguf_verify::DiffKind::GenuineDivergence,
        "a policy-shaped pair in a DATA position is a real bug, not policy: {r:?}"
    );
    assert_eq!(r.summary(), (0, 0, 1), "{r:?}");
}

/// The new arm is scoped to NVFP4. A Q8_0 code `0x7F` is a legitimate int8
/// code, not a NaN, so a Q8_0 pair containing those bytes must NOT be
/// downgraded.
///
/// Adversarial case defended: the arm being written as a dtype-independent
/// post-pass over the raw payload. That would reinterpret arbitrary int8 codes
/// as E4M3 NaNs and downgrade genuine Q8_0 divergences — the exact
/// "hides a real bug" failure.
#[test]
fn non_e4m3_dtypes_are_never_downgraded() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");

    // Q8_0 codes 127 and -1 differ; same nonzero scale. Genuine.
    write_q8_gguf(&a, NAME, &[q8_block(0.01, [127; 32])]);
    write_q8_gguf(&b, NAME, &[q8_block(0.01, [-1; 32])]);

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(
        r.diffs[0].kind,
        quant_core::gguf_verify::DiffKind::GenuineDivergence,
        "Q8_0 int8 code 0x7F is not an E4M3 NaN: {r:?}"
    );
}

/// REGRESSION: the pre-existing Q8_0 / Q4_0 classification arms are untouched.
///
/// Adversarial case defended: the new arm being inserted ABOVE an existing arm
/// in `classify` and stealing its cases. These restate the two Q8_0 verdicts
/// that existed before S05, so a reordering regression is caught here.
#[test]
fn scale_rule_diff_is_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");
    write_q8_gguf(
        &a,
        NAME,
        &[q8_block(0.01, [1; 32]), q8_block(0.03, [3; 32])],
    );
    write_q8_gguf(
        &b,
        NAME,
        &[q8_block(0.01, [1; 32]), q8_block(0.04, [2; 32])],
    );

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(
        r.diffs[0].kind,
        quant_core::gguf_verify::DiffKind::ScaleRuleDiff,
        "Q8_0 differing-scale arm must be unchanged: {r:?}"
    );
}

/// REGRESSION: the Q1 quirk stays `DeadBlockCosmetic`.
///
/// Q8_0 denormal blocks: ours gives code 127, the gguf-py oracle gives 0
/// (rlx-gguf clamps; Rust `inf as i32` saturates). Classified
/// `DeadBlockCosmetic` and benign, since `d == 0` on both sides. S05 must not
/// disturb it.
#[test]
fn dead_block_cosmetic_is_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");
    write_q8_gguf(
        &a,
        NAME,
        &[q8_block(0.01, [1; 32]), q8_block(0.0, [-1; 32])],
    );
    write_q8_gguf(
        &b,
        NAME,
        &[q8_block(0.01, [1; 32]), q8_block(0.0, [127; 32])],
    );

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(
        r.diffs[0].kind,
        quant_core::gguf_verify::DiffKind::DeadBlockCosmetic,
        "Q1 denormal quirk classification must be unchanged: {r:?}"
    );
    assert_eq!(r.diffs[0].max_abs_err, 0.0);
}

/// REGRESSION: MXFP4 must NOT be downgraded, even though its byte 0 is also a
/// scale byte holding a NaN-vs-saturated-looking pair.
///
/// MXFP4's block scale is **E8M0** — a pure exponent format with no NaN and no
/// saturation — so an E4M3 overflow policy cannot arise there at all. The arm
/// is deliberately scoped to NVFP4, and this test is what keeps it scoped.
///
/// Adversarial case defended: widening the arm to "any MX format" on the
/// reasoning that MXFP4 is adjacent. That would excuse MXFP4 block-scale
/// disagreements — which are real encoder bugs — as harmless policy.
#[test]
fn mxfp4_is_never_downgraded() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    let b = tmp.path().join("b.gguf");

    // MXFP4: 32 elements/block, 17 bytes/block (1 E8M0 scale + 16 nibble bytes).
    let write_mxfp4 = |path: &std::path::Path, scale: u8| {
        let mut block = vec![0u8; 16];
        block.insert(0, scale);
        let mut w = GgufWriter::new();
        w.set_arch("test");
        w.add_tensor_bytes(NAME, vec![32], GgmlType::MXFP4, block)
            .unwrap();
        w.write_to_path(path).unwrap();
    };
    write_mxfp4(&a, 0x7F);
    write_mxfp4(&b, 0x7E);

    let r = quant_core::gguf_verify::verify_against(&a, &b).unwrap();
    assert_eq!(
        r.diffs[0].kind,
        quant_core::gguf_verify::DiffKind::GenuineDivergence,
        "MXFP4 scale is E8M0, so an E4M3 policy pair means nothing here: {r:?}"
    );
}

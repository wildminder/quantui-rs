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

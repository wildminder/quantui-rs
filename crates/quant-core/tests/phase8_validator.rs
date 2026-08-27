//! Phase 8 E2E tests: structural + numeric validator.
//!
//! Positive: every golden output (INT8/FP8/MXFP8/NVFP4 across all cases, plus
//! the sharded outputs) validates clean, both structural-only and with the
//! numeric pass.
//!
//! Negative: a hand-corrupted corpus — each file yields the specific issue
//! codes from the reference `Issue` model (parse errors, missing tensors,
//! wrong dtypes/shapes, quantized bias, orphans, bad input_scale, bad header).

use std::path::{Path, PathBuf};

use quant_core::st_io::align_header_to_8;
use quant_core::validator::{validate_comfy_quant, IssueLevel};

fn golden(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.join("tests/golden").join(name)
}

// --------------------------------------------------------------------------- //
// Minimal safetensors builder for the negative corpus.
// --------------------------------------------------------------------------- //

/// Build a valid single-file safetensors from (name, dtype, shape, bytes).
fn build_st(tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
    let mut obj = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len() as u64;
        data.extend_from_slice(bytes);
        let end = data.len() as u64;
        let mut entry = serde_json::Map::new();
        entry.insert("dtype".into(), serde_json::Value::from(*dtype));
        entry.insert(
            "shape".into(),
            serde_json::Value::Array(shape.iter().map(|&d| serde_json::Value::from(d)).collect()),
        );
        entry.insert("data_offsets".into(), serde_json::json!([start, end]));
        obj.insert((*name).to_string(), serde_json::Value::Object(entry));
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(obj)).unwrap();
    let header = align_header_to_8(&header);
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    out
}

fn write_tmp(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

fn int8_weight_bytes(m: usize, n: usize) -> Vec<u8> {
    // All values within [-127, 127].
    vec![7i8 as u8; m * n]
}

fn f32_scale_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn marker_bytes(json: &str) -> Vec<u8> {
    json.as_bytes().to_vec()
}

const INT8_BLOB: &str =
    r#"{"format": "int8_blockwise", "orig_dtype": "torch.bfloat16", "group_size": 128}"#;

/// Assemble a valid int8_blockwise file for one 256x128 layer.
fn valid_int8_file() -> Vec<u8> {
    build_st(&[
        (
            "blk.weight",
            "I8",
            vec![256, 128],
            int8_weight_bytes(256, 128),
        ),
        (
            "blk.weight_scale",
            "F32",
            vec![2, 1],
            f32_scale_bytes(&[0.5, 0.5]),
        ),
        ("blk.input_scale", "F32", vec![], f32_scale_bytes(&[1.0])),
        (
            "blk.comfy_quant",
            "U8",
            vec![INT8_BLOB.len() as u64],
            marker_bytes(INT8_BLOB),
        ),
    ])
}

// --------------------------------------------------------------------------- //
// Positive: all goldens validate clean.
// --------------------------------------------------------------------------- //

fn all_golden_outputs() -> Vec<PathBuf> {
    let root = golden("");
    let mut files = Vec::new();
    for case in std::fs::read_dir(&root).unwrap() {
        let case = case.unwrap().path();
        if !case.is_dir() {
            continue;
        }
        for f in std::fs::read_dir(&case).unwrap() {
            let f = f.unwrap().path();
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            if f.is_dir() {
                // Sharded outputs live one level deeper.
                for sf in std::fs::read_dir(&f).unwrap() {
                    let sf = sf.unwrap().path();
                    if sf
                        .file_name()
                        .map(|n| n.to_string_lossy().ends_with(".safetensors"))
                        .unwrap_or(false)
                    {
                        files.push(sf);
                    }
                }
            } else if name.ends_with(".safetensors") {
                files.push(f);
            }
        }
    }
    files.sort();
    files
}

#[test]
fn positive_all_goldens_structural() {
    let files = all_golden_outputs();
    assert!(!files.is_empty(), "must find golden outputs");
    for f in &files {
        let report = validate_comfy_quant(f, false);
        assert!(
            report.ok,
            "{}: expected ok, got errors {:?} warnings {:?}",
            f.display(),
            report.errors,
            report.warnings
        );
    }
}

#[test]
fn positive_all_goldens_numeric() {
    for f in &all_golden_outputs() {
        let report = validate_comfy_quant(f, true);
        assert!(
            report.ok,
            "{}: numeric pass expected ok, got errors {:?}",
            f.display(),
            report.errors
        );
    }
}

#[test]
fn positive_golden_summary_sane() {
    let f = golden("linear_basic_bf16").join("output.safetensors");
    let report = validate_comfy_quant(&f, false);
    assert!(report.ok);
    assert_eq!(report.summary.quantized_matrices, 1);
    assert!(report.formats.contains("int8_blockwise"));
    assert_eq!(report.summary.formats_found, vec!["int8_blockwise"]);
    // blocks.0 is 256x128 = 32768 quantized params.
    assert_eq!(report.summary.quantized_params, 256 * 128);
    assert_eq!(report.layers.len(), 1);
    assert_eq!(report.layers[0].prefix, "blocks.0");
}

// --------------------------------------------------------------------------- //
// Negative corpus: each corruption yields a specific issue.
// --------------------------------------------------------------------------- //

fn assert_has_error(report: &quant_core::validator::ValidationReport, needle: &str) {
    assert!(
        !report.ok,
        "expected failure, but report is ok (errors={:?})",
        report.errors
    );
    assert!(
        report.errors.iter().any(|e| e.contains(needle)),
        "expected an error containing {needle:?}, got {:?}",
        report.errors
    );
}

#[test]
fn negative_file_not_found() {
    let report = validate_comfy_quant("/nonexistent/nope.safetensors", false);
    assert_has_error(&report, "file not found");
}

#[test]
fn negative_invalid_header() {
    let tmp = tempfile::tempdir().unwrap();
    let p = write_tmp(
        tmp.path(),
        "bad_header.safetensors",
        b"not a safetensors file at all",
    );
    let report = validate_comfy_quant(&p, false);
    assert_has_error(&report, "invalid safetensors header");
}

#[test]
fn negative_no_markers_is_ok_with_warning() {
    let tmp = tempfile::tempdir().unwrap();
    let bytes = build_st(&[("plain.weight", "BF16", vec![4, 4], vec![0u8; 32])]);
    let p = write_tmp(tmp.path(), "plain.safetensors", &bytes);
    let report = validate_comfy_quant(&p, false);
    assert!(
        report.ok,
        "no-marker file must be ok, got {:?}",
        report.errors
    );
    assert!(report
        .warnings
        .iter()
        .any(|w| w.contains("no .comfy_quant markers")));
}

#[test]
fn negative_truncated_marker_json() {
    let tmp = tempfile::tempdir().unwrap();
    let truncated = r#"{"format": "int8_blockwise", "orig_dtype": "torch.bfloat16""#;
    let bytes = build_st(&[
        (
            "blk.weight",
            "I8",
            vec![256, 128],
            int8_weight_bytes(256, 128),
        ),
        (
            "blk.weight_scale",
            "F32",
            vec![2, 1],
            f32_scale_bytes(&[0.5, 0.5]),
        ),
        ("blk.input_scale", "F32", vec![], f32_scale_bytes(&[1.0])),
        (
            "blk.comfy_quant",
            "U8",
            vec![truncated.len() as u64],
            marker_bytes(truncated),
        ),
    ]);
    let p = write_tmp(tmp.path(), "trunc.safetensors", &bytes);
    let report = validate_comfy_quant(&p, false);
    assert_has_error(&report, "not valid JSON");
}

#[test]
fn negative_unknown_format() {
    let tmp = tempfile::tempdir().unwrap();
    let blob = r#"{"format": "convrot_w4a4", "orig_dtype": "torch.bfloat16"}"#;
    let bytes = build_st(&[
        (
            "blk.weight",
            "I8",
            vec![256, 128],
            int8_weight_bytes(256, 128),
        ),
        (
            "blk.weight_scale",
            "F32",
            vec![2, 1],
            f32_scale_bytes(&[0.5, 0.5]),
        ),
        (
            "blk.comfy_quant",
            "U8",
            vec![blob.len() as u64],
            marker_bytes(blob),
        ),
    ]);
    let p = write_tmp(tmp.path(), "unknown.safetensors", &bytes);
    let report = validate_comfy_quant(&p, false);
    assert_has_error(&report, "unknown .comfy_quant format");
}

#[test]
fn negative_missing_format_field() {
    let tmp = tempfile::tempdir().unwrap();
    let blob = r#"{"orig_dtype": "torch.bfloat16"}"#;
    let bytes = build_st(&[
        (
            "blk.weight",
            "I8",
            vec![256, 128],
            int8_weight_bytes(256, 128),
        ),
        (
            "blk.weight_scale",
            "F32",
            vec![2, 1],
            f32_scale_bytes(&[0.5, 0.5]),
        ),
        (
            "blk.comfy_quant",
            "U8",
            vec![blob.len() as u64],
            marker_bytes(blob),
        ),
    ]);
    let p = write_tmp(tmp.path(), "noformat.safetensors", &bytes);
    let report = validate_comfy_quant(&p, false);
    assert_has_error(&report, "missing the \"format\" field");
}

#[test]
fn negative_missing_weight() {
    let tmp = tempfile::tempdir().unwrap();
    // Marker + scale but no weight.
    let bytes = build_st(&[
        (
            "blk.weight_scale",
            "F32",
            vec![2, 1],
            f32_scale_bytes(&[0.5, 0.5]),
        ),
        (
            "blk.comfy_quant",
            "U8",
            vec![INT8_BLOB.len() as u64],
            marker_bytes(INT8_BLOB),
        ),
    ]);
    let p = write_tmp(tmp.path(), "noweight.safetensors", &bytes);
    let report = validate_comfy_quant(&p, false);
    assert_has_error(&report, "missing quantized weight tensor");
}

#[test]
fn negative_missing_scale() {
    let tmp = tempfile::tempdir().unwrap();
    let bytes = build_st(&[
        (
            "blk.weight",
            "I8",
            vec![256, 128],
            int8_weight_bytes(256, 128),
        ),
        ("blk.input_scale", "F32", vec![], f32_scale_bytes(&[1.0])),
        (
            "blk.comfy_quant",
            "U8",
            vec![INT8_BLOB.len() as u64],
            marker_bytes(INT8_BLOB),
        ),
    ]);
    let p = write_tmp(tmp.path(), "noscale.safetensors", &bytes);
    let report = validate_comfy_quant(&p, false);
    assert_has_error(&report, "missing weight_scale tensor");
}

#[test]
fn negative_wrong_weight_dtype() {
    let tmp = tempfile::tempdir().unwrap();
    // int8_blockwise but weight stored as F32.
    let bytes = build_st(&[
        (
            "blk.weight",
            "F32",
            vec![256, 128],
            vec![0u8; 256 * 128 * 4],
        ),
        (
            "blk.weight_scale",
            "F32",
            vec![2, 1],
            f32_scale_bytes(&[0.5, 0.5]),
        ),
        ("blk.input_scale", "F32", vec![], f32_scale_bytes(&[1.0])),
        (
            "blk.comfy_quant",
            "U8",
            vec![INT8_BLOB.len() as u64],
            marker_bytes(INT8_BLOB),
        ),
    ]);
    let p = write_tmp(tmp.path(), "wrongdtype.safetensors", &bytes);
    let report = validate_comfy_quant(&p, false);
    assert_has_error(&report, "weight dtype");
}

#[test]
fn negative_wrong_scale_shape() {
    let tmp = tempfile::tempdir().unwrap();
    // int8_blockwise 256x128 g=128 expects scale [2,1]; give [3,1].
    let bytes = build_st(&[
        (
            "blk.weight",
            "I8",
            vec![256, 128],
            int8_weight_bytes(256, 128),
        ),
        (
            "blk.weight_scale",
            "F32",
            vec![3, 1],
            f32_scale_bytes(&[0.5, 0.5, 0.5]),
        ),
        ("blk.input_scale", "F32", vec![], f32_scale_bytes(&[1.0])),
        (
            "blk.comfy_quant",
            "U8",
            vec![INT8_BLOB.len() as u64],
            marker_bytes(INT8_BLOB),
        ),
    ]);
    let p = write_tmp(tmp.path(), "wrongshape.safetensors", &bytes);
    let report = validate_comfy_quant(&p, false);
    assert_has_error(&report, "weight_scale shape");
}

#[test]
fn negative_missing_input_scale_for_blockwise() {
    let tmp = tempfile::tempdir().unwrap();
    let bytes = build_st(&[
        (
            "blk.weight",
            "I8",
            vec![256, 128],
            int8_weight_bytes(256, 128),
        ),
        (
            "blk.weight_scale",
            "F32",
            vec![2, 1],
            f32_scale_bytes(&[0.5, 0.5]),
        ),
        (
            "blk.comfy_quant",
            "U8",
            vec![INT8_BLOB.len() as u64],
            marker_bytes(INT8_BLOB),
        ),
    ]);
    let p = write_tmp(tmp.path(), "noinputscale.safetensors", &bytes);
    let report = validate_comfy_quant(&p, false);
    assert_has_error(&report, "requires 'blk.input_scale'");
}

#[test]
fn negative_quantized_bias() {
    let tmp = tempfile::tempdir().unwrap();
    let bytes = build_st(&[
        (
            "blk.weight",
            "I8",
            vec![256, 128],
            int8_weight_bytes(256, 128),
        ),
        (
            "blk.weight_scale",
            "F32",
            vec![2, 1],
            f32_scale_bytes(&[0.5, 0.5]),
        ),
        ("blk.input_scale", "F32", vec![], f32_scale_bytes(&[1.0])),
        ("blk.bias", "I8", vec![256], vec![1u8; 256]),
        (
            "blk.comfy_quant",
            "U8",
            vec![INT8_BLOB.len() as u64],
            marker_bytes(INT8_BLOB),
        ),
    ]);
    let p = write_tmp(tmp.path(), "quantbias.safetensors", &bytes);
    let report = validate_comfy_quant(&p, false);
    assert_has_error(&report, "bias was quantized");
}

#[test]
fn negative_orphan_scale() {
    let tmp = tempfile::tempdir().unwrap();
    // A weight_scale with no matching .comfy_quant marker.
    let bytes = build_st(&[
        (
            "blk.weight",
            "I8",
            vec![256, 128],
            int8_weight_bytes(256, 128),
        ),
        (
            "blk.weight_scale",
            "F32",
            vec![2, 1],
            f32_scale_bytes(&[0.5, 0.5]),
        ),
        ("blk.input_scale", "F32", vec![], f32_scale_bytes(&[1.0])),
        (
            "blk.comfy_quant",
            "U8",
            vec![INT8_BLOB.len() as u64],
            marker_bytes(INT8_BLOB),
        ),
        (
            "orphan.weight_scale",
            "F32",
            vec![1],
            f32_scale_bytes(&[0.1]),
        ),
    ]);
    let p = write_tmp(tmp.path(), "orphanscale.safetensors", &bytes);
    let report = validate_comfy_quant(&p, false);
    assert_has_error(&report, "orphan weight_scale");
}

#[test]
fn negative_orphan_marker() {
    let tmp = tempfile::tempdir().unwrap();
    // A .comfy_quant marker whose layer has no weight_scale.
    let bytes = build_st(&[
        (
            "blk.weight",
            "I8",
            vec![256, 128],
            int8_weight_bytes(256, 128),
        ),
        (
            "blk.comfy_quant",
            "U8",
            vec![INT8_BLOB.len() as u64],
            marker_bytes(INT8_BLOB),
        ),
    ]);
    let p = write_tmp(tmp.path(), "orphanmarker.safetensors", &bytes);
    let report = validate_comfy_quant(&p, false);
    // Missing scale is reported AND the marker is an orphan.
    assert_has_error(&report, "orphan .comfy_quant");
}

#[test]
fn negative_numeric_input_scale_not_one() {
    let tmp = tempfile::tempdir().unwrap();
    let bytes = build_st(&[
        (
            "blk.weight",
            "I8",
            vec![256, 128],
            int8_weight_bytes(256, 128),
        ),
        (
            "blk.weight_scale",
            "F32",
            vec![2, 1],
            f32_scale_bytes(&[0.5, 0.5]),
        ),
        ("blk.input_scale", "F32", vec![], f32_scale_bytes(&[2.5])),
        (
            "blk.comfy_quant",
            "U8",
            vec![INT8_BLOB.len() as u64],
            marker_bytes(INT8_BLOB),
        ),
    ]);
    let p = write_tmp(tmp.path(), "badinputscale.safetensors", &bytes);
    // Structural pass is fine (input_scale present + F32 scalar).
    let structural = validate_comfy_quant(&p, false);
    assert!(
        structural.ok,
        "structural should pass: {:?}",
        structural.errors
    );
    // Numeric pass flags input_scale != 1.0.
    let numeric = validate_comfy_quant(&p, true);
    assert_has_error(&numeric, "input_scale != 1.0");
}

#[test]
fn negative_numeric_nonpositive_scale() {
    let tmp = tempfile::tempdir().unwrap();
    let bytes = build_st(&[
        (
            "blk.weight",
            "I8",
            vec![256, 128],
            int8_weight_bytes(256, 128),
        ),
        (
            "blk.weight_scale",
            "F32",
            vec![2, 1],
            f32_scale_bytes(&[0.5, -0.5]),
        ),
        ("blk.input_scale", "F32", vec![], f32_scale_bytes(&[1.0])),
        (
            "blk.comfy_quant",
            "U8",
            vec![INT8_BLOB.len() as u64],
            marker_bytes(INT8_BLOB),
        ),
    ]);
    let p = write_tmp(tmp.path(), "negscale.safetensors", &bytes);
    let numeric = validate_comfy_quant(&p, true);
    assert_has_error(&numeric, "non-finite / non-positive scales");
}

#[test]
fn negative_numeric_int8_overflow() {
    let tmp = tempfile::tempdir().unwrap();
    // 0x80 as i8 = -128 → unsigned_abs 128 > 127 → overflow.
    let mut w = int8_weight_bytes(256, 128);
    w[0] = 0x80;
    let bytes = build_st(&[
        ("blk.weight", "I8", vec![256, 128], w),
        (
            "blk.weight_scale",
            "F32",
            vec![2, 1],
            f32_scale_bytes(&[0.5, 0.5]),
        ),
        ("blk.input_scale", "F32", vec![], f32_scale_bytes(&[1.0])),
        (
            "blk.comfy_quant",
            "U8",
            vec![INT8_BLOB.len() as u64],
            marker_bytes(INT8_BLOB),
        ),
    ]);
    let p = write_tmp(tmp.path(), "overflow.safetensors", &bytes);
    let numeric = validate_comfy_quant(&p, true);
    assert_has_error(&numeric, "outside the valid range");
}

#[test]
fn valid_file_reports_pass_issues() {
    let tmp = tempfile::tempdir().unwrap();
    let p = write_tmp(tmp.path(), "valid.safetensors", &valid_int8_file());
    let report = validate_comfy_quant(&p, true);
    assert!(report.ok, "valid file must pass: {:?}", report.errors);
    let issues = quant_core::validator::report_to_issues(&report);
    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].level, IssueLevel::Info);
}

#[test]
fn format_report_renders_pass_and_fail() {
    let tmp = tempfile::tempdir().unwrap();
    let good = write_tmp(tmp.path(), "valid.safetensors", &valid_int8_file());
    let report = validate_comfy_quant(&good, false);
    let text = quant_core::validator::format_report(&report);
    assert!(text.contains("PASS"), "got:\n{text}");
    assert!(text.contains("formats found"), "got:\n{text}");

    let bad = write_tmp(
        tmp.path(),
        "noweight.safetensors",
        &build_st(&[
            (
                "blk.weight_scale",
                "F32",
                vec![2, 1],
                f32_scale_bytes(&[0.5, 0.5]),
            ),
            (
                "blk.comfy_quant",
                "U8",
                vec![INT8_BLOB.len() as u64],
                marker_bytes(INT8_BLOB),
            ),
        ]),
    );
    let report = validate_comfy_quant(&bad, false);
    let text = quant_core::validator::format_report(&report);
    assert!(text.contains("FAIL"), "got:\n{text}");
}

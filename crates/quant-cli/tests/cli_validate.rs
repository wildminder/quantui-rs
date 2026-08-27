//! Phase 8.3 CLI integration tests for `quantui-rs validate`.
//!
//! Exit-code contract: 0 = all files pass, 1 = at least one failure,
//! 2 = usage error (missing/bad path, no safetensors in directory).

use std::path::PathBuf;
use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_quantui-rs"))
}

fn golden(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.join("tests/golden").join(name)
}

// --------------------------------------------------------------------------- //
// Minimal safetensors builder (mirrors the one in phase8_validator.rs).
// --------------------------------------------------------------------------- //

fn align_header_to_8(header: &[u8]) -> Vec<u8> {
    let rem = header.len() % 8;
    if rem == 0 {
        return header.to_vec();
    }
    let mut out = header.to_vec();
    out.extend(std::iter::repeat_n(b' ', 8 - rem));
    out
}

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

fn tmp_dir() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

// --------------------------------------------------------------------------- //
// Tests
// --------------------------------------------------------------------------- //

#[test]
fn validate_golden_file_exits_zero_and_prints_pass() {
    let path = golden("linear_basic_bf16/output.safetensors");
    let out = bin()
        .args(["validate", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "expected exit 0, got {:?}\nstdout:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("PASS"), "stdout:\n{stdout}");
    assert!(stdout.contains("int8_blockwise"), "stdout:\n{stdout}");
}

#[test]
fn validate_golden_file_numeric_exits_zero() {
    let path = golden("linear_basic_bf16/output.safetensors");
    let out = bin()
        .args(["validate", "--numeric", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "expected exit 0, got {:?}\nstdout:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn validate_directory_expands_shards_and_passes() {
    // The sharded output directory contains several .safetensors shards.
    let dir = golden("sharded_model/output_sharded");
    let out = bin()
        .args(["validate", dir.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "expected exit 0, got {:?}\nstdout:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // At least the three quantized shards report PASS.
    assert!(
        stdout.matches("PASS").count() >= 3,
        "expected >= 3 PASS reports, stdout:\n{stdout}"
    );
}

#[test]
fn validate_corrupted_file_exits_one_and_prints_fail() {
    // Marker present but weight tensor missing -> structural failure.
    let blob = r#"{"format": "int8_blockwise", "orig_dtype": "torch.bfloat16", "group_size": 128}"#;
    let bytes = build_st(&[(
        "blk.comfy_quant",
        "U8",
        vec![blob.len() as u64],
        blob.as_bytes().to_vec(),
    )]);
    let dir = tmp_dir();
    let path = dir.path().join("corrupt.safetensors");
    std::fs::write(&path, &bytes).unwrap();

    let out = bin()
        .args(["validate", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(1),
        "expected exit 1\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("FAIL"), "stdout:\n{stdout}");
}

#[test]
fn validate_mixed_files_exits_one_if_any_fails() {
    let good = golden("linear_basic_bf16/output.safetensors");
    let blob = r#"{"format": "int8_blockwise", "orig_dtype": "torch.bfloat16", "group_size": 128}"#;
    let bytes = build_st(&[(
        "blk.comfy_quant",
        "U8",
        vec![blob.len() as u64],
        blob.as_bytes().to_vec(),
    )]);
    let dir = tmp_dir();
    let bad = dir.path().join("corrupt.safetensors");
    std::fs::write(&bad, &bytes).unwrap();

    let out = bin()
        .args(["validate", good.to_str().unwrap(), bad.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("PASS"), "good file should PASS:\n{stdout}");
    assert!(stdout.contains("FAIL"), "bad file should FAIL:\n{stdout}");
}

#[test]
fn validate_missing_path_exits_two() {
    let out = bin()
        .args(["validate", "C:/_does_not_exist_/nope.safetensors"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("does not exist"), "stderr:\n{stderr}");
}

#[test]
fn validate_empty_directory_exits_two() {
    let dir = tmp_dir();
    let out = bin()
        .args(["validate", dir.path().to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no .safetensors files"),
        "stderr:\n{stderr}"
    );
}

#[test]
fn validate_no_args_is_usage_error() {
    let out = bin().arg("validate").output().unwrap();
    // clap exits 2 on missing required arguments.
    assert_eq!(out.status.code(), Some(2));
}

//! Phase 9 CLI integration tests: `quantize` + `info` + `--help` surface.
//!
//! Exit-code contract: 0 ok, 1 runtime failure, 2 usage error.
//! Byte-parity: CLI quantize output must equal the golden fixtures exactly.

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

fn files_equal(a: &std::path::Path, b: &std::path::Path) -> bool {
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(ma), Ok(mb)) if ma.len() == mb.len() => {}
        _ => return false,
    }
    std::fs::read(a).unwrap() == std::fs::read(b).unwrap()
}

// --------------------------------------------------------------------------- //
// quantize — single file
// --------------------------------------------------------------------------- //

#[test]
fn quantize_single_file_byte_exact() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");
    let status = bin()
        .args([
            "quantize",
            golden("linear_basic_bf16/input.safetensors")
                .to_str()
                .unwrap(),
            out.to_str().unwrap(),
            "--no-progress",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(files_equal(
        &out,
        &golden("linear_basic_bf16/output.safetensors")
    ));
}

#[test]
fn quantize_single_file_resume_is_idempotent() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");
    let input = golden("linear_basic_bf16/input.safetensors");
    for _ in 0..2 {
        let status = bin()
            .args([
                "quantize",
                input.to_str().unwrap(),
                out.to_str().unwrap(),
                "--no-progress",
            ])
            .status()
            .unwrap();
        assert!(status.success());
    }
    assert!(files_equal(
        &out,
        &golden("linear_basic_bf16/output.safetensors")
    ));
}

#[test]
fn quantize_auto_naming_uses_ctq_tags() {
    // No OUTPUT arg → auto-name `<base>-int8_block-simple-heur.safetensors`
    // next to the input (effective config: simple on, heur on by default).
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("mymodel.safetensors");
    std::fs::copy(golden("linear_basic_bf16/input.safetensors"), &input).unwrap();

    let status = bin()
        .args(["quantize", input.to_str().unwrap(), "--no-progress"])
        .status()
        .unwrap();
    assert!(status.success());

    let expected = tmp
        .path()
        .join("mymodel-int8_block-simple-heur.safetensors");
    assert!(
        expected.is_file(),
        "auto-named output must exist at {}",
        expected.display()
    );
    assert!(files_equal(
        &expected,
        &golden("linear_basic_bf16/output.safetensors")
    ));
}

// --------------------------------------------------------------------------- //
// quantize — sharded folder
// --------------------------------------------------------------------------- //

#[test]
fn quantize_sharded_mode_byte_exact() {
    let tmp = tempfile::tempdir().unwrap();
    let out_dir = tmp.path().join("out");
    let status = bin()
        .args([
            "quantize",
            golden("sharded_model/input").to_str().unwrap(),
            out_dir.to_str().unwrap(),
            "--no-progress",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let golden_dir = golden("sharded_model/output_sharded");
    for entry in std::fs::read_dir(&golden_dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.ends_with(".safetensors") {
            continue;
        }
        assert!(
            files_equal(&out_dir.join(&*name), &entry.path()),
            "shard {name} must be byte-exact"
        );
    }
    // Index + global manifest copied/written.
    assert!(out_dir.join("model.safetensors.index.json").is_file());
    assert!(out_dir.join(".quant-manifest.json").is_file());
}

#[test]
fn quantize_sharded_single_mode_byte_exact() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("merged.safetensors");
    let status = bin()
        .args([
            "quantize",
            golden("sharded_model/input").to_str().unwrap(),
            out.to_str().unwrap(),
            "--output-mode",
            "single",
            "--no-progress",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(files_equal(
        &out,
        &golden("sharded_model/output_single.safetensors")
    ));
}

// --------------------------------------------------------------------------- //
// quantize — usage errors
// --------------------------------------------------------------------------- //

#[test]
fn quantize_missing_input_exits_two() {
    let out = bin()
        .args(["quantize", "C:/_does_not_exist_/nope.safetensors"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn quantize_invalid_scaling_mode_exits_two() {
    let out = bin()
        .args([
            "quantize",
            golden("linear_basic_bf16/input.safetensors")
                .to_str()
                .unwrap(),
            "--scaling-mode",
            "bogus",
        ])
        .output()
        .unwrap();
    // clap rejects invalid enum values with exit code 2.
    assert_eq!(out.status.code(), Some(2));
}

// --------------------------------------------------------------------------- //
// info
// --------------------------------------------------------------------------- //

#[test]
fn info_golden_output_lists_tensors_and_format() {
    let out = bin()
        .args([
            "info",
            golden("linear_basic_bf16/output.safetensors")
                .to_str()
                .unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("blocks.0.weight"), "stdout:\n{stdout}");
    assert!(stdout.contains("int8_blockwise"), "stdout:\n{stdout}");
    assert!(stdout.contains("quantized layers (1)"), "stdout:\n{stdout}");
}

#[test]
fn info_raw_prints_valid_json_header() {
    let out = bin()
        .args([
            "info",
            golden("linear_basic_bf16/output.safetensors")
                .to_str()
                .unwrap(),
            "--raw",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("raw header must be JSON");
    assert!(v.get("blocks.0.weight").is_some());
}

#[test]
fn info_missing_file_exits_one() {
    let out = bin()
        .args(["info", "C:/_does_not_exist_/nope.safetensors"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
}

// --------------------------------------------------------------------------- //
// --help surface (Phase 9.1 snapshot-style assertions)
// --------------------------------------------------------------------------- //

#[test]
fn help_lists_all_subcommands() {
    let out = bin().arg("--help").output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    for cmd in ["quantize", "validate", "info"] {
        assert!(stdout.contains(cmd), "help must list {cmd}:\n{stdout}");
    }
}

#[test]
fn quantize_help_documents_p0_options() {
    let out = bin().args(["quantize", "--help"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    for opt in [
        "--scaling-mode",
        "--block-size",
        "--heur",
        "--no-heur",
        "--exclude-layers",
        "--output-mode",
        "--orig-dtype",
        "--calib-seed",
    ] {
        assert!(stdout.contains(opt), "help must document {opt}:\n{stdout}");
    }
}

// --------------------------------------------------------------------------- //
// Phase A.4: format values + fixed-scaling validation
// --------------------------------------------------------------------------- //

#[test]
fn quantize_help_lists_all_format_values() {
    let out = bin().args(["quantize", "--help"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    for fmt in ["int8", "fp8_e4m3", "mxfp8", "nvfp4"] {
        assert!(
            stdout.contains(fmt),
            "help must list format {fmt}:\n{stdout}"
        );
    }
}

#[test]
fn mxfp8_block_size_override_is_usage_error() {
    let tmp = tempfile::tempdir().unwrap();
    let out = bin()
        .args([
            "quantize",
            golden("linear_basic_bf16/input.safetensors")
                .to_str()
                .unwrap(),
            tmp.path().join("o.safetensors").to_str().unwrap(),
            "--format",
            "mxfp8",
            "--block-size",
            "64",
        ])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(2),
        "explicit --block-size with mxfp8 must be exit 2"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--block-size"),
        "stderr must name the offending flag:\n{stderr}"
    );
}

#[test]
fn nvfp4_scaling_mode_override_is_usage_error() {
    let tmp = tempfile::tempdir().unwrap();
    let out = bin()
        .args([
            "quantize",
            golden("linear_basic_bf16/input.safetensors")
                .to_str()
                .unwrap(),
            tmp.path().join("o.safetensors").to_str().unwrap(),
            "--format",
            "nvfp4",
            "--scaling-mode",
            "row",
        ])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(2),
        "explicit --scaling-mode with nvfp4 must be exit 2"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--scaling-mode"),
        "stderr must name the offending flag:\n{stderr}"
    );
}

/// TEMPORARY (plan Phase A.4 → replaced by parity tests in Phase C/E): the
/// orchestrator is INT8-only so far, so a non-INT8 run must fail with a
/// clean runtime error (exit 1) — never silently emit INT8 output.
#[test]
fn unwired_format_fails_cleanly() {
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("o.safetensors");
    let out = bin()
        .args([
            "quantize",
            golden("linear_basic_bf16/input.safetensors")
                .to_str()
                .unwrap(),
            out_path.to_str().unwrap(),
            "--format",
            "mxfp8",
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(1),
        "unwired format must be a runtime failure"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not yet implemented"),
        "stderr must explain the format is not wired yet:\n{stderr}"
    );
    assert!(
        !out_path.exists(),
        "no output file may be created for an unwired format"
    );
}

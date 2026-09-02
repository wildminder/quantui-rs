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

/// Phase C.2 removed the "only int8" guard: non-INT8 formats are now wired,
/// so a `--format mxfp8` run must SUCCEED (exit 0) and produce an output file.
/// (Full per-format CLI payload parity lives in Phase E `cli_all_formats.rs`.)
#[test]
fn wired_format_runs_via_cli() {
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
        Some(0),
        "mxfp8 is wired and must succeed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out_path.exists(),
        "a wired format must produce an output file"
    );
}

// --------------------------------------------------------------------------- //
// Phase 7.0: int8_convrot honesty guard (plan decision Q3)
// --------------------------------------------------------------------------- //

/// Selecting int8_convrot exits 2 with the specific not-implemented
/// message, BEFORE any output file is created — never silently emits
/// plain INT8 under a ConvRot name.
#[test]
fn int8_convrot_is_rejected_with_exit_2() {
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
            "int8_convrot",
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(2),
        "int8_convrot must be a usage error until the kernel lands"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("int8_convrot"),
        "stderr must name the format:/n{stderr}"
    );
    assert!(
        stderr.contains("rotation kernel is not implemented"),
        "stderr must state the specific cause:/n{stderr}"
    );
    assert!(
        stderr.contains("Phase 7.1"),
        "stderr must name the planned phase:/n{stderr}"
    );
    // Honesty guard: NOTHING may be emitted under the ConvRot name.
    assert!(
        !out_path.exists(),
        "no output file may be created for a rejected format"
    );
}

/// The guard fires before input classification: even a nonexistent input
/// path with --format int8_convrot reports the ConvRot rejection, not an
/// input error (the format check is the more specific diagnosis).
#[test]
fn int8_convrot_guard_precedes_input_classification() {
    let out = bin()
        .args([
            "quantize",
            "does_not_exist.safetensors",
            "o.safetensors",
            "--format",
            "int8_convrot",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("rotation kernel is not implemented"),
        "format rejection must precede input classification:/n{stderr}"
    );
    assert!(
        !stderr.contains("unusable input"),
        "input error must not mask the format rejection:/n{stderr}"
    );
}

/// Counterpart: plain --format int8 is completely unaffected — full
/// byte-parity with the golden output still holds after the guard lands.
#[test]
fn plain_int8_unaffected_by_convrot_guard() {
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("o.safetensors");
    let out = bin()
        .args([
            "quantize",
            golden("linear_basic_bf16/input.safetensors")
                .to_str()
                .unwrap(),
            out_path.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "plain int8 must keep working: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(files_equal(
        &out_path,
        &golden("linear_basic_bf16/output.safetensors")
    ));
}

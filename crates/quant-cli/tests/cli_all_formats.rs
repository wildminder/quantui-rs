//! Phase D (plan D.3/D.4) CLI integration tests for the new formats.
//!
//! D.3 — auto-naming: `--format fp8_e4m3|mxfp8|nvfp4` with no OUTPUT arg must
//! suggest `<base>-<fmt>-simple-heur.safetensors` (registry id + effective
//! config tags: simple on by default, heur on by default).
//!
//! D.4 — `validate` (incl. `--numeric`) and `info` must pass on our streamed
//! new-format outputs (the validator already supports all 7 ctq formats).

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

/// Copy the golden input into a temp dir under a known base name and return
/// its path (so auto-naming derives from `mymodel`).
fn staged_input(tmp: &std::path::Path) -> PathBuf {
    let input = tmp.join("mymodel.safetensors");
    std::fs::copy(golden("linear_basic_bf16/input.safetensors"), &input).unwrap();
    input
}

// --------------------------------------------------------------------------- //
// D.3 — auto-naming
// --------------------------------------------------------------------------- //

#[test]
fn auto_name_fp8_e4m3_simple_heur() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let status = bin()
        .args([
            "quantize",
            input.to_str().unwrap(),
            "--format",
            "fp8_e4m3",
            "--no-progress",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let expected = tmp.path().join("mymodel-fp8_e4m3-simple-heur.safetensors");
    assert!(
        expected.is_file(),
        "auto-named fp8_e4m3 output must exist at {}",
        expected.display()
    );
}

#[test]
fn auto_name_mxfp8_simple_heur() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let status = bin()
        .args([
            "quantize",
            input.to_str().unwrap(),
            "--format",
            "mxfp8",
            "--no-progress",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let expected = tmp.path().join("mymodel-mxfp8-simple-heur.safetensors");
    assert!(
        expected.is_file(),
        "auto-named mxfp8 output must exist at {}",
        expected.display()
    );
}

#[test]
fn auto_name_nvfp4_simple_heur() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let status = bin()
        .args([
            "quantize",
            input.to_str().unwrap(),
            "--format",
            "nvfp4",
            "--no-progress",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let expected = tmp.path().join("mymodel-nvfp4-simple-heur.safetensors");
    assert!(
        expected.is_file(),
        "auto-named nvfp4 output must exist at {}",
        expected.display()
    );
}

// --------------------------------------------------------------------------- //
// D.4 — validate + info on streamed new-format outputs
// --------------------------------------------------------------------------- //

/// Quantize the golden input to `out` with `--format <fmt>` and return `out`.
fn quantize_to(tmp: &std::path::Path, fmt: &str) -> PathBuf {
    let out = tmp.join(format!("out_{fmt}.safetensors"));
    let status = bin()
        .args([
            "quantize",
            golden("linear_basic_bf16/input.safetensors")
                .to_str()
                .unwrap(),
            out.to_str().unwrap(),
            "--format",
            fmt,
            "--no-progress",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "quantize --format {fmt} must succeed");
    out
}

#[test]
fn validate_passes_on_streamed_mxfp8() {
    let tmp = tempfile::tempdir().unwrap();
    let out = quantize_to(tmp.path(), "mxfp8");
    let res = bin()
        .args(["validate", out.to_str().unwrap(), "--numeric"])
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "validate --numeric on streamed mxfp8 must pass; stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&res.stdout),
        String::from_utf8_lossy(&res.stderr)
    );
}

#[test]
fn validate_passes_on_streamed_nvfp4() {
    let tmp = tempfile::tempdir().unwrap();
    let out = quantize_to(tmp.path(), "nvfp4");
    let res = bin()
        .args(["validate", out.to_str().unwrap(), "--numeric"])
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "validate --numeric on streamed nvfp4 must pass; stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&res.stdout),
        String::from_utf8_lossy(&res.stderr)
    );
}

#[test]
fn validate_passes_on_streamed_fp8() {
    let tmp = tempfile::tempdir().unwrap();
    let out = quantize_to(tmp.path(), "fp8_e4m3");
    let res = bin()
        .args(["validate", out.to_str().unwrap(), "--numeric"])
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "validate --numeric on streamed fp8_e4m3 must pass; stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&res.stdout),
        String::from_utf8_lossy(&res.stderr)
    );
}

#[test]
fn info_detects_format_mxfp8() {
    let tmp = tempfile::tempdir().unwrap();
    let out = quantize_to(tmp.path(), "mxfp8");
    let res = bin()
        .args(["info", out.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(res.status.success());
    let stdout = String::from_utf8_lossy(&res.stdout);
    assert!(
        stdout.contains("mxfp8"),
        "info must detect the mxfp8 format:\n{stdout}"
    );
    assert!(
        stdout.contains("quantized layers"),
        "info must list quantized layers:\n{stdout}"
    );
}

#[test]
fn info_detects_format_nvfp4() {
    let tmp = tempfile::tempdir().unwrap();
    let out = quantize_to(tmp.path(), "nvfp4");
    let res = bin()
        .args(["info", out.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(res.status.success());
    let stdout = String::from_utf8_lossy(&res.stdout);
    assert!(
        stdout.contains("nvfp4"),
        "info must detect the nvfp4 format:\n{stdout}"
    );
}

#[test]
fn info_detects_format_fp8() {
    let tmp = tempfile::tempdir().unwrap();
    let out = quantize_to(tmp.path(), "fp8_e4m3");
    let res = bin()
        .args(["info", out.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(res.status.success());
    let stdout = String::from_utf8_lossy(&res.stdout);
    assert!(
        stdout.contains("float8_e4m3fn"),
        "info must detect an fp8 (float8_e4m3fn*) format:\n{stdout}"
    );
}

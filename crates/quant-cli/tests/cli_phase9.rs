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
    for fmt in ["int8", "fp8_e4m3", "mxfp8", "nvfp4", "int8_convrot"] {
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
// Phase 7.1: int8_convrot — INT8 row-wise + group-wise Hadamard rotation
// --------------------------------------------------------------------------- //

/// Read one tensor's payload out of a `.safetensors` file.
fn read_tensor(path: &std::path::Path, name: &str) -> Vec<u8> {
    let reader = quant_core::st_io::reader::SafetensorsReader::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    reader
        .tensor_bytes(name)
        .unwrap_or_else(|_| panic!("{name} missing from {}", path.display()))
        .to_vec()
}

/// Write a minimal `.safetensors` file (u64 header length + JSON header +
/// payload) from `(name, dtype, shape, bytes)` tuples.
fn write_safetensors(path: &std::path::Path, tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len() as u64;
        data.extend_from_slice(bytes);
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, data.len() as u64],
            }),
        );
    }
    let mut header_json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    // safetensors pads the header to a multiple of 8 with spaces.
    while header_json.len() % 8 != 0 {
        header_json.push(b' ');
    }
    let mut out = (header_json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header_json);
    out.extend_from_slice(&data);
    std::fs::write(path, out).unwrap();
}

/// Deterministic bf16 payload for an `[m, n]` weight.
fn bf16_weight(m: usize, n: usize) -> Vec<u8> {
    (0..m * n)
        .flat_map(|i| {
            let v = ((i as f32) * 0.017).sin() + ((i % 7) as f32) * 0.003;
            half::bf16::from_f32(v).to_le_bytes()
        })
        .collect()
}

/// `int8_convrot` is a working format now (exit 0, output written). On the
/// shared golden fixture neither `blocks.0.weight [256,128]` nor
/// `blocks.1.weight [128,64]` has in_features divisible by the group size
/// (256), so NO layer rotates: the run warns per skipped tensor and falls
/// back to plain row-wise INT8 (no `convrot` key in the blob).
#[test]
fn int8_convrot_runs_and_warns_on_undivisible_layers() {
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
        Some(0),
        "int8_convrot must succeed now that the kernel landed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_path.exists(), "a successful run must write an output");

    // One warning per skipped tensor, naming the offending in_features.
    // `blocks.1.weight [128,64]` is dropped by the skip-inefficient
    // heuristic (64 < block size 128) before ConvRot ever sees it, so only
    // blocks.0 warns — the warning is per QUANTIZED tensor.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("skipping ConvRot for blocks.0.weight: in_features 128 not divisible"),
        "stderr must warn about the skipped layer:\n{stderr}"
    );
    assert!(
        !stderr.contains("in_features 64 not divisible"),
        "a heur-skipped layer never reaches the ConvRot decision:\n{stderr}"
    );

    // Row-wise INT8 emission: [m,1] weight_scale, no input_scale, and a
    // family-A blob with NO convrot key (the layer was not rotated).
    let blob = String::from_utf8(read_tensor(&out_path, "blocks.0.comfy_quant")).unwrap();
    assert!(
        !blob.contains("convrot"),
        "an unrotated layer must not claim ConvRot:\n{blob}"
    );
    assert!(
        blob.starts_with(r#"{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16""#),
        "unrotated row-wise INT8 keeps the plain tensorwise blob:\n{blob}"
    );
}

/// A layer whose in_features IS divisible by the group size gets the ConvRot
/// blob (`convrot`, `convrot_groupsize`, `per_row`) — the rotation is baked
/// into the payload. A sibling layer it does not divide stays plain.
#[test]
fn int8_convrot_rotates_divisible_layers() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("convrot_in.safetensors");
    write_safetensors(
        &input,
        &[
            // 256 % 256 == 0 → rotated.  (128, 256 both divisible by the
            // skip-heuristic's block size 128 → layer is quantized.)
            ("rot.weight", "BF16", vec![128, 256], bf16_weight(128, 256)),
            // 384 % 256 != 0 → plain.    (128, 384 → quantized as well.)
            (
                "plain.weight",
                "BF16",
                vec![128, 384],
                bf16_weight(128, 384),
            ),
        ],
    );
    let out_path = tmp.path().join("o.safetensors");
    let out = bin()
        .args([
            "quantize",
            input.to_str().unwrap(),
            out_path.to_str().unwrap(),
            "--format",
            "int8_convrot",
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "int8_convrot must succeed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // ---- the rotated layer ---- //
    let rot = String::from_utf8(read_tensor(&out_path, "rot.comfy_quant")).unwrap();
    assert_eq!(
        rot,
        r#"{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "convrot": true, "convrot_groupsize": 256, "per_row": true}"#,
        "the rotated layer's blob must carry the ConvRot keys in reference order"
    );
    for key in [
        r#""convrot": true"#,
        r#""convrot_groupsize": 256"#,
        r#""per_row": true"#,
    ] {
        assert!(rot.contains(key), "rot blob must contain {key}:\n{rot}");
    }
    // Row-wise INT8 emission for the rotated layer: [m,1] scale, no input_scale.
    let hdr = quant_core::st_io::reader::SafetensorsReader::open(&out_path).unwrap();
    assert_eq!(
        hdr.header().get("rot.weight").unwrap().shape,
        vec![128, 256]
    );
    assert_eq!(
        hdr.header().get("rot.weight_scale").unwrap().shape,
        vec![128, 1]
    );
    assert!(hdr.header().get("rot.input_scale").is_none());

    // ---- the non-divisible layer stays plain ---- //
    let plain = String::from_utf8(read_tensor(&out_path, "plain.comfy_quant")).unwrap();
    assert!(
        !plain.contains("convrot"),
        "a non-divisible layer must not claim ConvRot:\n{plain}"
    );
    assert_eq!(
        plain, r#"{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16"}"#,
        "a non-divisible layer stays plain row-wise INT8"
    );
}

/// Determinism: two `int8_convrot` runs with the pinned default seed produce
/// byte-identical output (the rotation is deterministic — no RNG involved).
#[test]
fn int8_convrot_is_deterministic() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in.safetensors");
    write_safetensors(
        &input,
        &[("rot.weight", "BF16", vec![128, 256], bf16_weight(128, 256))],
    );
    let a = tmp.path().join("a.safetensors");
    let b = tmp.path().join("b.safetensors");
    for out_path in [&a, &b] {
        let out = bin()
            .args([
                "quantize",
                input.to_str().unwrap(),
                out_path.to_str().unwrap(),
                "--format",
                "int8_convrot",
                "--no-progress",
            ])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(0));
    }
    assert_eq!(
        std::fs::read(&a).unwrap(),
        std::fs::read(&b).unwrap(),
        "two int8_convrot runs must be byte-identical"
    );
}

/// Counterpart: plain --format int8 is completely unaffected — full
/// byte-parity with the golden output still holds after ConvRot lands.
#[test]
fn plain_int8_unaffected_by_convrot_path() {
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

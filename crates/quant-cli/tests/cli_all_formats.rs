//! Phase D (plan D.3/D.4) + Phase E (plan E.1/E.3/E.4) CLI integration tests
//! for the new formats.
//!
//! D.3 — auto-naming: `--format fp8_e4m3|mxfp8|nvfp4` with no OUTPUT arg must
//! suggest `<base>-<fmt>-simple-heur.safetensors` (registry id + effective
//! config tags: simple on by default, heur on by default).
//!
//! D.4 — `validate` (incl. `--numeric`) and `info` must pass on our streamed
//! new-format outputs (the validator already supports all 7 ctq formats).
//!
//! E.1 — CLI E2E payload parity: `quantize --format <fmt>` over each golden
//! input must produce per-tensor payload + dtype/shape parity vs the ctq
//! goldens (whole-file compare is impossible: ctq writes in its own order).
//!
//! E.3 — resume through the CLI (idempotent re-run; cancel-then-resume).
//! E.4 — `--output-mode single` for sharded input + new format.

use std::path::{Path, PathBuf};
use std::process::Command;

use quant_core::st_io::reader::SafetensorsReader;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_quantui-rs"))
}

fn golden(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.join("tests/golden").join(name)
}

fn load(path: &Path) -> SafetensorsReader {
    SafetensorsReader::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
}

/// Per-tensor payload + dtype/shape parity vs a ctq golden (plan §4.3).
/// ctq writes tensors in its own processing order and `save_file` sorts the
/// header keys, so whole-file compare is structurally impossible for the new
/// formats — compare every tensor's payload bytes and `(dtype, shape)`, and
/// require identical name sets. `__metadata__` is compared separately (D.2).
fn assert_cli_payload_parity(ours: &Path, golden_path: &Path, label: &str) {
    let g = load(golden_path);
    let o = load(ours);

    let mut golden_names: Vec<&String> = Vec::new();
    for (name, info) in g.header().iter() {
        if name == "__metadata__" {
            continue;
        }
        golden_names.push(name);

        let o_info = o
            .header()
            .get(name)
            .unwrap_or_else(|| panic!("{label}: missing tensor {name} in CLI output"));
        assert_eq!(
            o_info.dtype_raw, info.dtype_raw,
            "{label}/{name}: dtype mismatch (ours {} vs golden {})",
            o_info.dtype_raw, info.dtype_raw
        );
        assert_eq!(
            o_info.shape, info.shape,
            "{label}/{name}: shape mismatch (ours {:?} vs golden {:?})",
            o_info.shape, info.shape
        );

        let g_bytes = g.tensor_bytes(name).unwrap();
        let o_bytes = o
            .tensor_bytes(name)
            .unwrap_or_else(|_| panic!("{label}: cannot read {name} from CLI output"));
        assert_eq!(
            o_bytes.len(),
            g_bytes.len(),
            "{label}/{name}: payload length mismatch (ours {} vs golden {})",
            o_bytes.len(),
            g_bytes.len()
        );
        assert!(
            o_bytes == g_bytes,
            "{label}/{name}: payload bytes differ (first diff at index {:?})",
            o_bytes.iter().zip(g_bytes.iter()).position(|(a, b)| a != b)
        );
    }

    let mut our_names: Vec<&String> = o
        .header()
        .iter()
        .filter(|(n, _)| n.as_str() != "__metadata__")
        .map(|(n, _)| n)
        .collect();
    let mut golden_sorted = golden_names.clone();
    golden_sorted.sort();
    our_names.sort();
    assert_eq!(
        our_names, golden_sorted,
        "{label}: tensor name sets differ (ours vs golden)"
    );
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

// --------------------------------------------------------------------------- //
// E.1 — CLI E2E payload parity (all 4 cases × 5 format variants)
// --------------------------------------------------------------------------- //

/// Run `quantize --format <fmt> [--scaling-mode <mode>]` over one golden case
/// and assert per-tensor payload parity vs the ctq golden.
fn cli_payload_parity(case: &str, fmt: &str, mode: Option<&str>, golden_suffix: &str) {
    let dir = golden(case);
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    let mut args = vec![
        "quantize".to_string(),
        dir.join("input.safetensors").to_str().unwrap().to_string(),
        out.to_str().unwrap().to_string(),
        "--format".to_string(),
        fmt.to_string(),
        "--no-progress".to_string(),
    ];
    if let Some(m) = mode {
        args.push("--scaling-mode".to_string());
        args.push(m.to_string());
    }
    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let res = bin().args(&arg_refs).output().unwrap();
    assert_eq!(
        res.status.code(),
        Some(0),
        "quantize --format {fmt} on {case} must exit 0; stderr:\n{}",
        String::from_utf8_lossy(&res.stderr)
    );

    assert_cli_payload_parity(
        &out,
        &dir.join(format!("output_{golden_suffix}.safetensors")),
        &format!("{case}/{golden_suffix} (CLI)"),
    );
}

#[test]
fn cli_quantize_fp8_block_payload_parity() {
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"] {
        cli_payload_parity(case, "fp8_e4m3", None, "fp8");
    }
}

#[test]
fn cli_quantize_fp8_tensor_payload_parity() {
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"] {
        cli_payload_parity(case, "fp8_e4m3", Some("tensor"), "fp8_tensor");
    }
}

#[test]
fn cli_quantize_fp8_row_payload_parity() {
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"] {
        cli_payload_parity(case, "fp8_e4m3", Some("row"), "fp8_row");
    }
}

#[test]
fn cli_quantize_mxfp8_payload_parity() {
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"] {
        cli_payload_parity(case, "mxfp8", None, "mxfp8");
    }
}

#[test]
fn cli_quantize_nvfp4_payload_parity() {
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"] {
        cli_payload_parity(case, "nvfp4", None, "nvfp4");
    }
}

// --------------------------------------------------------------------------- //
// E.3 — resume through the CLI
// --------------------------------------------------------------------------- //

/// Re-running a completed CLI quantize must be idempotent: the output stays
/// payload-identical to the golden (resume path, manifest hit).
#[test]
fn cli_nvfp4_resume_idempotent() {
    let dir = golden("linear_basic_bf16");
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    for _ in 0..2 {
        let status = bin()
            .args([
                "quantize",
                dir.join("input.safetensors").to_str().unwrap(),
                out.to_str().unwrap(),
                "--format",
                "nvfp4",
                "--no-progress",
            ])
            .status()
            .unwrap();
        assert!(status.success());
    }

    assert_cli_payload_parity(
        &out,
        &dir.join("output_nvfp4.safetensors"),
        "linear_basic_bf16/nvfp4 (CLI resumed)",
    );
}

// --------------------------------------------------------------------------- //
// E.4 — sharded input, single output mode
// --------------------------------------------------------------------------- //

/// `--output-mode single` over the sharded fixture with a new format must
/// produce a single output file whose tensors match the core library's union
/// streaming run over the same shards. (Per-format sharded ctq goldens are
/// generated in E.5; until then the gate is CLI-vs-union payload parity.)
#[test]
fn cli_mxfp8_sharded_input_single_output_payload_parity() {
    let input_dir = golden("sharded_model/input");
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("merged.safetensors");

    let status = bin()
        .args([
            "quantize",
            input_dir.to_str().unwrap(),
            out.to_str().unwrap(),
            "--format",
            "mxfp8",
            "--output-mode",
            "single",
            "--no-progress",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "sharded input + single mode must succeed");
    assert!(out.is_file(), "single mode must produce one output file");

    // Reference: the core library's union streaming over the same shards.
    let ref_out = tmp.path().join("ref.safetensors");
    let shards: Vec<PathBuf> = [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    .iter()
    .map(|s| input_dir.join(s))
    .collect();
    let config = quant_core::manifest::QuantConfig {
        format: quant_core::manifest::Format::Mxfp8,
        target_format: "mxfp8".into(),
        int8: false,
        scaling_mode: quant_core::manifest::ScalingMode::Block,
        block_size: 32,
        ..quant_core::manifest::QuantConfig::default()
    };
    quant_core::stream::stream_quantize_shards(&shards, &ref_out, &config).unwrap();

    // CLI single-mode output must be payload-identical to the union run.
    assert_cli_payload_parity(&out, &ref_out, "sharded_model/mxfp8 single (CLI vs union)");
}

// --------------------------------------------------------------------------- //
// E.3 — cancel-then-resume through the CLI via a REAL Ctrl-C signal (Windows).
// --------------------------------------------------------------------------- //
//
// The core library's cancel path is covered in-process (phase12 mxfp8 resume
// test). Here we exercise the full CLI: spawn the binary in its own process
// group, wait until the per-tensor manifest shows >= 2 tensors done, deliver
// CTRL_BREAK_EVENT, assert exit 130 + a loadable partial output, then re-run
// to resume and require payload parity against an uninterrupted reference run.
//
// Windows recipe (Git Bash `kill -INT` never reaches SetConsoleCtrlHandler):
// spawn with CREATE_NEW_PROCESS_GROUP, then GenerateConsoleCtrlEvent(
// CTRL_BREAK_EVENT, child_pid). Targeting the child's own process-group id
// keeps the break event away from the test harness.

#[cfg(windows)]
mod cli_cancel_resume {
    use super::{assert_cli_payload_parity, bin, load};
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const CTRL_BREAK_EVENT: u32 = 1;
    const N_TENSORS: usize = 32;

    extern "system" {
        #[link_name = "GenerateConsoleCtrlEvent"]
        fn generate_console_ctrl_event(ctrl_event: u32, process_group_id: u32) -> i32;
    }

    fn synth_f32(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Minimal safetensors serializer: (name, shape, payload) triples, BF16.
    fn build_safetensors_bytes(tensors: &[(String, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
        let mut data = Vec::new();
        let mut map = serde_json::Map::new();
        for (name, shape, bytes) in tensors {
            let start = data.len() as u64;
            data.extend_from_slice(bytes);
            let end = data.len() as u64;
            let mut info = serde_json::Map::new();
            info.insert("dtype".into(), serde_json::json!("BF16"));
            info.insert("shape".into(), serde_json::json!(shape));
            info.insert("data_offsets".into(), serde_json::json!([start, end]));
            map.insert(name.clone(), serde_json::Value::Object(info));
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(map)).unwrap();
        let pad = (8 - header.len() % 8) % 8;
        let mut padded = header;
        padded.extend(std::iter::repeat_n(b' ', pad));
        let mut out = Vec::new();
        out.extend_from_slice(&(padded.len() as u64).to_le_bytes());
        out.extend_from_slice(&padded);
        out.extend_from_slice(&data);
        out
    }

    /// A synthetic model with many quantizable 2D weights so an interrupt has a
    /// reliable mid-run window. No biases → no bias correction (keeps the run
    /// focused on the cancel/resume mechanism).
    fn write_big_model(path: &Path) {
        let mut tensors = Vec::with_capacity(N_TENSORS);
        for i in 0..N_TENSORS {
            let name = format!("model.layers.{i}.mlp.gate_proj.weight");
            let vals = synth_f32(1024 * 1024, 1000 + i as u64);
            let bytes: Vec<u8> = vals
                .iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
                .collect();
            tensors.push((name, vec![1024, 1024], bytes));
        }
        std::fs::write(path, build_safetensors_bytes(&tensors)).unwrap();
    }

    /// Number of tensors recorded done in `<out>.quant-manifest.json`, if the
    /// manifest exists and parses. The manifest is flushed atomically after
    /// every tensor, so this is a safe, deterministic progress probe.
    fn manifest_done_count(out: &Path) -> Option<usize> {
        let mut s = out.to_path_buf().into_os_string();
        s.push(".quant-manifest.json");
        let text = std::fs::read_to_string(PathBuf::from(s)).ok()?;
        let v: serde_json::Value = serde_json::from_str(&text).ok()?;
        Some(v["done"].as_array()?.len())
    }

    #[test]
    fn cli_mxfp8_cancel_then_resume_to_parity() {
        use std::os::windows::process::CommandExt;

        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("big.safetensors");
        write_big_model(&input);
        let ref_out = tmp.path().join("ref.safetensors");
        let out = tmp.path().join("out.safetensors");

        // 1) Uninterrupted reference run (the "golden" for this synthetic input).
        let status = bin()
            .args([
                "quantize",
                input.to_str().unwrap(),
                ref_out.to_str().unwrap(),
                "--format",
                "mxfp8",
                "--no-progress",
            ])
            .status()
            .unwrap();
        assert!(status.success(), "uninterrupted reference run must exit 0");

        // 2) Spawn the run to cancel in its own process group.
        let mut child = bin()
            .args([
                "quantize",
                input.to_str().unwrap(),
                out.to_str().unwrap(),
                "--format",
                "mxfp8",
                "--no-progress",
            ])
            .creation_flags(CREATE_NEW_PROCESS_GROUP)
            .spawn()
            .unwrap();
        let pid = child.id();

        // Wait for a deterministic mid-run window: >= 2 tensors done.
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if manifest_done_count(&out).is_some_and(|n| n >= 2) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "run never reached 2 done tensors within 60s"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        // Deliver a real CTRL_BREAK to the child's process group.
        let sent = unsafe { generate_console_ctrl_event(CTRL_BREAK_EVENT, pid) != 0 };
        assert!(sent, "GenerateConsoleCtrlEvent failed (no shared console?)");

        let status = child.wait().unwrap();
        assert_eq!(
            status.code(),
            Some(130),
            "cancelled run must exit 130 (SIGINT convention)"
        );

        // Partial output must be loadable and hold a strict subset of tensors.
        let partial = load(&out);
        let n_partial = partial
            .header()
            .iter()
            .filter(|(n, _)| n.as_str() != "__metadata__")
            .count();
        assert!(
            (2..N_TENSORS).contains(&n_partial),
            "partial output holds {n_partial} of {N_TENSORS} tensors"
        );

        // 3) Re-run the identical command → resume to completion.
        let status = bin()
            .args([
                "quantize",
                input.to_str().unwrap(),
                out.to_str().unwrap(),
                "--format",
                "mxfp8",
                "--no-progress",
            ])
            .status()
            .unwrap();
        assert!(status.success(), "resumed run must exit 0");

        // 4) Resumed output must be payload-identical to the uninterrupted run.
        assert_cli_payload_parity(&out, &ref_out, "big/mxfp8 (cancel+resume vs reference)");
    }
}

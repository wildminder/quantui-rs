//! Phase 5 E2E tests: streaming orchestrator byte-parity + resume semantics.
//!
//! For each golden fixture: run stream_quantize on input.safetensors and
//! byte-compare the complete output against output.safetensors. Then verify
//! manifest JSON is semantically equal, and exercise kill-simulation resume.

use std::path::PathBuf;

use quant_core::manifest::QuantConfig;
use quant_core::stream::stream_quantize;

fn golden(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.join("tests/golden").join(name)
}

fn ref_config(exclude: Option<&str>) -> QuantConfig {
    QuantConfig {
        exclude_layers: exclude.map(|s| s.to_string()),
        ..QuantConfig::default()
    }
}

fn files_equal(a: &std::path::Path, b: &std::path::Path) -> bool {
    if std::fs::metadata(a).unwrap().len() != std::fs::metadata(b).unwrap().len() {
        return false;
    }
    std::fs::read(a).unwrap() == std::fs::read(b).unwrap()
}

#[test]
fn e2e_parity_linear_basic_bf16() {
    let dir = golden("linear_basic_bf16");
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    let result = stream_quantize(dir.join("input.safetensors"), &out, &ref_config(None)).unwrap();

    assert!(
        files_equal(&out, &dir.join("output.safetensors")),
        "output bytes must equal golden"
    );
    // Manifest semantic equality.
    let ours: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.with_extension("safetensors.quant-manifest.json")).unwrap())
            .unwrap();
    let theirs: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("output.safetensors.quant-manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(ours["config_hash"], theirs["config_hash"]);
    assert_eq!(ours["order"], theirs["order"]);
    assert_eq!(ours["done"], theirs["done"]);
    assert_eq!(result.config_hash, "56920c6553cfa241");
}

#[test]
fn e2e_parity_odd_shapes_with_exclusion() {
    let dir = golden("odd_shapes");
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    stream_quantize(
        dir.join("input.safetensors"),
        &out,
        &ref_config(Some("attn_norm")),
    )
    .unwrap();

    assert!(
        files_equal(&out, &dir.join("output.safetensors")),
        "output bytes must equal golden (excluded layer keeps original dtype)"
    );
}

#[test]
fn e2e_parity_conv_net_f16() {
    let dir = golden("conv_net");
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    stream_quantize(dir.join("input.safetensors"), &out, &ref_config(None)).unwrap();

    assert!(
        files_equal(&out, &dir.join("output.safetensors")),
        "f16 weight quantization + f32 4D passthrough must match golden"
    );
}

#[test]
fn kill_simulation_resume_mid_run_equals_golden() {
    // Simulate a crash after N tensors: run once, capture manifest+file state,
    // then delete the last few tensors from done by truncating our own progress:
    // simplest faithful simulation — fresh run in a temp dir where we manually
    // stop early by running the orchestrator, then removing trailing entries
    // from done/order and rewinding the file via IncrementalWriter resume rules.
    //
    // Simpler robust approach mirroring Phase 1 kill-sims: take the GOLDEN
    // OUTPUT, truncate it mid-data-section exactly like a Python-crashed writer,
    // craft a matching partial manifest, then resume from it.
    let dir = golden("linear_basic_bf16");
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    let golden_bytes = std::fs::read(dir.join("output.safetensors")).unwrap();
    // Truncate inside blocks.1.weight's data region (after header slot).
    // Header slot for this file is 65536 → data starts at 65544.
    let data_start = 8 + u64::from_le_bytes(golden_bytes[0..8].try_into().unwrap()) as usize;
    let truncated_len = data_start + (golden_bytes.len() - data_start) / 2;
    std::fs::write(&out, &golden_bytes[..truncated_len]).unwrap();

    // Partial manifest claiming only blocks.0.* tensors are done.
    let manifest = r#"{"version":1,"config_hash":"56920c6553cfa241","order":["blocks.0.bias","blocks.0.weight"],"done":["blocks.0.bias"]}"#;
    std::fs::write(
        tmp.path().join("out.safetensors.quant-manifest.json"),
        manifest,
    )
    .unwrap();

    let result = stream_quantize(dir.join("input.safetensors"), &out, &ref_config(None)).unwrap();

    assert!(
        files_equal(&out, &dir.join("output.safetensors")),
        "resumed run must complete to byte-identical golden"
    );
    assert_eq!(result.done.len(), 5);
    assert_eq!(result.order.len(), 5);
}

#[test]
fn config_hash_mismatch_restarts_clean() {
    // A partial output+manifest from a DIFFERENT config must be discarded:
    // fresh run overwrites from zero and still matches the golden.
    let dir = golden("conv_net");
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    // Plant garbage "partial" output + mismatched-hash manifest.
    std::fs::write(&out, b"garbage-from-another-run-not-even-safetensors").unwrap();
    std::fs::write(
        tmp.path().join("out.safetensors.quant-manifest.json"),
        r#"{"version":1,"config_hash":"0000000000000000","order":["x"],"done":["x"]}"#,
    )
    .unwrap();

    stream_quantize(dir.join("input.safetensors"), &out, &ref_config(None)).unwrap();
    assert!(
        files_equal(&out, &dir.join("output.safetensors")),
        "hash mismatch must trigger clean restart producing exact golden"
    );
}

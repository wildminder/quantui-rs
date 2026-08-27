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
    let ours: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.with_extension("safetensors.quant-manifest.json")).unwrap(),
    )
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

// --------------------------------------------------------------------------- //
// Progress callback semantics (Phase 9.4 — mirrors reference on_progress).
// --------------------------------------------------------------------------- //

#[test]
fn progress_callback_single_file_counts_tensors() {
    use quant_core::stream::stream_quantize_with_progress;

    let dir = golden("linear_basic_bf16");
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    let mut events: Vec<(usize, usize)> = Vec::new();
    {
        let mut cb = |cur: usize, total: usize| events.push((cur, total));
        stream_quantize_with_progress(
            dir.join("input.safetensors"),
            &out,
            &ref_config(None),
            Some(&mut cb),
        )
        .unwrap();
    }

    // One event per tensor; total is constant = number of input tensors.
    assert!(!events.is_empty());
    let total = events[0].1;
    assert!(events.iter().all(|&(_, t)| t == total));
    // cur strictly increases by 1 and ends at total.
    for (i, &(cur, _)) in events.iter().enumerate() {
        assert_eq!(cur, i + 1, "cur must count up 1..=total");
    }
    assert_eq!(events.last().unwrap().0, total);
    // Output still byte-exact with the callback attached.
    assert!(files_equal(&out, &dir.join("output.safetensors")));
}

#[test]
fn progress_callback_sharded_counts_shards() {
    use quant_core::discover::discover_shards;
    use quant_core::stream::stream_quantize_sharded_with_progress;

    let input = golden("sharded_model/input");
    let model = discover_shards(&input).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let out_dir = tmp.path().join("out");

    let mut events: Vec<(usize, usize)> = Vec::new();
    {
        let mut cb = |cur: usize, total: usize| events.push((cur, total));
        stream_quantize_sharded_with_progress(&model, &out_dir, &ref_config(None), Some(&mut cb))
            .unwrap();
    }

    // Shard-level progress only: one event per shard, 1-based, total = #shards.
    let n = model.shard_files.len();
    assert_eq!(events.len(), n);
    for (i, &(cur, total)) in events.iter().enumerate() {
        assert_eq!(cur, i + 1);
        assert_eq!(total, n);
    }
}

#[test]
fn progress_callback_resume_starts_above_zero() {
    use quant_core::stream::stream_quantize_with_progress;

    // Run once fully, then re-run (resume): all tensors already done, so the
    // loop body never executes and NO progress events fire on the second run —
    // matching the reference (callback only fires inside the remaining loop).
    let dir = golden("linear_basic_bf16");
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    stream_quantize(dir.join("input.safetensors"), &out, &ref_config(None)).unwrap();

    let mut events: Vec<(usize, usize)> = Vec::new();
    {
        let mut cb = |cur: usize, total: usize| events.push((cur, total));
        stream_quantize_with_progress(
            dir.join("input.safetensors"),
            &out,
            &ref_config(None),
            Some(&mut cb),
        )
        .unwrap();
    }
    assert!(
        events.is_empty(),
        "fully-resumed run processes no tensors, so no progress events"
    );
}

// --------------------------------------------------------------------------- //
// Ctrl-C graceful stop (Phase 9.6 / plan 8.5): cancel mid-run → valid partial
// output + manifest, then resume to byte-exact golden.
// --------------------------------------------------------------------------- //

#[test]
fn cancel_mid_run_leaves_resumable_partial_output() {
    use quant_core::st_io::reader::SafetensorsReader;
    use quant_core::stream::{stream_quantize_cancellable, StreamError};
    use std::sync::atomic::{AtomicBool, Ordering};

    let dir = golden("linear_basic_bf16");
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    // Cancel after the 2nd tensor completes.
    let cancel = AtomicBool::new(false);
    let mut events = 0usize;
    let result = {
        let mut cb = |_cur: usize, _total: usize| {
            events += 1;
            if events >= 2 {
                cancel.store(true, Ordering::Relaxed);
            }
        };
        stream_quantize_cancellable(
            dir.join("input.safetensors"),
            &out,
            &ref_config(None),
            Some(&mut cb),
            &cancel,
        )
    };

    // Must report Cancelled with the right counts.
    match result {
        Err(StreamError::Cancelled { done, total }) => {
            assert_eq!(done, 2, "cancelled after exactly 2 tensors");
            assert_eq!(total, 5);
        }
        other => panic!("expected Cancelled, got {other:?}"),
    }

    // Partial output must be loadable by the safetensors reader.
    let reader = SafetensorsReader::open(&out).expect("partial output must be loadable");
    assert_eq!(reader.header().len(), 2, "partial file holds 2 tensors");

    // Manifest must exist and list exactly the 2 done tensors.
    let manifest_path = out.with_extension("safetensors.quant-manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["done"].as_array().unwrap().len(), 2);

    // Resume: re-run without cancellation → completes to byte-exact golden.
    let resumed = stream_quantize(dir.join("input.safetensors"), &out, &ref_config(None)).unwrap();
    assert_eq!(resumed.done.len(), 5);
    assert!(
        files_equal(&out, &dir.join("output.safetensors")),
        "resumed run must complete to byte-identical golden"
    );
}

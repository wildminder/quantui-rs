//! Phase 12 (plan Phase C.2/C.4) streaming E2E payload parity — MXFP8.
//!
//! For each golden case, run the streaming orchestrator (MXFP8 config) over
//! `input.safetensors` and byte-compare EVERY emitted tensor (weight F8_E4M3
//! at padded shape, weight_scale U8 tiled, comfy_quant family-B blob,
//! corrected biases, and all copied/skipped tensors) against
//! `output_mxfp8.safetensors`.
//!
//! Parity is per-tensor payload + `(dtype, shape)` (plan §3.3). Phase D adds
//! the file-level `__metadata__._quantization_metadata` byte-parity check.

mod common;

use common::{assert_metadata_parity, assert_payload_parity, golden};
use quant_core::manifest::{Format, QuantConfig, ScalingMode};
use quant_core::stream::stream_quantize;

fn mxfp8_config() -> QuantConfig {
    QuantConfig {
        format: Format::Mxfp8,
        target_format: "mxfp8".into(),
        int8: false,
        scaling_mode: ScalingMode::Block,
        block_size: 32,
        ..QuantConfig::default()
    }
}

fn run_mxfp8(case: &str) {
    let dir = golden(case);
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    stream_quantize(dir.join("input.safetensors"), &out, &mxfp8_config()).unwrap();

    assert_payload_parity(
        &out,
        &dir.join("output_mxfp8.safetensors"),
        &format!("{case}/mxfp8"),
    );
}

#[test]
fn mxfp8_linear_basic_bf16() {
    run_mxfp8("linear_basic_bf16");
}

#[test]
fn mxfp8_odd_shapes() {
    run_mxfp8("odd_shapes");
}

#[test]
fn mxfp8_conv_net() {
    run_mxfp8("conv_net");
}

#[test]
fn mxfp8_zero_blocks() {
    run_mxfp8("zero_blocks");
}

// ---- Phase D.2: file-level __metadata__ parity ---------------------------- //

/// D.2 gate: the streamed MXFP8 output's `__metadata__` must equal the
/// golden's byte-for-byte (the `_quantization_metadata` JSON string), for
/// every golden case.
#[test]
fn mxfp8_file_metadata_matches_golden() {
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"] {
        let dir = golden(case);
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out.safetensors");

        stream_quantize(dir.join("input.safetensors"), &out, &mxfp8_config()).unwrap();

        assert_metadata_parity(
            &out,
            &dir.join("output_mxfp8.safetensors"),
            &format!("{case}/mxfp8 metadata"),
        );
    }
}

// ---- determinism + resume (plan C.5) -------------------------------------- //

/// Double-run determinism: two fresh runs over the same input must produce
/// byte-identical output files (pinned seed, no GPU, deterministic writer).
#[test]
fn mxfp8_double_run_identical() {
    let dir = golden("linear_basic_bf16");
    let tmp = tempfile::tempdir().unwrap();
    let out1 = tmp.path().join("run1.safetensors");
    let out2 = tmp.path().join("run2.safetensors");

    stream_quantize(dir.join("input.safetensors"), &out1, &mxfp8_config()).unwrap();
    stream_quantize(dir.join("input.safetensors"), &out2, &mxfp8_config()).unwrap();

    assert_eq!(
        std::fs::read(&out1).unwrap(),
        std::fs::read(&out2).unwrap(),
        "two fresh MXFP8 runs must be byte-identical"
    );
}

/// Resume smoke: cancel mid-run (after 2 tensors), then re-run to resume; the
/// final output must be payload-identical to an uninterrupted full run.
#[test]
fn mxfp8_resume_to_parity() {
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
            &mxfp8_config(),
            Some(&mut cb),
            &cancel,
        )
    };
    assert!(
        matches!(result, Err(StreamError::Cancelled { .. })),
        "expected Cancelled, got {result:?}"
    );

    // Resume to completion.
    stream_quantize(dir.join("input.safetensors"), &out, &mxfp8_config()).unwrap();

    // Payload-identical to the golden (which an uninterrupted run matches).
    assert_payload_parity(
        &out,
        &dir.join("output_mxfp8.safetensors"),
        "linear_basic_bf16/mxfp8 (resumed)",
    );
}

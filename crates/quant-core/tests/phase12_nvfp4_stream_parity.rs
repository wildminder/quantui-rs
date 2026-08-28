//! Phase 12 (plan Phase C.2/C.4) streaming E2E payload parity — NVFP4.
//!
//! For each golden case, run the streaming orchestrator (NVFP4 config) over
//! `input.safetensors` and byte-compare EVERY emitted tensor (weight U8 packed
//! at `(m_pad, n_pad/2)`, weight_scale F8_E4M3 tiled, weight_scale_2 F32 `[]`,
//! comfy_quant family-B blob, corrected biases, and all copied/skipped tensors)
//! against `output_nvfp4.safetensors`.
//!
//! Parity is per-tensor payload + `(dtype, shape)` (plan §3.3). Phase D adds
//! the file-level `__metadata__._quantization_metadata` byte-parity check.

mod common;

use common::{assert_metadata_parity, assert_payload_parity, golden};
use quant_core::manifest::{Format, QuantConfig, ScalingMode};
use quant_core::stream::stream_quantize;

fn nvfp4_config() -> QuantConfig {
    QuantConfig {
        format: Format::Nvfp4,
        target_format: "nvfp4".into(),
        int8: false,
        scaling_mode: ScalingMode::Block,
        block_size: 16,
        ..QuantConfig::default()
    }
}

fn run_nvfp4(case: &str) {
    let dir = golden(case);
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    stream_quantize(dir.join("input.safetensors"), &out, &nvfp4_config()).unwrap();

    assert_payload_parity(
        &out,
        &dir.join("output_nvfp4.safetensors"),
        &format!("{case}/nvfp4"),
    );
}

#[test]
fn nvfp4_linear_basic_bf16() {
    run_nvfp4("linear_basic_bf16");
}

#[test]
fn nvfp4_odd_shapes() {
    run_nvfp4("odd_shapes");
}

#[test]
fn nvfp4_conv_net() {
    run_nvfp4("conv_net");
}

#[test]
fn nvfp4_zero_blocks() {
    run_nvfp4("zero_blocks");
}

// ---- Phase D.2: file-level __metadata__ parity ---------------------------- //

/// D.2 gate: the streamed NVFP4 output's `__metadata__` must equal the
/// golden's byte-for-byte (the `_quantization_metadata` JSON string), for
/// every golden case.
#[test]
fn nvfp4_file_metadata_matches_golden() {
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"] {
        let dir = golden(case);
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out.safetensors");

        stream_quantize(dir.join("input.safetensors"), &out, &nvfp4_config()).unwrap();

        assert_metadata_parity(
            &out,
            &dir.join("output_nvfp4.safetensors"),
            &format!("{case}/nvfp4 metadata"),
        );
    }
}

// ---- determinism + resume (plan C.5) -------------------------------------- //

/// Double-run determinism: two fresh runs over the same input must produce
/// byte-identical output files (pinned seed, no GPU, deterministic writer).
#[test]
fn nvfp4_double_run_identical() {
    let dir = golden("linear_basic_bf16");
    let tmp = tempfile::tempdir().unwrap();
    let out1 = tmp.path().join("run1.safetensors");
    let out2 = tmp.path().join("run2.safetensors");

    stream_quantize(dir.join("input.safetensors"), &out1, &nvfp4_config()).unwrap();
    stream_quantize(dir.join("input.safetensors"), &out2, &nvfp4_config()).unwrap();

    assert_eq!(
        std::fs::read(&out1).unwrap(),
        std::fs::read(&out2).unwrap(),
        "two fresh NVFP4 runs must be byte-identical"
    );
}

/// Resume smoke: cancel mid-run (after 2 tensors), then re-run to resume; the
/// final output must be payload-identical to an uninterrupted full run.
#[test]
fn nvfp4_resume_to_parity() {
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
            &nvfp4_config(),
            Some(&mut cb),
            &cancel,
        )
    };
    assert!(
        matches!(result, Err(StreamError::Cancelled { .. })),
        "expected Cancelled, got {result:?}"
    );

    // Resume to completion.
    stream_quantize(dir.join("input.safetensors"), &out, &nvfp4_config()).unwrap();

    // Payload-identical to the golden (which an uninterrupted run matches).
    assert_payload_parity(
        &out,
        &dir.join("output_nvfp4.safetensors"),
        "linear_basic_bf16/nvfp4 (resumed)",
    );
}

// ---- skip-heuristic per-format block size (plan C.3) ---------------------- //

/// The skip-inefficient heuristic uses the FORMAT's block size (plan §3.6):
/// NVFP4=16, MXFP8=32, FP8=64 (tensor/row) / config (block). A `[48,48]`
/// weight is divisible by 16 (NVFP4 quantizes) but NOT by 32 (MXFP8 skips)
/// and is below 64/128 (FP8 skips). This pins the per-format predicate.
#[test]
fn skip_heur_per_format_block_size() {
    use common::{bf16_weight_bytes, load, write_input};
    use quant_core::manifest::{Format, QuantConfig, ScalingMode};
    use quant_core::stream::stream_quantize;

    let tmp = tempfile::tempdir().unwrap();
    let inp = tmp.path().join("in.safetensors");
    write_input(
        &inp,
        &[(
            "w.weight",
            quant_core::dtype::DType::Bf16,
            vec![48, 48],
            bf16_weight_bytes(48, 48),
        )],
    );

    // NVFP4 (heur bs=16): 48 % 16 == 0 → quantized (U8 packed).
    let out = tmp.path().join("nvfp4.safetensors");
    stream_quantize(
        &inp,
        &out,
        &QuantConfig {
            format: Format::Nvfp4,
            target_format: "nvfp4".into(),
            int8: false,
            scaling_mode: ScalingMode::Block,
            block_size: 16,
            ..QuantConfig::default()
        },
    )
    .unwrap();
    assert_eq!(
        load(&out).header().get("w.weight").unwrap().dtype_raw,
        "U8",
        "nvfp4 (heur bs=16) must quantize [48,48]"
    );

    // MXFP8 (heur bs=32): 48 % 32 != 0 → skipped (stays BF16).
    let out = tmp.path().join("mxfp8.safetensors");
    stream_quantize(
        &inp,
        &out,
        &QuantConfig {
            format: Format::Mxfp8,
            target_format: "mxfp8".into(),
            int8: false,
            scaling_mode: ScalingMode::Block,
            block_size: 32,
            ..QuantConfig::default()
        },
    )
    .unwrap();
    assert_eq!(
        load(&out).header().get("w.weight").unwrap().dtype_raw,
        "BF16",
        "mxfp8 (heur bs=32) must skip [48,48]"
    );

    // FP8 tensor (heur bs=64): 48 < 64 → skipped (stays BF16).
    let out = tmp.path().join("fp8.safetensors");
    stream_quantize(
        &inp,
        &out,
        &QuantConfig {
            format: Format::Fp8E4m3,
            target_format: "fp8".into(),
            int8: false,
            scaling_mode: ScalingMode::Tensor,
            block_size: 128,
            ..QuantConfig::default()
        },
    )
    .unwrap();
    assert_eq!(
        load(&out).header().get("w.weight").unwrap().dtype_raw,
        "BF16",
        "fp8 tensor (heur bs=64) must skip [48,48]"
    );
}

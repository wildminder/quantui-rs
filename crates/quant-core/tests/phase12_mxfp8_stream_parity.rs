//! Phase 12 (plan Phase C.2/C.4) streaming E2E payload parity — MXFP8.
//!
//! For each golden case, run the streaming orchestrator (MXFP8 config) over
//! `input.safetensors` and byte-compare EVERY emitted tensor (weight F8_E4M3
//! at padded shape, weight_scale U8 tiled, comfy_quant family-B blob,
//! corrected biases, and all copied/skipped tensors) against
//! `output_mxfp8.safetensors`.
//!
//! Parity is per-tensor payload + `(dtype, shape)` (plan §3.3). `__metadata__`
//! is carried by the MXFP8 goldens but emitted in Phase D, so the helper
//! ignores it here.

mod common;

use common::{assert_payload_parity, golden};
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

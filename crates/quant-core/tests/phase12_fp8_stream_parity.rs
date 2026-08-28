//! Phase 12 (plan Phase C.2/C.4) streaming E2E payload parity — FP8 E4M3.
//!
//! For each golden case and each FP8 scaling mode (block/tensor/row), run the
//! streaming orchestrator over `input.safetensors` and byte-compare EVERY
//! emitted tensor (weight, weight_scale, comfy_quant, corrected biases, and
//! all copied/skipped tensors) against `output_fp8*.safetensors`.
//!
//! Parity is per-tensor payload + `(dtype, shape)` (plan §3.3): ctq writes in
//! its own processing order, so whole-file compare is not possible for the new
//! formats. `__metadata__` is absent for FP8 (plan §3.3) and is ignored by the
//! helper regardless (Phase D concern).

mod common;

use common::{assert_payload_parity, golden, load};
use quant_core::manifest::{Format, QuantConfig, ScalingMode};
use quant_core::stream::stream_quantize;

/// FP8 config for a given scaling mode. Block mode uses the golden's
/// `block_size=128`; tensor/row ignore block_size for quantization and use the
/// mode-aware heuristic block size of 64 internally (see `heur_block_size`).
fn fp8_config(mode: ScalingMode) -> QuantConfig {
    QuantConfig {
        format: Format::Fp8E4m3,
        target_format: "fp8".into(),
        int8: false,
        scaling_mode: mode,
        block_size: 128,
        ..QuantConfig::default()
    }
}

fn run_fp8(case: &str, golden_suffix: &str, mode: ScalingMode) {
    let dir = golden(case);
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    stream_quantize(dir.join("input.safetensors"), &out, &fp8_config(mode)).unwrap();

    assert_payload_parity(
        &out,
        &dir.join(format!("output_{golden_suffix}.safetensors")),
        &format!("{case}/{golden_suffix}"),
    );
}

// ---- block scaling (golden `fp8`, block_size=128) ------------------------- //

#[test]
fn fp8_block_linear_basic_bf16() {
    run_fp8("linear_basic_bf16", "fp8", ScalingMode::Block);
}

#[test]
fn fp8_block_odd_shapes() {
    run_fp8("odd_shapes", "fp8", ScalingMode::Block);
}

#[test]
fn fp8_block_conv_net() {
    run_fp8("conv_net", "fp8", ScalingMode::Block);
}

#[test]
fn fp8_block_zero_blocks() {
    run_fp8("zero_blocks", "fp8", ScalingMode::Block);
}

// ---- tensor scaling (golden `fp8_tensor`) --------------------------------- //

#[test]
fn fp8_tensor_linear_basic_bf16() {
    run_fp8("linear_basic_bf16", "fp8_tensor", ScalingMode::Tensor);
}

#[test]
fn fp8_tensor_odd_shapes() {
    run_fp8("odd_shapes", "fp8_tensor", ScalingMode::Tensor);
}

#[test]
fn fp8_tensor_conv_net() {
    run_fp8("conv_net", "fp8_tensor", ScalingMode::Tensor);
}

#[test]
fn fp8_tensor_zero_blocks() {
    run_fp8("zero_blocks", "fp8_tensor", ScalingMode::Tensor);
}

// ---- row scaling (golden `fp8_row`) --------------------------------------- //

#[test]
fn fp8_row_linear_basic_bf16() {
    run_fp8("linear_basic_bf16", "fp8_row", ScalingMode::Row);
}

#[test]
fn fp8_row_odd_shapes() {
    run_fp8("odd_shapes", "fp8_row", ScalingMode::Row);
}

#[test]
fn fp8_row_conv_net() {
    run_fp8("conv_net", "fp8_row", ScalingMode::Row);
}

#[test]
fn fp8_row_zero_blocks() {
    run_fp8("zero_blocks", "fp8_row", ScalingMode::Row);
}

// ---- heuristic block-size discrimination (plan C.3 finding) --------------- //

/// `linear_basic_bf16` `blocks.1.weight [128,64]` must be BF16-skipped in
/// block mode (heur bs=128) but F8_E4M3-quantized in tensor/row mode (heur
/// bs=64). This pins the mode-aware heuristic against regression.
#[test]
fn fp8_heur_block_size_is_mode_aware() {
    let dir = golden("linear_basic_bf16");

    // Block mode: [128,64] skipped → stays BF16.
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("block.safetensors");
    stream_quantize(
        dir.join("input.safetensors"),
        &out,
        &fp8_config(ScalingMode::Block),
    )
    .unwrap();
    let r = load(&out);
    assert_eq!(
        r.header().get("blocks.1.weight").unwrap().dtype_raw,
        "BF16",
        "block mode (heur bs=128) must skip [128,64]"
    );

    // Tensor mode: [128,64] quantized → F8_E4M3.
    let out2 = tmp.path().join("tensor.safetensors");
    stream_quantize(
        dir.join("input.safetensors"),
        &out2,
        &fp8_config(ScalingMode::Tensor),
    )
    .unwrap();
    let r2 = load(&out2);
    assert_eq!(
        r2.header().get("blocks.1.weight").unwrap().dtype_raw,
        "F8_E4M3",
        "tensor mode (heur bs=64) must quantize [128,64]"
    );
}

// ---- divisibility policy (plan §3.6 / C.3) -------------------------------- //

/// FP8 block mode with the skip heuristic OFF must NOT error on indivisible
/// dims — the kernel falls back to row-wise scaling (reference behavior). The
/// output is F8_E4M3 with a row-wise `[m]` scale.
#[test]
fn fp8_block_indivisible_falls_back_to_row_no_heur() {
    use common::{bf16_weight_bytes, write_input};

    let tmp = tempfile::tempdir().unwrap();
    let inp = tmp.path().join("in.safetensors");
    // [130,130] is not divisible by 128 → would be NotDivisible for INT8.
    write_input(
        &inp,
        &[(
            "w.weight",
            quant_core::dtype::DType::Bf16,
            vec![130, 130],
            bf16_weight_bytes(130, 130),
        )],
    );

    let out = tmp.path().join("out.safetensors");
    let mut cfg = fp8_config(ScalingMode::Block);
    cfg.skip_inefficient = false; // heur OFF → quantize anyway
    stream_quantize(&inp, &out, &cfg).unwrap();

    let r = load(&out);
    assert_eq!(
        r.header().get("w.weight").unwrap().dtype_raw,
        "F8_E4M3",
        "fp8 block (heur off) must quantize indivisible dims via row fallback"
    );
    // Row-wise fallback → scale shape [m] = [130].
    assert_eq!(
        r.header().get("w.weight_scale").unwrap().shape,
        vec![130],
        "row fallback must emit a [m] scale"
    );
}

/// The `odd_shapes` `odd.weight [130,130]` is heur-skipped in ALL four
/// formats (130 is not divisible by 16/32/64/128) and stays BF16. The payload
/// parity tests already assert this against every golden; this test pins it
/// explicitly for the FP8 modes.
#[test]
fn odd_shapes_odd_weight_skipped_all_fp8_modes() {
    let dir = golden("odd_shapes");
    for (suffix, mode) in [
        ("fp8", ScalingMode::Block),
        ("fp8_tensor", ScalingMode::Tensor),
        ("fp8_row", ScalingMode::Row),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out.safetensors");
        stream_quantize(dir.join("input.safetensors"), &out, &fp8_config(mode)).unwrap();
        let r = load(&out);
        assert_eq!(
            r.header().get("odd.weight").unwrap().dtype_raw,
            "BF16",
            "{suffix}: odd.weight [130,130] must stay BF16 (heur-skipped)"
        );
    }
}

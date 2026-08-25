//! Phase 2 parity tests: Rust INT8 kernels vs Python-produced golden outputs.
//!
//! For each golden fixture, loads the input safetensors, quantizes each 2D
//! `.weight` tensor with the reference config (block mode, bs=128, heur on,
//! bf16 output dtype), and byte-compares against the corresponding tensors in
//! the Python-produced output.safetensors.

use std::path::PathBuf;

use quant_core::dtype::{bf16_bits_to_f32, f32_to_bf16_bits};
use quant_core::quant::{quantize_int8_weight, should_skip_shape, ScalingMode};
use quant_core::st_io::reader::SafetensorsReader;

fn golden(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.join("tests/golden").join(name)
}

fn load(path: &std::path::Path) -> SafetensorsReader {
    SafetensorsReader::open(path).expect("open safetensors")
}

/// Reference QuantConfig defaults used by gen_golden.py (block/128/heur/bf16).
const BLOCK_SIZE: usize = 128;

#[test]
fn int8_blockwise_matches_golden_linear_basic_bf16() {
    let dir = golden("linear_basic_bf16");
    let input = load(&dir.join("input.safetensors"));
    let output = load(&dir.join("output.safetensors"));

    // blocks.0.weight [256,128] bf16 → quantized.
    let w = input.tensor_bytes("blocks.0.weight").unwrap();
    let shape = input.header().get("blocks.0.weight").unwrap().shape.clone();
    assert_eq!(shape, vec![256u64, 128]);
    let w_f32: Vec<f32> = w
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    let r = quantize_int8_weight(&w_f32, 256, 128, ScalingMode::Block, BLOCK_SIZE);

    // qdata bytes must match exactly.
    let q_golden = output.tensor_bytes("blocks.0.weight").unwrap();
    let q_bytes: Vec<u8> = r.qdata.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(q_bytes.len(), q_golden.len());
    assert!(
        q_bytes == q_golden,
        "int8 payload mismatch for blocks.0.weight"
    );

    // Scale [2,1] F32 must match bit-for-bit.
    let s_golden = output.tensor_bytes("blocks.0.weight_scale").unwrap();
    let s_bytes: Vec<u8> = r.scale.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(s_bytes, s_golden, "scale bytes mismatch for blocks.0");

    // input_scale scalar 1.0 present in golden.
    let iscale = output.tensor_bytes("blocks.0.input_scale").unwrap();
    assert_eq!(iscale, 1.0f32.to_le_bytes());

    // comfy_quant blob: {"format": "int8_blockwise", "orig_dtype": "torch.bfloat16", "group_size": 128}
    let blob_golden = output.tensor_bytes("blocks.0.comfy_quant").unwrap();
    assert_eq!(
        String::from_utf8_lossy(blob_golden),
        r#"{"format": "int8_blockwise", "orig_dtype": "torch.bfloat16", "group_size": 128}"#
    );
}

#[test]
fn int8_blockwise_matches_golden_odd_shapes() {
    let dir = golden("odd_shapes");
    let input = load(&dir.join("input.safetensors"));
    let output = load(&dir.join("output.safetensors"));

    let name = "model.diffusion_model.double_block.0.weight";
    let w = input.tensor_bytes(name).unwrap(); // f32 [384,256]
    assert_eq!(w.len(), 384 * 256 * 4);
    let w_f32: Vec<f32> = w
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let r = quantize_int8_weight(&w_f32, 384, 256, ScalingMode::Block, BLOCK_SIZE);

    let q_golden = output.tensor_bytes(name).unwrap();
    let q_bytes: Vec<u8> = r.qdata.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(q_bytes, q_golden, "int8 payload mismatch for {name}");

    let s_golden = output
        .tensor_bytes("model.diffusion_model.double_block.0.weight_scale")
        .unwrap();
    let s_bytes: Vec<u8> = r.scale.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(s_bytes, s_golden, "scale mismatch for {name}");
}

#[test]
fn int8_blockwise_scalar_scale_matches_golden_conv_net() {
    let dir = golden("conv_net");
    let input = load(&dir.join("input.safetensors"));
    let output = load(&dir.join("output.safetensors"));

    // head.weight [128,128] f16 → quantized; scale is [1,1] → squeezed to [] scalar.
    let w = input.tensor_bytes("head.weight").unwrap();
    let w_f32: Vec<f32> = w
        .chunks_exact(2)
        .map(|c| quant_core::dtype::f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    let r = quantize_int8_weight(&w_f32, 128, 128, ScalingMode::Block, BLOCK_SIZE);

    let q_golden = output.tensor_bytes("head.weight").unwrap();
    let q_bytes: Vec<u8> = r.qdata.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(q_bytes, q_golden);

    let s_golden = output.tensor_bytes("head.weight_scale").unwrap();
    let s_bytes: Vec<u8> = r.scale.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(s_bytes, s_golden);
}

#[test]
fn skipped_weight_cast_to_bf16_matches_golden() {
    // blocks.1.weight [128,64] bf16: cols < block_size → skip-heur copy + cast to
    // output dtype bf16. Input already bf16 so bytes pass through unchanged.
    let dir = golden("linear_basic_bf16");
    let input = load(&dir.join("input.safetensors"));
    let output = load(&dir.join("output.safetensors"));

    assert!(should_skip_shape(&[128, 64], BLOCK_SIZE));
    let w_in = input.tensor_bytes("blocks.1.weight").unwrap();
    let w_out = output.tensor_bytes("blocks.1.weight").unwrap();
    assert_eq!(
        w_in, w_out,
        "skipped bf16 weight must pass through verbatim"
    );
    // Header records BF16 (cast target), not the raw U16 tolerance form.
    use quant_core::st_io::header::Header as _;
    let info = output.header().get("blocks.1.weight").unwrap();
    assert_eq!(info.dtype_raw, "BF16");
}

#[test]
fn odd_shape_skipped_weight_stays_bf16() {
    let dir = golden("odd_shapes");
    let input = load(&dir.join("input.safetensors"));
    let output = load(&dir.join("output.safetensors"));

    assert!(should_skip_shape(&[130, 130], BLOCK_SIZE));
    let w_in = input.tensor_bytes("odd.weight").unwrap();
    let w_out = output.tensor_bytes("odd.weight").unwrap();
    assert_eq!(w_in, w_out);
}

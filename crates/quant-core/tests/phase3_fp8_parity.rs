//! Phase 3.2 parity tests: Rust FP8 E4M3 kernels vs ctq golden outputs.
//!
//! For each format golden (`output_fp8.safetensors` = block, `output_fp8_tensor`
//! = tensor, `output_fp8_row` = row), loads the shared input, runs the matching
//! FP8 kernel over every quantized (F8_E4M3) weight tensor, and byte-compares
//! both the E4M3 payload and the dequant-scale payload against the golden.
//!
//! ctq writes tensors in processing order (not input order), so parity here is
//! per-tensor payload byte-compare, not whole-file compare (plan §3.2).

use std::path::PathBuf;

use quant_core::dtype::{bf16_bits_to_f32, f16_bits_to_f32};
use quant_core::quant_fp8::{quantize_fp8_weight, Fp8ScalingMode};
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

/// Convert raw input tensor bytes to f32 based on the header dtype string.
fn to_f32(reader: &SafetensorsReader, name: &str) -> (Vec<f32>, Vec<u64>) {
    let info = reader.header().get(name).unwrap_or_else(|| panic!("input tensor {name}"));
    let bytes = reader.tensor_bytes(name).unwrap();
    let shape = info.shape.clone();
    let vals = match info.dtype_raw.as_str() {
        "BF16" | "U16" => bytes
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "F16" => bytes
            .chunks_exact(2)
            .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "F32" => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("unsupported input dtype {other} for {name}"),
    };
    (vals, shape)
}

/// Run one parity check for a given case dir + format + scaling mode.
fn check_case(case: &str, fmt: &str, mode: Fp8ScalingMode, block_size: usize) {
    let dir = golden(case);
    let input = load(&dir.join("input.safetensors"));
    let output = load(&dir.join(format!("output_{fmt}.safetensors")));

    // Collect quantized weight tensor names (dtype F8_E4M3) from the golden.
    let quantized: Vec<String> = output
        .header()
        .iter()
        .filter(|(_, info)| info.dtype_raw == "F8_E4M3")
        .map(|(name, _)| name.clone())
        .collect();

    assert!(
        !quantized.is_empty(),
        "{case}/{fmt}: golden has no F8_E4M3 tensors"
    );

    for name in quantized {
        let prefix = name
            .strip_suffix(".weight")
            .unwrap_or_else(|| panic!("quantized tensor {name} not a .weight"));
        let scale_name = format!("{prefix}.weight_scale");

        // Input weight → f32.
        let (w_f32, shape) = to_f32(&input, &name);
        assert_eq!(shape.len(), 2, "{name}: expected 2D weight");
        let (m, n) = (shape[0] as usize, shape[1] as usize);
        assert_eq!(w_f32.len(), m * n);

        let r = quantize_fp8_weight(&w_f32, m, n, mode, block_size);

        // E4M3 payload must match byte-for-byte.
        let q_golden = output.tensor_bytes(&name).unwrap();
        assert_eq!(
            r.qdata.len(),
            q_golden.len(),
            "{case}/{fmt}/{name}: qdata length mismatch"
        );
        assert!(
            r.qdata == q_golden,
            "{case}/{fmt}/{name}: E4M3 payload mismatch (first diff at {:?})",
            r.qdata
                .iter()
                .zip(q_golden.iter())
                .position(|(a, b)| a != b)
        );

        // Dequant-scale payload must match bit-for-bit. (Shape normalization
        // [1,1]→[] at emit time doesn't change the bytes, so raw compare works.)
        let s_golden = output
            .tensor_bytes(&scale_name)
            .unwrap_or_else(|_| panic!("{case}/{fmt}: missing {scale_name}"));
        let s_bytes: Vec<u8> = r.scale.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            s_bytes, s_golden,
            "{case}/{fmt}/{name}: scale bytes mismatch"
        );
    }
}

#[test]
fn fp8_blockwise_matches_golden() {
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"] {
        check_case(case, "fp8", Fp8ScalingMode::Block, 128);
    }
}

#[test]
fn fp8_tensorwise_matches_golden() {
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"] {
        check_case(case, "fp8_tensor", Fp8ScalingMode::Tensor, 128);
    }
}

#[test]
fn fp8_rowwise_matches_golden() {
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"] {
        check_case(case, "fp8_row", Fp8ScalingMode::Row, 128);
    }
}

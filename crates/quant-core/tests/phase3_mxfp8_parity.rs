//! Phase 3.3 parity tests: Rust MXFP8 kernel vs ctq golden outputs.
//!
//! For each `output_mxfp8.safetensors` golden, loads the shared input, runs
//! the MXFP8 kernel over every quantized (F8_E4M3) weight tensor, and
//! byte-compares both the E4M3 payload and the E8M0 `weight_scale` payload
//! (cuBLAS tiled layout) against the golden.
//!
//! ctq writes tensors in processing order (not input order), so parity here is
//! per-tensor payload byte-compare, not whole-file compare (plan §3.3).

use std::path::PathBuf;

use quant_core::dtype::{bf16_bits_to_f32, f16_bits_to_f32};
use quant_core::quant_mxfp8::quantize_mxfp8_weight;
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

fn check_case(case: &str) {
    let dir = golden(case);
    let input = load(&dir.join("input.safetensors"));
    let output = load(&dir.join("output_mxfp8.safetensors"));

    // Quantized weight tensors (dtype F8_E4M3) from the golden.
    let quantized: Vec<String> = output
        .header()
        .iter()
        .filter(|(_, info)| info.dtype_raw == "F8_E4M3")
        .map(|(name, _)| name.clone())
        .collect();

    assert!(
        !quantized.is_empty(),
        "{case}/mxfp8: golden has no F8_E4M3 tensors"
    );

    for name in quantized {
        let prefix = name
            .strip_suffix(".weight")
            .unwrap_or_else(|| panic!("quantized tensor {name} not a .weight"));
        let scale_name = format!("{prefix}.weight_scale");

        let (w_f32, shape) = to_f32(&input, &name);
        assert_eq!(shape.len(), 2, "{name}: expected 2D weight");
        let (m, n) = (shape[0] as usize, shape[1] as usize);
        assert_eq!(w_f32.len(), m * n);

        let r = quantize_mxfp8_weight(&w_f32, m, n);

        // E4M3 payload must match byte-for-byte.
        let q_golden = output.tensor_bytes(&name).unwrap();
        assert_eq!(
            r.qdata.len(),
            q_golden.len(),
            "{case}/mxfp8/{name}: qdata length mismatch (rust {:?} vs golden {})",
            r.qdata_shape,
            q_golden.len()
        );
        assert!(
            r.qdata == q_golden,
            "{case}/mxfp8/{name}: E4M3 payload mismatch (first diff at {:?})",
            r.qdata
                .iter()
                .zip(q_golden.iter())
                .position(|(a, b)| a != b)
        );

        // E8M0 tiled-layout scale payload must match byte-for-byte.
        let s_golden = output
            .tensor_bytes(&scale_name)
            .unwrap_or_else(|_| panic!("{case}/mxfp8: missing {scale_name}"));
        assert_eq!(
            r.scale.len(),
            s_golden.len(),
            "{case}/mxfp8/{name}: scale length mismatch (rust {:?} vs golden {})",
            r.scale_shape,
            s_golden.len()
        );
        assert!(
            r.scale == s_golden,
            "{case}/mxfp8/{name}: E8M0 scale payload mismatch (first diff at {:?})",
            r.scale
                .iter()
                .zip(s_golden.iter())
                .position(|(a, b)| a != b)
        );
    }
}

#[test]
fn mxfp8_matches_golden() {
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"] {
        check_case(case);
    }
}

//! Phase 3.4 parity tests: Rust NVFP4 kernel vs ctq golden outputs.
//!
//! For each `output_nvfp4.safetensors` golden, loads the shared input, runs
//! the NVFP4 kernel over every quantized weight tensor (identified by a
//! matching `.weight_scale`), and byte-compares the packed U8 qdata, the
//! F8_E4M3 `weight_scale` payload (cuBLAS tiled layout), and the F32
//! `weight_scale_2` scalar against the golden.
//!
//! ctq writes tensors in processing order (not input order), so parity here is
//! per-tensor payload byte-compare, not whole-file compare (plan §3.4).
//!
//! NOTE: the goldens were generated with CUDA hidden (`CUDA_VISIBLE_DEVICES=-1`)
//! so the eager backend ran; see `quant_nvfp4.rs` module docs for the
//! provenance rationale.

use std::path::PathBuf;

use quant_core::dtype::{bf16_bits_to_f32, f16_bits_to_f32};
use quant_core::quant_nvfp4::quantize_nvfp4_weight;
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
    let info = reader
        .header()
        .get(name)
        .unwrap_or_else(|| panic!("input tensor {name}"));
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
    let output = load(&dir.join("output_nvfp4.safetensors"));

    // Quantized weight tensors: U8 `.weight` with a matching `.weight_scale`.
    let quantized: Vec<String> = output
        .header()
        .iter()
        .filter(|(name, info)| {
            info.dtype_raw == "U8"
                && name.ends_with(".weight")
                && output
                    .header()
                    .contains_key(&format!("{}.weight_scale", &name[..name.len() - 7]))
        })
        .map(|(name, _)| name.clone())
        .collect();

    assert!(
        !quantized.is_empty(),
        "{case}/nvfp4: golden has no quantized weight tensors"
    );

    for name in quantized {
        let prefix = name
            .strip_suffix(".weight")
            .unwrap_or_else(|| panic!("quantized tensor {name} not a .weight"));
        let scale_name = format!("{prefix}.weight_scale");
        let scale2_name = format!("{prefix}.weight_scale_2");

        let (w_f32, shape) = to_f32(&input, &name);
        assert_eq!(shape.len(), 2, "{name}: expected 2D weight");
        let (m, n) = (shape[0] as usize, shape[1] as usize);
        assert_eq!(w_f32.len(), m * n);

        let r = quantize_nvfp4_weight(&w_f32, m, n);

        // Packed U8 qdata must match byte-for-byte.
        let q_golden = output.tensor_bytes(&name).unwrap();
        assert_eq!(
            r.qdata.len(),
            q_golden.len(),
            "{case}/nvfp4/{name}: qdata length mismatch (rust {:?} vs golden {})",
            r.qdata_shape,
            q_golden.len()
        );
        assert!(
            r.qdata == q_golden,
            "{case}/nvfp4/{name}: packed qdata mismatch (first diff at {:?})",
            r.qdata
                .iter()
                .zip(q_golden.iter())
                .position(|(a, b)| a != b)
        );

        // F8_E4M3 tiled-layout block-scale payload must match byte-for-byte.
        let s_golden = output
            .tensor_bytes(&scale_name)
            .unwrap_or_else(|_| panic!("{case}/nvfp4: missing {scale_name}"));
        assert_eq!(
            r.scale.len(),
            s_golden.len(),
            "{case}/nvfp4/{name}: scale length mismatch (rust {:?} vs golden {})",
            r.scale_shape,
            s_golden.len()
        );
        assert!(
            r.scale == s_golden,
            "{case}/nvfp4/{name}: block-scale payload mismatch (first diff at {:?})",
            r.scale
                .iter()
                .zip(s_golden.iter())
                .position(|(a, b)| a != b)
        );

        // F32 per-tensor scale (weight_scale_2) must match bit-for-bit.
        let s2_golden = output
            .tensor_bytes(&scale2_name)
            .unwrap_or_else(|_| panic!("{case}/nvfp4: missing {scale2_name}"));
        assert_eq!(
            s2_golden.len(),
            4,
            "{case}/nvfp4/{name}: weight_scale_2 not scalar"
        );
        let golden_pts =
            f32::from_le_bytes([s2_golden[0], s2_golden[1], s2_golden[2], s2_golden[3]]);
        assert_eq!(
            r.per_tensor_scale.to_bits(),
            golden_pts.to_bits(),
            "{case}/nvfp4/{name}: weight_scale_2 mismatch (rust {} vs golden {})",
            r.per_tensor_scale,
            golden_pts
        );
    }
}

#[test]
fn nvfp4_matches_golden() {
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"] {
        check_case(case);
    }
}

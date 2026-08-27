//! Phase 3.5 parity tests: Rust `.comfy_quant` blob encoder vs ctq goldens.
//!
//! For every golden output file, extracts each `.comfy_quant` U8 blob and
//! byte-compares it against the blob produced by `comfy_schema::encode_*` for
//! the matching format. This covers both serialization families:
//!   * family A (INT8/FP8): `format, orig_dtype, group_size`
//!   * family B (MXFP8/NVFP4): `format, group_size, orig_dtype, orig_shape`
//!
//! `orig_shape` for family B is the PRE-padding input shape, read from the
//! shared input fixture. `orig_dtype` is the output-dtype policy string
//! (`torch.bfloat16` for all fixtures).

use std::path::PathBuf;

use quant_core::comfy_schema::{
    encode_block_format, encode_standard, ComfyFormat, FP8_BLOCKWISE, FP8_ROWWISE, FP8_TENSOR,
    INT8_BLOCKWISE, MXFP8, NVFP4,
};
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

/// (golden file suffix, encoder) pairs. `None` suffix = INT8 `output.safetensors`.
const CASES: [&str; 4] = ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"];

/// Extract all `.comfy_quant` blobs from a golden output, keyed by base name.
fn extract_blobs(output: &SafetensorsReader) -> Vec<(String, Vec<u8>)> {
    output
        .header()
        .iter()
        .filter(|(name, info)| info.dtype_raw == "U8" && name.ends_with(".comfy_quant"))
        .map(|(name, _)| {
            let bytes = output.tensor_bytes(name).unwrap().to_vec();
            let base = name.strip_suffix(".comfy_quant").unwrap().to_string();
            (base, bytes)
        })
        .collect()
}

/// Input shape for a weight (pre-padding), from the shared input fixture.
fn input_shape(input: &SafetensorsReader, weight_name: &str) -> Vec<u64> {
    input
        .header()
        .get(weight_name)
        .unwrap_or_else(|| panic!("input weight {weight_name}"))
        .shape
        .clone()
}

fn check_case(case: &str) {
    let dir = golden(case);
    let input = load(&dir.join("input.safetensors"));

    // ---- family A: INT8 + FP8 -------------------------------------------- //
    // (suffix, format string, group_size)
    let family_a: [(&str, &str, Option<u32>); 4] = [
        ("", INT8_BLOCKWISE, Some(128)),
        ("_fp8", FP8_BLOCKWISE, Some(128)),
        ("_fp8_tensor", FP8_TENSOR, None),
        ("_fp8_row", FP8_ROWWISE, None),
    ];
    for (suffix, fmt, group) in family_a {
        let out_path = dir.join(format!("output{suffix}.safetensors"));
        if !out_path.exists() {
            continue; // e.g. zero_blocks has no INT8 golden
        }
        let output = load(&out_path);
        let blobs = extract_blobs(&output);
        assert!(
            !blobs.is_empty(),
            "{case}/output{suffix}: no comfy_quant blobs"
        );
        for (base, golden_blob) in blobs {
            let rust_blob = encode_standard(fmt, "torch.bfloat16", group);
            assert_eq!(
                rust_blob,
                golden_blob,
                "{case}/output{suffix}/{base}.comfy_quant mismatch:\n  rust   = {}\n  golden = {}",
                String::from_utf8_lossy(&rust_blob),
                String::from_utf8_lossy(&golden_blob)
            );
        }
    }

    // ---- family B: MXFP8 + NVFP4 ----------------------------------------- //
    let family_b: [(&str, &str, u32); 2] = [("_mxfp8", MXFP8, 32), ("_nvfp4", NVFP4, 16)];
    for (suffix, fmt, group) in family_b {
        let out_path = dir.join(format!("output{suffix}.safetensors"));
        if !out_path.exists() {
            continue;
        }
        let output = load(&out_path);
        let blobs = extract_blobs(&output);
        assert!(
            !blobs.is_empty(),
            "{case}/output{suffix}: no comfy_quant blobs"
        );
        for (base, golden_blob) in blobs {
            let weight_name = format!("{base}.weight");
            let shape = input_shape(&input, &weight_name);
            let rust_blob = encode_block_format(fmt, group, "torch.bfloat16", &shape);
            assert_eq!(
                rust_blob,
                golden_blob,
                "{case}/output{suffix}/{base}.comfy_quant mismatch:\n  rust   = {}\n  golden = {}",
                String::from_utf8_lossy(&rust_blob),
                String::from_utf8_lossy(&golden_blob)
            );
        }
    }
}

#[test]
fn comfy_quant_blobs_match_golden() {
    for case in CASES {
        check_case(case);
    }
}

#[test]
fn dispatcher_covers_all_golden_formats() {
    // Sanity: the ComfyFormat dispatcher reproduces the exact family encoders
    // used above, so wiring it into the orchestrator is equivalent.
    use quant_core::comfy_schema::encode_comfy_quant;
    assert_eq!(
        encode_comfy_quant(
            ComfyFormat::Int8Blockwise,
            "torch.bfloat16",
            Some(128),
            None
        ),
        encode_standard(INT8_BLOCKWISE, "torch.bfloat16", Some(128))
    );
    assert_eq!(
        encode_comfy_quant(
            ComfyFormat::Mxfp8,
            "torch.bfloat16",
            None,
            Some(&[256, 128])
        ),
        encode_block_format(MXFP8, 32, "torch.bfloat16", &[256, 128])
    );
    assert_eq!(
        encode_comfy_quant(ComfyFormat::Nvfp4, "torch.bfloat16", None, Some(&[128, 64])),
        encode_block_format(NVFP4, 16, "torch.bfloat16", &[128, 64])
    );
}

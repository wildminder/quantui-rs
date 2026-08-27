//! Phase 4.2 parity tests: parse every `.comfy_quant` blob in the golden
//! fixtures and validate layer structure against the golden headers.
//!
//! Two checks per golden output file:
//!   1. **Parse round-trip** — every `.comfy_quant` U8 blob parses to a typed
//!      [`ComfyQuantConfig`] whose `to_blob()` re-encoding is byte-identical
//!      to the golden blob (proves the parser is the exact inverse of the
//!      reference encoder for all real-world blobs).
//!   2. **Structural layout** — for each parsed config, the child-addressed
//!      config tensor, the quantized weight, and every sibling-addressed
//!      scale required by the format are present in the header.

use std::path::PathBuf;

use quant_core::comfy_schema::{layer_prefix, parse_blob, validate_layer_layout, ComfyFormat};
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

const CASES: [&str; 4] = ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"];

/// All golden output suffixes ("" = INT8 blockwise).
const SUFFIXES: [&str; 6] = ["", "_fp8", "_fp8_tensor", "_fp8_row", "_mxfp8", "_nvfp4"];

/// Expected registry format per suffix.
fn expected_format(suffix: &str) -> ComfyFormat {
    match suffix {
        "" => ComfyFormat::Int8Blockwise,
        "_fp8" => ComfyFormat::Fp8Blockwise,
        "_fp8_tensor" => ComfyFormat::Fp8Tensor,
        "_fp8_row" => ComfyFormat::Fp8Rowwise,
        "_mxfp8" => ComfyFormat::Mxfp8,
        "_nvfp4" => ComfyFormat::Nvfp4,
        _ => unreachable!(),
    }
}

fn check_case(case: &str) {
    let dir = golden(case);
    for suffix in SUFFIXES {
        let out_path = dir.join(format!("output{suffix}.safetensors"));
        if !out_path.exists() {
            continue; // e.g. zero_blocks has no INT8 golden
        }
        let output = load(&out_path);
        let names: Vec<String> = output.header().iter().map(|(n, _)| n.clone()).collect();

        let config_keys: Vec<String> = names
            .iter()
            .filter(|n| n.ends_with(".comfy_quant"))
            .cloned()
            .collect();
        assert!(
            !config_keys.is_empty(),
            "{case}/output{suffix}: no .comfy_quant tensors"
        );

        for key in &config_keys {
            let blob = output.tensor_bytes(key).unwrap().to_vec();

            // 1. Parse + byte-exact round-trip.
            let cfg = parse_blob(&blob)
                .unwrap_or_else(|e| panic!("{case}/output{suffix}/{key}: parse failed: {e}"));
            assert_eq!(
                cfg.format,
                expected_format(suffix),
                "{case}/output{suffix}/{key}: format mismatch"
            );
            assert_eq!(
                cfg.orig_dtype, "torch.bfloat16",
                "{case}/output{suffix}/{key}: orig_dtype"
            );
            assert_eq!(
                cfg.to_blob(),
                blob,
                "{case}/output{suffix}/{key}: round-trip re-encode mismatch"
            );

            // 2. Structural layout vs the header.
            let prefix = layer_prefix(key).unwrap();
            let issues = validate_layer_layout(prefix, &cfg, &names);
            assert!(
                issues.is_empty(),
                "{case}/output{suffix}/{key}: structural issues: {issues:?}"
            );
        }
    }
}

#[test]
fn parse_and_validate_all_goldens() {
    for case in CASES {
        check_case(case);
    }
}

#[test]
fn family_b_orig_shape_matches_input_weight() {
    // Family-B `orig_shape` must equal the PRE-padding input weight shape.
    for case in CASES {
        let dir = golden(case);
        let input = load(&dir.join("input.safetensors"));
        for suffix in ["_mxfp8", "_nvfp4"] {
            let out_path = dir.join(format!("output{suffix}.safetensors"));
            if !out_path.exists() {
                continue;
            }
            let output = load(&out_path);
            for (name, _info) in output.header().iter() {
                if !name.ends_with(".comfy_quant") {
                    continue;
                }
                let blob = output.tensor_bytes(name).unwrap();
                let cfg = parse_blob(blob).unwrap();
                let prefix = layer_prefix(name).unwrap();
                let weight_name = format!("{prefix}.weight");
                let want = input
                    .header()
                    .get(&weight_name)
                    .unwrap_or_else(|| panic!("input weight {weight_name}"))
                    .shape
                    .clone();
                assert_eq!(
                    cfg.orig_shape.as_deref(),
                    Some(want.as_slice()),
                    "{case}/output{suffix}/{name}: orig_shape != input weight shape"
                );
            }
        }
    }
}

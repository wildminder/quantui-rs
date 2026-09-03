//! Phase 7.2 (plan Phase 7 / decision Q3): ConvRot streaming E2E payload
//! parity — INT8 row-wise + group-wise Hadamard rotation.
//!
//! For each golden case and each group size (256 = the CLI preset default,
//! 64 = the smaller ladder step), run the streaming orchestrator over
//! `input.safetensors` and byte-compare EVERY emitted tensor (rotated/plain
//! int8 weight, [m,1] weight_scale, comfy_quant blob, corrected biases, and
//! all copied/skipped tensors) against `output_int8_convrot*.safetensors`.
//!
//! The goldens were produced by ctq's WHOLE-FILE batch driver
//! (`convert_to_fp8_scaled`, `--int8 --scaling_mode row --convrot`), NOT the
//! streaming reference (which never implemented convrot) — so two contract
//! details differ from the legacy INT8 goldens:
//!   * the calibration draw order is SORTED (`safe_open.keys()`,
//!     fp8_conversion.py:154/:301) — the stream pins it for convrot configs;
//!   * `per_row: true` appears on EVERY int8 row-wise layer (scaling-mode
//!     keyed, fp8_conversion.py:583-584 / tensor_quant.py:254-255), while the
//!     `convrot`/`convrot_groupsize` keys appear only on tensors that were
//!     actually rotated (convrot_applied recomputed per tensor, :476-492).
//!
//! Coverage per case (group size 256 vs 64):
//!   * linear_basic_bf16 — gs256: blocks.0 [256,128] stays PLAIN row-wise
//!     (per_row only; generic bias path); gs64: rotated (ConvRot bias path).
//!     blocks.1 [128,64] is heur-skipped in both (cols < 128) → BF16 cast
//!     passthrough. norm.weight 1D passthrough.
//!   * odd_shapes — double_block.0 [384,256] rotated in both; odd.weight
//!     [130,130] heur-skipped → BF16 cast.
//!   * conv_net — head.weight [128,128] F16: gs256 → plain (per_row only);
//!     gs64 → rotated. conv.net.weight 4D → passthrough uncast; conv.bias
//!     passthrough.
//!   * zero_blocks — blk.weight [256,256] with zeroed rows 64..127,
//!     rotated in both (zero-row scale → clamp_min(1e-12) path).
//!
//! Parity tier: BYTE-EXACT for every tensor. The kchunk128 GEMM emulation
//! was probe-validated against the reference chain
//! (`tools/probe_convrot_chain.py`, `tools/probe_convrot_fma.py`); the only
//! residual observed there (1 element in 2²⁹, torch's fused FMA vs our
//! double rounding) sits far below the f32 rounding of a bias element after
//! the /3072 mean — if it ever surfaces here it must be INVESTIGATED, not
//! tolerated. No tolerance downgrade is pre-authorized.

mod common;

use common::{assert_payload_parity, golden, load};
use quant_core::manifest::{Format, QuantConfig, ScalingMode};
use quant_core::st_io::reader::SafetensorsReader;

/// ConvRot config: INT8 row-wise, rotation on, group size `gs`.
///
/// `block_size` stays 128 — it feeds the heur skip rule (the batch driver
/// uses the int8 format default, fp8_conversion.py:184) and the heur-OFF
/// divisibility check, never the row quantizer.
fn convrot_config(gs: u32) -> QuantConfig {
    QuantConfig {
        format: Format::Int8,
        target_format: "int8".into(),
        int8: true,
        scaling_mode: ScalingMode::Row,
        block_size: 128,
        convrot: true,
        convrot_group_size: gs,
        ..QuantConfig::default()
    }
}

fn run_convrot(case: &str, golden_suffix: &str, gs: u32) {
    let dir = golden(case);
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");

    quant_core::stream::stream_quantize(dir.join("input.safetensors"), &out, &convrot_config(gs))
        .unwrap();

    assert_payload_parity(
        &out,
        &dir.join(format!("output_{golden_suffix}.safetensors")),
        &format!("{case}/{golden_suffix}"),
    );
}

// ---- group size 256 (the int8_convrot CLI preset default) ---------------- //

#[test]
fn convrot_gs256_linear_basic_bf16() {
    run_convrot("linear_basic_bf16", "int8_convrot", 256);
}

#[test]
fn convrot_gs256_odd_shapes() {
    run_convrot("odd_shapes", "int8_convrot", 256);
}

#[test]
fn convrot_gs256_conv_net() {
    run_convrot("conv_net", "int8_convrot", 256);
}

#[test]
fn convrot_gs256_zero_blocks() {
    run_convrot("zero_blocks", "int8_convrot", 256);
}

// ---- group size 64 (smaller ladder step; different rotated set) ----------- //

#[test]
fn convrot_gs64_linear_basic_bf16() {
    run_convrot("linear_basic_bf16", "int8_convrot_gs64", 64);
}

#[test]
fn convrot_gs64_odd_shapes() {
    run_convrot("odd_shapes", "int8_convrot_gs64", 64);
}

#[test]
fn convrot_gs64_conv_net() {
    run_convrot("conv_net", "int8_convrot_gs64", 64);
}

#[test]
fn convrot_gs64_zero_blocks() {
    run_convrot("zero_blocks", "int8_convrot_gs64", 64);
}

/// Structural: the goldens themselves must encode the per-tensor rotation
/// decision — gs256 conv_net/head.weight [128,128] carries `per_row` but NO
/// `convrot` keys (128 % 256 != 0 → plain), while the gs64 golden carries
/// both (128 % 64 == 0 → rotated). This pins the blob contract independently
/// of the payload compare (and would catch a golden regenerated with a
/// different ctq build silently changing blob semantics).
#[test]
fn golden_blob_encodes_per_tensor_rotation_decision() {
    let blob = |path: &std::path::Path| -> String {
        let r = load(path);
        let name = "head.comfy_quant";
        String::from_utf8(r.tensor_bytes(name).unwrap().to_vec()).unwrap()
    };
    let dir = golden("conv_net");
    let plain = blob(&dir.join("output_int8_convrot.safetensors"));
    let rotated = blob(&dir.join("output_int8_convrot_gs64.safetensors"));

    assert_eq!(
        plain, r#"{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "per_row": true}"#,
        "gs256: head.weight [128,128] is NOT rotated (128 % 256 != 0) — per_row only"
    );
    assert_eq!(
        rotated,
        r#"{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "convrot": true, "convrot_groupsize": 64, "per_row": true}"#,
        "gs64: head.weight [128,128] IS rotated (128 % 64 == 0)"
    );
}

/// Structural: no convrot golden carries `__metadata__` (the batch driver's
/// `save_quant_metadata` defaults to False, and the gen script does not pass
/// it), and the quantized weights are I8 with [m,1] F32 row scales.
#[test]
fn golden_structure_metadata_and_scales() {
    for case in ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"] {
        for suffix in ["int8_convrot", "int8_convrot_gs64"] {
            let r = load(&golden(case).join(format!("output_{suffix}.safetensors")));
            assert!(
                !r.header().contains_key("__metadata__"),
                "{case}/{suffix}: convrot goldens must not carry __metadata__"
            );
            for (name, info) in r.header().iter() {
                if name.ends_with(".weight") && info.dtype_raw == "I8" {
                    assert_eq!(
                        info.shape.len(),
                        2,
                        "{case}/{suffix}/{name}: int8 weight must stay 2D"
                    );
                    let base = name.strip_suffix(".weight").unwrap();
                    let sinfo = r
                        .header()
                        .get(&format!("{base}.weight_scale"))
                        .unwrap_or_else(|| {
                            panic!("{case}/{suffix}/{name}: missing {base}.weight_scale")
                        });
                    assert_eq!(sinfo.dtype_raw, "F32");
                    assert_eq!(
                        sinfo.shape,
                        vec![info.shape[0], 1],
                        "{case}/{suffix}/{name}: row scale must be [m,1]"
                    );
                }
            }
        }
    }
}

/// Sanity helper reference for SafetensorsReader (keeps the import honest in
/// case the structural tests above are refactored away from it).
#[allow(dead_code)]
fn _reader_type_check(p: &std::path::Path) -> SafetensorsReader {
    load(p)
}

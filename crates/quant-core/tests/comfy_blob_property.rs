//! WP6 / NTH-001 — property tests for the ComfyUI `.comfy_quant` blob
//! schema (`comfy_schema.rs`).
//!
//!   * round-trip: `parse_blob(encode_*(...))` recovers the exact config
//!     for arbitrary (format, orig_dtype, group_size, orig_shape) across
//!     ALL 7 registry formats — including the rowwise/convrot INT8 tails;
//!   * totality: arbitrary bytes → typed `Err`, never a panic.
//!
//! The round-trip mirrors what the encoder can express: family A never
//! carries `orig_shape` (parse yields `None`), and `group_size` is only
//! emitted for block-based formats (`int8_blockwise`, `fp8_blockwise`)
//! and family B (whose value is the format's FIXED size, not the input).

use proptest::prelude::*;

use quant_core::comfy_schema::{parse_blob, ComfyFormat, MXFP8_GROUP_SIZE, NVFP4_GROUP_SIZE};

/// Every registry format, so the property covers all 7 encoders.
fn format_strategy() -> impl Strategy<Value = ComfyFormat> {
    proptest::sample::select(vec![
        ComfyFormat::Int8Tensorwise,
        ComfyFormat::Int8Blockwise,
        ComfyFormat::Fp8Tensor,
        ComfyFormat::Fp8Rowwise,
        ComfyFormat::Fp8Blockwise,
        ComfyFormat::Mxfp8,
        ComfyFormat::Nvfp4,
    ])
}

/// dtype strings the reference emits (subset is fine — the parser treats
/// it as an opaque string).
fn orig_dtype_strategy() -> impl Strategy<Value = String> {
    proptest::sample::select(vec![
        "torch.bfloat16",
        "torch.float16",
        "torch.float32",
        "torch.float8_e4m3fn",
    ])
    .prop_map(|s| s.to_string())
}

#[test]
fn comfy_blob_round_trip() {
    proptest!(|(
        format in format_strategy(),
        orig_dtype in orig_dtype_strategy(),
        group in 1u32..=1024,
        shape in proptest::collection::vec(1u64..=8192, 1..=4),
        rowwise in proptest::bool::ANY,
        convrot_gs in 1u32..=512,
    )| {
        let blob: Vec<u8> = if format.is_block_format() {
            // Family B: group size is FIXED by the format, orig_shape required.
            quant_core::comfy_schema::encode_block_format(
                format.as_str(),
                format.fixed_group_size().expect("family B has fixed gs"),
                &orig_dtype,
                &shape,
            )
        } else if rowwise && format == ComfyFormat::Int8Tensorwise {
            // The INT8 row-wise tails: plain per_row and the ConvRot variant.
            if convrot_gs % 2 == 0 {
                quant_core::comfy_schema::encode_comfy_quant_int8_convrot(
                    &orig_dtype,
                    convrot_gs,
                )
            } else {
                quant_core::comfy_schema::encode_comfy_quant_int8_rowwise(&orig_dtype)
            }
        } else {
            // Family A plain: group_size only survives for block-based formats.
            let gs = match format {
                ComfyFormat::Int8Blockwise | ComfyFormat::Fp8Blockwise => Some(group),
                _ => None,
            };
            quant_core::comfy_schema::encode_standard(format.as_str(), &orig_dtype, gs)
        };

        let cfg = parse_blob(&blob).expect("encoder output must always parse");
        prop_assert_eq!(cfg.format, format, "format round-trip");
        prop_assert_eq!(&cfg.orig_dtype, &orig_dtype, "orig_dtype round-trip");

        if format.is_block_format() {
            prop_assert_eq!(
                cfg.group_size,
                Some(match format {
                    ComfyFormat::Mxfp8 => MXFP8_GROUP_SIZE,
                    ComfyFormat::Nvfp4 => NVFP4_GROUP_SIZE,
                    _ => unreachable!("block format exhausted"),
                }),
                "family B group size is the format's fixed value"
            );
            prop_assert_eq!(cfg.orig_shape.as_deref(), Some(shape.as_slice()));
        } else {
            prop_assert!(cfg.orig_shape.is_none(), "family A never has orig_shape");
            prop_assert_eq!(
                cfg.group_size,
                match format {
                    ComfyFormat::Int8Blockwise | ComfyFormat::Fp8Blockwise => Some(group),
                    _ => None,
                },
                "family A group_size only for block-based"
            );
        }

        // per_row: true only for the INT8 row-wise tails.
        let expect_row = !format.is_block_format()
            && rowwise
            && format == ComfyFormat::Int8Tensorwise;
        prop_assert_eq!(cfg.per_row, expect_row, "per_row round-trip");
    });
}

/// Arbitrary bytes → typed `Err`, never a panic (invalid UTF-8, broken
/// JSON, unknown formats, wrong field types — all are `ComfyQuantError`).
#[test]
fn comfy_blob_never_panics_on_arbitrary_bytes() {
    proptest!(|(blob in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..256))| {
        let _ = parse_blob(&blob);
    });
}

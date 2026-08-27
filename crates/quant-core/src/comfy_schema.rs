//! `.comfy_quant` blob encoding (plan Phase 3.5; Phase 4.1 extends to a full
//! parser/validator).
//!
//! Emits the `.comfy_quant` U8 JSON blobs byte-identical to the reference
//! `convert_to_quant` simple paths. Two serialization families exist in the
//! reference, and the key ORDER differs between them, so both are ported
//! explicitly:
//!
//! * **Family A** — `utils/comfy_quant.py::create_comfy_quant_tensor`
//!   (INT8 + FP8). Key order `format, orig_dtype, group_size`; `group_size`
//!   is present only when a block size is given AND the format is in
//!   `BLOCK_BASED_FORMATS = ("int8_blockwise", "float8_e4m3fn_blockwise")`.
//!   FP8 rowwise does NOT add `per_row` (only the INT8 rowwise path does, and
//!   the fixtures never exercise it).
//! * **Family B** — direct `dict_to_tensor(metadata)` in
//!   `formats/mxfp8_conversion.py` / `formats/nvfp4_conversion.py`
//!   (MXFP8 + NVFP4). Key order `format, group_size, orig_dtype, orig_shape`,
//!   where `orig_shape` is the PRE-padding input shape.
//!
//! Both families use `json.dumps` default separators (`", "` between items,
//! `": "` between key and value, `", "` between array items), so the strings
//! are hand-built to avoid serde_json formatting drift (same approach as
//! `manifest.rs::config_hash`). Verified byte-for-byte against blobs extracted
//! from every golden fixture (see `tests/phase3_comfy_quant_blob.rs`).
//!
//! `orig_dtype` is the OUTPUT-dtype policy string (`torch.bfloat16` /
//! `torch.float16`), not the true input dtype — mirroring
//! `resolve_output_dtype` → `str(resolved_output_dtype)`.

/// Format strings matching the reference ComfyFormat registry.
pub const INT8_TENSORWISE: &str = "int8_tensorwise";
pub const INT8_BLOCKWISE: &str = "int8_blockwise";
pub const FP8_TENSOR: &str = "float8_e4m3fn";
pub const FP8_ROWWISE: &str = "float8_e4m3fn_rowwise";
pub const FP8_BLOCKWISE: &str = "float8_e4m3fn_blockwise";
pub const MXFP8: &str = "mxfp8";
pub const NVFP4: &str = "nvfp4";

/// Formats that carry a `group_size` key in family-A blobs
/// (`create_comfy_quant_tensor`'s `BLOCK_BASED_FORMATS`).
const BLOCK_BASED_FORMATS: &[&str] = &[INT8_BLOCKWISE, FP8_BLOCKWISE];

/// Fixed group sizes for the block formats (family B).
pub const MXFP8_GROUP_SIZE: u32 = 32;
pub const NVFP4_GROUP_SIZE: u32 = 16;

/// High-level format selector for the streaming orchestrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComfyFormat {
    Int8Tensorwise,
    Int8Blockwise,
    Fp8Tensor,
    Fp8Rowwise,
    Fp8Blockwise,
    Mxfp8,
    Nvfp4,
}

impl ComfyFormat {
    /// Registry format string.
    pub fn as_str(&self) -> &'static str {
        match self {
            ComfyFormat::Int8Tensorwise => INT8_TENSORWISE,
            ComfyFormat::Int8Blockwise => INT8_BLOCKWISE,
            ComfyFormat::Fp8Tensor => FP8_TENSOR,
            ComfyFormat::Fp8Rowwise => FP8_ROWWISE,
            ComfyFormat::Fp8Blockwise => FP8_BLOCKWISE,
            ComfyFormat::Mxfp8 => MXFP8,
            ComfyFormat::Nvfp4 => NVFP4,
        }
    }

    /// True for the family-B (direct-dict, `orig_shape`-bearing) formats.
    pub fn is_block_format(&self) -> bool {
        matches!(self, ComfyFormat::Mxfp8 | ComfyFormat::Nvfp4)
    }

    /// Fixed group size for family-B formats; `None` for family-A (where the
    /// block size comes from the quantization config, if block-based).
    pub fn fixed_group_size(&self) -> Option<u32> {
        match self {
            ComfyFormat::Mxfp8 => Some(MXFP8_GROUP_SIZE),
            ComfyFormat::Nvfp4 => Some(NVFP4_GROUP_SIZE),
            _ => None,
        }
    }
}

/// Family-A blob (INT8 / FP8): `create_comfy_quant_tensor` key order
/// `format, orig_dtype, group_size`. `group_size` is emitted only when `Some`
/// AND `format` is block-based.
pub fn encode_standard(format: &str, orig_dtype: &str, group_size: Option<u32>) -> Vec<u8> {
    let mut s = format!(r#"{{"format": "{format}", "orig_dtype": "{orig_dtype}""#);
    if let Some(g) = group_size {
        if BLOCK_BASED_FORMATS.contains(&format) {
            s.push_str(&format!(r#", "group_size": {g}"#));
        }
    }
    s.push('}');
    s.into_bytes()
}

/// Family-B blob (MXFP8 / NVFP4): direct-dict key order
/// `format, group_size, orig_dtype, orig_shape`.
pub fn encode_block_format(
    format: &str,
    group_size: u32,
    orig_dtype: &str,
    orig_shape: &[u64],
) -> Vec<u8> {
    let shape = orig_shape
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"{{"format": "{format}", "group_size": {group_size}, "orig_dtype": "{orig_dtype}", "orig_shape": [{shape}]}}"#
    )
    .into_bytes()
}

/// Dispatch on [`ComfyFormat`].
///
/// * Family A (`Int8*` / `Fp8*`): `block_group_size` supplies the block size
///   for the blockwise variants (ignored otherwise); `orig_shape` unused.
/// * Family B (`Mxfp8` / `Nvfp4`): `orig_shape` is required (pre-padding
///   input shape); the group size is the format's fixed value.
pub fn encode_comfy_quant(
    format: ComfyFormat,
    orig_dtype: &str,
    block_group_size: Option<u32>,
    orig_shape: Option<&[u64]>,
) -> Vec<u8> {
    if format.is_block_format() {
        let g = format
            .fixed_group_size()
            .expect("block format has fixed group size");
        let shape = orig_shape.expect("family-B format requires orig_shape");
        encode_block_format(format.as_str(), g, orig_dtype, shape)
    } else {
        // Only block-based family-A formats carry group_size.
        let g = if BLOCK_BASED_FORMATS.contains(&format.as_str()) {
            block_group_size
        } else {
            None
        };
        encode_standard(format.as_str(), orig_dtype, g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_a_int8_blockwise() {
        let b = encode_standard(INT8_BLOCKWISE, "torch.bfloat16", Some(128));
        assert_eq!(
            String::from_utf8(b).unwrap(),
            r#"{"format": "int8_blockwise", "orig_dtype": "torch.bfloat16", "group_size": 128}"#
        );
    }

    #[test]
    fn family_a_fp8_variants() {
        assert_eq!(
            String::from_utf8(encode_standard(FP8_BLOCKWISE, "torch.bfloat16", Some(128))).unwrap(),
            r#"{"format": "float8_e4m3fn_blockwise", "orig_dtype": "torch.bfloat16", "group_size": 128}"#
        );
        assert_eq!(
            String::from_utf8(encode_standard(FP8_TENSOR, "torch.bfloat16", None)).unwrap(),
            r#"{"format": "float8_e4m3fn", "orig_dtype": "torch.bfloat16"}"#
        );
        // Rowwise: no group_size, no per_row.
        assert_eq!(
            String::from_utf8(encode_standard(FP8_ROWWISE, "torch.bfloat16", None)).unwrap(),
            r#"{"format": "float8_e4m3fn_rowwise", "orig_dtype": "torch.bfloat16"}"#
        );
    }

    #[test]
    fn family_a_ignores_group_size_for_non_block() {
        // int8_tensorwise with a stray block size must NOT emit group_size.
        let b = encode_standard(INT8_TENSORWISE, "torch.bfloat16", Some(128));
        assert_eq!(
            String::from_utf8(b).unwrap(),
            r#"{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16"}"#
        );
    }

    #[test]
    fn family_b_mxfp8_and_nvfp4() {
        assert_eq!(
            String::from_utf8(encode_block_format(
                MXFP8,
                32,
                "torch.bfloat16",
                &[256, 128]
            ))
            .unwrap(),
            r#"{"format": "mxfp8", "group_size": 32, "orig_dtype": "torch.bfloat16", "orig_shape": [256, 128]}"#
        );
        assert_eq!(
            String::from_utf8(encode_block_format(NVFP4, 16, "torch.bfloat16", &[128, 64]))
                .unwrap(),
            r#"{"format": "nvfp4", "group_size": 16, "orig_dtype": "torch.bfloat16", "orig_shape": [128, 64]}"#
        );
    }

    #[test]
    fn dispatcher_matches_families() {
        // Family A via dispatcher.
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
            encode_comfy_quant(ComfyFormat::Fp8Tensor, "torch.float16", None, None),
            encode_standard(FP8_TENSOR, "torch.float16", None)
        );
        // Family B via dispatcher (group size is fixed, shape required).
        assert_eq!(
            encode_comfy_quant(
                ComfyFormat::Mxfp8,
                "torch.bfloat16",
                None,
                Some(&[384, 256])
            ),
            encode_block_format(MXFP8, 32, "torch.bfloat16", &[384, 256])
        );
        assert_eq!(
            encode_comfy_quant(
                ComfyFormat::Nvfp4,
                "torch.bfloat16",
                None,
                Some(&[256, 256])
            ),
            encode_block_format(NVFP4, 16, "torch.bfloat16", &[256, 256])
        );
    }

    #[test]
    fn float16_orig_dtype() {
        assert_eq!(
            String::from_utf8(encode_standard(FP8_TENSOR, "torch.float16", None)).unwrap(),
            r#"{"format": "float8_e4m3fn", "orig_dtype": "torch.float16"}"#
        );
    }
}

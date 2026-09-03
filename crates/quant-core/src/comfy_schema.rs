//! `.comfy_quant` blob encoding, parsing, and structural validation
//! primitives (plan Phase 3.5 encoder; Phase 4.2 parser/validator).
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

/// ConvRot INT8 row-wise blob (plan Phase 7.1).
///
/// Still family A — same `create_comfy_quant_tensor` insertion order
/// (`format`, `orig_dtype`, [`group_size`], [`full_precision_matrix_mult`],
/// `convrot`, `convrot_groupsize`, `per_row` — `utils/comfy_quant.py:24-60`),
/// with the ConvRot tail keys present. ConvRot is INT8 row-wise ONLY, so:
///
/// * `format` is `int8_tensorwise` (`fp8_conversion.py:580-582`: both `tensor`
///   and `row` map to it), which is NOT in `BLOCK_BASED_FORMATS` → no
///   `group_size` key even though ConvRot has a group size of its own;
/// * `full_precision_matrix_mult` is never set by the streaming path → omitted;
/// * `convrot=true` + `convrot_groupsize=<gs>` (`fp8_conversion.py:594` passes
///   the group size only when `convrot_applied`);
/// * `per_row=true` because `converter.scaling_mode == "row"` (`:594`).
///
/// Exact emitted string (json.dumps default separators):
/// `{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "convrot": true, "convrot_groupsize": 256, "per_row": true}`
pub fn encode_comfy_quant_int8_convrot(orig_dtype: &str, convrot_groupsize: u32) -> Vec<u8> {
    format!(
        r#"{{"format": "{INT8_TENSORWISE}", "orig_dtype": "{orig_dtype}", "convrot": true, "convrot_groupsize": {convrot_groupsize}, "per_row": true}}"#
    )
    .into_bytes()
}

/// Plain INT8 row-wise blob (Phase 7.2 resolution of the 7.1 divergence).
///
/// BOTH references key `per_row` off the SCALING MODE, not off ConvRot —
/// every INT8 row-wise layer carries it, rotated or not:
/// * batch driver `fp8_conversion.py:583-594`: `per_row = True` when
///   `converter.scaling_mode == "row"`, passed to `create_comfy_quant_tensor`;
/// * streaming reference `docs/ref/quantui/quantui/tensor_quant.py:252-255`:
///   `per_row = scaling == "row"`, passed unconditionally.
///
/// A ConvRot run whose tensor is NOT rotated (in_features not divisible by
/// the group size) lands here too: the batch driver recomputes
/// `convrot_applied` per tensor (`:476-492`) and omits the `convrot` keys,
/// but keeps `per_row` (probe evidence: `tools/probe_convrot_blob_per_row.py`).
///
/// Exact emitted string (json.dumps default separators):
/// `{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "per_row": true}`
pub fn encode_comfy_quant_int8_rowwise(orig_dtype: &str) -> Vec<u8> {
    format!(r#"{{"format": "{INT8_TENSORWISE}", "orig_dtype": "{orig_dtype}", "per_row": true}}"#)
        .into_bytes()
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

// --------------------------------------------------------------------------- //
// Parsing (Phase 4.2 — port of ctq `tensor_to_dict` + typed extraction).
// --------------------------------------------------------------------------- //

/// Typed errors for `.comfy_quant` blob parsing and structural validation.
///
/// Parse variants mirror the failure modes of the reference
/// `tensor_to_dict` (`bytes.decode("utf-8")` + `json.loads`) plus typed field
/// extraction; the `Missing*` variants are the structural (header-only)
/// layout checks used by the validator primitives below.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ComfyQuantError {
    /// Blob bytes are not valid UTF-8.
    #[error(".comfy_quant blob is not valid UTF-8")]
    InvalidUtf8,
    /// Blob is not valid JSON (truncated / malformed).
    #[error(".comfy_quant blob is not valid JSON: {0}")]
    InvalidJson(String),
    /// Top-level JSON value is not an object.
    #[error(".comfy_quant payload is not a JSON object")]
    NotAnObject,
    /// The mandatory `"format"` field is absent.
    #[error(".comfy_quant config is missing the \"format\" field")]
    MissingFormat,
    /// `"format"` is present but not a known registry string.
    #[error("unknown .comfy_quant format {0:?}")]
    UnknownFormat(String),
    /// A field required by the format family is absent (e.g. family-B
    /// formats require `orig_shape`).
    #[error(".comfy_quant config is missing required field {0:?}")]
    MissingField(&'static str),
    /// A field is present but has the wrong JSON type.
    #[error(".comfy_quant field {field:?} has wrong type (expected {expected})")]
    BadFieldType {
        field: &'static str,
        expected: &'static str,
    },
    /// Structural: the child-addressed `<prefix>.comfy_quant` config tensor
    /// is missing.
    #[error("layer {prefix:?} is missing its child-addressed config tensor {name:?}")]
    MissingConfig { prefix: String, name: String },
    /// Structural: the layer's quantized `<prefix>.weight` tensor is missing.
    #[error("layer {prefix:?} is missing its quantized weight tensor {name:?}")]
    MissingWeight { prefix: String, name: String },
    /// Structural: a required sibling-addressed scale tensor is missing.
    #[error("layer {prefix:?} is missing required sibling scale tensor {name:?}")]
    MissingScale { prefix: String, name: String },
}

/// Parsed `.comfy_quant` config — the typed view of a blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComfyQuantConfig {
    /// Resolved registry format.
    pub format: ComfyFormat,
    /// Output-dtype policy string (e.g. `torch.bfloat16`).
    pub orig_dtype: String,
    /// Quantization group/block size (family-A blockwise + family B).
    pub group_size: Option<u32>,
    /// Pre-padding input shape (family B only).
    pub orig_shape: Option<Vec<u64>>,
    /// `per_row: true` — INT8 row-wise marker (Phase 7.2). Both references
    /// key it off the scaling mode (`fp8_conversion.py:583-584`,
    /// `tensor_quant.py:254-255`), so every `int8_tensorwise` row-mode
    /// layer carries it — ConvRot-skipped layers included. Tensor-mode
    /// blobs omit it. Affects the expected `weight_scale` shape: row mode
    /// is `[m,1]`, tensor mode is a squeezed scalar.
    pub per_row: bool,
}

impl ComfyFormat {
    /// Resolve a registry format string to a [`ComfyFormat`].
    pub fn from_registry(s: &str) -> Option<ComfyFormat> {
        Some(match s {
            INT8_TENSORWISE => ComfyFormat::Int8Tensorwise,
            INT8_BLOCKWISE => ComfyFormat::Int8Blockwise,
            FP8_TENSOR => ComfyFormat::Fp8Tensor,
            FP8_ROWWISE => ComfyFormat::Fp8Rowwise,
            FP8_BLOCKWISE => ComfyFormat::Fp8Blockwise,
            MXFP8 => ComfyFormat::Mxfp8,
            NVFP4 => ComfyFormat::Nvfp4,
            _ => return None,
        })
    }

    /// Sibling-addressed scale tensor suffixes required by this format,
    /// relative to the layer prefix (i.e. `<prefix>.<suffix>`). All formats
    /// need `weight_scale`; NVFP4 additionally needs the per-tensor
    /// `weight_scale_2`.
    pub fn required_scale_suffixes(&self) -> &'static [&'static str] {
        match self {
            ComfyFormat::Nvfp4 => &["weight_scale", "weight_scale_2"],
            _ => &["weight_scale"],
        }
    }
}

impl ComfyQuantConfig {
    /// Re-encode this config to its canonical blob bytes. This is the exact
    /// inverse of [`parse_blob`] for blobs produced by the reference encoder
    /// (verified byte-for-byte against every golden in the Phase 4 tests).
    pub fn to_blob(&self) -> Vec<u8> {
        encode_comfy_quant(
            self.format,
            &self.orig_dtype,
            self.group_size,
            self.orig_shape.as_deref(),
        )
    }
}

/// Parse a `.comfy_quant` U8 JSON blob into a typed [`ComfyQuantConfig`].
///
/// Port of ctq `utils/tensor_utils.py::tensor_to_dict`
/// (`bytes(...).decode("utf-8")` + `json.loads`) with typed field extraction
/// on top. Unknown extra fields are tolerated — the reference tolerates them
/// too (`edit_comfy_quant` freely adds/removes keys).
pub fn parse_blob(blob: &[u8]) -> Result<ComfyQuantConfig, ComfyQuantError> {
    let text = std::str::from_utf8(blob).map_err(|_| ComfyQuantError::InvalidUtf8)?;
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| ComfyQuantError::InvalidJson(e.to_string()))?;
    parse_value(&value)
}

/// Typed extraction from an already-decoded JSON value.
fn parse_value(value: &serde_json::Value) -> Result<ComfyQuantConfig, ComfyQuantError> {
    let obj = value.as_object().ok_or(ComfyQuantError::NotAnObject)?;

    // "format" — mandatory, must be a known registry string.
    let fmt_raw = obj
        .get("format")
        .ok_or(ComfyQuantError::MissingFormat)?
        .as_str()
        .ok_or(ComfyQuantError::BadFieldType {
            field: "format",
            expected: "string",
        })?;
    let format = ComfyFormat::from_registry(fmt_raw)
        .ok_or_else(|| ComfyQuantError::UnknownFormat(fmt_raw.to_string()))?;

    // "orig_dtype" — mandatory string (present in every ctq blob).
    let orig_dtype = obj
        .get("orig_dtype")
        .ok_or(ComfyQuantError::MissingField("orig_dtype"))?
        .as_str()
        .ok_or(ComfyQuantError::BadFieldType {
            field: "orig_dtype",
            expected: "string",
        })?
        .to_string();

    // "group_size" — optional non-negative integer.
    let group_size = match obj.get("group_size") {
        None => None,
        Some(v) => Some(v.as_u64().and_then(|g| u32::try_from(g).ok()).ok_or(
            ComfyQuantError::BadFieldType {
                field: "group_size",
                expected: "non-negative integer",
            },
        )?),
    };

    // "orig_shape" — optional array of non-negative integers; REQUIRED for
    // family-B formats.
    let orig_shape = match obj.get("orig_shape") {
        None => {
            if format.is_block_format() {
                return Err(ComfyQuantError::MissingField("orig_shape"));
            }
            None
        }
        Some(v) => {
            let arr = v.as_array().ok_or(ComfyQuantError::BadFieldType {
                field: "orig_shape",
                expected: "array of non-negative integers",
            })?;
            let mut shape = Vec::with_capacity(arr.len());
            for dim in arr {
                shape.push(dim.as_u64().ok_or(ComfyQuantError::BadFieldType {
                    field: "orig_shape",
                    expected: "array of non-negative integers",
                })?);
            }
            Some(shape)
        }
    };

    // "per_row" — optional boolean (INT8 row-wise marker, Phase 7.2).
    let per_row = match obj.get("per_row") {
        None => false,
        Some(v) => v.as_bool().ok_or(ComfyQuantError::BadFieldType {
            field: "per_row",
            expected: "boolean",
        })?,
    };

    Ok(ComfyQuantConfig {
        format,
        orig_dtype,
        group_size,
        orig_shape,
        per_row,
    })
}

// --------------------------------------------------------------------------- //
// Structural validator primitives (Phase 4.2).
//
// On-disk addressing conventions (ctq):
//   * child-addressed config:   `<prefix>.comfy_quant`   (U8 JSON blob)
//   * quantized weight:         `<prefix>.weight`
//   * sibling-addressed scales: `<prefix>.weight_scale`  (all formats) and
//                               `<prefix>.weight_scale_2` (NVFP4 only).
// These are header-only checks — no tensor bytes are read.
// --------------------------------------------------------------------------- //

/// Child-addressed `.comfy_quant` config key for a layer prefix.
pub fn config_key(prefix: &str) -> String {
    format!("{prefix}.comfy_quant")
}

/// Recover the layer prefix from a `<prefix>.comfy_quant` tensor name.
pub fn layer_prefix(config_key_name: &str) -> Option<&str> {
    config_key_name.strip_suffix(".comfy_quant")
}

/// Structural validation of one quantized layer against the tensor names
/// present in a file. Returns every issue found (empty = structurally valid).
///
/// `tensor_names` is the full set of tensor names in the file (all shards /
/// the whole header). The check verifies the child-addressed config, the
/// quantized weight, and every sibling-addressed scale required by the
/// parsed config's format.
pub fn validate_layer_layout<S: AsRef<str>>(
    prefix: &str,
    config: &ComfyQuantConfig,
    tensor_names: &[S],
) -> Vec<ComfyQuantError> {
    let mut issues = Vec::new();
    let has = |suffix: &str| {
        let want = format!("{prefix}.{suffix}");
        tensor_names.iter().any(|n| n.as_ref() == want)
    };

    if !has("comfy_quant") {
        issues.push(ComfyQuantError::MissingConfig {
            prefix: prefix.to_string(),
            name: config_key(prefix),
        });
    }
    if !has("weight") {
        issues.push(ComfyQuantError::MissingWeight {
            prefix: prefix.to_string(),
            name: format!("{prefix}.weight"),
        });
    }
    for suffix in config.format.required_scale_suffixes() {
        if !has(suffix) {
            issues.push(ComfyQuantError::MissingScale {
                prefix: prefix.to_string(),
                name: format!("{prefix}.{suffix}"),
            });
        }
    }
    issues
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

    /// Phase 7.2: plain INT8 row-wise carries `per_row: true` in BOTH
    /// references (`fp8_conversion.py:583-584`, `tensor_quant.py:254-255`)
    /// — keyed off the scaling mode, not off ConvRot.
    #[test]
    fn family_a_int8_rowwise_per_row_exact_bytes() {
        assert_eq!(
            String::from_utf8(encode_comfy_quant_int8_rowwise("torch.bfloat16")).unwrap(),
            r#"{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "per_row": true}"#
        );
        assert_eq!(
            String::from_utf8(encode_comfy_quant_int8_rowwise("torch.float16")).unwrap(),
            r#"{"format": "int8_tensorwise", "orig_dtype": "torch.float16", "per_row": true}"#
        );
        // Distinct from the convrot blob (convrot keys present) and from
        // the tensor-mode blob (no per_row).
        assert_ne!(
            encode_comfy_quant_int8_rowwise("torch.bfloat16"),
            encode_comfy_quant_int8_convrot("torch.bfloat16", 256)
        );
        assert_ne!(
            encode_comfy_quant_int8_rowwise("torch.bfloat16"),
            encode_standard(INT8_TENSORWISE, "torch.bfloat16", None)
        );
    }

    /// Phase 7.1: the ConvRot blob is family A with the ConvRot tail keys, in
    /// `create_comfy_quant_tensor` order — `format`, `orig_dtype`, `convrot`,
    /// `convrot_groupsize`, `per_row` (no `group_size`: `int8_tensorwise` is
    /// not block-based, so the ConvRot group size lives in its own key).
    #[test]
    fn family_a_int8_convrot_exact_bytes() {
        let b = encode_comfy_quant_int8_convrot("torch.bfloat16", 256);
        assert_eq!(
            String::from_utf8(b).unwrap(),
            r#"{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "convrot": true, "convrot_groupsize": 256, "per_row": true}"#
        );
        // float16 policy + a different group size keep the same key order.
        assert_eq!(
            String::from_utf8(encode_comfy_quant_int8_convrot("torch.float16", 64)).unwrap(),
            r#"{"format": "int8_tensorwise", "orig_dtype": "torch.float16", "convrot": true, "convrot_groupsize": 64, "per_row": true}"#
        );
    }

    /// The ConvRot blob must not collide with the plain row-wise INT8 blob:
    /// the plain path emits no `convrot`/`per_row` keys at all (its goldens
    /// are already locked without them).
    #[test]
    fn convrot_blob_differs_from_plain_int8_rowwise() {
        assert_ne!(
            encode_comfy_quant_int8_convrot("torch.bfloat16", 256),
            encode_standard(INT8_TENSORWISE, "torch.bfloat16", None)
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

    // ---------------- parser: round-trips ---------------- //

    #[test]
    fn parse_round_trip_family_a() {
        for (fmt, group) in [
            (ComfyFormat::Int8Tensorwise, None),
            (ComfyFormat::Int8Blockwise, Some(128u32)),
            (ComfyFormat::Fp8Tensor, None),
            (ComfyFormat::Fp8Rowwise, None),
            (ComfyFormat::Fp8Blockwise, Some(128u32)),
        ] {
            let blob = encode_comfy_quant(fmt, "torch.bfloat16", group, None);
            let cfg = parse_blob(&blob).unwrap();
            assert_eq!(cfg.format, fmt);
            assert_eq!(cfg.orig_dtype, "torch.bfloat16");
            assert_eq!(cfg.group_size, group);
            assert_eq!(cfg.orig_shape, None);
            // Exact inverse: re-encoding reproduces the blob byte-for-byte.
            assert_eq!(cfg.to_blob(), blob);
        }
    }

    #[test]
    fn parse_round_trip_family_b() {
        for (fmt, shape) in [
            (ComfyFormat::Mxfp8, vec![256u64, 128]),
            (ComfyFormat::Nvfp4, vec![128, 64]),
            (ComfyFormat::Nvfp4, vec![384, 256]),
        ] {
            let blob = encode_comfy_quant(fmt, "torch.bfloat16", None, Some(&shape));
            let cfg = parse_blob(&blob).unwrap();
            assert_eq!(cfg.format, fmt);
            assert_eq!(cfg.orig_dtype, "torch.bfloat16");
            assert_eq!(cfg.group_size, fmt.fixed_group_size());
            assert_eq!(cfg.orig_shape.as_deref(), Some(shape.as_slice()));
            assert_eq!(cfg.to_blob(), blob);
        }
    }

    #[test]
    fn parse_known_golden_strings() {
        // Verbatim blob strings as they appear in the golden fixtures.
        let cfg = parse_blob(
            br#"{"format": "int8_blockwise", "orig_dtype": "torch.bfloat16", "group_size": 128}"#,
        )
        .unwrap();
        assert_eq!(cfg.format, ComfyFormat::Int8Blockwise);
        assert_eq!(cfg.group_size, Some(128));

        let cfg = parse_blob(
            br#"{"format": "mxfp8", "group_size": 32, "orig_dtype": "torch.bfloat16", "orig_shape": [256, 128]}"#,
        )
        .unwrap();
        assert_eq!(cfg.format, ComfyFormat::Mxfp8);
        assert_eq!(cfg.orig_shape.as_deref(), Some([256u64, 128].as_slice()));

        let cfg = parse_blob(
            br#"{"format": "nvfp4", "group_size": 16, "orig_dtype": "torch.bfloat16", "orig_shape": [128, 64]}"#,
        )
        .unwrap();
        assert_eq!(cfg.format, ComfyFormat::Nvfp4);
        assert_eq!(cfg.group_size, Some(16));
    }

    // ---------------- parser: negative cases ---------------- //

    #[test]
    fn parse_rejects_non_utf8() {
        assert_eq!(
            parse_blob(&[0xFF, 0xFE, 0x7B]),
            Err(ComfyQuantError::InvalidUtf8)
        );
    }

    #[test]
    fn parse_rejects_truncated_json() {
        let blob = br#"{"format": "mxfp8", "group_size": 32, "orig_dtype": "torch.bfloat16""#;
        match parse_blob(blob) {
            Err(ComfyQuantError::InvalidJson(_)) => {}
            other => panic!("expected InvalidJson, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_non_object() {
        assert_eq!(
            parse_blob(br#"[1, 2, 3]"#),
            Err(ComfyQuantError::NotAnObject)
        );
        assert_eq!(parse_blob(br#""nvfp4""#), Err(ComfyQuantError::NotAnObject));
    }

    #[test]
    fn parse_rejects_missing_format() {
        assert_eq!(
            parse_blob(br#"{"orig_dtype": "torch.bfloat16"}"#),
            Err(ComfyQuantError::MissingFormat)
        );
    }

    #[test]
    fn parse_rejects_unknown_format() {
        assert_eq!(
            parse_blob(br#"{"format": "convrot_w4a4", "orig_dtype": "torch.bfloat16"}"#),
            Err(ComfyQuantError::UnknownFormat("convrot_w4a4".to_string()))
        );
    }

    #[test]
    fn parse_rejects_wrong_types() {
        assert_eq!(
            parse_blob(br#"{"format": 42, "orig_dtype": "torch.bfloat16"}"#),
            Err(ComfyQuantError::BadFieldType {
                field: "format",
                expected: "string"
            })
        );
        assert_eq!(
            parse_blob(
                br#"{"format": "mxfp8", "orig_dtype": "torch.bfloat16", "group_size": "32"}"#
            ),
            Err(ComfyQuantError::BadFieldType {
                field: "group_size",
                expected: "non-negative integer"
            })
        );
        assert_eq!(
            parse_blob(
                br#"{"format": "mxfp8", "orig_dtype": "torch.bfloat16", "orig_shape": [1, "x"]}"#
            ),
            Err(ComfyQuantError::BadFieldType {
                field: "orig_shape",
                expected: "array of non-negative integers"
            })
        );
        // Negative group_size is not a u64.
        assert_eq!(
            parse_blob(br#"{"format": "mxfp8", "orig_dtype": "torch.bfloat16", "group_size": -1}"#),
            Err(ComfyQuantError::BadFieldType {
                field: "group_size",
                expected: "non-negative integer"
            })
        );
    }

    #[test]
    fn parse_family_b_requires_orig_shape() {
        assert_eq!(
            parse_blob(br#"{"format": "nvfp4", "group_size": 16, "orig_dtype": "torch.bfloat16"}"#),
            Err(ComfyQuantError::MissingField("orig_shape"))
        );
    }

    #[test]
    fn parse_tolerates_extra_fields() {
        // The reference tolerates extra keys (edit_comfy_quant adds/removes).
        let cfg = parse_blob(
            br#"{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "per_row": true}"#,
        )
        .unwrap();
        assert_eq!(cfg.format, ComfyFormat::Int8Tensorwise);
    }

    // ---------------- structural validator primitives ---------------- //

    #[test]
    fn layer_layout_ok_for_full_nvfp4_layer() {
        let cfg = ComfyQuantConfig {
            format: ComfyFormat::Nvfp4,
            orig_dtype: "torch.bfloat16".to_string(),
            group_size: Some(16),
            orig_shape: Some(vec![128, 64]),
            per_row: false,
        };
        let names = [
            "blocks.0.weight",
            "blocks.0.weight_scale",
            "blocks.0.weight_scale_2",
            "blocks.0.comfy_quant",
            "blocks.0.bias",
        ];
        assert!(validate_layer_layout("blocks.0", &cfg, &names).is_empty());
    }

    #[test]
    fn layer_layout_flags_missing_scale_2_for_nvfp4() {
        let cfg = ComfyQuantConfig {
            format: ComfyFormat::Nvfp4,
            orig_dtype: "torch.bfloat16".to_string(),
            group_size: Some(16),
            orig_shape: Some(vec![128, 64]),
            per_row: false,
        };
        let names = [
            "blocks.0.weight",
            "blocks.0.weight_scale",
            "blocks.0.comfy_quant",
        ];
        let issues = validate_layer_layout("blocks.0", &cfg, &names);
        assert_eq!(issues.len(), 1);
        assert_eq!(
            issues[0],
            ComfyQuantError::MissingScale {
                prefix: "blocks.0".to_string(),
                name: "blocks.0.weight_scale_2".to_string()
            }
        );
    }

    #[test]
    fn layer_layout_flags_missing_weight_and_config() {
        let cfg = ComfyQuantConfig {
            format: ComfyFormat::Int8Blockwise,
            orig_dtype: "torch.bfloat16".to_string(),
            group_size: Some(128),
            orig_shape: None,
            per_row: false,
        };
        let names = ["blocks.0.weight_scale"];
        let issues = validate_layer_layout("blocks.0", &cfg, &names);
        assert_eq!(issues.len(), 2);
        assert!(issues.contains(&ComfyQuantError::MissingConfig {
            prefix: "blocks.0".to_string(),
            name: "blocks.0.comfy_quant".to_string()
        }));
        assert!(issues.contains(&ComfyQuantError::MissingWeight {
            prefix: "blocks.0".to_string(),
            name: "blocks.0.weight".to_string()
        }));
    }

    #[test]
    fn layer_prefix_helpers() {
        assert_eq!(config_key("blocks.0"), "blocks.0.comfy_quant");
        assert_eq!(
            layer_prefix("model.double_block.0.comfy_quant"),
            Some("model.double_block.0")
        );
        assert_eq!(layer_prefix("blocks.0.weight"), None);
    }

    #[test]
    fn required_scale_suffixes() {
        assert_eq!(
            ComfyFormat::Nvfp4.required_scale_suffixes(),
            &["weight_scale", "weight_scale_2"]
        );
        assert_eq!(
            ComfyFormat::Mxfp8.required_scale_suffixes(),
            &["weight_scale"]
        );
        assert_eq!(
            ComfyFormat::Int8Blockwise.required_scale_suffixes(),
            &["weight_scale"]
        );
    }
}

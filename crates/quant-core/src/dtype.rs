//! Dtype handling for safetensors IO: header dtype-string enum.
//!
//! Mirrors `_TORCH_DTYPE_TO_STR` from the reference `tensor_quant.py`.
//! Numeric cast kernels (bf16/f16/fp8 ↔ f32) land in Phase 2; this module
//! currently covers what st_io needs: parse/serialize of header dtype strings.

use std::fmt;

/// Header dtype strings used by the reference implementation, plus UINT16
/// tolerance (numpy-written bf16 files use "U16" — see PHASE0_NOTES.md).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType {
    F64,
    F32,
    F16,
    Bf16,
    I64,
    I32,
    I16,
    I8,
    U8,
    U16,
    Bool,
}

impl DType {
    /// Byte width per element. Returns None for BOOL (1 byte but semantically
    /// distinct) — actually BOOL is 1 byte too, so all variants are Some.
    pub fn elem_size(&self) -> Option<u64> {
        Some(match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
        })
    }

    /// Map a raw tensor to its canonical output dtype string, mirroring
    /// `torch_dtype_to_str` behavior for dtypes the reference emits.
    pub fn from_header_str(s: &str) -> Option<DType> {
        Some(match s {
            "F64" => DType::F64,
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::Bf16,
            "I64" => DType::I64,
            "I32" => DType::I32,
            "I16" => DType::I16,
            "I8" => DType::I8,
            "U8" => DType::U8,
            // Tolerance quirk: safetensors.numpy writes bf16 tensors with a
            // "U16" header because numpy has no native bf16.
            "U16" => DType::U16,
            "BOOL" => DType::Bool,
            _ => return None,
        })
    }

    /// Canonical header string for this dtype.
    pub fn as_header_str(&self) -> &'static str {
        match self {
            DType::F64 => "F64",
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::Bf16 => "BF16",
            DType::I64 => "I64",
            DType::I32 => "I32",
            DType::I16 => "I16",
            DType::I8 => "I8",
            DType::U8 => "U8",
            DType::U16 => "U16",
            DType::Bool => "BOOL",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_header_str())
    }
}

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

/// Raw bf16 bits → f32 (exact: shift into the high 16 bits).
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Raw f32 → bf16 bits, round-to-nearest-even.
///
/// Matches `torch.Tensor.to(torch.bfloat16)` and the `half` crate's
/// `bf16::from_f32` semantics. NaN/Inf propagate.
pub fn f32_to_bf16_bits(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

/// Raw f32 bits → bf16 bits, round-to-nearest-even (no intermediate f32 value).
pub fn f32_bits_to_bf16_bits(bits: u32) -> u16 {
    half::bf16::from_f32(f32::from_bits(bits)).to_bits()
}

/// Raw f16 bits → f32 (exact).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Raw f32 → f16 bits, round-to-nearest-even.
pub fn f32_to_f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip_exact_values() {
        // Known vectors: 1.0, 0.5, -2.0 are exact in bf16.
        for (bits, val) in [(0x3F80u16, 1.0f32), (0x3F00, 0.5), (0xC000, -2.0)] {
            assert_eq!(bf16_bits_to_f32(bits), val);
            assert_eq!(f32_to_bf16_bits(val), bits);
        }
    }

    #[test]
    fn bf16_rne_truncation() {
        // 1.0 + 2^-8 = 1.00390625 is exactly halfway between two bf16 values;
        // round-to-nearest-even picks 0x3F80 (even mantissa), not 0x3F81.
        let v = f32::from_bits(0x3F80_8000);
        assert_eq!(f32_to_bf16_bits(v), 0x3F80);
        // Just above halfway rounds up to 0x3F81.
        let v2 = f32::from_bits(0x3F80_8001);
        assert_eq!(f32_to_bf16_bits(v2), 0x3F81);
        // Just below halfway truncates to 0x3F80.
        let v3 = f32::from_bits(0x3F80_7FFF);
        assert_eq!(f32_to_bf16_bits(v3), 0x3F80);
    }

    #[test]
    fn bf16_nan_inf_propagate() {
        assert_eq!(f32_to_bf16_bits(f32::INFINITY), 0x7F80);
        let nan = f32_to_bf16_bits(f32::NAN);
        assert_eq!(nan & 0x7F80, 0x7F80); // exponent all ones
        assert_ne!(nan & 0x007F, 0); // nonzero mantissa
    }

    #[test]
    fn f16_roundtrip_and_subnormals() {
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
        // Smallest f16 subnormal: 2^-24.
        assert_eq!(f16_bits_to_f32(0x0001), 2f32.powi(-24));
        // f32 2^-26 rounds to zero; 2^-25 exactly halfway -> ties-to-even -> 0x0000;
        // 1.5*2^-25 just above halfway -> 0x0001 (verified against torch).
        assert_eq!(f32_to_f16_bits(2f32.powi(-26)), 0x0000);
        assert_eq!(f32_to_f16_bits(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_f16_bits(2f32.powi(-25) * 1.5), 0x0001);
    }

    #[test]
    fn slice_casts_match_scalar() {
        let vals: [f32; 4] = [1.0, -0.75, 1234.5, 1e-30];
        let bf: Vec<u16> = vals.iter().map(|&v| f32_to_bf16_bits(v)).collect();
        // bf16 of 1234.5 is 1232.0 (mantissa truncated at 7 bits); 1e-30 stays
        // representable (bf16 min normal ≈ 1.18e-38); others exact.
        let expected = [1.0f32, -0.75, 1232.0, 9.984_021e-31];
        for (b, &e) in bf.iter().zip(&expected) {
            assert_eq!(bf16_bits_to_f32(*b), e);
        }
        // f16 of 1234.5 rounds to 1234.0 (torch-verified: bits 25810).
        assert_eq!(f32_to_f16_bits(1234.5), 25810u16);
        assert_eq!(f16_bits_to_f32(25810), 1234.0);
    }
}

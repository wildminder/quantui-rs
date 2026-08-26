//! Dtype handling for safetensors IO: header dtype-string enum.
//!
//! Mirrors `_TORCH_DTYPE_TO_STR` from the reference `tensor_quant.py`.
//! Numeric cast kernels (bf16/f16/fp8 ↔ f32) land in Phase 2; this module
//! currently covers what st_io needs: parse/serialize of header dtype strings.

use std::fmt;

/// Header dtype strings used by the reference implementation, plus UINT16
/// tolerance (numpy-written bf16 files use "U16" — see PHASE0_NOTES.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

// --- FP8 E4M3 (float8_e4m3fn) -------------------------------------------- //
//
// Verbatim port of c10::detail::fp8e4m3fn_from_fp32_value /
// fp8e4m3fn_to_fp32_value from torch's Float8_e4m3fn.h (the exact code path
// torch CPU uses for `.to(torch.float8_e4m3fn)` and back). A hand-rolled port
// was chosen over the `float8` crate so that saturation, NaN payload, and
// denormal rounding behavior are bit-identical to torch by construction.
//
// Format: s eeee mmm, bias 7, no infinity (0x7F/0xFF are NaN), max finite
// 448.0 (0x7E), smallest normal 2^-6, smallest subnormal 2^-9.

/// f32 → FP8 E4M3 bits (round-to-nearest-even, saturating).
///
/// Mirrors c10 exactly:
/// - NaN input → 0x7F (positive NaN payload, sign dropped)
/// - |x| ≥ 480.0 (first unrepresentable value) or inf → saturate to 0x7E (448)
/// - |x| < 2^-6 → denormal path via magic-add rounding
/// - otherwise RNE via mant_odd + 0x7FFFF bias trick; rounding carry into the
///   NaN pattern (0x7F) is saturated back to 0x7E
pub fn f32_to_fp8_e4m3_bits(f: f32) -> u8 {
    // Binary representation of 480.0f, the first value not representable:
    // 0 10000111 11100000000000000000000 → 1087 << 20.
    const FP8_MAX: u32 = 1087 << 20;
    // Magic for the denormal path: ((127 - 7) + (23 - 3) + 1) = 141.
    const DENORM_MASK: u32 = 141 << 23;

    let mut f_bits = f.to_bits();
    let sign = f_bits & 0x8000_0000;
    f_bits ^= sign; // work with |f|

    let result: u8 = if f_bits >= FP8_MAX {
        if f_bits > 0x7F80_0000 {
            0x7F // NaN input → NaN output
        } else {
            0x7E // finite overflow or +inf → saturate to max finite (448)
        }
    } else if f_bits < (121 << 23) {
        // |f| < 2^-6: below smallest normal → denormal representation.
        // Adding the magic float (exponent 141) and subtracting it back as
        // an integer performs round-to-nearest-even into the 3-bit grid.
        let shifted = f32::from_bits(f_bits) + f32::from_bits(DENORM_MASK);
        (shifted.to_bits() - DENORM_MASK) as u8
    } else {
        // Normal path: RNE via the mant_odd + rounding-bias trick.
        let mant_odd = ((f_bits >> 20) & 1) as u32;
        // Update exponent (7 - 127 bias delta) + rounding bias part 1.
        let mut b = f_bits.wrapping_add(((7u32.wrapping_sub(127)) << 23) + 0x7_FFFF);
        // Rounding bias part 2.
        b += mant_odd;
        let mut r = (b >> 20) as u8;
        // Rounding may carry into the NaN bit pattern; saturate to max.
        if r == 0x7F {
            r = 0x7E;
        }
        r
    };

    result | ((sign >> 24) as u8)
}

/// FP8 E4M3 bits → f32 (exact; no FP arithmetic).
///
/// Verbatim port of c10::detail::fp8e4m3fn_to_fp32_value.
pub fn fp8_e4m3_bits_to_f32(input: u8) -> f32 {
    let w = (input as u32) << 24;
    let sign = w & 0x8000_0000;
    let nonsign = w & 0x7FFF_FFFF;

    // Renorm shift for denormals (clz-based, mirroring c10's branching).
    let renorm_shift = if nonsign == 0 {
        32u32
    } else {
        nonsign.leading_zeros()
    };
    let renorm_shift = if renorm_shift > 4 { renorm_shift - 4 } else { 0 };

    // All-ones exponent+mantissa (NaN pattern) → exponent becomes 0xFF.
    let inf_nan_mask = (((nonsign as i32).wrapping_add(0x0100_0000) >> 8) as u32) & 0x7F80_0000;
    // Zero input → mask out everything.
    let zero_mask = ((nonsign as i32).wrapping_sub(1) >> 31) as u32;

    let result = sign
        | (((((nonsign << renorm_shift) >> 4) + ((0x78u32.wrapping_sub(renorm_shift)) << 23))
            | inf_nan_mask)
            & !zero_mask);
    f32::from_bits(result)
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

    // --- FP8 E4M3 -------------------------------------------------------- //

    #[test]
    fn fp8_e4m3_known_values() {
        // Exact representables.
        assert_eq!(f32_to_fp8_e4m3_bits(1.0), 0x38);
        assert_eq!(f32_to_fp8_e4m3_bits(-1.0), 0xB8);
        assert_eq!(f32_to_fp8_e4m3_bits(448.0), 0x7E);
        assert_eq!(f32_to_fp8_e4m3_bits(-448.0), 0xFE);
        assert_eq!(f32_to_fp8_e4m3_bits(0.0), 0x00);
        assert_eq!(f32_to_fp8_e4m3_bits(-0.0), 0x80);
        // Smallest normal 2^-6 = 0x08; smallest subnormal 2^-9 = 0x01.
        assert_eq!(f32_to_fp8_e4m3_bits(2f32.powi(-6)), 0x08);
        assert_eq!(f32_to_fp8_e4m3_bits(2f32.powi(-9)), 0x01);
        // Saturation: 480 is the first unrepresentable value; 464 rounds up
        // into the NaN pattern and must saturate back to 0x7E (torch-verified).
        assert_eq!(f32_to_fp8_e4m3_bits(480.0), 0x7E);
        assert_eq!(f32_to_fp8_e4m3_bits(464.0), 0x7E);
        assert_eq!(f32_to_fp8_e4m3_bits(1e30), 0x7E);
        assert_eq!(f32_to_fp8_e4m3_bits(f32::INFINITY), 0x7E);
        assert_eq!(f32_to_fp8_e4m3_bits(f32::NEG_INFINITY), 0xFE);
        // NaN → 0x7F / 0xFF: c10 preserves the sign bit on NaN input
        // (result |= sign >> 24), payload is dropped.
        assert_eq!(f32_to_fp8_e4m3_bits(f32::NAN), 0x7F);
        assert_eq!(f32_to_fp8_e4m3_bits(-f32::NAN), 0xFF);
        // Underflow: below half the smallest subnormal → 0.
        assert_eq!(f32_to_fp8_e4m3_bits(1e-40), 0x00);
        assert_eq!(f32_to_fp8_e4m3_bits(2f32.powi(-10)), 0x00);
    }

    #[test]
    fn fp8_e4m3_rne_ties() {
        // 1.5*2^-9 is exactly halfway between 2^-9 (0x01) and 2*2^-9 (0x02);
        // ties-to-even picks 0x02 (even mantissa). 1.25*2^-9 → 0x01.
        assert_eq!(f32_to_fp8_e4m3_bits(1.5 * 2f32.powi(-9)), 0x02);
        assert_eq!(f32_to_fp8_e4m3_bits(1.25 * 2f32.powi(-9)), 0x01);
        // Normal-range tie: 1.0 + 0.5*ulp(1.0)=1.0625 is halfway between 1.0
        // (0x38) and 1.125 (0x39); ties-to-even keeps 0x38.
        assert_eq!(f32_to_fp8_e4m3_bits(1.0625), 0x38);
        assert_eq!(f32_to_fp8_e4m3_bits(1.1875), 0x3A); // tie → even 0x3A
    }

    #[test]
    fn fp8_e4m3_roundtrip_all_256_codes() {
        // Every code decodes to a finite f32 and re-encodes to itself
        // (except the two NaN codes 0x7F/0xFF, which decode to NaN).
        for code in 0u8..=255 {
            let v = fp8_e4m3_bits_to_f32(code);
            if code & 0x7F == 0x7F {
                assert!(v.is_nan(), "code {code:#04x} should decode to NaN");
                // Re-encoding NaN reproduces the sign-bearing NaN code.
                assert_eq!(f32_to_fp8_e4m3_bits(v), code, "code {code:#04x}");
            } else {
                assert!(v.is_finite(), "code {code:#04x} decoded non-finite");
                assert_eq!(f32_to_fp8_e4m3_bits(v), code, "code {code:#04x}");
            }
        }
        // Spot-check decoded magnitudes.
        assert_eq!(fp8_e4m3_bits_to_f32(0x7E), 448.0);
        assert_eq!(fp8_e4m3_bits_to_f32(0x08), 2f32.powi(-6));
        assert_eq!(fp8_e4m3_bits_to_f32(0x01), 2f32.powi(-9));
        assert_eq!(fp8_e4m3_bits_to_f32(0x00), 0.0);
        assert_eq!(fp8_e4m3_bits_to_f32(0x80), -0.0);
        assert!(fp8_e4m3_bits_to_f32(0x7F).is_nan());
        assert!(fp8_e4m3_bits_to_f32(0xFF).is_nan());
    }
}

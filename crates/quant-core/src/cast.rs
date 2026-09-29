//! dtype casting for the `cast` subcommand — a **lossless-or-error** contract.
//!
//! This module is deliberately NOT a quantizer. It performs no scaling, no
//! rounding search, and writes no metadata. It exists so a user can turn a
//! model's float tensors into `bf16` / `f16` / `f32` without the output
//! claiming to be a ComfyUI-quantized model (which would load as broken).
//!
//! # The four cases
//!
//! | case | behaviour |
//! |---|---|
//! | same dtype | **verbatim byte copy.** No decode, no re-encode. |
//! | wider → narrower float (`f32`→`bf16`/`f16`) | RNE narrowing via [`crate::dtype`] |
//! | narrower → wider float (`bf16`/`f16`→`f32`) | exact widening |
//! | narrowing that OVERFLOWS (`bf16`/`f32`→`f16` above 65504) | **error**, never `Inf` |
//! | non-float (`I64`/`U8`/`Bool`/`F8_E4M3`/…) | pass through untouched |
//! | `f64` source | **rejected** (see below) |
//!
//! # Why the same-dtype case must copy bytes
//!
//! A `bf16`→`bf16` round trip *through* `f32` is numerically lossless — every
//! `bf16` value is exactly representable in `f32`. But it is not a **byte**
//! copy, and the byte-copy guarantee is the entire point of that fast path:
//!
//! - a signalling-NaN payload can be quieted by the `f32` round trip, changing
//!   the stored bits of a value nobody intended to change;
//! - it costs a decode + encode per element for no numerical benefit.
//!
//! So `cast_tensor` returns [`CastOutcome::Verbatim`] and the caller writes the
//! source slice straight through. `same_dtype_payload_is_byte_identical` and
//! `verbatim_path_preserves_nan_payload_that_a_round_trip_would_erase` in
//! `tests/cast_dtype.rs` guard this.
//!
//! # Why overflow is an error and not saturation
//!
//! `bf16`'s range vastly exceeds `f16`'s (`bf16` max ≈ 3.39e38, `f16` max
//! 65504). Every value above 65504 has **no** `f16` representation, and
//! `half::f16::from_f32` returns `Inf` for them.
//!
//! Emitting `Inf` where the user expected a number is **silent data
//! corruption**: the file is written, the run "succeeds", and the damage only
//! appears as garbage activations at inference time. So this module returns
//! [`CastError::F16Overflow`] naming the tensor and element index, and the CLI
//! exits 1.
//!
//! Subnormal-range collapse is *not* an error: `f16` has a much smaller
//! exponent range than `bf16` on the low side too, but a tiny value flushing
//! to `±0.0` is the standard, expected, information-losing-but-honest `f16`
//! narrowing that every framework performs silently. Only *overflow* — a
//! value that becomes `Inf`, and therefore poisons downstream arithmetic — is
//! refused.
//!
//! # Why `f64` sources are rejected outright
//!
//! `f64` → `bf16` cannot be done in one rounding. Going through `f32` first is
//! a **double rounding** whose result depends on that intermediate step: a
//! value can round to a `f32` neighbour and then to a *different* `bf16`
//! neighbour than it would have directly. Silently picking one is how you get
//! a reproducibility bug nobody can find. There is no `f64` input in the target
//! models, so the honest answer is to refuse rather than guess.
//!
//! # Non-circularity of the rounding tests
//!
//! `tests/cast_dtype.rs` pins **literal expected bit patterns** for the
//! rounding cases, chosen to sit exactly on a tie — the only input that
//! discriminates round-to-nearest-even from truncation. The tests never call
//! `f32_to_bf16_bits` to compute what they expect.

use crate::dtype::{bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits, DType};

/// Largest finite `f16`. `half::f16::MAX` is 65504.0; the value above it
/// rounds to `Inf` under RNE, so this is the exact overflow threshold.
const F16_MAX: f32 = 65504.0;

/// Why a cast could not be performed.
///
/// Every variant is a **data** error (CLI exit 1), never a usage error: the
/// command line was well-formed, the model just cannot be represented.
#[derive(Debug, Clone, PartialEq)]
pub enum CastError {
    /// A value above `f16`'s finite range has no `f16` representation.
    /// `half::f16::from_f32` would silently return `Inf`.
    F16Overflow {
        /// Tensor name, so the user can find the offending weight.
        tensor: String,
        /// Index of the first offending element within the tensor.
        element: usize,
        /// The value that does not fit.
        value: f32,
    },
    /// The source tensor's dtype is `F64`, which this command refuses rather
    /// than double-round.
    UnsupportedSourceDtype {
        /// Tensor name.
        tensor: String,
        /// Always [`DType::F64`] today; named so the error reads clearly if the
        /// policy ever widens.
        dtype: DType,
    },
    /// The declared element count does not match the payload length for the
    /// dtype — i.e. the source file is internally inconsistent.
    PayloadSizeMismatch {
        /// Tensor name.
        tensor: String,
        /// **Bytes** implied by `shape` × the source dtype's element size.
        expected: usize,
        /// Bytes actually present.
        got: usize,
    },
}

impl std::fmt::Display for CastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CastError::F16Overflow {
                tensor,
                element,
                value,
            } => write!(
                f,
                "tensor {tensor:?}: element {element} is {value}, which overflows \
                 f16 (max finite 65504) and would be written as Inf. \
                 Refusing to emit Inf where a number was expected."
            ),
            CastError::UnsupportedSourceDtype { tensor, dtype } => write!(
                f,
                "tensor {tensor:?}: source dtype {dtype} is not supported. \
                 f64 sources are refused because f64->f32->bf16 is a double \
                 rounding whose result depends on the intermediate step. \
                 Cast from f32/bf16/f16 instead."
            ),
            CastError::PayloadSizeMismatch {
                tensor,
                expected,
                got,
            } => write!(
                f,
                "tensor {tensor:?}: header implies {expected} bytes but the \
                 payload holds {got}. The source file is inconsistent."
            ),
        }
    }
}

impl std::error::Error for CastError {}

/// What `cast_tensor` did with one tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastOutcome {
    /// Source dtype was already the target: the payload was NOT touched.
    Verbatim,
    /// The payload was re-encoded (or validated and passed through).
    Converted,
}

impl CastOutcome {
    /// `true` for [`CastOutcome::Verbatim`].
    pub fn is_verbatim(self) -> bool {
        matches!(self, CastOutcome::Verbatim)
    }
}

/// Number of elements implied by `shape`, or `None` on overflow.
fn element_count(shape: &[u64]) -> Option<u64> {
    shape.iter().try_fold(1u64, |a, &d| a.checked_mul(d))
}

/// Is this a dtype the command accepts as a float source?
///
/// `F64` is excluded — see the module docs.
fn is_supported_float_source(d: DType) -> bool {
    matches!(d, DType::F32 | DType::F16 | DType::Bf16)
}

/// Is this a float dtype the command can produce?
fn is_valid_target(d: DType) -> bool {
    matches!(d, DType::Bf16 | DType::F16 | DType::F32)
}

/// Cast one tensor payload from `src` (`src_dtype`) to `target`.
///
/// Returns the new payload plus what was done. `src` is returned **by slice
/// reference** in the verbatim case, so a no-op cast allocates nothing.
///
/// `shape` is used only to cross-check the payload length; the byte length is
/// what is actually converted.
pub fn cast_tensor(
    name: &str,
    src: &[u8],
    src_dtype: DType,
    target: DType,
    shape: &[u64],
) -> Result<(Vec<u8>, CastOutcome), CastError> {
    if !is_valid_target(target) {
        // Defensive: the CLI only ever passes Bf16/F16/F32. Reaching this is a
        // programming error, not a user error.
        return Err(CastError::UnsupportedSourceDtype {
            tensor: name.to_string(),
            dtype: target,
        });
    }

    // Non-float dtypes pass through untouched — including their original
    // header spelling, which the caller preserves separately.
    if !is_supported_float_source(src_dtype) {
        if src_dtype == DType::F64 {
            return Err(CastError::UnsupportedSourceDtype {
                tensor: name.to_string(),
                dtype: src_dtype,
            });
        }
        return Ok((src.to_vec(), CastOutcome::Verbatim));
    }

    // Same dtype: the verbatim fast path. No decode, no re-encode.
    if src_dtype == target {
        return Ok((src.to_vec(), CastOutcome::Verbatim));
    }

    let elems = element_count(shape).unwrap_or(0);
    let esz = src_dtype.elem_size().unwrap_or(0) as usize;
    let expected = elems as usize * esz;
    if expected != src.len() {
        return Err(CastError::PayloadSizeMismatch {
            tensor: name.to_string(),
            expected,
            got: src.len(),
        });
    }

    let out = match (src_dtype, target) {
        // f32 -> bf16: RNE narrowing.
        (DType::F32, DType::Bf16) => {
            let mut v = Vec::with_capacity(src.len() / 2);
            for c in src.chunks_exact(4) {
                let bits = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                v.extend_from_slice(&f32_to_bf16_bits(f32::from_bits(bits)).to_le_bytes());
            }
            v
        }
        // f32 -> f16: RNE narrowing, with the overflow refusal.
        (DType::F32, DType::F16) => {
            let mut v = Vec::with_capacity(src.len() / 2);
            for (i, c) in src.chunks_exact(4).enumerate() {
                let bits = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let x = f32::from_bits(bits);
                check_f16_overflow(name, i, x)?;
                v.extend_from_slice(&f32_to_f16_bits(x).to_le_bytes());
            }
            v
        }
        // bf16 -> f16: RNE narrowing, with the overflow refusal. This is the
        // case the target model hits if a user asks for f16 on a bf16 model.
        (DType::Bf16, DType::F16) => {
            let mut v = Vec::with_capacity(src.len());
            for (i, c) in src.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                let x = bf16_bits_to_f32(bits);
                check_f16_overflow(name, i, x)?;
                v.extend_from_slice(&f32_to_f16_bits(x).to_le_bytes());
            }
            v
        }
        // bf16 -> f32: exact widening (every bf16 is exactly an f32).
        (DType::Bf16, DType::F32) => {
            let mut v = Vec::with_capacity(src.len() * 2);
            for c in src.chunks_exact(2) {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                v.extend_from_slice(&bf16_bits_to_f32(bits).to_bits().to_le_bytes());
            }
            v
        }
        // f16 -> f32: exact widening.
        (DType::F16, DType::F32) => {
            let mut v = Vec::with_capacity(src.len() * 2);
            for c in src.chunks_exact(2) {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                v.extend_from_slice(&f16_bits_to_f32(bits).to_bits().to_le_bytes());
            }
            v
        }
        // f16 -> bf16: RNE narrowing. f16's range is a strict subset of bf16's,
        // so this can never overflow.
        (DType::F16, DType::Bf16) => {
            let mut v = Vec::with_capacity(src.len());
            for c in src.chunks_exact(2) {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                v.extend_from_slice(&f32_to_bf16_bits(f16_bits_to_f32(bits)).to_le_bytes());
            }
            v
        }
        (DType::F64, _) => unreachable!("F64 rejected above"),
        _ => unreachable!("non-float dtypes returned above"),
    };

    Ok((out, CastOutcome::Converted))
}

/// Refuse a value that would become `Inf` in `f16`.
///
/// `x.is_finite()` is load-bearing, not incidental — it carves out the two
/// non-finite cases, and both are correct to pass through:
///
/// - **NaN** propagates to an `f16` NaN. Legitimate, and unchanged by the cast.
/// - **`Inf`** maps to `f16` `Inf`. `f16` has a true infinity, so this is an
///   *exact*, lossless mapping — refusing it would reject a conversion that
///   loses nothing.
///
/// Only a **finite** value above [`F16_MAX`] is refused: that is the case where
/// a real number would silently become `Inf`, which is unrecoverable.
fn check_f16_overflow(name: &str, element: usize, x: f32) -> Result<(), CastError> {
    if x.is_finite() && x.abs() > F16_MAX {
        return Err(CastError::F16Overflow {
            tensor: name.to_string(),
            element,
            value: x,
        });
    }
    Ok(())
}

/// The dtype a cast target should be written as in the output header.
///
/// `U16`-spelled bf16 (the numpy tolerance quirk, see
/// [`DType::from_header_str`]) is normalised to the canonical `BF16` when it is
/// the source dtype, but non-float tensors keep their ORIGINAL spelling — the
/// caller passes `src.dtype_raw` through for those.
pub fn output_dtype_for(src: DType, target: DType) -> DType {
    if is_supported_float_source(src) {
        target
    } else {
        src
    }
}

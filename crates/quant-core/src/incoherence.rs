//! QuaRot incoherence diagnostic (QuaRot Eq. 3).
//!
//! This is a **measurement**, not a gate. It exists so a caller can ask "how
//! incoherent is this weight matrix?" and report the number. **Nothing in the
//! crate calls it**, which is deliberate: it is inert until a consumer opts
//! in, so it cannot perturb any output bytes.
//!
//! # What the metric is
//!
//! For a weight matrix `W` of shape `(m, n)` with `mn` entries:
//!
//! ```text
//!   mu = max |W_ij|
//!   gamma = mu / (||W||_F / sqrt(m*n))
//! ```
//!
//! `||W||_F / sqrt(mn)` is the root-mean-square magnitude, so `gamma` is the
//! ratio of the worst entry to the average magnitude. It is always `>= 1`,
//! with equality **iff** all `|W_ij|` are equal.
//!
//! # What this module deliberately does NOT claim
//!
//! **A single Hadamard rotation does not reliably reduce `gamma`.** Measured
//! on this crate's own `convrot_int8_weight` path with a right-multiplied
//! Hadamard:
//!
//! ```text
//!   128x128  before = 3.4959  after = 3.8311  ratio = 1.0959   <- goes UP
//!    64x64   before = 3.4216  after = 3.3852  ratio = 0.9894
//! ```
//!
//! QuaRot's incoherence bound is a worst-case *bound*, and one Hadamard
//! application on a random Gaussian does not monotonically reduce `max|W|`.
//! So any test asserting "rotation reduces gamma" is **flaky by construction**
//! and must not be written. The tests assert the metric's own algebra only.
//!
//! # Precision
//!
//! The Frobenius norm accumulates in `f64` even though the inputs are `f32`.
//! An `f32` accumulator over `mn` terms loses precision on large tensors, and
//! this is a diagnostic where a quietly wrong number is worse than no number.

/// Frobenius (Euclidean) norm of `w`, accumulated in `f64`.
///
/// Returns `0.0` for an empty slice.
pub fn frobenius_norm(w: &[f32]) -> f64 {
    w.iter()
        .map(|&v| {
            let v = f64::from(v);
            v * v
        })
        .sum::<f64>()
        .sqrt()
}

/// The two QuaRot terms: `(mu, bound)` where `bound = ||W||_F / sqrt(mn)`.
///
/// Exposed separately because a diagnostic report usually wants to show *why*
/// `gamma` came out the way it did, and `(mu, bound)` says that directly while
/// the ratio alone does not.
///
/// Returns `None` for an empty tensor or a zero Frobenius norm — in both cases
/// `gamma` would be `0/0` or `x/0`, i.e. `NaN` or `inf`, and a `NaN` in a
/// report field is worse than an explicit absence.
pub fn incoherence_terms(w: &[f32], rows: usize, cols: usize) -> Option<(f64, f64)> {
    debug_assert_eq!(w.len(), rows * cols);
    if rows == 0 || cols == 0 || w.len() != rows * cols {
        return None;
    }
    let frob = frobenius_norm(w);
    if frob == 0.0 {
        return None;
    }
    let mu = w.iter().fold(0.0f64, |a, &v| a.max(f64::from(v.abs())));
    let bound = frob / ((rows as f64) * (cols as f64)).sqrt();
    Some((mu, bound))
}

/// QuaRot Eq. 3 incoherence proxy: `max|W| / (||W||_F / sqrt(mn))`.
///
/// Returns `None` for an empty tensor or `||W||_F == 0`.
///
/// Always `>= 1.0` for a non-empty tensor, with equality iff all `|W_ij|` are
/// equal.
pub fn incoherence(w: &[f32], rows: usize, cols: usize) -> Option<f64> {
    let (mu, bound) = incoherence_terms(w, rows, cols)?;
    Some(mu / bound)
}

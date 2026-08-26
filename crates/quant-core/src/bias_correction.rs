//! Bias correction (plan Phase 7.3) — port of reference
//! `stream_quant.py::_build_torch_calibration_cache` + `_correct_bias_torch`,
//! which themselves mirror ctq `formats/fp8_conversion.py`.
//!
//! Pipeline:
//! 1. One shared MT19937 generator seeded once with `calib_seed`
//!    (pinned 233983427); for each unique `in_features` among 2D `.weight`
//!    tensors, **in file order**, draw `randn(3072, in_features)` (normal_fill
//!    path — bit-exact per Phase 7.2 tests).
//! 2. For each quantized weight with a bias: `err = W_orig - W_dequant`;
//!    `corr = mean(X @ err.T, dim=0)`; output bias = `b_orig - corr`, cast
//!    back to the original bias dtype (F32 in all fixtures).
//!
//! The matmul is f32×f32→f32 accumulation. torch's CPU GEMM (oneDNN/MKL)
//! may use FMA/blocked accumulation orders that differ from a naive triple
//! loop; parity for the corrected-bias bytes is asserted by E2E golden
//! comparison, which pins the required order empirically.

use crate::torch_rng::TorchRng;

pub const CALIB_SAMPLES: usize = 3072;

/// Cache of simulated calibration data keyed by in_features.
pub struct CalibCache {
    entries: std::collections::HashMap<usize, Vec<f32>>,
}

impl CalibCache {
    /// Build the calibration cache: one shared generator seeded once, drawing
    /// randn(CALIB_SAMPLES, in_features) per unique in_features in file order.
    pub fn build(shapes_in_file_order: impl Iterator<Item = Option<usize>>, seed: u64) -> Self {
        let mut rng = TorchRng::manual_seed(seed);
        let mut entries = std::collections::HashMap::new();
        for n in shapes_in_file_order {
            let Some(n) = n else { continue };
            if entries.contains_key(&n) {
                continue;
            }
            entries.insert(n, rng.randn_f32(CALIB_SAMPLES * n));
        }
        Self { entries }
    }

    pub fn get(&self, in_features: usize) -> Option<&[f32]> {
        self.entries.get(&in_features).map(|v| v.as_slice())
    }
}

/// Mirror `_correct_bias_torch`: b_new = b_orig - mean(X @ (W_orig - W_dq).T, dim=0).
///
/// - `x`: calib data `(S, N)` row-major; `w_orig`, `w_dq`: `(M, N)` row-major;
///   `bias`: `(M,)`; all f32. Returns corrected bias `(M,)` as raw LE bytes.
pub fn correct_bias(
    x: &[f32],
    w_orig: &[f32],
    w_dq: &[f32],
    bias: &[f32],
    m: usize,
    n: usize,
) -> Vec<u8> {
    debug_assert_eq!(x.len(), CALIB_SAMPLES * n);
    debug_assert_eq!(w_orig.len(), m * n);
    debug_assert_eq!(w_dq.len(), m * n);
    debug_assert_eq!(bias.len(), m);

    // out_err[s, i] = sum_j X[s, j] * err[i, j]; corr[i] = mean over s.
    // Accumulate column sums directly to avoid materializing (3072, M).
    let mut corr = vec![0.0f32; m];
    for (i, c) in corr.iter_mut().enumerate() {
        let row = &w_orig[i * n..(i + 1) * n];
        let dq = &w_dq[i * n..(i + 1) * n];
        // dot over samples: sum_s X[s,j]*err[j], accumulated in f64 then
        // narrowed? No — torch computes mean(dim=0) of an f32 GEMM result,
        // i.e. f32 elementwise products summed by torch's reduction kernel.
        // We mirror the mathematical op; byte-parity is gated by E2E tests.
        let mut acc = 0.0f64;
        for s in 0..CALIB_SAMPLES {
            let xs = &x[s * n..(s + 1) * n];
            let mut d = 0.0f32;
            for j in 0..n {
                d += xs[j] * (row[j] - dq[j]);
            }
            acc += d as f64;
        }
        *c = (acc / CALIB_SAMPLES as f64) as f32;
    }

    let mut out_bytes = Vec::with_capacity(m * 4);
    for (b, c) in bias.iter().zip(corr.iter()) {
        out_bytes.extend_from_slice((b - c).to_le_bytes().as_slice());
    }
    out_bytes
}

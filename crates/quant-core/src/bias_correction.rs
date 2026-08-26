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
//! The numerics replicate torch CPU bit-exactly (validated by
//! `target/probe_sum.py` against torch 2.13.0+cpu on fixture + random data,
//! thread counts 1/3/12):
//!
//! - GEMM (`X @ err.T`, f32): oneDNN's sgemm microkernel resets its f32
//!   accumulator every 128 elements of K and sums the per-chunk partials
//!   left-to-right; within a chunk each step is a correctly-rounded FMA
//!   `part = round_f32(f64(x_k) * f64(e_k) + f64(part))`. Bit-exact vs torch
//!   2.13.0+cpu at S=3072 for K in {64, 128, 256}.
//! - `sum(dim=0)`: dispatches to `cascade_sum` in
//!   `aten/src/ATen/native/cpu/SumKernel.cpp` (NOT Reduce.h). With
//!   VF = Vectorized<f32>::size() = 8 (AVX2):
//!     - C >= VF -> `vectorized_outer_sum`: columns in 32-groups use
//!       `multi_row_sum` (per-column 4-level plain-f32 cascade); remaining
//!       columns use `row_sum<vacc_t>` (ilp_factor=4 strided partials).
//!     - C < VF -> `scalar_outer_sum`: columns in 4-groups use the cascade;
//!       last C%4 columns use scalar `row_sum` (same ilp structure).
//! - Column parallelization splits columns only; per-column order unchanged.
//! - mean = cascade sum then f32 division by S; bias update b - corr in f32.

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

/// Emulates `multi_row_sum<scalar_t, nl>` from SumKernel.cpp: a 4-level
/// plain-f32 cascade over `get(i, lane)` elements, i ascending in `[0, size)`,
/// filling `out[lane]` with each lane's partial (acc[0]) sum.
///
/// level_power = max(4, bit_length(size-1) // 4); all arithmetic is plain f32,
/// matching the C++ scalar accumulation exactly.
pub fn multi_row_cascade(
    get: impl Fn(usize, usize) -> f32,
    nrows: usize,
    size: usize,
    out: &mut [f32],
) {
    const NL: usize = 4;
    // bit_length(size-1); must use the *same integer width* for the BITS
    // constant and leading_zeros, otherwise on a 64-bit target the u32 cast
    // below would under-count by 32 (64 - 20 vs 32 - 20). This mirrors
    // C++ utils::CeilLog2(size) used by SumKernel.cpp::multi_row_sum.
    let bits = if size == 0 {
        0usize
    } else {
        (usize::BITS - (size - 1).leading_zeros()) as usize
    };
    let lp = usize::max(4, bits / NL);
    let ls = 1usize << lp;
    let lm = ls - 1;
    let mut acc = vec![[0.0f32; NL_MAX_LANES]; NL];
    let mut i = 0usize;
    while i + ls <= size {
        for _ in 0..ls {
            for k in 0..nrows {
                acc[0][k] += get(i, k);
            }
            i += 1;
        }
        for j in 1..NL {
            for k in 0..nrows {
                acc[j][k] += acc[j - 1][k];
                acc[j - 1][k] = 0.0;
            }
            if (i & (lm << (j * lp))) != 0 {
                break;
            }
        }
    }
    while i < size {
        for k in 0..nrows {
            acc[0][k] += get(i, k);
        }
        i += 1;
    }
    for j in 1..NL {
        for k in 0..nrows {
            acc[0][k] += acc[j][k];
        }
    }
    for k in 0..nrows {
        out[k] = acc[0][k];
    }
}
const NL_MAX_LANES: usize = 4;

/// Column summed via `multi_row_sum` (per-column independent cascade).
pub fn plain_col_sum(col: &[f32]) -> f32 {
    let mut out = [0.0f32; 1];
    multi_row_cascade(|i, _| col[i], 1, col.len(), &mut out);
    out[0]
}

/// Column summed via `row_sum<scalar_t>`: ilp_factor=4 strided partials.
///
/// p[k] = cascade over elements {i*4 + k}, i in [0, size/4); tail elements
/// added ascending into p0; result = p0; p0 += p1; p0 += p2; p0 += p3.
fn ilp_col_sum(col: &[f32]) -> f32 {
    let size = col.len();
    let size_ilp = size / 4;
    let mut ps = [0.0f32; 4];
    multi_row_cascade(|i, k| col[i * 4 + k], 4, size_ilp, &mut ps);
    let mut s = ps[0];
    for &v in &col[size_ilp * 4..] {
        s += v;
    }
    for &p in &ps[1..4] {
        s += p;
    }
    s
}

/// Bit-exact emulation of torch `.sum(dim=0)` for contiguous f32 input.
///
/// Dispatch (SumKernel.cpp `cascade_sum`, VF = Vectorized<f32>::size() = 8):
/// - C >= 8 (`vectorized_outer_sum`): columns in 32-groups use the plain
///   per-column cascade (`multi_row_sum`, 4 vectors); remaining columns use
///   `row_sum<vacc_t>` (ilp strided partials).
/// - C < 8 (`scalar_outer_sum`): columns in 4-groups use the cascade; last
///   C%4 columns use scalar `row_sum` (same ilp structure).
///
/// Thread parallelization splits columns only; per-column order is unchanged.
fn emulate_sum_dim0(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(mat.len(), rows * cols);
    let mut out = vec![0.0f32; cols];
    let at = |j: usize, i: usize| mat[i * cols + j];
    let mut j = 0usize;
    if cols >= 8 {
        while j + 32 <= cols {
            for jj in j..j + 32 {
                let c: Vec<f32> = (0..rows).map(|i| at(jj, i)).collect();
                out[jj] = plain_col_sum(&c);
            }
            j += 32;
        }
        while j < cols {
            let c: Vec<f32> = (0..rows).map(|i| at(j, i)).collect();
            out[j] = ilp_col_sum(&c);
            j += 1;
        }
    } else {
        while j + 4 <= cols {
            for jj in j..j + 4 {
                let c: Vec<f32> = (0..rows).map(|i| at(jj, i)).collect();
                out[jj] = plain_col_sum(&c);
            }
            j += 4;
        }
        while j < cols {
            let c: Vec<f32> = (0..rows).map(|i| at(j, i)).collect();
            out[j] = ilp_col_sum(&c);
            j += 1;
        }
    }
    out
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

    // GEMM stage: out_err[s, i] = sum_j X[s, j] * err[i, j], emulating
    // oneDNN's f32 sgemm microkernel. The kernel does NOT run one unbroken
    // FMA chain over K: it resets the f32 accumulator every 128 elements of
    // K (GEMM_K_CHUNK) and sums the per-chunk partials left-to-right. Within
    // a chunk each step is a correctly-rounded FMA (product exact in f64,
    // added to the previous rounded f32 partial, rounded once back to f32).
    // For K <= 128 this degenerates to a single chain. Validated bit-exact
    // vs torch 2.13.0+cpu at S=3072 for K in {64, 128, 256}
    // (target/probe_gemm_final.py). Materializing (S, M) costs ~3MB.
    const GEMM_K_CHUNK: usize = 128;
    let mut out_err = vec![0.0f32; CALIB_SAMPLES * m];
    for s in 0..CALIB_SAMPLES {
        let xs = &x[s * n..(s + 1) * n];
        for i in 0..m {
            let ro = &w_orig[i * n..(i + 1) * n];
            let rd = &w_dq[i * n..(i + 1) * n];
            let mut acc = 0.0f32;
            let mut j0 = 0usize;
            let mut first = true;
            while j0 < n {
                let j1 = (j0 + GEMM_K_CHUNK).min(n);
                let mut part = 0.0f32;
                for j in j0..j1 {
                    let e = ro[j] - rd[j]; // err tensor itself is an f32 torch op
                    part = ((xs[j] as f64) * (e as f64) + (part as f64)) as f32;
                }
                acc = if first {
                    part
                } else {
                    ((acc as f64) + (part as f64)) as f32
                };
                first = false;
                j0 = j1;
            }
            out_err[s * m + i] = acc;
        }
    }

    // Reduction stage: sum(dim=0) via the cascade algorithm, then mean =
    // elementwise f32 division by S; bias update b - corr in plain f32.
    let sums = emulate_sum_dim0(&out_err, CALIB_SAMPLES, m);

    if std::env::var("BC_DEBUG").is_ok() {
        let mut bytes: Vec<u8> = x[..CALIB_SAMPLES * n.min(128)].iter().flat_map(|v| v.to_le_bytes()).collect();
        bytes.extend(out_err.iter().flat_map(|v| v.to_le_bytes()));
        bytes.extend(sums.iter().flat_map(|v| v.to_le_bytes()));
        std::fs::write("out_err_rust.bin", &bytes).unwrap();
    }

    let mut out_bytes = Vec::with_capacity(m * 4);
    for (b, s) in bias.iter().zip(sums.iter()) {
        let corr = s / CALIB_SAMPLES as f32;
        out_bytes.extend_from_slice((b - corr).to_le_bytes().as_slice());
    }
    out_bytes
}

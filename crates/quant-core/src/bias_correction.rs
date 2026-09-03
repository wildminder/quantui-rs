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

use crate::manifest::CalibOrder;
use crate::torch_rng::TorchRng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

pub const CALIB_SAMPLES: usize = 3072;

/// Cache of simulated calibration data keyed by in_features.
pub struct CalibCache {
    entries: std::collections::HashMap<usize, Vec<f32>>,
}

impl CalibCache {
    /// Build the calibration cache: one shared generator seeded once, drawing
    /// randn(CALIB_SAMPLES, in_features) per unique in_features.
    ///
    /// `pairs` yields `(tensor_name, Option<in_features>)` for every tensor in
    /// FILE order (`None` for non-2D-`.weight` tensors). The DRAW ORDER
    /// depends on the format family (plan §3.4, parity-critical):
    ///
    /// - [`CalibOrder::FileOrderAll2D`] (INT8/FP8, ctq
    ///   `convert_to_fp8_scaled`): iterate in file order, draw for every
    ///   `Some(n)`, dedup by `n` — exactly the legacy INT8 behavior.
    /// - [`CalibOrder::SortedWeightsOnly`] (MXFP8/NVFP4, ctq
    ///   `mxfp8_conversion.py:193-206` / `nvfp4_conversion.py`): collect the
    ///   `Some(n)` entries, sort by NAME, then draw per unique `n` in that
    ///   sorted order.
    ///
    /// Skipping a draw (or changing the order) shifts the shared RNG stream
    /// and corrupts every subsequent bias correction — the order is part of
    /// the byte-parity contract.
    pub fn build(
        pairs: impl Iterator<Item = (String, Option<usize>)>,
        order: CalibOrder,
        seed: u64,
    ) -> Self {
        let mut rng = TorchRng::manual_seed(seed);
        let mut entries = std::collections::HashMap::new();
        match order {
            CalibOrder::FileOrderAll2D => {
                for (_name, n) in pairs {
                    let Some(n) = n else { continue };
                    if entries.contains_key(&n) {
                        continue;
                    }
                    entries.insert(n, rng.randn_f32(CALIB_SAMPLES * n));
                }
            }
            CalibOrder::SortedWeightsOnly => {
                let mut weighted: Vec<(String, usize)> =
                    pairs.filter_map(|(name, n)| n.map(|n| (name, n))).collect();
                weighted.sort_by(|a, b| a.0.cmp(&b.0));
                for (_name, n) in weighted {
                    if entries.contains_key(&n) {
                        continue;
                    }
                    entries.insert(n, rng.randn_f32(CALIB_SAMPLES * n));
                }
            }
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
///
/// Public for `convrot.rs` — the ConvRot bias correction runs the same
/// cascade `sum(dim=0)` over its (S, m) diff matrix.
pub fn ilp_col_sum(col: &[f32]) -> f32 {
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
///
/// Bit-safe parallelization: each output column depends only on its own input
/// column (a self-contained cascade), and every column writes a distinct `out`
/// slot. The per-column algorithm (plain cascade vs ilp) is a pure function of
/// the column index `j` and `cols`, so scheduling columns across threads in any
/// order reproduces the sequential result bit-for-bit.
fn emulate_sum_dim0(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(mat.len(), rows * cols);
    let mut out = vec![0.0f32; cols];
    // Sequential dispatch groups: cols >= 8 -> 32-groups of plain cascade then
    // ilp tail; cols < 8 -> 4-groups of plain cascade then ilp tail. In both
    // cases the first `plain_count` columns use the plain per-column cascade
    // and the remainder use the ilp strided partials.
    let group = if cols >= 8 { 32 } else { 4 };
    let plain_count = (cols / group) * group;
    out.par_iter_mut().enumerate().for_each(|(j, slot)| {
        let c: Vec<f32> = (0..rows).map(|i| mat[i * cols + j]).collect();
        *slot = if j < plain_count {
            plain_col_sum(&c)
        } else {
            ilp_col_sum(&c)
        };
    });
    out
}

/// Mirror `_correct_bias_torch`: b_new = b_orig - mean(X @ (W_orig - W_dq).T, dim=0).
///
/// - `x`: calib data `(S, N)` row-major; `w_orig`, `w_dq`: `(M, N)` row-major;
///   `bias`: `(M,)`; all f32. Returns the corrected bias `(M,)` as **f32 values**.
/// - `cancel`: optional cooperative-cancellation flag (Ctrl-C). When set, the
///   GEMM stops scheduling new work promptly and this returns `None`. The
///   caller treats `None` as "abort the run at a tensor boundary".
///
/// The caller is responsible for casting the result back to the bias's original
/// dtype (mirroring the reference's `(b_orig - corr).to(dtype=bias.dtype)`):
/// for an F32 bias the cast is the identity (byte-exact), for a BF16/F16 bias
/// the f32 result is rounded to that dtype before serialization.
pub fn correct_bias(
    x: &[f32],
    w_orig: &[f32],
    w_dq: &[f32],
    bias: &[f32],
    m: usize,
    n: usize,
    cancel: Option<&AtomicBool>,
) -> Option<Vec<f32>> {
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
    // (target/probe_gemm_final.py).
    //
    // Parallelization (bit-safe): each output element out_err[s, i] owns its
    // own K-accumulation chain, independent of every other element, so we may
    // schedule the (s, i) work across threads in any order without changing a
    // single accumulated bit. We parallelize over `s` (rows of the output /
    // calibration rows), keeping the reference's inner loop order (i, then j)
    // intact inside each task — this preserves the original cache behavior
    // (the calibration row `xs` is loaded once and reused across all `i`).
    // Same order-independence argument as the block-amax pass (Phase 11.1).
    //
    // The `err` matrix (w_orig - w_dq, "the err tensor itself is an f32 torch
    // op") is materialized ONCE up front and shared read-only across tasks.
    // f32 subtraction is deterministic, so this is bit-identical to computing
    // `ro[j] - rd[j]` inline, and it removes (S-1)·m·n redundant subtractions.
    //
    // Cancellation: `try_for_each` short-circuits — once any task sees the
    // flag set it returns Err and rayon stops scheduling further tasks, so a
    // Ctrl-C takes effect within one task's worth of work (a few ms), not
    // after the whole (S, M) GEMM.
    const GEMM_K_CHUNK: usize = 128;
    let s_count = CALIB_SAMPLES;
    let err: Vec<f32> = (0..w_orig.len())
        .into_par_iter()
        .map(|k| w_orig[k] - w_dq[k])
        .collect();
    let mut out_err = vec![0.0f32; s_count * m];
    let gemm_cancelled = out_err
        .par_chunks_mut(m)
        .enumerate()
        .try_for_each(|(s, row)| {
            if let Some(flag) = cancel {
                if flag.load(Ordering::Relaxed) {
                    return Err(());
                }
            }
            let xs = &x[s * n..(s + 1) * n];
            for i in 0..m {
                let er = &err[i * n..(i + 1) * n];
                let mut acc = 0.0f32;
                let mut j0 = 0usize;
                let mut first = true;
                while j0 < n {
                    let j1 = (j0 + GEMM_K_CHUNK).min(n);
                    let mut part = 0.0f32;
                    for j in j0..j1 {
                        part = ((xs[j] as f64) * (er[j] as f64) + (part as f64)) as f32;
                    }
                    acc = if first {
                        part
                    } else {
                        ((acc as f64) + (part as f64)) as f32
                    };
                    first = false;
                    j0 = j1;
                }
                row[i] = acc;
            }
            Ok(())
        })
        .is_err();
    if gemm_cancelled {
        return None;
    }

    // Reduction stage: sum(dim=0) via the cascade algorithm, then mean =
    // elementwise f32 division by S; bias update b - corr in plain f32.
    let sums = emulate_sum_dim0(&out_err, s_count, m);

    if std::env::var("BC_DEBUG").is_ok() {
        let mut bytes: Vec<u8> = x[..CALIB_SAMPLES * n.min(128)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        bytes.extend(out_err.iter().flat_map(|v| v.to_le_bytes()));
        bytes.extend(sums.iter().flat_map(|v| v.to_le_bytes()));
        std::fs::write("out_err_rust.bin", &bytes).unwrap();
    }

    let mut out = Vec::with_capacity(m);
    for (b, s) in bias.iter().zip(sums.iter()) {
        let corr = s / CALIB_SAMPLES as f32;
        out.push(b - corr);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    /// A.3 gate: `CalibOrder::FileOrderAll2D` must reproduce the LEGACY
    /// build loop byte-for-byte (iterate in file order, draw per unique
    /// in_features, dedup by n). The reference below is an inline
    /// reimplementation of the pre-A.3 code path.
    #[test]
    fn calib_cache_file_order_matches_legacy() {
        // Fixture: mixed tensor names in a deliberately non-sorted file order,
        // with duplicate in_features and non-weight / non-2D entries (None).
        let pairs: Vec<(String, Option<usize>)> = vec![
            ("zeta.weight".into(), Some(96)),
            ("alpha.bias".into(), None),
            ("alpha.weight".into(), Some(64)),
            ("norm.weight".into(), Some(64)), // duplicate n=64 → dedup
            ("conv.weight".into(), None),     // 4D conv → None
            ("beta.weight".into(), Some(128)),
            ("gamma.weight".into(), Some(96)), // duplicate n=96 → dedup
        ];

        let got = CalibCache::build(
            pairs.clone().into_iter(),
            CalibOrder::FileOrderAll2D,
            233983427,
        );

        // Legacy algorithm (verbatim pre-A.3 loop).
        let mut rng = TorchRng::manual_seed(233983427);
        let mut legacy: std::collections::HashMap<usize, Vec<f32>> =
            std::collections::HashMap::new();
        for (_name, n) in pairs.iter() {
            let Some(n) = n else { continue };
            if legacy.contains_key(n) {
                continue;
            }
            legacy.insert(*n, rng.randn_f32(CALIB_SAMPLES * n));
        }

        assert_eq!(got.entries.len(), legacy.len(), "same unique-n count");
        for (n, want) in &legacy {
            let have = got.get(*n).expect("entry present");
            assert_eq!(have.len(), want.len());
            assert!(
                have.iter()
                    .zip(want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "FileOrderAll2D entry n={n} must be bit-identical to legacy"
            );
        }
    }

    /// A.3: `SortedWeightsOnly` draws per unique in_features in NAME-sorted
    /// order (ctq mxfp8/nvfp4 modules). When file order != sorted order the
    /// RNG stream differs from FileOrderAll2D — this test pins that semantic.
    #[test]
    fn calib_cache_sorted_weights_only_draws_in_sorted_order() {
        // File order puts zeta (n=96) BEFORE alpha (n=64); sorted order is
        // alpha, zeta. The two orders must therefore produce different bytes
        // for the same (name, n) set.
        let pairs: Vec<(String, Option<usize>)> = vec![
            ("zeta.weight".into(), Some(96)),
            ("alpha.weight".into(), Some(64)),
        ];

        let file_order = CalibCache::build(
            pairs.clone().into_iter(),
            CalibOrder::FileOrderAll2D,
            233983427,
        );
        let sorted = CalibCache::build(pairs.into_iter(), CalibOrder::SortedWeightsOnly, 233983427);

        // Both caches hold the same keys…
        assert_eq!(file_order.entries.len(), 2);
        assert_eq!(sorted.entries.len(), 2);
        // …but the draw order differs → different bytes for at least one n.
        let differs = [64usize, 96]
            .iter()
            .any(|n| file_order.get(*n).unwrap() != sorted.get(*n).unwrap());
        assert!(
            differs,
            "sorted order must shift the RNG stream vs file order"
        );

        // And SortedWeightsOnly must equal a reference that draws in sorted
        // name order: alpha (n=64) first, then zeta (n=96).
        let mut rng = TorchRng::manual_seed(233983427);
        let want_64 = rng.randn_f32(CALIB_SAMPLES * 64);
        let want_96 = rng.randn_f32(CALIB_SAMPLES * 96);
        assert!(sorted
            .get(64)
            .unwrap()
            .iter()
            .zip(&want_64)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(sorted
            .get(96)
            .unwrap()
            .iter()
            .zip(&want_96)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// Deterministic pseudo-random f32 in [-1, 1) from a 64-bit state (SplitMix64).
    fn next_f32(state: &mut u64) -> f32 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Map to [-1, 1).
        ((z >> 40) as f32 / (1u64 << 24) as f32) - 1.0
    }

    /// Naive SEQUENTIAL reference of the GEMM + reduction + bias update,
    /// mirroring the original (pre-parallelization) loop order exactly:
    /// s outer, i middle, j inner with 128-element K chunks. Used to prove the
    /// parallelized `correct_bias` is bit-identical.
    fn sequential_correct_bias(
        x: &[f32],
        w_orig: &[f32],
        w_dq: &[f32],
        bias: &[f32],
        m: usize,
        n: usize,
    ) -> Vec<f32> {
        const K_CHUNK: usize = 128;
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
                    let j1 = (j0 + K_CHUNK).min(n);
                    let mut part = 0.0f32;
                    for j in j0..j1 {
                        let e = ro[j] - rd[j];
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
        let sums = emulate_sum_dim0(&out_err, CALIB_SAMPLES, m);
        let mut out = Vec::with_capacity(m);
        for (b, s) in bias.iter().zip(sums.iter()) {
            let corr = s / CALIB_SAMPLES as f32;
            out.push(b - corr);
        }
        out
    }

    #[test]
    fn parallel_correct_bias_is_bit_identical_to_sequential() {
        // Small but non-trivial dims; n not a multiple of 128 to exercise the
        // K-chunk tail, m spanning both the 32-group and ilp-tail reduction paths.
        let (m, n) = (40usize, 300usize);
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let x: Vec<f32> = (0..CALIB_SAMPLES * n)
            .map(|_| next_f32(&mut state))
            .collect();
        let w_orig: Vec<f32> = (0..m * n).map(|_| next_f32(&mut state)).collect();
        let w_dq: Vec<f32> = (0..m * n).map(|_| next_f32(&mut state)).collect();
        let bias: Vec<f32> = (0..m).map(|_| next_f32(&mut state)).collect();

        let par = correct_bias(&x, &w_orig, &w_dq, &bias, m, n, None).expect("no cancel");
        let seq = sequential_correct_bias(&x, &w_orig, &w_dq, &bias, m, n);

        assert_eq!(par.len(), seq.len());
        for (i, (a, b)) in par.iter().zip(seq.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bias[{i}] differs: parallel={a} sequential={b}"
            );
        }
    }

    #[test]
    fn correct_bias_returns_none_when_cancelled() {
        let (m, n) = (8usize, 64usize);
        let mut state = 42u64;
        let x: Vec<f32> = (0..CALIB_SAMPLES * n)
            .map(|_| next_f32(&mut state))
            .collect();
        let w_orig: Vec<f32> = (0..m * n).map(|_| next_f32(&mut state)).collect();
        let w_dq: Vec<f32> = (0..m * n).map(|_| next_f32(&mut state)).collect();
        let bias: Vec<f32> = (0..m).map(|_| next_f32(&mut state)).collect();

        // Flag already set before the call → must abort and return None.
        let cancel = AtomicBool::new(true);
        let res = correct_bias(&x, &w_orig, &w_dq, &bias, m, n, Some(&cancel));
        assert!(res.is_none(), "pre-set cancel flag must abort the GEMM");

        // Flag not set → completes normally.
        let cancel = AtomicBool::new(false);
        let res = correct_bias(&x, &w_orig, &w_dq, &bias, m, n, Some(&cancel));
        assert!(res.is_some(), "unset flag must complete");
    }
}

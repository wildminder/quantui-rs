//! Bit-exactness proofs for the bias-correction GEMM inner product.
//!
//! # Why this module exists
//!
//! The bias-correction GEMM (`bias_correction.rs::correct_bias`, and the
//! mirrored pair in `convrot.rs::correct_bias_convrot`) is ~97% of a real
//! `quantize` run's wall-clock (see `docs/plans/2026-10-06-bias-gemm-...`).
//! Its inner loop is a software FMA written in f64:
//!
//! ```text
//! part = ((xs[j] as f64) * (er[j] as f64) + (part as f64)) as f32
//! ```
//!
//! which defeats hardware FMA and blocks vectorization (0.68 GFLOP/s
//! measured). Replacing it with `f32::mul_add` is only legal if the two
//! expressions are **bitwise identical**, because the emitted bytes are the
//! byte-parity oracle.
//!
//! # What this module proves
//!
//! Step 2.1 of the plan. Three claims, each tested directly against the
//! verbatim pre-change code, on adversarial inputs:
//!
//! 1. [`fma_equals_f64_emulation_bitwise`] — a fused `mul_add` chain equals
//!    the f64-emulated chain, per K-chunk AND across chunk boundaries.
//! 2. [`non_fused_form_is_proven_wrong`] — the non-fused `a * b + c` form
//!    DIFFERS. This is the negative control: it proves test 1 can fail, so
//!    test 1 is not vacuous.
//! 3. [`partial_add_is_plain_f32_add`] — the per-chunk partial add
//!    `((acc as f64) + (part as f64)) as f32` equals plain `acc + part`,
//!    so the partial-add line can also drop the f64 detour.
//!
//! If (1) fails on ANY input, the vectorization plan is void for that range
//! and must fall back (plan §10). If (2) never finds a difference, the proof
//! methodology is broken and must be re-derived before proceeding.
//!
//! # The trap this module guards against
//!
//! `SimdFloat::mul_add(self, a, b)` computes `self * a + b` — argument order
//! is SEMANTICS. `acc.mul_add(ec, x)` computes `acc*ec + x`, a different
//! expression, and silently produces wrong-but-plausible numbers. This was
//! hit once while building the plan (32768/32768 mismatches); the tests below
//! are written so that class of mistake cannot survive.
//!
//! # Step 2.2 measured negative (why production stays on the f64 detour)
//!
//! The scalar `mul_add` swap the plan originally specified for Step 2.2 is
//! bitwise correct but a measured NEGATIVE end-to-end (31 s -> 122 s on
//! `bench_bias_probe`): this crate ships baseline x86-64 (no `target-cpu`
//! in the release profile or CI), where `f32::mul_add` lowers to a software
//! `fmaf` that is 2.2x slower than the f64 detour it would replace.
//! Hardware FMA is therefore obtained ONLY through
//! `#[target_feature(enable = "avx2,fma")]` + `is_x86_feature_detected!`
//! dispatch (Step 2.4); the scalar f64 detour stays the portable fallback.

// Production code begins at Step 2.3. Step 2.1 was proofs-only (the module
// was empty in release builds); from Step 2.3 on it owns the packed-Err
// layout and (Step 2.4) the AVX2/FMA microkernel for the bias GEMM.

/// Deterministic LCG pseudo-random f32 in `[-1.0, 1.0)`.
/// No RNG crate: the proofs must run with zero new dependencies and be
/// reproducible bit-for-bit on every platform.
#[cfg(test)]
fn next_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32 / (1 << 24) as f32) - 1.0
}

use rayon::prelude::*;

/// Transpose `err` from row-major `[m, n]` to `[n, m]` so the `m` values
/// belonging to one K-element are CONTIGUOUS: in the packed microkernel,
/// one 256-bit load replaces 8 strided 4-byte loads.
///
/// Bit-exactness-neutral: a pure data movement, no arithmetic. The packed
/// layout only changes WHICH ADDRESS holds a value, never a computed bit.
///
/// The transpose is done ONCE per `correct_bias` call and reused across all
/// `S = 3072` calibration rows, so its cost is amortized ~3072x. Blocked:
/// output strips of `JR = 8` rows per rayon task (contiguous, disjoint
/// writes), walked in `IR = 32`-row tiles of `err` (contiguous reads, 8 hot
/// output lines per tile). A naive scalar transpose measured 85.8 ms for
/// 4096^2; this must stay far under the microkernel's cost (plan gate:
/// < 5% of the GEMM at S = 3072, asserted by `bench_transpose_err`).
pub fn transpose_err(err: &[f32], m: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(err.len(), m * n);
    let mut out = vec![0.0f32; m * n];
    if m == 0 || n == 0 {
        debug_assert!(out.is_empty());
        return out;
    }
    const IR: usize = 32;
    const JR: usize = 8;
    // Disjointness: task jb exclusively owns out rows [jb*JR, jb*JR + rows),
    // a contiguous strip of the output — no cross-task writes, no false
    // sharing within a strip's interior.
    out.par_chunks_mut(m * JR)
        .enumerate()
        .for_each(|(jb, chunk)| {
            let j0 = jb * JR;
            let rows = chunk.len() / m;
            let mut i = 0usize;
            while i < m {
                let i1 = (i + IR).min(m);
                for ii in i..i1 {
                    // Contiguous run of `rows` K-elements from err row ii.
                    let src = &err[ii * n + j0..ii * n + j0 + rows];
                    for (jj, v) in src.iter().enumerate() {
                        chunk[jj * m + ii] = *v;
                    }
                }
                i = i1;
            }
        });
    out
}

/// The K-block size the GEMM resets its accumulator at — MUST mirror
/// `bias_correction.rs::GEMM_K_CHUNK` and `convrot.rs::GEMM_K_CHUNK`
/// (asserted by `proofs::k_chunk_constant_mirrors_the_kernel`).
const GEMM_K_CHUNK: usize = 128;

/// The verbatim production kernel: a K-chunk-128 f64-emulated FMA chain,
/// chunk partials summed left-to-right. THE bit-exactness oracle for this
/// module — the microkernel, its tails, and the scalar fallback must all
/// reproduce this function's bits exactly (the line-level form is pinned by
/// `proofs::f64_emulation_line_is_verbatim_everywhere`).
fn scalar_dot_kchunk128(xs: &[f32], er: &[f32]) -> f32 {
    let n = xs.len();
    debug_assert_eq!(er.len(), n);
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
    acc
}

/// Scalar fallback over the ROW-major `err` — identical arithmetic and
/// cache behavior to the pre-SIMD production loop, so CPUs without
/// AVX2+FMA (and the aarch64 musl/darwin releases) run exactly what ran
/// before this module existed.
fn gemm_block_scalar(err: &[f32], x_blk: &[f32], m: usize, n: usize, out_blk: &mut [f32]) {
    let s_blk = out_blk.len() / m;
    debug_assert_eq!(err.len(), m * n);
    debug_assert_eq!(x_blk.len(), s_blk * n);
    for q in 0..s_blk {
        let xs = &x_blk[q * n..(q + 1) * n];
        let row = &mut out_blk[q * m..(q + 1) * m];
        for i in 0..m {
            let er = &err[i * n..(i + 1) * n];
            row[i] = scalar_dot_kchunk128(xs, er);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::GEMM_K_CHUNK;
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    /// One block of calibration rows against all `m` outputs, reading the
    /// PACKED err (`err_t`, [n, m] row-major from [`super::transpose_err`]).
    /// Rows are grouped internally into <= 8-row passes — any `s_blk` is
    /// accepted.
    ///
    /// Lanes are 8 consecutive output values `i`; the unrolled `q` axis is
    /// the calibration row. Per K-element `j`: one 256-bit `err_t` load
    /// (the transpose exists for exactly this load) plus one broadcast-FMA
    /// per calibration row. Each lane accumulates its OWN K-chain, so the
    /// arithmetic for output element (q, i) is element-for-element the
    /// scalar reference chain:
    ///
    /// - within a 128-element chunk: `vfmadd(x[j], err[j], vpart)` — the
    ///   fused FMA, proven bitwise-identical to the f64 detour in
    ///   [`super::proofs`] (Step 2.1);
    /// - across chunks: plain `_mm256_add_ps` — proven identical to the
    ///   f64 partial add (Step 2.1, `partial_add_is_plain_f32_add`);
    /// - the `first`-chunk structure is reproduced exactly (`vsum` starts
    ///   as the first chunk's partial, later chunks add left-to-right).
    ///
    /// NaN payloads are outside the parity contract (every golden fixture
    /// and real weight set is finite); the f64-detour paths remain the
    /// reference for pathological inputs.
    ///
    /// # Safety
    ///
    /// The caller must have verified AVX2 and FMA via
    /// `is_x86_feature_detected!` (done once in
    /// [`super::avx2_fma_detected`]) and must satisfy the length
    /// contracts debug-asserted below. All memory accesses derive from
    /// `j < n`, `i0 + 8 <= m`, `q < s8 <= 8` under the documented
    /// layouts, so every pointer stays in bounds.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn microkernel_8x8(
        x_blk: &[f32],
        err_t: &[f32],
        m: usize,
        n: usize,
        out_blk: &mut [f32],
    ) {
        debug_assert_eq!(err_t.len(), m * n);
        debug_assert_eq!(x_blk.len() / n, out_blk.len() / m);
        if out_blk.is_empty() || m == 0 {
            return;
        }
        const LANES: usize = 8;
        // Any block size is accepted: group into <= 8-row passes.
        for (xg, outg) in x_blk.chunks(LANES * n).zip(out_blk.chunks_mut(LANES * m)) {
            let s8 = outg.len() / m; // 1..=8
            debug_assert!(s8 <= 8);
            debug_assert_eq!(xg.len(), s8 * n);
            for i0 in (0..m).step_by(LANES) {
                if i0 + LANES <= m {
                    // Full 8-lane block: the packed path.
                    let mut vsum: [__m256; 8] = [_mm256_setzero_ps(); 8];
                    let mut vpart: [__m256; 8] = [_mm256_setzero_ps(); 8];
                    let mut first = true;
                    let mut j0 = 0usize;
                    while j0 < n {
                        let j1 = (j0 + GEMM_K_CHUNK).min(n);
                        for q in 0..s8 {
                            // Per-chunk accumulator reset: mirrors the scalar
                            // `part = 0.0` at every chunk head (invariant I2).
                            vpart[q] = _mm256_setzero_ps();
                        }
                        for j in j0..j1 {
                            // SAFETY: j < n and i0 + 8 <= m with
                            // err_t.len() == m*n, so the 8 loaded floats are
                            // in bounds. Unaligned by design (buffers are
                            // not guaranteed 32B-aligned).
                            let v_err = _mm256_loadu_ps(err_t.as_ptr().add(j * m + i0));
                            for q in 0..s8 {
                                // Slice indexing (bounds-checked) is
                                // deliberate: safe load of the broadcast
                                // value. set1 lowers to the same vbroadcast
                                // as broadcast_ss would.
                                let vx = _mm256_set1_ps(xg[q * n + j]);
                                // vfmadd(vx, v_err, vpart[q]): lane i
                                // computes x[q][j]*err[i][j] + part with
                                // ONE rounding — the fused FMA. Argument
                                // order is semantics (plan §0.3):
                                // vx*v_err + vpart, nothing else.
                                vpart[q] = _mm256_fmadd_ps(vx, v_err, vpart[q]);
                            }
                        }
                        for q in 0..s8 {
                            // Chunk-partial add: plain f32 add,
                            // left-to-right — the `first`-flag structure of
                            // the reference.
                            vsum[q] = if first {
                                vpart[q]
                            } else {
                                _mm256_add_ps(vsum[q], vpart[q])
                            };
                        }
                        first = false;
                        j0 = j1;
                    }
                    for q in 0..s8 {
                        // SAFETY: outg.len() == s8*m, q < s8, i0 + 8 <= m.
                        _mm256_storeu_ps(outg.as_mut_ptr().add(q * m + i0), vsum[q]);
                    }
                } else {
                    // i-tail: fewer than 8 output values remain. Scalar
                    // per element, reading the packed layout column-wise;
                    // bounded by 7 columns x s8 rows and shares cache lines
                    // with the block above. Same reference order, same
                    // f64 emulation.
                    for i in i0..m {
                        for q in 0..s8 {
                            let xs = &xg[q * n..(q + 1) * n];
                            let mut acc = 0.0f32;
                            let mut j0 = 0usize;
                            let mut first = true;
                            while j0 < n {
                                let j1 = (j0 + GEMM_K_CHUNK).min(n);
                                let mut part = 0.0f32;
                                for j in j0..j1 {
                                    let e = err_t[j * m + i];
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
                            outg[q * m + i] = acc;
                        }
                    }
                }
            }
        }
    }
}

/// Cached AVX2+FMA detection: one CPUID decision for the whole process.
#[cfg(target_arch = "x86_64")]
fn avx2_fma_detected() -> bool {
    static DETECT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DETECT.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_fma_detected() -> bool {
    false
}

/// Runtime-dispatch engine for the bias-correction GEMM.
///
/// Build ONE per [`crate::bias_correction::correct_bias`] call. Construction
/// performs the (only) dispatch decision: on x86_64 with AVX2+FMA it pays the
/// one-time `Err` transpose ([`transpose_err`], amortized across all S = 3072
/// calibration rows); otherwise it keeps the row-major `err` and every block
/// runs the scalar fallback — byte-for-byte today's production loop. Both
/// paths are bit-identical (Step 2.1 proofs + the microkernel tests), so the
/// emitted artifact does not depend on the CPU the run happens to use.
///
/// # Example
///
/// `f32::mul_add` argument order is semantics (plan §0.3): `a.mul_add(b, c)`
/// is fused `a*b + c` — never `b*c + a`. (Illustrative; the executable
/// guard is `microkernel_tests::f32_mul_add_semantics_documented` — this
/// workspace has no doctest harness by policy.)
///
/// ```text
/// // 2*3 + 4 == 10; any other argument order gives a different number.
/// assert_eq!(2.0f32.mul_add(3.0, 4.0), 10.0);
/// // And the fused form is exactly the f64 emulation the scalar path uses:
/// let (a, b, c) = (1.000_000_1f32, 1.000_000_1f32, -1.0f32);
/// let fused = a.mul_add(b, c);
/// let emulated = ((a as f64) * (b as f64) + (c as f64)) as f32;
/// assert_eq!(fused.to_bits(), emulated.to_bits());
/// ```
pub struct GemmBias<'a> {
    /// Row-major [m, n] — the caller's (w_orig - w_dq) buffer, read by the
    /// scalar fallback.
    err: &'a [f32],
    /// Packed [n, m] — present only when the microkernel will run.
    err_t: Option<Vec<f32>>,
    m: usize,
    n: usize,
}

impl<'a> GemmBias<'a> {
    /// `err` is the caller's (w_orig - w_dq) buffer, [m, n] row-major.
    pub fn new(err: &'a [f32], m: usize, n: usize) -> Self {
        debug_assert_eq!(err.len(), m * n);
        let err_t = if avx2_fma_detected() {
            Some(transpose_err(err, m, n))
        } else {
            None
        };
        Self { err, err_t, m, n }
    }

    /// Compute one block of calibration rows: `x_blk` is [s_blk, n]
    /// row-major, `out_blk` is [s_blk, m] row-major and receives
    /// `out[q][i] = dot_kchunk128(x[q], err[i])`. Any `s_blk` is accepted;
    /// the packed path internally chunks into <= 8-row passes.
    pub fn gemm_block(&self, x_blk: &[f32], out_blk: &mut [f32]) {
        let (m, n) = (self.m, self.n);
        if out_blk.is_empty() {
            return;
        }
        let s_blk = out_blk.len() / m;
        debug_assert_eq!(out_blk.len(), s_blk * m, "out_blk must be whole rows");
        debug_assert_eq!(x_blk.len(), s_blk * n);
        #[cfg(target_arch = "x86_64")]
        if let Some(err_t) = &self.err_t {
            // SAFETY: err_t is Some(..) only when avx2_fma_detected() held
            // at construction; every length contract is debug-asserted
            // above; the kernel groups rows internally and stays in bounds
            // by construction (see its Safety section).
            unsafe { x86::microkernel_8x8(x_blk, err_t, m, n, out_blk) };
            return;
        }
        gemm_block_scalar(self.err, x_blk, m, n, out_blk);
    }
}

#[cfg(test)]
mod transpose_tests {
    use super::next_f32;
    use super::transpose_err;

    /// The reference: an index-by-index double loop. Slow on purpose — it
    /// is the definition the blocked kernel must reproduce.
    fn naive(err: &[f32], m: usize, n: usize) -> Vec<f32> {
        assert_eq!(err.len(), m * n);
        let mut out = vec![0.0f32; m * n];
        for j in 0..n {
            for i in 0..m {
                out[j * m + i] = err[i * n + j];
            }
        }
        out
    }

    fn fill(m: usize, n: usize, seed: u64) -> Vec<f32> {
        let mut st = seed;
        (0..m * n).map(|_| next_f32(&mut st)).collect()
    }

    const SHAPES: &[(usize, usize)] = &[
        (1, 1),
        (1, 5),
        (5, 1),
        (2, 2),
        (7, 3),
        (3, 7),
        (8, 8),
        (8, 1),
        (1, 8),
        (9, 16),
        (16, 9),
        (31, 33),
        (33, 31),
        (32, 32),
        (40, 300),
        (300, 40),
        (128, 4),
        (4, 128),
        (64, 64), // multiple rayon strips, exact JR boundary
    ];

    /// A transpose must move values, never change them: multiset equality
    /// of the bit patterns (NaN payloads and -0.0 included).
    #[test]
    fn transpose_is_a_permutation() {
        for &(m, n) in SHAPES {
            let err = fill(m, n, 0x7000_0000_0000_0001u64 ^ m as u64 ^ (n as u64) << 32);
            let mut a: Vec<u32> = err.iter().map(|v| v.to_bits()).collect();
            a.sort_unstable();
            let mut b: Vec<u32> = transpose_err(&err, m, n)
                .iter()
                .map(|v| v.to_bits())
                .collect();
            b.sort_unstable();
            assert_eq!(a, b, "m={m} n={n}: transpose changed the value multiset");
        }
    }

    /// Roundtrip: transpose(N x M) back must be the identity — including
    /// the rayon strip remainder handling on both passes.
    #[test]
    fn transpose_roundtrips() {
        for &(m, n) in SHAPES {
            let err = fill(m, n, 0x5EED_0000_0000_00AAu64 ^ n as u64 ^ (m as u64) << 32);
            let once = transpose_err(&err, m, n);
            let back = transpose_err(&once, n, m);
            for (i, (x, y)) in once.iter().zip(back.iter()).enumerate() {
                debug_assert_eq!(err.len(), back.len());
                assert_eq!(
                    y.to_bits(),
                    err[i].to_bits(),
                    "m={m} n={n} idx={i}: roundtrip differs"
                );
                let _ = x;
            }
        }
    }

    /// Non-square is the NORM (m = out-dim, n = in-dim of the layer).
    /// Bitwise agreement with the naive double loop on every element,
    /// plus specials that must survive any code path verbatim: ±0.0,
    /// denormals, NaN (payload bits), infinities, extremes.
    #[test]
    fn transpose_handles_non_square() {
        for &(m, n) in SHAPES {
            let mut err = fill(m, n, 0xC0FF_EE00_DEAD_0001u64 ^ m as u64 ^ n as u64);
            // Sprinkle specials at deterministic positions.
            let specials = [
                0.0f32,
                -0.0,
                f32::MIN_POSITIVE,
                1e-40,
                f32::MAX,
                f32::MIN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
            ];
            for (k, s) in specials.iter().enumerate() {
                if err.is_empty() {
                    break;
                }
                let idx = (k * 977 + 3) % err.len();
                err[idx] = *s;
            }
            let got = transpose_err(&err, m, n);
            let want = naive(&err, m, n);
            assert_eq!(got.len(), want.len(), "m={m} n={n}: length");
            for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "m={m} n={n} out[{j}]: got {g:e} want {w:e}"
                );
            }
        }
    }

    /// Degenerate axes: a transpose of nothing is nothing, and a 0-length
    /// axis must not panic in the blocked/rayon paths.
    #[test]
    fn transpose_handles_empty_axes() {
        assert!(transpose_err(&[], 0, 0).is_empty());
        assert!(transpose_err(&[], 0, 5).is_empty());
        assert!(transpose_err(&[], 5, 0).is_empty());
    }
}

#[cfg(test)]
mod proofs {
    use super::next_f32;

    /// The K-block size the GEMM resets its accumulator at. MUST mirror
    /// `bias_correction.rs::GEMM_K_CHUNK` — if that constant ever moves, this
    /// test module must move with it (asserted by
    /// [`k_chunk_constant_mirrors_the_kernel`]).
    const K_CHUNK: usize = 128;

    // ---------------------------------------------------------------------
    // The verbatim PRE-CHANGE kernels. Kept here, unmodified, as the oracle
    // the optimized path must reproduce. Copy-of-code, not copy-of-test.
    // ---------------------------------------------------------------------

    /// `bias_correction.rs:304-317` exactly as it was before Step 2.2.
    fn gemm_f64_emulated(xs: &[f32], er: &[f32]) -> f32 {
        let n = xs.len();
        debug_assert_eq!(er.len(), n);
        let mut acc = 0.0f32;
        let mut j0 = 0usize;
        let mut first = true;
        while j0 < n {
            let j1 = (j0 + K_CHUNK).min(n);
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
        acc
    }

    /// The Step 2.2 candidate: fused `mul_add`, plain f32 partial add.
    fn gemm_mul_add(xs: &[f32], er: &[f32]) -> f32 {
        let n = xs.len();
        debug_assert_eq!(er.len(), n);
        let mut acc = 0.0f32;
        let mut j0 = 0usize;
        let mut first = true;
        while j0 < n {
            let j1 = (j0 + K_CHUNK).min(n);
            let mut part = 0.0f32;
            for j in j0..j1 {
                // mul_add(self=a, m=b, c=acc) == a*b + acc with ONE rounding.
                // The f64 emulation widens a,b to f64 (exact), takes the exact
                // product (exact in f64: 24+24 <= 53 significand bits), adds
                // the f32 acc exactly in f64, then rounds once to f32 — which
                // is the fused definition. Argument order is load-bearing:
                // `part.mul_add(er[j], ...)` would compute part*er + xs[j].
                part = xs[j].mul_add(er[j], part);
            }
            acc = if first { part } else { acc + part };
            first = false;
            j0 = j1;
        }
        acc
    }

    // ---------------------------------------------------------------------
    // Proof 1: mul_add == f64 emulation, bitwise, adversarial inputs.
    // ---------------------------------------------------------------------

    /// Inputs span every shape class the real GEMM sees: `n` below, at, and
    /// above the K-chunk boundary, plus the 4096 production width. For each
    /// `n` we test a uniform-scale draw AND a wide-dynamic-range draw — the
    /// case where a double-rounding difference would surface first.
    #[test]
    fn fma_equals_f64_emulation_bitwise() {
        let shapes = [1usize, 2, 3, 127, 128, 129, 255, 256, 257, 300, 4096];
        let scales: [f32; 5] = [1.0, 1e6, 1e-6, 1e20, 1e-20];
        let mut checked = 0usize;

        for &n in &shapes {
            for &sa in &scales {
                for &sb in &scales {
                    let mut st = 0x1234_5678_9ABC_DEF0u64
                        ^ (n as u64)
                        ^ (sa.to_bits() as u64).rotate_left(17)
                        ^ (sb.to_bits() as u64).rotate_left(33);
                    let xs: Vec<f32> = (0..n).map(|_| next_f32(&mut st) * sa).collect();
                    let er: Vec<f32> = (0..n).map(|_| next_f32(&mut st) * sb).collect();
                    let want = gemm_f64_emulated(&xs, &er);
                    let got = gemm_mul_add(&xs, &er);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "n={n} sa={sa} sb={sb}: mul_add {:#010x} != f64-emulated {:#010x}",
                        got.to_bits(),
                        want.to_bits()
                    );
                    checked += 1;
                }
            }
        }

        // All-zero and all-one blocks: the real GEMM hits these (zero_blocks
        // golden exercises an all-zero tile).
        for &n in &shapes {
            let zeros = vec![0.0f32; n];
            let ones = vec![1.0f32; n];
            for (xs, er) in [
                (&zeros[..], &zeros[..]),
                (&ones[..], &zeros[..]),
                (&zeros[..], &ones[..]),
                (&ones[..], &ones[..]),
            ] {
                let want = gemm_f64_emulated(xs, er);
                let got = gemm_mul_add(xs, er);
                assert_eq!(got.to_bits(), want.to_bits(), "degenerate n={n}");
                checked += 1;
            }
        }

        assert!(checked > 300, "expected a broad sweep, ran {checked}");
    }

    // ---------------------------------------------------------------------
    // Proof 2: the negative control. `a * b + c` MUST differ somewhere.
    // ---------------------------------------------------------------------

    /// If this test ever passes, Proof 1 is vacuous — the sweep cannot
    /// distinguish fused from non-fused, and the whole methodology is void.
    #[test]
    fn non_fused_form_is_proven_wrong() {
        let mut st = 0xC0FF_EE00_DEAD_BEEFu64;
        let mut found = 0usize;
        let mut first_example = None;
        for _ in 0..100_000 {
            let a = next_f32(&mut st);
            let b = next_f32(&mut st);
            let c = next_f32(&mut st);
            let fused = a.mul_add(b, c);
            let separate = a * b + c;
            if fused.to_bits() != separate.to_bits() {
                found += 1;
                if first_example.is_none() {
                    first_example = Some((a, b, c, fused, separate));
                }
            }
        }
        assert!(
            found > 0,
            "VACUOUS PROOF: non-fused a*b+c never differed from mul_add in \
             100000 samples — the sweep cannot detect a rounding-semantics \
             regression, so Proof 1 proves nothing"
        );
        let (a, b, c, fused, separate) = first_example.unwrap();
        assert!(
            found > 100,
            "non-fused form differs only {found}/100000 times; suspiciously \
             rare — widen the input distribution before trusting Proof 1"
        );
        // Silence unused-variable warnings while keeping the example in the
        // failure message context. The values are the evidence.
        let _ = (a, b, c, fused, separate);
    }

    // ---------------------------------------------------------------------
    // Proof 3: the partial-add detour is a plain f32 add.
    // ---------------------------------------------------------------------

    /// `((acc as f64) + (part as f64)) as f32` must equal `acc + part`
    /// bitwise. Widening two f32 to f64 is exact; the f64 sum rounds once;
    /// narrowing rounds again — the classic double-rounding hazard. For f32
    /// operands widened to f64 the double rounding is innocuous (the exact
    /// sum's significand fits: 24+24+1 <= 53), but this test PROVES it for
    /// the adversarial cases rather than trusting the theorem.
    #[test]
    fn partial_add_is_plain_f32_add() {
        let mut st = 0xFEED_FACE_CAFE_F00Du64;
        let mut checked = 0usize;
        // Uniform pairs, then extreme-exponent pairs (1e30 + 1e-30 class),
        // then sign flips — the cases where innocuity is least obvious.
        let scales: [f32; 6] = [1.0, 1e6, 1e-6, 1e30, 1e-30, -1.0];
        for &sa in &scales {
            for &sb in &scales {
                for _ in 0..5_000 {
                    let acc = next_f32(&mut st) * sa;
                    let part = next_f32(&mut st) * sb;
                    let want = ((acc as f64) + (part as f64)) as f32;
                    let got = acc + part;
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "acc={acc:e} part={part:e}: plain {:#010x} != f64 {:#010x}",
                        got.to_bits(),
                        want.to_bits()
                    );
                    checked += 1;
                }
            }
        }
        // Degenerate pairs: zeros, negatives, both zero, opposite signs.
        for (acc, part) in [
            (0.0f32, 0.0f32),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, -1.0),
            (-0.0, 0.0),
            (f32::MIN_POSITIVE, f32::MIN_POSITIVE),
            (f32::MIN, f32::MAX),
        ] {
            let want = ((acc as f64) + (part as f64)) as f32;
            let got = acc + part;
            assert_eq!(got.to_bits(), want.to_bits(), "acc={acc} part={part}");
            checked += 1;
        }
        assert!(checked > 100_000, "expected a broad sweep, ran {checked}");
    }

    // ---------------------------------------------------------------------
    // Structural guard: the mirrored constant.
    // ---------------------------------------------------------------------

    /// `K_CHUNK` here must equal the K-block constant in every kernel. The
    /// proofs are about THE reference order; if the constants drift the
    /// proofs silently stop testing the real code path. Read the source and
    /// compare literally — a doc test would not see runtime drift.
    ///
    /// Since Step 2.5, bias_correction.rs delegates its GEMM to this module
    /// (see `correct_bias_goes_through_the_dispatcher`), so its pin is the
    /// sequential f64 ORACLE's local `K_CHUNK`; the production constants are
    /// the ones below.
    #[test]
    fn k_chunk_constant_mirrors_the_kernel() {
        let src = include_str!("bias_correction.rs");
        assert!(
            src.contains("const K_CHUNK: usize = 128;"),
            "bias_correction.rs no longer declares its sequential oracle's \
             K_CHUNK = 128; the oracle, this module, and the proofs must \
             move together"
        );
        let src2 = include_str!("convrot.rs");
        assert!(
            src2.contains("const GEMM_K_CHUNK: usize = 128;"),
            "convrot.rs no longer declares GEMM_K_CHUNK = 128; the ConvRot \
             GEMM and the proofs must move together"
        );
        let src3 = include_str!("bias_gemm.rs");
        assert!(
            src3.contains("const GEMM_K_CHUNK: usize = 128;"),
            "bias_gemm.rs no longer declares GEMM_K_CHUNK = 128; the \
             microkernel, the fallback, and the proofs must move together"
        );
        assert_eq!(K_CHUNK, 128);
    }

    /// Step 2.5 wiring pin: `correct_bias` must run its GEMM through
    /// `bias_gemm::GemmBias` (packed microkernel or the proven-identical
    /// scalar fallback) — not through a private re-inline. A re-inline is
    /// not automatically wrong, but it must be a CONSCIOUS change: this pin
    /// fails and forces moving the bit-identity battery in the same commit.
    #[test]
    fn correct_bias_goes_through_the_dispatcher() {
        let src = include_str!("bias_correction.rs");
        assert!(
            src.contains("GemmBias::new("),
            "correct_bias no longer builds bias_gemm::GemmBias; if the GEMM \
             moved, the k_chunk/f64-line pins and the microkernel battery \
             must move with it in the same commit"
        );
    }

    /// The f64-emulation line and the partial-add line must stay VERBATIM in
    /// every file that carries a copy of the kernel. A copy that drifts is a
    /// silent parity break: this module's oracle, the Step 2.4 fallback, the
    /// microkernel tail, the sequential oracle in bias_correction, and both
    /// production kernels must round identically. The pins are per-file
    /// because the copies name their operands differently. Since Step 2.5,
    /// bias_correction.rs's copy is its sequential ORACLE (the production
    /// GEMM lives in this module).
    #[test]
    fn f64_emulation_line_is_verbatim_everywhere() {
        let pins: &[(&str, &[&str])] = &[
            (
                "bias_correction.rs",
                &[
                    // sequential_correct_bias oracle (the standing f64
                    // reference; operand named `e` there):
                    "part = ((xs[j] as f64) * (e as f64) + (part as f64)) as f32;",
                    "((acc as f64) + (part as f64)) as f32",
                ],
            ),
            (
                "convrot.rs",
                &[
                    "part = ((a[j] as f64) * (b[j] as f64) + (part as f64)) as f32;",
                    "((acc as f64) + (part as f64)) as f32",
                ],
            ),
            (
                "bias_gemm.rs",
                &[
                    // scalar_dot_kchunk128 (fallback oracle):
                    "part = ((xs[j] as f64) * (er[j] as f64) + (part as f64)) as f32;",
                    // microkernel i-tail (packed-layout variant):
                    "part = ((xs[j] as f64) * (e as f64) + (part as f64)) as f32;",
                    "((acc as f64) + (part as f64)) as f32",
                ],
            ),
        ];
        for (file, lines) in pins {
            let src = match *file {
                "bias_correction.rs" => include_str!("bias_correction.rs"),
                "convrot.rs" => include_str!("convrot.rs"),
                _ => include_str!("bias_gemm.rs"),
            };
            for line in *lines {
                assert!(
                    src.contains(line),
                    "{file} no longer contains the verbatim kernel line:\n  {line}\n\
                     Every copy of the f64 emulation must round identically — \
                     mirror the edit in ALL files or revert it."
                );
            }
        }
    }
}

#[cfg(test)]
mod microkernel_tests {
    use super::next_f32;
    use super::{transpose_err, GemmBias};

    /// INDEPENDENT spec implementation written from the reference
    /// description — out[q][i] = K-chunk-128 dot of x row q and err row i,
    /// f64-emulated FMA, partials left-to-right — NOT by calling production
    /// code. Every path under test (dispatch, direct microkernel, scalar
    /// fallback) is compared against THIS, so a bug shared by two
    /// implementations cannot hide.
    fn spec_gemm(x_blk: &[f32], err: &[f32], m: usize, n: usize) -> Vec<f32> {
        let s = x_blk.len() / n;
        let mut out = vec![0.0f32; s * m];
        for q in 0..s {
            for i in 0..m {
                let mut acc = 0.0f32;
                let mut j0 = 0usize;
                let mut first = true;
                while j0 < n {
                    let j1 = (j0 + 128).min(n); // hardcoded: the invariant
                    let mut part = 0.0f32;
                    for j in j0..j1 {
                        let xj = x_blk[q * n + j];
                        let ej = err[i * n + j];
                        part = ((xj as f64) * (ej as f64) + (part as f64)) as f32;
                    }
                    acc = if first {
                        part
                    } else {
                        ((acc as f64) + (part as f64)) as f32
                    };
                    first = false;
                    j0 = j1;
                }
                out[q * m + i] = acc;
            }
        }
        out
    }

    /// What reassociation would produce: ONE unbroken f64-emulated chain
    /// over all of K. The kernel must NEVER match this (invariant I2).
    fn unbroken_gemm(x_blk: &[f32], err: &[f32], m: usize, n: usize) -> Vec<f32> {
        let s = x_blk.len() / n;
        let mut out = vec![0.0f32; s * m];
        for q in 0..s {
            for i in 0..m {
                let mut part = 0.0f32;
                for j in 0..n {
                    let xj = x_blk[q * n + j];
                    let ej = err[i * n + j];
                    part = ((xj as f64) * (ej as f64) + (part as f64)) as f32;
                }
                out[q * m + i] = part;
            }
        }
        out
    }

    fn make(m: usize, n: usize, s: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut st = seed;
        let x: Vec<f32> = (0..s * n).map(|_| next_f32(&mut st) * 3.0).collect();
        let err: Vec<f32> = (0..m * n).map(|_| next_f32(&mut st) * 5.0).collect();
        (x, err)
    }

    fn assert_bits(got: &[f32], want: &[f32], what: &str, m: usize, n: usize, s: usize) {
        assert_eq!(got.len(), want.len(), "{what} m={m} n={n} s={s}: length");
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what} m={m} n={n} s={s} idx={idx}: got {g:e} want {w:e} \
                 (bits {:#010x} vs {:#010x})",
                g.to_bits(),
                w.to_bits()
            );
        }
    }

    const MS: &[usize] = &[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 40, 128,
    ];
    const SS: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 16];

    /// The core tail-shape sweep: every m-tail (0..=17 covers m<8, m=8,
    /// boundary crossings; 31/32/33/40/128 the larger ones) x every s-tail
    /// x a K-tail (300 = 2x128 + 44). ALL THREE routes — runtime dispatch,
    /// the direct microkernel (when the host has AVX2+FMA), and the scalar
    /// fallback — must equal the independent spec bitwise.
    #[test]
    fn microkernel_matches_scalar_for_all_tail_shapes() {
        const N: usize = 300;
        for &m in MS {
            for &s in SS {
                let seed = 0x600D_0000_0000_0001u64 ^ (m as u64) ^ ((s as u64) << 32);
                let (x, err) = make(m, N, s, seed);
                let want = spec_gemm(&x, &err, m, N);

                let g = GemmBias::new(&err, m, N); // dispatches per host
                let mut via_dispatch = vec![0.0f32; s * m];
                g.gemm_block(&x, &mut via_dispatch);
                assert_bits(&via_dispatch, &want, "dispatch", m, N, s);

                let mut via_scalar = vec![0.0f32; s * m];
                super::gemm_block_scalar(&err, &x, m, N, &mut via_scalar);
                assert_bits(&via_scalar, &want, "scalar-fallback", m, N, s);

                #[cfg(target_arch = "x86_64")]
                if super::avx2_fma_detected() {
                    let err_t = transpose_err(&err, m, N);
                    let mut via_mk = vec![0.0f32; s * m];
                    // SAFETY: host AVX2+FMA verified; lengths asserted in
                    // the kernel; test buffers match the documented layouts.
                    unsafe { super::x86::microkernel_8x8(&x, &err_t, m, N, &mut via_mk) };
                    assert_bits(&via_mk, &want, "microkernel", m, N, s);
                }
            }
        }
    }

    /// K-tails: n below, at, above the 128 boundary, plus 4096 (the
    /// production width, 32 chunks).
    #[test]
    fn microkernel_handles_k_tail() {
        for &n in &[1usize, 127, 128, 129, 255, 256, 4096] {
            for &(m, s) in &[(40usize, 8usize), (33, 3), (8, 1), (128, 5)] {
                let seed = 0xD1CE_0000_0000_0007u64 ^ (n as u64).rotate_left(11) ^ m as u64;
                let (x, err) = make(m, n, s, seed);
                let want = spec_gemm(&x, &err, m, n);

                let g = GemmBias::new(&err, m, n);
                let mut via_dispatch = vec![0.0f32; s * m];
                g.gemm_block(&x, &mut via_dispatch);
                assert_bits(&via_dispatch, &want, "dispatch", m, n, s);

                let mut via_scalar = vec![0.0f32; s * m];
                super::gemm_block_scalar(&err, &x, m, n, &mut via_scalar);
                assert_bits(&via_scalar, &want, "scalar-fallback", m, n, s);

                #[cfg(target_arch = "x86_64")]
                if super::avx2_fma_detected() {
                    let err_t = transpose_err(&err, m, n);
                    let mut via_mk = vec![0.0f32; s * m];
                    // SAFETY: host AVX2+FMA verified; layouts documented.
                    unsafe { super::x86::microkernel_8x8(&x, &err_t, m, n, &mut via_mk) };
                    assert_bits(&via_mk, &want, "microkernel", m, n, s);
                }
            }
        }
    }

    /// On AVX2+FMA hosts, the GemmBias dispatch result must equal the
    /// microkernel called directly (i.e. dispatch really reached the
    /// packed path). On fallback hosts, dispatch equals the scalar
    /// fallback. Either way both equal the spec.
    #[test]
    fn dispatch_matches_microkernel_when_supported() {
        let (m, n, s) = (40usize, 300usize, 8usize);
        let (x, err) = make(m, n, s, 0x0DD1_B1A5_0000_0003u64);
        let want = spec_gemm(&x, &err, m, n);

        let g = GemmBias::new(&err, m, n);
        let mut via_dispatch = vec![0.0f32; s * m];
        g.gemm_block(&x, &mut via_dispatch);
        assert_bits(&via_dispatch, &want, "dispatch", m, n, s);

        #[cfg(target_arch = "x86_64")]
        if super::avx2_fma_detected() {
            let err_t = transpose_err(&err, m, n);
            let mut direct = vec![0.0f32; s * m];
            // SAFETY: host AVX2+FMA verified; layouts documented.
            unsafe { super::x86::microkernel_8x8(&x, &err_t, m, n, &mut direct) };
            assert_bits(&direct, &want, "direct-microkernel", m, n, s);
        }
    }

    /// The zero_blocks golden exercises an all-zero tile; zero rows and
    /// zero inputs hit the +0.0/-0.0 edge of the FMA chain. The spec
    /// decides the bits; all routes must agree with it.
    #[test]
    fn microkernel_handles_zero_tiles() {
        let (m, n, s) = (40usize, 300usize, 8usize);
        let (mut x, mut err) = make(m, n, s, 0x2E40_0000_0000_0009u64);

        let cases: Vec<(&str, Vec<f32>, Vec<f32>)> = {
            let mut v = Vec::new();
            v.push(("x-zero", vec![0.0f32; s * n], err.clone()));
            v.push(("err-zero", x.clone(), vec![0.0f32; m * n]));
            v.push(("both-zero", vec![0.0f32; s * n], vec![0.0f32; m * n]));
            // Interior zero block in err (rows 16..24) — the zero_blocks
            // golden's shape class; those outputs must be exactly the
            // spec's zeros while the rest stay non-trivial.
            let mut ez = err.clone();
            for i in 16..24 {
                for j in 0..n {
                    ez[i * n + j] = 0.0;
                }
            }
            v.push(("err-interior-zero-block", x.clone(), ez));
            v
        };

        for (name, xc, ec) in &cases {
            let want = spec_gemm(xc, ec, m, n);
            let g = GemmBias::new(ec, m, n);
            let mut via_dispatch = vec![0.0f32; s * m];
            g.gemm_block(xc, &mut via_dispatch);
            assert_bits(&via_dispatch, &want, name, m, n, s);

            let mut via_scalar = vec![0.0f32; s * m];
            super::gemm_block_scalar(ec, xc, m, n, &mut via_scalar);
            assert_bits(&via_scalar, &want, name, m, n, s);

            // The zero cases must genuinely produce zeros where expected.
            if *name == "err-interior-zero-block" {
                for q in 0..s {
                    for i in 16..24 {
                        let idx = q * m + i;
                        assert_eq!(
                            via_dispatch[idx].to_bits(),
                            0u32,
                            "zero-block output must be +0.0, got {}",
                            via_dispatch[idx]
                        );
                    }
                }
            }
        }

        // Silence later-mutation warnings; the bases were consumed above.
        x.clear();
        err.clear();
    }

    /// Invariant I2, behaviorally: with adversarial cancellation data, the
    /// chunked-128 result differs from an unbroken chain (the guard must be
    /// ABLE to bite), and BOTH dispatch routes produce the chunked-128
    /// bits — the spec hardcodes 128, so any change to GEMM_K_CHUNK
    /// anywhere in the kernel chain fails here.
    #[test]
    fn gemm_k_chunk_boundary_is_respected() {
        let (m, n, s) = (8usize, 256usize, 8usize);
        let mut trials_with_difference = 0usize;
        for trial in 0..64u64 {
            let mut st = 0xAB00_0000_0000_0001u64 ^ trial.rotate_left(13);
            let mut x = vec![0.0f32; s * n];
            let mut err = vec![0.0f32; m * n];
            // Adversarial magnitudes: products are O(1) but partial sums
            // mix 1e-6..1e6 terms, so chunk boundaries change the rounding.
            for j in 0..n {
                let mag = 10f32.powi(((j % 7) as i32) - 3);
                let inv = 10f32.powi(3 - ((j % 7) as i32));
                for q in 0..s {
                    x[q * n + j] = next_f32(&mut st) * mag;
                }
                for i in 0..m {
                    err[i * n + j] = next_f32(&mut st) * inv;
                }
            }

            let want = spec_gemm(&x, &err, m, n); // hardcodes 128
            let wrong = unbroken_gemm(&x, &err, m, n);
            if want
                .iter()
                .zip(wrong.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                trials_with_difference += 1;
            }

            let g = GemmBias::new(&err, m, n);
            let mut via_dispatch = vec![0.0f32; s * m];
            g.gemm_block(&x, &mut via_dispatch);
            assert_bits(&via_dispatch, &want, "dispatch", m, n, s);

            let mut via_scalar = vec![0.0f32; s * m];
            super::gemm_block_scalar(&err, &x, m, n, &mut via_scalar);
            assert_bits(&via_scalar, &want, "scalar-fallback", m, n, s);

            #[cfg(target_arch = "x86_64")]
            if super::avx2_fma_detected() {
                let err_t = transpose_err(&err, m, n);
                let mut via_mk = vec![0.0f32; s * m];
                // SAFETY: host AVX2+FMA verified; layouts documented.
                unsafe { super::x86::microkernel_8x8(&x, &err_t, m, n, &mut via_mk) };
                assert_bits(&via_mk, &want, "microkernel", m, n, s);
            }
        }
        assert!(
            trials_with_difference > 0,
            "VACUOUS GUARD: chunked-128 == unbroken on all 64 adversarial \
             trials — this test can no longer detect reassociation; widen \
             the input distribution before trusting invariant I2 coverage"
        );
    }

    /// Named unit test for the §0.3 trap (the doctest on `GemmBias` covers
    /// the same property; this exists so the guard is visible by name in
    /// the test list and in the §6.5 drill).
    #[test]
    fn f32_mul_add_semantics_documented() {
        // mul_add(self, a, b) == self*a + b, fused. Any other reading of
        // the argument order produces a different number here.
        assert_eq!(2.0f32.mul_add(3.0, 4.0), 10.0);
        assert_eq!(2.0f32.mul_add(4.0, 3.0), 11.0);
        // And the fused form is the f64 emulation, bitwise (Step 2.1).
        let (a, b, c) = (1.000_000_1f32, 1.000_000_1f32, -1.0f32);
        let fused = a.mul_add(b, c);
        let emulated = ((a as f64) * (b as f64) + (c as f64)) as f32;
        assert_eq!(fused.to_bits(), emulated.to_bits());
    }
}

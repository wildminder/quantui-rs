//! Phase 1 Step 2.6 (bias-GEMM SIMD plan) — kernel-level benchmarks.
//!
//! Three GEMM implementations of the SAME arithmetic (out[s][i] =
//! K-chunk-128 dot of x row s and err row i; all bit-identical, proven in
//! `bias_gemm::proofs`), plus the Err transpose the packed path needs:
//!
//! - `gemm/scalar_f64` — the verbatim pre-SIMD kernel (the baseline that
//!   ran in production until 2026-10). Kept here so the comparison is
//!   permanent even though the kernel is gone from production code.
//! - `gemm/scalar_mul_add` — the Step 2.2 candidate. MEASURED NEGATIVE on
//!   this crate's default target (no `target-cpu` in the profile: baseline
//!   x86-64 has no FMA3, so `f32::mul_add` lowers to software `fmaf` —
//!   2.2x slower than the f64 detour per-op, 31 s -> 122 s end-to-end).
//!   Benchmarked anyway: the number is the recorded negative, and on a
//!   `target-cpu=native` build it flips to ~3x faster, which demonstrates
//!   exactly what the crate gives up by shipping baseline x86-64.
//! - `gemm/packed_avx2` — the shipped microkernel via `GemmBias`
//!   (single-threaded here; production adds rayon over 8-row blocks).
//! - `transpose/err_4096` — the one-time packed-layout build. The plan
//!   gate: < 5% of the GEMM at S=3072; the prelude below prints the
//!   measured share directly.
//!
//! GFLOP/s are PER-CORE: the GEMM closures run without rayon. Production
//! parallelism is measured end-to-end instead: `quantui-rs quantize` on
//! `tests/bench/bench_bias_probe.safetensors` (8x [4096,4096] bf16 layers
//! with `.bias` siblings) went 31 s -> 4 s with this change.
//!
//! Shape: m = 4096, n = 4096 (production dims), S = 96 (a 32x reduction of
//! CALIB_SAMPLES so the scalar iterations stay ~1 s; a multiple of 8 so
//! the packed path never sees a block tail). FLOPs = S*m*n*2.
//!
//! Run with: `cargo bench -p quant-core --bench bias_gemm`

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use quant_core::bias_gemm::{transpose_err, GemmBias};

const M: usize = 4096; // out-dim (rows of err / out)
const N: usize = 4096; // in-dim (K)
const S_BENCH: usize = 96; // calibration rows per iteration (multiple of 8)

/// Deterministic LCG fill — same shape of generator as the test modules.
fn next_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32 / (1 << 24) as f32) - 1.0
}

fn fixtures() -> (Vec<f32>, Vec<f32>) {
    let mut st = 0xBEEF_C0FF_EE00_0001u64;
    let x: Vec<f32> = (0..S_BENCH * N).map(|_| next_f32(&mut st) * 3.0).collect();
    let err: Vec<f32> = (0..M * N).map(|_| next_f32(&mut st) * 5.0).collect();
    (x, err)
}

/// The verbatim PRE-SIMD production kernel (bias_correction.rs as it stood
/// before Phase 1): f64-emulated FMA, K-chunk-128, partials left-to-right.
/// Copy-of-code on purpose — this is the permanent baseline.
fn dot_scalar_f64(xs: &[f32], er: &[f32]) -> f32 {
    let n = xs.len();
    let mut acc = 0.0f32;
    let mut j0 = 0usize;
    let mut first = true;
    while j0 < n {
        let j1 = (j0 + 128).min(n);
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

/// The Step 2.2 candidate (reverted — measured negative, see module doc):
/// identical order, fused `mul_add` instead of the f64 detour.
fn dot_scalar_mul_add(xs: &[f32], er: &[f32]) -> f32 {
    let n = xs.len();
    let mut acc = 0.0f32;
    let mut j0 = 0usize;
    let mut first = true;
    while j0 < n {
        let j1 = (j0 + 128).min(n);
        let mut part = 0.0f32;
        for j in j0..j1 {
            part = xs[j].mul_add(er[j], part);
        }
        acc = if first { part } else { acc + part };
        first = false;
        j0 = j1;
    }
    acc
}

fn run_gemm<F>(x: &Vec<f32>, err: &Vec<f32>, dot: F) -> Vec<f32>
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    let mut out = vec![0.0f32; S_BENCH * M];
    for s in 0..S_BENCH {
        let xs = &x[s * N..(s + 1) * N];
        for i in 0..M {
            out[s * M + i] = dot(xs, &err[i * N..(i + 1) * N]);
        }
    }
    out
}

fn flops() -> u64 {
    (S_BENCH * M * N * 2) as u64
}

/// One-shot wall-clock share of the Err transpose against the packed GEMM
/// at the PRODUCTION S=3072 (the plan's < 5% gate, printed directly —
/// criterion medians of differently-shaped benches can't express a ratio).
fn print_transpose_share(err: &[f32]) {
    let t0 = std::time::Instant::now();
    let err_t = transpose_err(err, M, N);
    let t_transpose = t0.elapsed();

    // One packed GEMM pass at S = 3072, single-threaded, via the real
    // dispatcher — the same per-row work production repeats 384 times.
    const S_FULL: usize = 3072;
    let mut st = 0x5EED_0000_0000_0002u64;
    let x_full: Vec<f32> = (0..S_FULL * N).map(|_| next_f32(&mut st)).collect();
    let g = GemmBias::new(err, M, N);
    let mut out8 = vec![0.0f32; 8 * M];
    let t0 = std::time::Instant::now();
    for s0 in (0..S_FULL).step_by(8) {
        let x_blk = &x_full[s0 * N..(s0 + 8) * N];
        g.gemm_block(x_blk, &mut out8);
    }
    let t_gemm = t0.elapsed();

    let share = t_transpose.as_secs_f64() / t_gemm.as_secs_f64() * 100.0;
    println!(
        "[bias_gemm] transpose share @ S=3072: {:.1}% ({:?} transpose vs \
         {:?} single-thread packed GEMM) — plan gate: < 5%",
        share, t_transpose, t_gemm
    );
    let _ = black_box(err_t.len());
}

fn bench_bias_gemm(c: &mut Criterion) {
    let (x, err) = fixtures();

    // The share prelude runs BEFORE the timed benches so its one-shot
    // numbers don't pollute group medians.
    print_transpose_share(&err);

    let mut group = c.benchmark_group("bias_gemm");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    group.throughput(Throughput::Elements(flops()));

    group.bench_function(BenchmarkId::new("gemm", "scalar_f64"), |b| {
        b.iter(|| black_box(run_gemm(&x, &err, dot_scalar_f64)))
    });
    group.bench_function(BenchmarkId::new("gemm", "scalar_mul_add"), |b| {
        b.iter(|| black_box(run_gemm(&x, &err, dot_scalar_mul_add)))
    });
    group.bench_function(BenchmarkId::new("gemm", "packed_avx2"), |b| {
        // Single-threaded, mirroring production's per-block dispatch minus
        // rayon. GemmBias::new pays the transpose ONCE outside the timed
        // closure — exactly as correct_bias does.
        let g = GemmBias::new(&err, M, N);
        b.iter(|| {
            let mut out = vec![0.0f32; S_BENCH * M];
            for s0 in (0..S_BENCH).step_by(8) {
                let x_blk = &x[s0 * N..(s0 + 8) * N];
                let out_blk = &mut out[s0 * M..(s0 + 8) * M];
                g.gemm_block(x_blk, out_blk);
            }
            black_box(out)
        })
    });

    group.finish();

    let mut tgroup = c.benchmark_group("transpose");
    tgroup.sample_size(20);
    tgroup.measurement_time(Duration::from_secs(10));
    tgroup.throughput(Throughput::Bytes((M * N * 4) as u64));
    tgroup.bench_function(BenchmarkId::new("err", "4096x4096"), |b| {
        b.iter(|| black_box(transpose_err(&err, M, N)))
    });
    tgroup.finish();
}

criterion_group!(benches, bench_bias_gemm);
criterion_main!(benches);

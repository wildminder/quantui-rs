//! Phase 11.1 — streaming-quantizer throughput benchmark.
//!
//! Measures end-to-end `stream_quantize` (read -> quantize -> write) on the
//! shared ~1 GB fixture produced by `tools/gen_bench_fixture.py`, reporting
//! GB/s via criterion's `Throughput::Bytes`. The Python reference harness
//! (`tools/bench_python_ref.py`) times the reference `stream_quantize` on the
//! SAME file, so the two numbers are directly comparable.
//!
//! Run with: `cargo bench -p quant-core --bench stream_throughput`
//!
//! The fixture is gitignored (~1 GB). If it is missing, the bench prints a
//! generation hint and exits cleanly instead of failing.

use std::path::PathBuf;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use quant_core::manifest::QuantConfig;
use quant_core::quant::{quantize_int8_weight, ScalingMode};
use quant_core::stream::stream_quantize;

/// Locate the shared fixture relative to the workspace root.
fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // crates/quant-core
    p.pop(); // crates/
    p.pop(); // workspace root
    p.push("tests/bench/bench_1gb.safetensors");
    p
}

fn stream_throughput(c: &mut Criterion) {
    let fixture = fixture_path();
    if !fixture.exists() {
        eprintln!(
            "[skip] benchmark fixture not found: {}\n       \
             generate it first with: python tools/gen_bench_fixture.py",
            fixture.display()
        );
        return;
    }

    let input_bytes = std::fs::metadata(&fixture).map(|m| m.len()).unwrap_or(0);

    // Reference config: block mode, bs=128, simple rounding, heur on, bf16 —
    // identical to the golden/parity config and the Python reference defaults.
    let config = QuantConfig::default();

    let mut group = c.benchmark_group("stream_quantize");
    // A ~1 GB run takes seconds; keep the sample count low and the measurement
    // window generous so the whole bench stays under a couple of minutes.
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));
    group.warm_up_time(Duration::from_secs(5));
    group.throughput(Throughput::Bytes(input_bytes));

    group.bench_function(BenchmarkId::new("int8_block_1gb", input_bytes), |b| {
        b.iter_batched(
            // Setup (untimed): a fresh output dir each iteration so the
            // resumable manifest never short-circuits an already-done tensor.
            || {
                let dir = tempfile::tempdir().expect("tempdir");
                let out = dir.path().join("out.safetensors");
                (dir, out)
            },
            |(_dir, out)| {
                stream_quantize(&fixture, &out, &config).expect("stream_quantize");
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Pure CPU-bound quantization kernel (no I/O, no orchestration).
///
/// The plan's ">=5x Python wall-clock on CPU-bound quantization" target refers
/// to the quantization math itself, not the end-to-end streaming path (which is
/// dominated by disk I/O + per-tensor manifest/header flushes and therefore
/// compresses the Rust/Python ratio toward ~1x). This bench isolates the kernel
/// on an in-memory 4096x4096 f32 weight in block mode, bs=128.
fn kernel_compute(c: &mut Criterion) {
    const M: usize = 4096;
    const N: usize = 4096;
    const BS: usize = 128;

    // Deterministic pseudo-random f32 weight (no RNG dep needed).
    let mut w = vec![0.0f32; M * N];
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for v in w.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map top bits to [-1, 1).
        *v = ((state >> 40) as f32 / (1 << 24) as f32) - 1.0;
    }
    let bytes = (M * N * std::mem::size_of::<f32>()) as u64;

    let mut group = c.benchmark_group("quantize_int8_kernel");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(20));
    group.throughput(Throughput::Bytes(bytes));

    group.bench_function(BenchmarkId::new("block_bs128_4096x4096", bytes), |b| {
        b.iter(|| {
            let r = quantize_int8_weight(criterion::black_box(&w), M, N, ScalingMode::Block, BS);
            criterion::black_box(r.qdata.len());
        });
    });

    group.finish();
}

criterion_group!(benches, stream_throughput, kernel_compute);
criterion_main!(benches);

//! WP10 / NTH-003 — small-fixture benchmark for nightly regression
//! tracking.
//!
//! Same structure as `stream_throughput.rs` (full `stream_quantize`:
//! read → quantize → write) but on the COMMITTED ~8 MB fixture
//! `tests/bench/bench_small.safetensors` (4 × [1024,1024] BF16). No
//! generation guard is needed — the fixture is in the repo — which is
//! what makes this target usable from the nightly workflow
//! (`.github/workflows/bench.yml`) with zero setup steps.
//!
//! Run with: `cargo bench -p quant-core --bench stream_small`
//! Baseline flow (nightly): `--save-baseline nightly`, then later runs
//! compare with `--baseline nightly`.

use std::path::PathBuf;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use quant_core::manifest::QuantConfig;
use quant_core::stream::stream_quantize;

/// Locate the committed fixture relative to the workspace root.
fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // crates/quant-core
    p.pop(); // crates/
    p.pop(); // workspace root
    p.push("tests/bench/bench_small.safetensors");
    p
}

fn stream_small(c: &mut Criterion) {
    let fixture = fixture_path();
    let input_bytes = std::fs::metadata(&fixture).map(|m| m.len()).unwrap_or(0);
    if input_bytes == 0 {
        eprintln!(
            "[skip] committed fixture missing: {}\n       \
             regenerate with: python tools/gen_bench_small.py",
            fixture.display()
        );
        return;
    }

    // Reference config (QuantConfig::default()): block mode, bs=128,
    // simple rounding, heur on, bf16 — the same defaults the parity
    // goldens pin.
    let config = QuantConfig::default();

    let mut group = c.benchmark_group("stream_quantize");
    // 8 MB per iteration: sample enough for a stable median while the
    // whole bench stays well under a minute (CI-friendly).
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(input_bytes));

    group.bench_function(BenchmarkId::new("int8_block_small", input_bytes), |b| {
        b.iter_batched(
            // Setup (untimed): a fresh output dir each iteration so the
            // resumable manifest never short-circuits an already-done
            // tensor.
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

criterion_group!(benches, stream_small);
criterion_main!(benches);

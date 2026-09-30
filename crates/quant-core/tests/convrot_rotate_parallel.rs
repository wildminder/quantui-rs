//! `rotate_weight` is now parallel over rows (the fix for "int8_convrot uses
//! one core"), and this file is the guard that makes that safe.
//!
//! # The performance fact that motivated the change
//!
//! Measured on this box, [4096, 4096] gs=256, release build:
//!
//! | stage      | time    | share |
//! |------------|---------|-------|
//! | rotate     | 6.261 s | 99.1% |
//! | quantize   | 0.048 s |  0.8% |
//! | dequantize | 0.011 s |  0.2% |
//!
//! The two small stages were *already* parallel (`quantize_scaled` uses
//! `par_iter`). `rotate_weight` was a plain sequential triple loop, so it
//! alone pinned a 14 GB / 7.1-billion-element run to a single core.
//!
//! # Why parallelizing it cannot change a byte
//!
//! Every output element `out[r, base + j]` is written by exactly one call to
//! `dot_kchunk128`, whose accumulation order is fixed by that element's own
//! inputs: 128-element K-chunks, each an f64 product chain rounded once to
//! f32, partials summed left-to-right. Nothing is ever accumulated ACROSS
//! rows, so distributing rows over threads splits no sum, reorders no
//! addition, and reassociates nothing. The float semantics pinned against
//! torch 2.13.0+cpu in `convrot.rs`'s module header (probe
//! `tools/probe_convrot_gs256.py`, zero mismatches at gs=256) survive intact.
//!
//! `sequential_rotate_reference` below re-implements the original loop from
//! the pre-parallel source and is the oracle these tests compare against.
//! `rotation_is_bit_identical_across_thread_counts` additionally pins that
//! the answer does not depend on how rayon happens to schedule the work.

use quant_core::convrot::{build_hadamard, rotate_weight};

/// Deterministic xorshift data — no RNG crate, no dependence on a seed.
fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    let mut x = seed | 1;
    for _ in 0..n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        v.push(((x % 40_000) as f32 / 20_000.0) - 1.0);
    }
    v
}

/// The ORIGINAL sequential loop, transcribed from `rotate_weight` before it
/// was parallelized. Deliberately written straightforwardly — no closures, no
/// chunking — so it is an independent check and not a refactor of the code
/// under test.
fn sequential_rotate_reference(w: &[f32], h: &[f32], rows: usize, n: usize, gs: usize) -> Vec<f32> {
    let n_groups = n / gs;
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        for g in 0..n_groups {
            let base = g * gs;
            for j in 0..gs {
                // sum_k W[r, base+k] * H[j, k], same 128-chunk f64->f32 chain.
                let h_row = &h[j * gs..j * gs + gs];
                let mut acc = 0.0f32;
                let mut first = true;
                let mut j0 = 0usize;
                while j0 < gs {
                    let j1 = (j0 + 128).min(gs);
                    let mut part = 0.0f32;
                    for k in j0..j1 {
                        part = ((w[r * n + base + k] as f64) * (h_row[k] as f64) + (part as f64))
                            as f32;
                    }
                    acc = if first {
                        part
                    } else {
                        ((acc as f64) + (part as f64)) as f32
                    };
                    first = false;
                    j0 = j1;
                }
                out[r * n + base + j] = acc;
            }
        }
    }
    out
}

/// Shapes chosen to straddle the `PAR_MIN_ELEMENTS` (1<<16) threshold, so both
/// the sequential and parallel arms of the production function are covered —
/// including the boundary itself.
const SHAPES: &[(usize, usize, u32)] = &[
    (64, 256, 256),    // 16  elems — far below threshold
    (256, 256, 256),   // 65_536 — EXACTLY at threshold
    (255, 256, 256),   // 65_280 — one row below threshold
    (257, 256, 256),   // 65_792 — one row above
    (512, 512, 256),   // 262 144 — above, multi-group rows
    (1024, 4096, 256), // 4 194 304 — the real Qwen-Image shape
    (300, 1024, 256),  // 307 200 — above, wide rows
    (64, 64, 64),      // 4 096 — below threshold, smaller group size
    (128, 512, 16),    // 65 536 — at threshold, gs=16
];

#[test]
fn parallel_rotation_is_bit_identical_to_sequential() {
    for (rows, n, gs) in SHAPES {
        let h = build_hadamard(*gs).unwrap();
        let w = synth(
            rows * n,
            0xDEAD_BEEF_CAFE_F00D ^ (*rows as u64) << 32 ^ *n as u64,
        );
        let ours = rotate_weight(&w, &h, *rows, *n, *gs).unwrap();
        let want = sequential_rotate_reference(&w, &h, *rows, *n, *gs as usize);
        assert_eq!(ours.len(), want.len(), "{rows}x{n} gs{gs}: length");
        assert!(
            ours == want,
            "{rows}x{n} gs{gs}: parallel rotation is NOT bit-identical — \
             this is a parity event, not a tolerance to widen"
        );
    }
}

/// The pin must not depend on how rayon happened to split the work. A
/// non-deterministic result here would mean some value is being accumulated
/// across rows, which is exactly the bug the parallel version must not have.
#[test]
fn rotation_is_bit_identical_across_thread_counts() {
    let (rows, n, gs) = (1024usize, 4096usize, 256u32);
    let h = build_hadamard(gs).unwrap();
    let w = synth(rows * n, 0x1234_5678_9ABC_DEF0);
    let reference = rotate_weight(&w, &h, rows, n, gs).unwrap();

    for threads in [1usize, 2, 3, 7, 8] {
        // A scoped pool so the setting cannot leak into other tests.
        let got = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| rotate_weight(&w, &h, rows, n, gs).unwrap());
        assert_eq!(
            got, reference,
            "result changed at {threads} thread(s) — the rotation is not \
             thread-count independent, so it is accumulating across rows"
        );
    }
}

/// The rotation is the overwhelming majority of the work, so a change here
/// moves end-to-end time by an order of magnitude. Pin the SHAPE of the cost
/// so a future "optimization" that accidentally re-serializes it is caught by
/// a test rather than by a user waiting hours.
///
/// This is deliberately a RATIO and not a wall-clock bound: absolute timings
/// are machine-dependent, but "rotation must dominate quantize+dequantize" is
/// a structural property that holds on any core.
#[test]
fn rotation_still_dominates_the_other_stages() {
    use quant_core::quant::{dequantize_int8, quantize_int8_weight, ScalingMode};
    use std::time::Instant;

    let (rows, n, gs) = (2048usize, 2048usize, 256u32);
    let h = build_hadamard(gs).unwrap();
    let w = synth(rows * n, 0xFEED_FACE_CAFE_BEEF);

    // Warm up rayon so the first measured stage does not pay pool startup.
    let _ = rotate_weight(&w[..1024], &h, 1, 1024, gs).unwrap();

    let t = Instant::now();
    let w_rot = rotate_weight(&w, &h, rows, n, gs).unwrap();
    let rot = t.elapsed();

    let t = Instant::now();
    let r = quantize_int8_weight(&w_rot, rows, n, ScalingMode::Row, 128);
    let q = t.elapsed();

    let t = Instant::now();
    let _dq = dequantize_int8(&r.qdata, &r.scale, rows, n, ScalingMode::Row, 128);
    let dq = t.elapsed();

    let other = q + dq;
    assert!(
        rot > other,
        "rotation ({rot:?}) no longer dominates quantize+dequantize ({other:?}) — \
         the cost model changed; re-check before assuming the fix still holds"
    );
}

/// A single-row input must still rotate correctly — the parallel path must not
/// silently skip work when `rows` is small (an easy way to "optimize" this
/// into a wrong answer).
#[test]
fn single_row_is_not_skipped() {
    let (n, gs) = (256usize, 256u32);
    let h = build_hadamard(gs).unwrap();
    let w = synth(n, 0xABCD_1234_5678_9999);
    let got = rotate_weight(&w, &h, 1, n, gs).unwrap();
    let want = sequential_rotate_reference(&w, &h, 1, n, gs as usize);
    assert_eq!(got, want, "1-row rotation diverged");
    // And it must not be all zeros — a "parallel" arm that never runs would
    // pass a == b comparison against a broken reference of its own.
    assert!(
        got.iter().any(|&v| v != 0.0),
        "single row produced all zeros"
    );
}

/// Orthogonality must survive parallelization: H·W_rot over a group must
/// reconstruct W. This is a functional check that would catch a transposed
/// or skipped write that a pure self-comparison could miss.
#[test]
fn rotation_stays_invertible() {
    let (rows, n, gs) = (512usize, 512usize, 256u32);
    let h = build_hadamard(gs).unwrap();
    let w = synth(rows * n, 0x0F0F_0F0F_1234_5678);
    let w_rot = rotate_weight(&w, &h, rows, n, gs).unwrap();
    // Undo: W ≈ W_rot @ H (H is symmetric and orthogonal once normalized).
    let back = rotate_weight(&w_rot, &h, rows, n, gs).unwrap();
    for (i, (a, b)) in w.iter().zip(back.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "element {i}: rotate∘rotate gave {b}, want ~{a} — H is orthogonal, \
             so this must round-trip"
        );
    }
}

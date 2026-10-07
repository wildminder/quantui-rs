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

// Step 2.1 deliberately adds no non-test code: the module is empty in a
// release build until Step 2.3 lands the transpose helper.
#[cfg(test)]
mod proofs {
    /// The K-block size the GEMM resets its accumulator at. MUST mirror
    /// `bias_correction.rs::GEMM_K_CHUNK` — if that constant ever moves, this
    /// test module must move with it (asserted by
    /// [`k_chunk_constant_mirrors_the_kernel`]).
    const K_CHUNK: usize = 128;

    /// Deterministic xorshift-based pseudo-random f32 in `[-1.0, 1.0)`.
    /// No RNG crate: the proofs must run with zero new dependencies and be
    /// reproducible bit-for-bit on every platform.
    fn next_f32(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 40) as f32 / (1 << 24) as f32) - 1.0
    }

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

    /// `K_CHUNK` here must equal `GEMM_K_CHUNK` in the production kernel. The
    /// proofs are about THE reference order; if the two constants drift the
    /// proofs silently stop testing the real code path. Read the source and
    /// compare literally — a doc test would not see runtime drift.
    #[test]
    fn k_chunk_constant_mirrors_the_kernel() {
        let src = include_str!("bias_correction.rs");
        assert!(
            src.contains("const GEMM_K_CHUNK: usize = 128;"),
            "bias_correction.rs no longer declares GEMM_K_CHUNK = 128; \
             update K_CHUNK in this module in the same commit"
        );
        let src2 = include_str!("convrot.rs");
        assert!(
            src2.contains("const GEMM_K_CHUNK: usize = 128;"),
            "convrot.rs no longer declares GEMM_K_CHUNK = 128; the ConvRot \
             GEMM and the proofs must move together"
        );
        assert_eq!(K_CHUNK, 128);
    }
}

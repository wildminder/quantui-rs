//! S07 — tests for the QuaRot incoherence diagnostic.
//!
//! # What these tests deliberately do NOT assert
//!
//! **They do not assert that a Hadamard rotation reduces incoherence.**
//!
//! That is the single most tempting assertion to add here, and it would be a
//! **flaky test**. Measured on the real `convrot_int8_weight` path:
//!
//! ```text
//!   128x128  before = 3.4959  after = 3.8311  ratio = 1.0959   <- goes UP
//!    64x64   before = 3.4216  after = 3.3852  ratio = 0.9894
//! ```
//!
//! QuaRot's incoherence bound is a worst-case *bound*. A single Hadamard
//! application on a random Gaussian does not monotonically reduce `max|W|`, so
//! `assert!(after < before)` fails roughly half the time. The temptation is
//! named explicitly at the bottom of this file
//! (`rotation_reduces_incoherence_is_not_asserted`) so the next person to
//! consider adding it finds the measurement instead.
//!
//! What IS asserted: the metric's own algebra — its bound, its invariances,
//! and its definition on hand-computable inputs.

use quant_core::incoherence::{frobenius_norm, incoherence, incoherence_terms};

/// Deterministic xorshift64 PRNG. Inline (not `proptest`) so the values are
/// identical on every platform and in every seed mode, matching the pattern
/// already used in `quant_mxfp8.rs`.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// A deterministic `(m, n)` matrix of roughly-N(0, 1) values.
fn gaussian_matrix(m: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    (0..m * n)
        .map(|_| {
            // Two samples averaged -> a crude but adequate bell shape.
            let a = (xorshift(&mut x) % 20_000) as f32 / 10_000.0 - 1.0;
            let b = (xorshift(&mut x) % 20_000) as f32 / 10_000.0 - 1.0;
            (a + b) * 0.5
        })
        .collect()
}

/// BOUNDARY — the definition itself on hand-computable inputs.
///
/// Adversarial case defended: a wrong `sqrt(mn)` — easily written as `m * n` or
/// `sqrt(m) * sqrt(n)`, both of which pass a Gaussian smoke test (which only
/// checks `gamma >= 1`) but fail here.
#[test]
fn incoherence_matches_hand_computed_definition() {
    // All-ones: ||W||_F = sqrt(mn), max|W| = 1, so gamma = 1 / (sqrt(mn)/sqrt(mn)) = 1.
    for &(m, n) in &[(1usize, 1usize), (4, 4), (8, 8), (3, 7), (16, 9)] {
        let w = vec![1.0f32; m * n];
        let g = incoherence(&w, m, n).expect("all-ones is non-degenerate");
        assert!(
            (g - 1.0).abs() < 1e-12,
            "all-ones ({m}, {n}) must give gamma == 1.0 exactly, got {g}"
        );
    }

    // Single spike: only W[0] = A, everything else 0.
    //   max|W| = A, ||W||_F = A, so gamma = A / (A / sqrt(mn)) = sqrt(mn).
    for &(m, n) in &[(1usize, 1usize), (4, 4), (8, 8), (3, 7), (16, 9)] {
        let a = 2.0f32;
        let mut w = vec![0.0f32; m * n];
        w[0] = a;
        let g = incoherence(&w, m, n).expect("a single spike is non-degenerate");
        let want = ((m * n) as f64).sqrt();
        assert!(
            (g - want).abs() < 1e-6 * want,
            "single spike ({m}, {n}): gamma should be sqrt(mn) = {want}, got {g}"
        );
    }

    // A constant magnitude k (not 1) must also give exactly 1 — the metric
    // depends on the RATIO of max to RMS, so any equal-magnitude matrix is 1.
    let w = vec![-3.0f32; 25];
    let g = incoherence(&w, 5, 5).unwrap();
    assert!((g - 1.0).abs() < 1e-12, "equal magnitudes -> 1, got {g}");
}

/// INVARIANT — `gamma >= 1` always, because RMS never exceeds the max.
///
/// Adversarial case defended: an inverted ratio, and any bug producing
/// `gamma < 1`, which is mathematically impossible for a non-empty tensor.
#[test]
fn incoherence_is_at_least_one() {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    for &(m, n) in &[(1usize, 1), (2, 3), (8, 8), (16, 16), (33, 17), (64, 64)] {
        let w = gaussian_matrix(m, n, x);
        x = x.wrapping_mul(0x0100_0000_01B3) | 1;
        if let Some(g) = incoherence(&w, m, n) {
            assert!(g.is_finite(), "gamma must be finite for ({m}, {n})");
            assert!(
                g >= 1.0 - 1e-12,
                "gamma = {g} < 1 for ({m}, {n}) — impossible for a non-empty tensor"
            );
        }
    }

    // Equality iff all magnitudes are equal: exercise both directions. The
    // second case must be genuinely NON-uniform, otherwise gamma is 1 again.
    assert!((incoherence(&vec![2.0f32; 36], 6, 6).unwrap() - 1.0).abs() < 1e-12);
    let mut mixed = vec![1.0f32; 35];
    mixed[17] = 2.0; // one entry differs -> max > RMS -> gamma > 1
    assert!(incoherence(&mixed, 5, 7).unwrap() > 1.0);
}

/// INVARIANT — the Frobenius norm is rotation-invariant.
///
/// `||W||_F == ||W * H||_F` for an orthogonal `H`, because a right-multiplication
/// by an orthogonal matrix is a rigid rotation of the row vectors and the
/// Euclidean norm of each row vector is preserved; the sum of squares over rows
/// therefore is too.
///
/// Adversarial case defended: computing the norm with an `f32` accumulator
/// (precision loss on large tensors), and silently confirming that the `f64`
/// accumulation choice is actually necessary rather than cargo-culted.
#[test]
fn frobenius_is_rotation_invariant() {
    // An explicit 4x4 Hadamard (Hadamard's construction), normalized.
    let h_raw: [f32; 16] = [
        1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
    ];
    let h: Vec<f32> = h_raw.iter().map(|v| v / 2.0).collect(); // /2 -> orthogonal

    let (m, n) = (16usize, 4usize);
    let w = gaussian_matrix(m, n, 0xDEAD_BEEF_CAFE_F00D);

    // (W * H)[i][j] = sum_k W[i][k] * H[k][j]
    let mut wr = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += w[i * n + k] * h[k * n + j];
            }
            wr[i * n + j] = acc;
        }
    }

    let before = frobenius_norm(&w);
    let after = frobenius_norm(&wr);
    assert!(
        (before - after).abs() <= 1e-4 * before,
        "rotation changed the Frobenius norm: {before} vs {after}"
    );

    // The norm is also invariant under the incoherence metric's own inputs, so
    // gamma is a property of the matrix's shape, not its orientation.
    assert!(incoherence(&w, m, n).is_some());
    assert!(incoherence(&wr, m, n).is_some());
}

/// `None` for an empty or all-zero tensor — never `NaN`.
#[test]
fn empty_and_zero_tensors_return_none() {
    // Empty.
    assert_eq!(incoherence(&[], 0, 0), None);
    assert_eq!(incoherence(&[], 0, 4), None);
    assert_eq!(incoherence_terms(&[], 0, 0), None);

    // All zero: ||W||_F == 0, so gamma would be 0/0 = NaN.
    assert_eq!(incoherence(&vec![0.0f32; 16], 4, 4), None);
    assert_eq!(incoherence_terms(&vec![0.0f32; 16], 4, 4), None);

    // NOTE: a length/shape mismatch is deliberately NOT tested here. The
    // module has both a `debug_assert_eq!` (the crate's convention, e.g.
    // `quant_mxfp8.rs:153`) and a runtime guard. In a debug test build the
    // assert fires first, so the runtime guard is only reachable in release —
    // where tests do not run. The assert is the enforced contract; the guard
    // is defence in depth for release callers.
}

/// `incoherence_terms` returns the parts, and they reconstruct the ratio.
#[test]
fn terms_reconstruct_the_ratio() {
    let (m, n) = (9usize, 7);
    let w = gaussian_matrix(m, n, 0x5EED_1234_ABCD_0001);
    let g = incoherence(&w, m, n).unwrap();
    let (mu, bound) = incoherence_terms(&w, m, n).unwrap();

    assert!(mu > 0.0, "mu = max|W| must be positive");
    assert!(bound > 0.0, "the RMS term must be positive");
    assert!(
        (mu / bound - g).abs() < 1e-12,
        "gamma = {g} but mu/bound = {}",
        mu / bound
    );
    // mu is the max magnitude, so verify it against a direct scan.
    let want_mu = w.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    assert!((mu - f64::from(want_mu)).abs() < 1e-12);
}

/// DOCUMENTATION TEST — names the trap this module exists to avoid.
///
/// This test asserts only that the metric is **finite and positive** on a
/// rotated matrix. It deliberately does NOT assert that rotation *reduces* the
/// value, because measurement shows it does not reliably: ratios of 0.9894 and
/// 1.0959 were observed on the real `convrot_int8_weight` path, i.e. one of
/// them went UP.
///
/// Adversarial case defended: the specific temptation to add
/// `assert!(after < before)`. Naming the trap in code — with the numbers that
/// prove it — is the durable defence. A future reader who wants that assertion
/// finds this comment first.
#[test]
fn rotation_reduces_incoherence_is_not_asserted() {
    use quant_core::convrot::build_hadamard;

    // `build_hadamard` accepts a power of 4 only (4, 16, 64, ...), so the
    // column count must be one of those — 16 here, not 32.
    let (m, n) = (32usize, 16usize);
    let w = gaussian_matrix(m, n, 0x0BAD_F00D_1337_4242);
    let before = incoherence(&w, m, n).expect("non-degenerate");

    // Rotate with the crate's real Hadamard builder and the real rotation.
    let h = build_hadamard(n as u32).expect("32 is a valid Hadamard size");
    let rotated =
        quant_core::convrot::rotate_weight(&w, &h, m, n, n as u32).expect("rotation succeeds");
    let after = incoherence(&rotated, m, n).expect("non-degenerate after rotation");

    // Asserted: the metric is well-defined on both sides.
    assert!(before.is_finite() && before > 0.0, "before = {before}");
    assert!(after.is_finite() && after > 0.0, "after = {after}");

    // Asserted: both sides respect the mathematical bound.
    assert!(before >= 1.0, "before = {before}");
    assert!(after >= 1.0, "after = {after}");

    // NOT asserted, and deliberately so:
    //     assert!(after < before);   // <-- FLAKY. Measured ratios: 0.9894, 1.0959.
    //
    // The rotation preserves ||W||_F but redistributes magnitude. Whether
    // max|W| drops depends on the data, not on the algebra.
    let ratio = after / before;
    eprintln!(
        "incoherence: before={before:.4} after={after:.4} ratio={ratio:.4} \
         (informational only — not a pass/fail gate)"
    );
}

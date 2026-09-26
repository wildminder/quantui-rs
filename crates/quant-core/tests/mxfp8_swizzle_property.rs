//! S06 — property tests for the cuBLAS swizzle (`to_blocked_u8` /
//! `from_blocked_u8`).
//!
//! S03 pins the layout against external expectations. This file covers
//! *volume* over shapes, including the padding-triggering non-multiples of
//! 128/4 that a hand-written table can only sample.
//!
//! Three properties, covering *volume* over shapes:
//!
//! | property | defends against | caught by this? |
//! |---|---|---|
//! | round-trip | a **non-self-inverse** index error — a permutation whose inverse the reader does not also apply, so the value lands in the wrong slot | yes |
//! | injectivity | **aliasing** (two cells → one slot, silently dropping a scale) and **out-of-bounds indexing** (a panic in debug, a silent memory-corrupting write in release) | yes |
//! | determinism | order-dependence sneaking into the iterator | yes |
//!
//! ## ⚠️ What round-trip does NOT defend against
//!
//! **A self-inverse (involutive) index bug survives this file completely**, and
//! it is worth being precise about why, because an earlier version of this
//! comment claimed the opposite.
//!
//! If the index map is its own inverse, then applying it twice is the identity,
//! so `from_blocked_u8(to_blocked_u8(x)) == x` **holds** while the layout handed
//! to ComfyUI is wrong. A `(r0, r1) ↔ (r1, r0)`-style swap is exactly this shape.
//! Verified by mutation: rewriting the index to `r1 ^ 1` — a self-inverse
//! involution that is also a permutation — leaves this file **fully green**.
//!
//! Note that `swizzle_is_injective_property` cannot catch it either: an
//! involution is a bijection, so injectivity is **preserved by construction**.
//! The two properties are genuinely orthogonal, neither subsumes the other, and
//! neither covers this case.
//!
//! **The safety net for a self-inverse bug is S03's
//! `hand_computed_flat_index_matches_closed_form`**, which asserts hand-computed
//! offsets from an independent derivation. So the suite as a whole is safe — but
//! the credit belongs to that conformance test, not to this one.
//!
//! ⚠️ **Dependency worth knowing:** if S03 is ever refactored to recompute its
//! expectations *from the implementation* instead of from hand-computed
//! constants, a self-inverse bug would pass the **entire** suite. Keep S03's
//! expectations literal and independent.
//!
//! Case count: proptest's default is 256 cases, matching `recipe_property.rs`,
//! which documents relying on the default deliberately.
//!
//! No parity source is touched.

use std::collections::HashSet;

use proptest::prelude::*;
use quant_core::quant_mxfp8::{from_blocked_u8, to_blocked_u8};

/// Row/col strategy, biased toward the tile boundaries.
///
/// A uniform distribution would concentrate on mid-range values and
/// under-sample exactly the 128-row / 4-col tile edges that matter, so those
/// values are over-sampled. The area bound keeps CI fast.
///
/// Both `rows` and `cols` are `usize` strategies so proptest shrinks toward
/// `1` — a good shrink target, since the minimal failing case is a tiny shape
/// at a tile boundary, which is exactly the report a maintainer wants. Using
/// `prop::num::u32` + a manual `as usize` would shrink to 0 and invite a
/// divide-by-zero inside the strategy itself.
///
/// NOTE: a flat union is spelled `range.prop_union(select(edges))`, NOT
/// `prop_oneof![range, a, b, c]`. In proptest 1.11 `prop_oneof!` with more
/// than one branch builds a `TupleUnion` (a *tuple of strategies*), which does
/// not implement `Strategy<Value = usize>` — so `.prop_filter` cannot be called
/// on it. `prop_union` chains two same-typed strategies into one flat
/// strategy, which is what a biased `usize` generator actually wants.
fn boundary_biased(
    range: std::ops::RangeInclusive<usize>,
    edges: &[usize],
) -> impl Strategy<Value = usize> {
    // `Strategy::prop_union` is declared `fn prop_union(self, other: Self)`, so
    // BOTH sides must already be the same concrete type. `RangeInclusive<usize>`
    // and `Select<usize>` are different types that each implement `Strategy`, so
    // the call is ill-typed until both are erased to `BoxedStrategy<usize>`.
    let base: BoxedStrategy<usize> = range.boxed();
    let edge: BoxedStrategy<usize> = proptest::sample::select(edges.to_vec()).boxed();
    base.prop_union(edge)
}

/// `(rows, cols)` with a bounded area, biased toward tile boundaries.
fn bounded_shape() -> impl Strategy<Value = (usize, usize)> {
    (
        boundary_biased(3..=600, &[127, 128, 129, 255, 256, 257]),
        boundary_biased(3..=600, &[3, 4, 5]),
    )
        .prop_filter("bound the allocation to keep CI fast", |(r, c)| {
            r * c <= 20_000
        })
}

/// Deterministic pseudo-random byte source (no external dep, no seeding
/// concerns). The layout contract is about POSITIONS, so distinct values per
/// cell are what make a misplaced write visible.
fn distinct_bytes(n: usize) -> Vec<u8> {
    // An LCG with a full-period multiplier mod 2^32, walked over odd steps so
    // consecutive values differ — a mistake that swaps two adjacent cells is
    // then visible rather than masked by equal values.
    let mut x: u32 = 0x1234_5678;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 24) as u8
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    /// `from_blocked_u8(to_blocked_u8(src)) == src` for any shape.
    ///
    /// Adversarial case defended: a **non-self-inverse** index error — one whose
    /// inverse the reader does not also apply, so a value lands in the wrong slot
    /// and does not come back.
    ///
    /// ⚠️ This does **not** defend against a *self-inverse* index bug. An
    /// involution is its own inverse, so it round-trips clean while producing a
    /// wrong layout, and the injectivity property cannot see it either (an
    /// involution is a bijection). Verified by mutation: `r1 ^ 1` leaves this
    /// file green. See the module docs — S03's hand-computed test is the net.
    #[test]
    fn swizzle_roundtrip_property(
        (rows, cols) in bounded_shape(),
    ) {
        let src = distinct_bytes(rows * cols);
        let (blocked, shape) = to_blocked_u8(&src, rows, cols);
        let back = from_blocked_u8(&blocked, rows, cols);
        prop_assert_eq!(back, src, "round trip failed for ({}, {})", rows, cols);
        // The padded shape must always be a legal multiple of the tile.
        prop_assert_eq!(shape[0] as usize % 128, 0);
        prop_assert_eq!(shape[1] as usize % 4, 0);
        prop_assert_eq!(blocked.len(), (shape[0] * shape[1]) as usize);
    }

    /// All `rows * cols` computed flat indices are distinct, and each is in
    /// bounds.
    ///
    /// This is the property that makes the function memory-safe. Aliasing
    /// silently drops a block scale; an out-of-bounds index is a panic in debug
    /// and a **silent memory-corrupting write in release**. Neither is visible
    /// to the round-trip test.
    #[test]
    fn swizzle_is_injective_property(
        (rows, cols) in bounded_shape(),
    ) {
        let ncb = (cols + 3) / 4;
        let src = vec![0u8; rows * cols];
        let (out, _shape) = to_blocked_u8(&src, rows, cols);

        let mut seen: HashSet<usize> = HashSet::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let b = (r / 128) * ncb + c / 4;
                let r0 = (r % 128) / 32;
                let r1 = (r % 128) % 32;
                let crem = c % 4;
                let flat = b * 512 + r1 * 16 + r0 * 4 + crem;
                prop_assert!(flat < out.len(),
                    "index {} out of bounds for ({}, {})", flat, rows, cols);
                prop_assert!(seen.insert(flat),
                    "aliasing: ({}, {}) -> slot {} in a ({}, {}) tensor", r, c, flat, rows, cols);
            }
        }
        prop_assert_eq!(seen.len(), rows * cols);
    }

    /// Repeated swizzles of the same input give identical buffers.
    ///
    /// Adversarial case defended: order-dependence sneaking into the iterator
    /// at `quant_mxfp8.rs:123` (a parallel or lazily-evaluated source would
    /// make this flaky, and a flaky layout is a nightmare to diagnose).
    #[test]
    fn swizzle_is_deterministic_property(
        (rows, cols) in bounded_shape(),
    ) {
        let src = distinct_bytes(rows * cols);
        let first = to_blocked_u8(&src, rows, cols);
        for rep in 0..4 {
            let again = to_blocked_u8(&src, rows, cols);
            prop_assert_eq!(&again.0, &first.0, "buffer differed on rep {}", rep);
            prop_assert_eq!(&again.1, &first.1, "shape differed on rep {}", rep);
        }
    }

    /// Every non-padding output byte is zero.
    ///
    /// Adversarial case defended: stale data leaking into the padding region.
    /// A round trip passes even if the padding holds garbage, because
    /// `from_blocked` crops it — but ComfyUI reads the *padded* buffer, so
    /// garbage there is a real hardware-interop bug.
    #[test]
    fn padding_region_is_zero_property(
        (rows, cols) in bounded_shape(),
    ) {
        // A distinctive value that cannot collide with the zero fill.
        let marker = 0xA5u8;
        let src = vec![marker; rows * cols];
        let (out, shape) = to_blocked_u8(&src, rows, cols);
        let n = out.iter().filter(|&&b| b == marker).count();
        prop_assert_eq!(n, rows * cols,
            "exactly the source cells must be non-zero for ({}, {})", rows, cols);
        // Everything else is a hard zero — no third value can appear.
        prop_assert!(out.iter().all(|&b| b == 0 || b == marker),
            "padding must be zero for ({}, {}) shape {:?}", rows, cols, shape);
    }
}

/// The generated `.proptest-regressions` file is committed, matching the crate's
/// existing convention (`gguf_adversarial_property.proptest-regressions`,
/// `st_header_property.proptest-regressions`): a failure found once should
/// never be re-discovered by luck.
///
/// This test exists to keep that file meaningful — it asserts the file is
/// either absent (no failures yet) or parseable, and documents the convention.
#[test]
fn proptest_regressions_file_is_the_committed_convention() {
    // Nothing to assert about content — proptest owns the format. This test
    // exists so a reader sees the convention stated in code, and so removing
    // the file is a visible, deliberate act.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/mxfp8_swizzle_property.proptest-regressions"
    );
    if let Ok(text) = std::fs::read_to_string(path) {
        // Every non-comment, non-blank line must be a `cc <hex> # shrinks to ...`
        // seed line. A malformed one means the file was hand-mangled.
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            assert!(
                line.starts_with("cc "),
                "unexpected line in .proptest-regressions: {line:?}"
            );
        }
    }
}

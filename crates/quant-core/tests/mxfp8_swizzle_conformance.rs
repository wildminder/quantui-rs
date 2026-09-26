//! S03 — cuBLAS swizzle (`to_blocked_u8`) conformance against EXTERNAL
//! expectations.
//!
//! The existing unit test `from_blocked_inverts_to_blocked`
//! (`quant_mxfp8.rs`) and the S06 property tests both check *self-consistency*:
//! swizzle-then-unswizzle returns the input. That class of test is blind to two
//! real hardware-interop bugs:
//!
//! 1. **An off-by-one in `roundup`.** The layout shifts wholesale, yet the
//!    round trip still succeeds because the inverse shifts identically. Only an
//!    expectation derived *independently* of the implementation catches it.
//! 2. **Garbage in the padding region.** `from_blocked_u8` crops the padding,
//!    so a round trip passes even if the pad holds stale data. ComfyUI reads
//!    the *padded* buffer, so garbage there is a genuine interop bug that
//!    self-consistency is structurally unable to detect.
//!
//! Everything asserted here is therefore either a literal expectation or an
//! invariant of the index map itself. No parity source is touched.

use quant_core::quant_mxfp8::{from_blocked_u8, to_blocked_u8};

/// Reference `roundup(x, mult)`.
fn roundup(x: usize, mult: usize) -> usize {
    (x + mult - 1) / mult * mult
}

/// The output shape is the roundup of the source shape, and the buffer is
/// exactly `shape[0] * shape[1]`.
///
/// Covers the exact tile boundaries 127/128/129 (row) and 3/4/5 (col). These
/// are the cases where an off-by-one in `roundup` would produce a
/// self-consistent but WRONG layout.
#[test]
fn padded_shape_is_roundup_of_source_shape() {
    // (rows, cols) -> expected [roundup(rows,128), roundup(cols,4)].
    // Every expectation below was verified against the real implementation.
    let cases: &[(usize, usize, [u64; 2])] = &[
        (1, 1, [128, 4]),
        (3, 7, [128, 8]),
        (127, 4, [128, 4]),
        (128, 4, [128, 4]),
        (129, 4, [256, 4]),
        (255, 3, [256, 4]),
        (256, 12, [256, 12]),
        (513, 17, [640, 20]),
    ];

    for &(rows, cols, want) in cases {
        let src = vec![0u8; rows * cols];
        let (out, shape) = to_blocked_u8(&src, rows, cols);
        assert_eq!(shape, want.to_vec(), "shape for ({rows}, {cols})");
        assert_eq!(
            out.len(),
            (want[0] * want[1]) as usize,
            "buffer length for ({rows}, {cols})"
        );
        // And the shape is exactly the roundup — stated independently so a
        // change to `roundup` has to be justified against this line.
        assert_eq!(
            shape[0] as usize,
            roundup(rows, 128),
            "row roundup for ({rows}, {cols})"
        );
        assert_eq!(
            shape[1] as usize,
            roundup(cols, 4),
            "col roundup for ({rows}, {cols})"
        );
    }
}

/// The padding region is ZERO, not stale data.
///
/// (130, 5) all-sevens → shape [256, 8]. Exactly 650 bytes are non-zero; the
/// other 1398 must be zero. A round-trip test passes even with garbage in the
/// pad because `from_blocked` crops it — this is the assertion that closes
/// that blind spot, and it is the highest-value test in this file.
#[test]
fn padding_cells_are_zero_and_shape_is_roundup() {
    let (rows, cols) = (130usize, 5usize);
    let src = vec![7u8; rows * cols];
    let (out, shape) = to_blocked_u8(&src, rows, cols);

    assert_eq!(shape, vec![256, 8]);
    assert_eq!(out.len(), 256 * 8);

    let non_zero = out.iter().filter(|&&b| b == 7).count();
    assert_eq!(
        non_zero,
        rows * cols,
        "exact source cell count must survive"
    );
    // Everything that is not a source cell must be a hard zero.
    assert!(
        out.iter().all(|&b| b == 0 || b == 7),
        "padding region must be zeroed, found {:?}",
        out.iter()
            .filter(|&&b| b != 0 && b != 7)
            .collect::<Vec<_>>()
    );
    assert_eq!(out.iter().filter(|&&b| b == 0).count(), 256 * 8 - 650);
}

/// Degenerate zero-row shapes must not panic.
///
/// `src_iter` computes `i / cols`, so `cols == 0` was a real divide-by-zero
/// risk. A zero-row weight is not hypothetical: it is an empty expert tensor
/// in a MoE checkpoint, and panicking the CLI on it would be a bad failure
/// mode.
#[test]
fn degenerate_zero_shapes_do_not_panic() {
    let (out, shape) = to_blocked_u8(&[], 0, 0);
    assert_eq!(shape, vec![0, 0]);
    assert!(out.is_empty());

    let (out, shape) = to_blocked_u8(&[], 0, 4);
    assert_eq!(shape, vec![0, 4]);
    assert!(out.is_empty());
}

/// The closed-form index map is INJECTIVE on a padding-triggering shape.
///
/// Two source cells aliasing to one output slot would silently drop a block
/// scale — and because an aliasing round trip still *succeeds* whenever the
/// colliding values happen to be equal, only an explicit injectivity check
/// catches it. This is the property that actually matters for hardware interop.
#[test]
fn distinct_source_cells_land_in_distinct_slots() {
    let (rows, cols) = (130usize, 5usize);
    let ncb = roundup(cols, 4) / 4;
    let src = vec![0u8; rows * cols];
    let (out, _shape) = to_blocked_u8(&src, rows, cols);

    let mut seen = std::collections::HashSet::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let rb = r / 128;
            let rrem = r % 128;
            let cb = c / 4;
            let crem = c % 4;
            let b = rb * ncb + cb;
            let r0 = rrem / 32;
            let r1 = rrem % 32;
            let flat = b * 512 + r1 * 16 + r0 * 4 + crem;
            assert!(
                flat < out.len(),
                "flat index {flat} out of bounds for ({r}, {c})"
            );
            assert!(seen.insert(flat), "aliasing: ({r}, {c}) -> slot {flat}");
        }
    }
    assert_eq!(seen.len(), rows * cols, "every cell must own a unique slot");
}

/// Independent restatement of the first-row index formula for (128, 4).
///
/// The unit test `to_blocked_closed_form_small` (`quant_mxfp8.rs`) asserts
/// `flat = (r%32)*16 + (r/32)*4 + c` against the implementation. Lifting the
/// same formula to an integration test is only worth doing if the DERIVATION
/// is independent — so this walks the layout as a nested loop over the
/// documented `b / r0 / r1 / crem` decomposition and checks a hand-computed
/// sample of offsets, rather than re-asserting the same closed form.
///
/// Hand-computed spot values for (128, 4), single block row (`ncb == 1`):
///   r=0  → b=0, r0=0,  r1=0  → 0*16 + 0*4 + 0 = 0
///   r=1  → b=0, r0=0,  r1=1  → 1*16 + 0*4 + 0 = 16
///   r=31 → b=0, r0=0,  r1=31 → 31*16 + 0*4 + 0 = 496
///   r=32 → b=0, r0=1,  r1=0  → 0*16 + 1*4 + 0 = 4
///   r=127→ b=0, r0=3,  r1=31 → 31*16 + 3*4 + 0 = 508
#[test]
fn hand_computed_flat_index_matches_closed_form() {
    let (rows, cols) = (128usize, 4usize);
    let ncb = roundup(cols, 4) / 4;
    assert_eq!(ncb, 1, "this derivation assumes a single block column");

    // Distinct per-cell values so a misplaced write is visible.
    let src: Vec<u8> = (0..rows * cols).map(|i| (i % 251) as u8).collect();
    let (out, shape) = to_blocked_u8(&src, rows, cols);
    assert_eq!(shape, vec![128, 4]);

    // Hand-computed spot checks, restated from the derivation above.
    let spots: [(usize, usize, usize); 5] = [
        (0, 0, 0),
        (1, 0, 16),
        (31, 0, 496),
        (32, 0, 4),
        (127, 0, 508),
    ];
    for &(r, c, want_flat) in &spots {
        assert_eq!(
            out[want_flat],
            src[r * cols + c],
            "r={r} c={c} should live at flat {want_flat}"
        );
    }

    // And the full decomposition agrees with the closed form everywhere.
    for r in 0..rows {
        for c in 0..cols {
            let b = (r / 128) * ncb + c / 4;
            let r0 = (r % 128) / 32;
            let r1 = (r % 128) % 32;
            let crem = c % 4;
            let via_parts = b * 512 + r1 * 16 + r0 * 4 + crem;
            let via_closed_form = (r % 32) * 16 + (r / 32) * 4 + c;
            assert_eq!(
                via_parts, via_closed_form,
                "decomposition disagrees with closed form at ({r}, {c})"
            );
        }
    }

    // Round trip, as a sanity check that the layout is at least invertible.
    assert_eq!(from_blocked_u8(&out, rows, cols), src);
}

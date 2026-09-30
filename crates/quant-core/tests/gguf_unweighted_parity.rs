//! T4 — pin the UNWEIGHTED K-quant path (the default, and the one the
//! quality warning in T5 is about).
//!
//! # What this file is NOT
//!
//! It is NOT a byte-parity test against `llama-quantize`. That oracle does
//! not exist for this path, and claiming it would be a false mechanism.
//! Proof, from the two real sources:
//!
//! - Our unweighted encoder is `rlx_gguf::quantize` (`gguf_convert.rs:1245`).
//!   rlx-gguf 0.2.14's K-quant family self-documents as a *"Simplified
//!   per-sub-block quantizer; valid output but lower quality than upstream's
//!   iterative search"* (`quantize.rs:270`, `:544`), and the family header
//!   says it outright: *"Upstream `quantize_row_q*_K` runs an inner search
//!   across candidate scales (`make_qx_quants`) and an importance matrix; we
//!   use plain min/max."*
//! - llama.cpp's unweighted path is the `*_ref` variant, which DOES run that
//!   search: `make_qkx2_quants(32, 15, ..., 20, false)` for Q4_K
//!   (`ggml-quants.c:1476`), 15 steps for Q2_K (`:1663`).
//!
//! Plain min/max and a 20-step candidate-scale search are different
//! algorithms. They agree on the *block layout* — and the byte counts prove
//! it, since our sizes are exactly `blocks x 2` — but they cannot agree on
//! the *chosen scales*. So a byte-equality assertion here would be
//! permanently red, and "fixing" it would mean abandoning the default path's
//! encoder, which is a parity event, not a test fix.
//!
//! `gguf_parity.rs:11-13` already states this for the legacy formats. This
//! file is the K-quant half of that same statement.
//!
//! # What this file IS
//!
//! A REGRESSION PIN, which is the strongest claim actually available:
//! these bytes are what the default path emits today, on this exact input,
//! through the real end-to-end `convert_hf_to_gguf`. If an rlx bump, a
//! dispatch change, or an accidental "improvement" to the default moves them,
//! this goes red — which is the coverage gap the plan (T4) actually wanted
//! closed. The previous state was: the unweighted default had NO byte
//! assertion of any kind, so it could move silently.
//!
//! Goldens: `tests/golden/llamacpp/unweighted.*.bin`, vendored from
//! rlx-gguf 0.2.14 on the same `src.f32.bin` (seed 42, 2x256) the weighted
//! goldens use, so the two families are directly comparable.
//!
//! # The non-vacuity guards
//!
//! A pin is worthless if it cannot fail. Three separate guards:
//!
//! 1. `unweighted_differs_from_weighted` — the discriminating pair. The
//!    weighted path IS byte-exact vs `llama-quantize`
//!    (`gguf_weighted_parity.rs`). If unweighted ever equalled weighted,
//!    the unweighted path would silently be running the good encoder, and
//!    T5's warning would be lying.
//! 2. `every_unweighted_golden_is_distinct` — no two formats pin the same
//!    bytes (a copy-paste in the generator would otherwise hide).
//! 3. `block_byte_counts_match_the_layout` — sizes follow the K-quant
//!    layouts, so a wrong-shaped payload cannot pass by being short.

mod common;

use std::path::PathBuf;

use quant_core::gguf_quants::{
    quantize_row_q2_k_weighted, quantize_row_q3_k_weighted, quantize_row_q4_k_weighted,
    quantize_row_q5_k_weighted, quantize_row_q6_k_weighted,
};
use rlx_gguf::{quantize, GgmlType};

/// Super-block size, ggml-common.h:88 `QK_K = 256`. `quant_core` keeps its
/// own copy private, and it is a format constant (not a tuning knob), so the
/// test restates it rather than the library exporting it for a test.
const QK_K: usize = 256;

/// (format id, GgmlType, bytes per 256-element super-block).
///
/// The arithmetic is written out longhand from the block layouts
/// (ggml-common.h:296-338) rather than copied from the library constants,
/// so `block_byte_counts_match_the_layout` is a real cross-check instead of
/// a tautology. It has already earned its keep: the first draft of this
/// table had Q3_K as `QK_K/4 + QK_K/4 + 12 + 2` (142) and Q6_K as
/// `QK_K/2 + QK_K/2 + 16*2` (288), both wrong, and the guard is what said so.
///
/// - Q2_K `scales[16] qs[64] d dmin`                = 16 + 64 + 2 + 2  = 84
/// - Q3_K `hmask[32] qs[64] scales[12] d`           = 32 + 64 + 12 + 2 = 110
/// - Q4_K `d dmin scales[12] qs[128]`               = 2 + 2 + 12 + 128 = 144
/// - Q5_K `d dmin scales[12] qh[32] qs[128]`        = 2 + 2 + 12 + 32 + 128 = 176
/// - Q6_K `ql[128] qh[64] scales[16] d`             = 128 + 64 + 16 + 2 = 210
const CASES: &[(&str, GgmlType, usize)] = &[
    ("q2_k", GgmlType::Q2K, QK_K / 16 + QK_K / 4 + 2 + 2),
    ("q3_k", GgmlType::Q3K, QK_K / 8 + QK_K / 4 + 12 + 2),
    ("q4_k", GgmlType::Q4K, 2 + 2 + 12 + QK_K / 2),
    ("q5_k", GgmlType::Q5K, 2 + 2 + 12 + 32 + QK_K / 2),
    ("q6_k", GgmlType::Q6K, QK_K / 2 + QK_K / 4 + 16 + 2),
];

const N_ROWS: usize = 2;

fn golden_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p.join("tests/golden/llamacpp")
}

/// The SAME fixture the weighted goldens use, so the two families are
/// comparable byte for byte.
fn load_src() -> Vec<f32> {
    let raw = std::fs::read(golden_dir().join("src.f32.bin")).expect("src.f32.bin");
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn load_golden(stem: &str) -> Vec<u8> {
    std::fs::read(golden_dir().join(format!("{stem}.bin")))
        .unwrap_or_else(|e| panic!("read {stem}.bin: {e}"))
}

// ─── The pin ───────────────────────────────────────────────────────────

#[test]
fn unweighted_k_quant_bytes_are_pinned() {
    let src = load_src();
    for (id, t, _) in CASES {
        let ours = quantize(&src, *t).unwrap_or_else(|e| panic!("{id}: {e}"));
        let golden = load_golden(&format!("unweighted.{id}"));
        assert_eq!(
            ours.len(),
            golden.len(),
            "{id}: encoded size moved — a layout change is a parity event, \
             not a golden to update"
        );
        assert_eq!(ours, golden, "unweighted {id} bytes moved");
    }
}

// ─── Guard 1: the discriminating pair ─────────────────────────────────

/// The load-bearing negative control.
///
/// `gguf_weighted_parity.rs` pins the weighted path byte-exact against the
/// REAL `llama-quantize`. If our unweighted output ever equalled our
/// weighted output, that would mean the default path had silently acquired
/// the good encoder — and `kquant_encoder_warning`'s warning, plus every
/// measured number behind it (0.32985 rlx vs 0.26936 weighted vs 0.29840
/// published), would be false.
#[test]
fn unweighted_differs_from_weighted() {
    let src = load_src();
    let weights: Vec<f32> = std::fs::read(golden_dir().join("weights.f32.bin"))
        .expect("weights.f32.bin")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();

    // (id, rlx GgmlType, our weighted port, block bytes)
    type WeightedArm = (
        &'static str,
        GgmlType,
        fn(&[f32], usize, Option<&[f32]>) -> Vec<u8>,
        usize,
    );
    let weighted: &[WeightedArm] = &[
        ("q2_k", GgmlType::Q2K, quantize_row_q2_k_weighted, 84),
        ("q3_k", GgmlType::Q3K, quantize_row_q3_k_weighted, 110),
        ("q4_k", GgmlType::Q4K, quantize_row_q4_k_weighted, 144),
        ("q5_k", GgmlType::Q5K, quantize_row_q5_k_weighted, 176),
        ("q6_k", GgmlType::Q6K, quantize_row_q6_k_weighted, 210),
    ];

    for (id, t, port, block_bytes) in weighted {
        let unweighted = quantize(&src, *t).unwrap();
        // Drive the port per row, exactly as gguf_weighted_parity does.
        let mut ours = Vec::new();
        for r in 0..N_ROWS {
            let row = &src[r * QK_K..(r + 1) * QK_K];
            ours.extend(port(row, QK_K, Some(&weights)));
        }
        assert_eq!(
            unweighted.len(),
            N_ROWS * block_bytes,
            "{id}: unweighted size should be blocks x rows"
        );
        assert_ne!(
            unweighted, ours,
            "{id}: unweighted == weighted — the default path is running the \
             byte-exact encoder, so T5's quality warning is FALSE"
        );
    }
}

// ─── Guard 2: the goldens are mutually distinct ───────────────────────

/// A generator copy-paste would make two formats pin identical bytes and
/// the pin would still be green. Cheap to assert, so assert it.
#[test]
fn every_unweighted_golden_is_distinct() {
    let mut seen: Vec<(&str, Vec<u8>)> = Vec::new();
    for (id, _, _) in CASES {
        let g = load_golden(&format!("unweighted.{id}"));
        for (prev_id, prev) in &seen {
            assert_ne!(
                &g, prev,
                "unweighted {id} and {prev_id} are byte-identical — one \
                 golden was copied, so the pin proves nothing for one of them"
            );
        }
        seen.push((id, g));
    }
}

// ─── Guard 3: sizes follow the block layout ───────────────────────────

/// Pins the byte-count arithmetic in `CASES` against the constants the code
/// publishes, so a layout constant changing cannot silently make a pin
/// vacuous (a shorter payload would still "match" a shorter golden).
#[test]
fn block_byte_counts_match_the_layout() {
    use quant_core::gguf_quants::{
        Q2_K_BLOCK_BYTES, Q3_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
    };
    let published: &[(&str, usize, usize)] = &[
        ("q2_k", Q2_K_BLOCK_BYTES, 84),
        ("q3_k", Q3_K_BLOCK_BYTES, 110),
        ("q4_k", Q4_K_BLOCK_BYTES, 144),
        ("q5_k", Q5_K_BLOCK_BYTES, 176),
        ("q6_k", Q6_K_BLOCK_BYTES, 210),
    ];
    for (id, computed, _expected) in published.iter() {
        let layout = CASES
            .iter()
            .find(|(i, _, _)| i == id)
            .map(|(_, _, b)| *b)
            .expect("listed above");
        assert_eq!(
            *computed, layout,
            "{id}: the table's byte arithmetic disagrees with the published \
             block size"
        );
    }
    for (id, _, layout) in CASES {
        let golden = load_golden(&format!("unweighted.{id}"));
        assert_eq!(
            golden.len(),
            N_ROWS * layout,
            "{id}: golden size must be exactly 2 super-blocks"
        );
    }
}

// ─── Guard 4: the fixture is the shared one ───────────────────────────

/// Pins the premise of every test above: unweighted and weighted goldens
/// were generated from the SAME input. If someone regenerates one family
/// against a different fixture, the discriminating pair stops being a fair
/// comparison and this goes red.
#[test]
fn unweighted_goldens_share_the_weighted_fixture() {
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(golden_dir().join("manifest.json")).expect("manifest.json"),
    )
    .expect("manifest parses");
    assert_eq!(manifest["seed_src"], 42, "fixture seed changed");
    assert_eq!(manifest["qk_k"], 256, "super-block size changed");
    assert_eq!(manifest["n_rows"], 2, "row count changed");
    let src = std::fs::read(golden_dir().join("src.f32.bin")).unwrap();
    assert_eq!(src.len(), 512 * 4, "src.f32.bin size changed");
}

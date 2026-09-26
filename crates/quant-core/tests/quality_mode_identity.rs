//! Tier 2 S01 — the opt-in mechanism's identity guarantees.
//!
//! # Why this file exists separately from `manifest.rs`'s unit tests
//!
//! `config_hash` is not a checksum, it is a **resume-compatibility key**, and
//! `StreamState::load_manifest` is the only thing that reads it:
//!
//! ```text
//! manifest.rs  load_manifest(&out) -> bool
//!     ... trusts the partial output if and only if payload.config_hash == self.config_hash
//! ```
//!
//! There is no second guard. No format check, no algorithm fingerprint, no
//! warning. So if two different algorithms can produce the same hash, a
//! resumed run silently concatenates tensors built by both.
//!
//! The concrete disaster this file exists to prevent:
//!
//! 1. user runs `--format nvfp4`, gets 40% of the way through, interrupts it;
//! 2. user re-runs with `--format nvfp4_l2` to get better quality;
//! 3. `load_manifest` sees a matching hash and **resumes into the byte-exact
//!    partial file**;
//! 4. the finished artifact is a mix of two quantizers, with no error and no
//!    warning anywhere.
//!
//! Every assertion below is therefore a *distinctness* assertion, and the
//! digests are additionally pinned so that a well-meaning reordering of the
//! payload string is caught rather than silently accepted.

use quant_core::manifest::{Format, QuantConfig, ScalingMode};
use quant_core::quality::Quality;

fn cfg(format: Format) -> QuantConfig {
    let mut c = QuantConfig::default();
    c.format = format;
    c.int8 = false;
    c.target_format = format.as_str().to_string();
    c
}

/// The default configuration must remain byte-identical to the pre-Tier-2
/// payload. The INT8 vector is captured from the Python reference and is not
/// ours to move.
#[test]
fn default_config_hash_is_unchanged_by_the_quality_mechanism() {
    assert_eq!(QuantConfig::default().config_hash(), "56920c6553cfa241");
}

/// Every quality id must yield a distinct hash **for every base format**, not
/// just NVFP4. A variant that only differs for one format would still permit
/// a cross-format resume mix-up.
#[test]
fn every_quality_variant_distinguishes_every_base_format() {
    for format in [Format::Int8, Format::Fp8E4m3, Format::Mxfp8, Format::Nvfp4] {
        let mut seen: Vec<(Option<&'static str>, String)> = Vec::new();
        for q in Quality::ALL {
            let mut c = cfg(format);
            c.quality = q;
            let hash = c.config_hash();
            for (prev_id, prev_hash) in &seen {
                assert_ne!(
                    &hash, prev_hash,
                    "{format:?} + {q:?} collides with {prev_id:?} — a resume could mix them"
                );
            }
            seen.push((q.id(), hash));
        }
    }
}

/// The core resume-safety invariant, stated directly: no base format's exact
/// hash may equal any quality variant's hash for that same format.
#[test]
fn a_quality_variant_never_shares_a_base_formats_hash() {
    for format in [Format::Int8, Format::Fp8E4m3, Format::Mxfp8, Format::Nvfp4] {
        let base = cfg(format).config_hash();
        for q in Quality::ALL.iter().filter(|q| **q != Quality::Exact) {
            let mut c = cfg(format);
            c.quality = *q;
            assert_ne!(
                c.config_hash(),
                base,
                "{format:?} + {q:?} would resume into a plain {format:?} partial file"
            );
        }
    }
}

/// `Quality::Exact.id()` is `None` precisely so the payload keeps its 9-key
/// form. If `Exact` ever started contributing a key, every committed vector
/// would move at once.
#[test]
fn exact_contributes_no_key_to_the_payload() {
    assert_eq!(Quality::Exact.id(), None);
    assert!(Quality::Exact.is_parity_exact());

    let mut c = cfg(Format::Nvfp4);
    c.quality = Quality::Exact;
    assert_eq!(
        c.config_hash(),
        "95ede677cf402b53",
        "explicit Exact == default"
    );
}

/// Pinned digests. These exist to catch a *reordering* of the payload string,
/// which the distinctness assertions above cannot see: a reorder keeps every
/// hash distinct while moving all of them.
///
/// The values are `sort_keys=True` order — `quality_tuning` sits between
/// `no_learned_rounding` and `scaling_mode`. The Tier 2 plan predicted
/// different values computed with the key appended last; those were wrong,
/// because the payload is a `json.dumps(sort_keys=True)` mirror and the sorted
/// ordering is the one under which all four committed vectors reproduce.
#[test]
fn quality_hashes_are_pinned_to_their_sorted_order_payloads() {
    let cases: [(Format, Quality, &str); 4] = [
        (Format::Nvfp4, Quality::Exact, "95ede677cf402b53"),
        (
            Format::Nvfp4,
            Quality::Nvfp4L2ScaleSearch,
            "85cf59c986e5d1a4",
        ),
        (
            Format::Nvfp4,
            Quality::Nvfp4HessianScaleSearch,
            "47437e5e8343725b",
        ),
        (
            Format::Mxfp8,
            Quality::Mxfp8E8m0Compensated,
            "ee7269a90020e9b5",
        ),
    ];
    for (format, q, expected) in cases {
        let mut c = cfg(format);
        c.quality = q;
        assert_eq!(c.config_hash(), expected, "{format:?} + {q:?}");
    }
}

/// A rotated NVFP4 run gets its distinctness from `convrot` +
/// `convrot_group_size`, **not** from a quality key — it is byte-exact against
/// the reference, just rotated. Pinned here because S05 depends on it and the
/// two mechanisms are easy to confuse.
#[test]
fn rotation_is_distinguished_by_its_own_fields_not_a_quality_key() {
    let mut plain = cfg(Format::Nvfp4);
    let plain_hash = plain.config_hash();

    plain.convrot = true;
    plain.convrot_group_size = 16;
    let rot16_hash = plain.config_hash();

    assert_ne!(rot16_hash, plain_hash);
    // Rotation is a parity-exact transformation, so no quality key appears.
    assert_eq!(plain.quality, Quality::Exact);
}

/// Scaling mode is hash-relevant for INT8/FP8 but deliberately ignored for the
/// fixed-block formats. This is pre-existing behaviour, re-asserted here so the
/// new field is not mistaken for a way to make it relevant.
#[test]
fn fixed_block_formats_still_ignore_scaling_mode_and_block_size() {
    let mut c = cfg(Format::Nvfp4);
    let base = c.config_hash();
    c.scaling_mode = ScalingMode::Row;
    c.block_size = 64;
    assert_eq!(c.config_hash(), base, "NVFP4 payload is fixed at block/16");

    let mut c = cfg(Format::Mxfp8);
    let base = c.config_hash();
    c.scaling_mode = ScalingMode::Row;
    c.block_size = 64;
    assert_eq!(c.config_hash(), base, "MXFP8 payload is fixed at block/32");
}

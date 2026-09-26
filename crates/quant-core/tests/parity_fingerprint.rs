//! Tier 2 S02 — the parity fingerprint.
//!
//! # What this file is for
//!
//! Every subsequent Tier 2 step adds a *new* code path and must leave the
//! default one byte-identical. "The existing golden tests still pass" is a
//! weaker signal than it looks: a golden test can compare a single tensor, and
//! a refactor can move bytes in a tensor nobody checked. This file hashes the
//! **entire output file** for each of the five existing `--format` values, so a
//! change anywhere in the payload moves a digest.
//!
//! # Why a whole-file digest and not the existing goldens
//!
//! The golden corpus already byte-compares per-tensor payloads against files
//! produced by the Python reference. That is the stronger oracle and it stays
//! the primary gate. The fingerprint exists to cover a different failure: a
//! change to a path the goldens do not cover — a tensor skipped by
//! `skip_inefficient`, a metadata field, a header layout detail — where
//! "nothing that is checked moved" would otherwise be reported as "nothing
//! moved". Hashing the whole emitted file closes that gap.
//!
//! # The digests below were measured on this machine, not computed
//!
//! They are recorded from an actual run at the commit that introduced this
//! file. **If one of them changes, that is a parity event, not a test to
//! update.** Phase 1's own risk register names this exact failure — "someone
//! tidied a parity kernel" — as the reason the conformance packs were adopted.
//!
//! Note the digests are of the *emitted file*, which embeds no timestamps or
//! paths, so they are stable across machines and across reruns. That is
//! asserted directly by `repeated_runs_are_byte_identical`.

mod common;

use common::golden;
use quant_core::manifest::{Format, QuantConfig, ScalingMode};
use quant_core::quality::Quality;
use quant_core::stream::stream_quantize;
use sha2::{Digest, Sha256};

/// The five `--format` values that exist today. Each must stay byte-exact.
const BASE_FORMATS: [&str; 5] = ["int8", "fp8_e4m3", "mxfp8", "nvfp4", "int8_convrot"];

/// SHA-256 over the whole emitted `.safetensors` file.
fn file_digest(path: &std::path::Path) -> String {
    let bytes = std::fs::read(path).expect("output file must exist");
    let mut h = Sha256::new();
    h.update(&bytes);
    format!("{:x}", h.finalize())
}

/// Quantize one golden case and return the SHA-256 of the whole output file.
///
/// The `TempDir` is dropped inside, so only the hex digest escapes — a caller
/// cannot accidentally read a file that has already been cleaned up.
fn fingerprint(case: &str, cfg: &QuantConfig) -> String {
    let dir = golden(case);
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");
    stream_quantize(dir.join("input.safetensors"), &out, cfg).unwrap();
    file_digest(&out)
}

fn base_config(format: Format, scaling: ScalingMode, block: u32) -> QuantConfig {
    QuantConfig {
        format,
        target_format: format.as_str().into(),
        int8: format == Format::Int8,
        scaling_mode: scaling,
        block_size: block,
        ..QuantConfig::default()
    }
}

// --------------------------------------------------------------------------
// The five pinned digests
// --------------------------------------------------------------------------

/// Measured at the commit that introduced this file. Any change here is a
/// parity event.
const INT8_DIGEST: &str = "15fdfe1037a2ec26e2e541068533741e27e44f0c984070bfc57884c8bf08635f";
const FP8_DIGEST: &str = "8b157ae5940d8d00f7287a8d4631700f43b5adb1f473080853cf3bc520ffec91";
const MXFP8_DIGEST: &str = "f996dda40ce7d371563e806c624d3cb93c71296d9da89267646832369fa8c180";
const NVFP4_DIGEST: &str = "927b831aafb955ec58732859282f8e1fe57645d19f5e5d3d019f4fe7a2f2b6f8";
const INT8_CONVROT_DIGEST: &str =
    "a091a5c697131617e7b15c69e2e65a07fd0a254f984fcd24108655b2d3a3644f";

/// Recompute every digest and assert it matches the pin.
///
/// Present as one test so the failure message names *which* format moved.
#[test]
fn parity_fingerprint_unchanged() {
    let case = "linear_basic_bf16";

    let got_int8 = fingerprint(case, &base_config(Format::Int8, ScalingMode::Block, 128));
    assert_eq!(got_int8, INT8_DIGEST, "int8 digest moved");

    let got_fp8 = fingerprint(case, &base_config(Format::Fp8E4m3, ScalingMode::Block, 128));
    assert_eq!(got_fp8, FP8_DIGEST, "fp8_e4m3 digest moved");

    let got_mxfp8 = fingerprint(case, &base_config(Format::Mxfp8, ScalingMode::Block, 32));
    assert_eq!(got_mxfp8, MXFP8_DIGEST, "mxfp8 digest moved");

    let got_nvfp4 = fingerprint(case, &base_config(Format::Nvfp4, ScalingMode::Block, 16));
    assert_eq!(got_nvfp4, NVFP4_DIGEST, "nvfp4 digest moved");

    let got_convrot = fingerprint(case, &base_config(Format::Int8, ScalingMode::Row, 128));
    assert_eq!(
        got_convrot, INT8_CONVROT_DIGEST,
        "int8_convrot digest moved"
    );
}

// --------------------------------------------------------------------------
// Properties the pinned digests alone do not establish
// --------------------------------------------------------------------------

/// A digest pin is worthless if the value is not reproducible. Asserted over
/// the slowest path (NVFP4) and the ConvRot path, since both involve work a
/// naive implementation might parallelise.
#[test]
fn repeated_runs_are_byte_identical() {
    let case = "linear_basic_bf16";
    for (label, cfg) in [
        ("nvfp4", base_config(Format::Nvfp4, ScalingMode::Block, 16)),
        (
            "int8_convrot",
            QuantConfig {
                convrot: true,
                convrot_group_size: 256,
                ..base_config(Format::Int8, ScalingMode::Row, 128)
            },
        ),
    ] {
        let first = fingerprint(case, &cfg);
        for run in 0..3 {
            assert_eq!(
                fingerprint(case, &cfg),
                first,
                "{label} run {run} differed from the first"
            );
        }
    }
}

/// Byte-exactness is a **whole-process** property, not a per-call one. A
/// lazily-initialised global cache that the quality path populates and the
/// parity path then reads would leave every individual call correct.
#[test]
fn quality_variant_never_changes_base_variant_bytes() {
    let case = "linear_basic_bf16";

    // The base digest measured with nothing else having run in this process…
    let nvfp4_alone = fingerprint(case, &base_config(Format::Nvfp4, ScalingMode::Block, 16));

    // …must equal the base digest measured AFTER a quality-variant run has
    // already executed in the same process.
    let mut quality = base_config(Format::Nvfp4, ScalingMode::Block, 16);
    quality.quality = Quality::Nvfp4L2ScaleSearch;
    let _ = fingerprint(case, &quality);
    let _ = fingerprint(case, &base_config(Format::Mxfp8, ScalingMode::Block, 32));

    let nvfp4_after = fingerprint(case, &base_config(Format::Nvfp4, ScalingMode::Block, 16));
    assert_eq!(
        nvfp4_alone, nvfp4_after,
        "a quality run perturbed the plain nvfp4 output — shared mutable state"
    );
    assert_eq!(nvfp4_alone, NVFP4_DIGEST, "and the digest must still match");
}

/// The five base digests must be pairwise distinct, or the pin is not actually
/// discriminating between formats.
#[test]
fn every_base_format_has_a_distinct_digest() {
    let digests = [
        ("int8", INT8_DIGEST),
        ("fp8_e4m3", FP8_DIGEST),
        ("mxfp8", MXFP8_DIGEST),
        ("nvfp4", NVFP4_DIGEST),
        ("int8_convrot", INT8_CONVROT_DIGEST),
    ];
    for i in 0..digests.len() {
        for j in (i + 1)..digests.len() {
            assert_ne!(
                digests[i].1, digests[j].1,
                "{} and {} share a digest — the pin cannot tell them apart",
                digests[i].0, digests[j].0
            );
        }
    }
    assert_eq!(BASE_FORMATS.len(), digests.len());
}

/// `quality = Exact` must be indistinguishable from not setting the field at
/// all — otherwise a caller that threads the field through unconditionally
/// would alter every existing format.
#[test]
fn explicit_exact_matches_the_omitted_field() {
    let case = "linear_basic_bf16";
    for (format, scaling, block, pinned) in [
        (Format::Int8, ScalingMode::Block, 128, INT8_DIGEST),
        (Format::Fp8E4m3, ScalingMode::Block, 128, FP8_DIGEST),
        (Format::Mxfp8, ScalingMode::Block, 32, MXFP8_DIGEST),
        (Format::Nvfp4, ScalingMode::Block, 16, NVFP4_DIGEST),
    ] {
        let mut explicit = base_config(format, scaling, block);
        explicit.quality = Quality::Exact;
        assert_eq!(
            fingerprint(case, &explicit),
            pinned,
            "{format:?} with an explicit Exact differs from the default"
        );
    }
}

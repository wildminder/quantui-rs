//! T5 — the K-quant "you are on the weaker encoder" warning.
//!
//! # Why a warning and not a default switch
//!
//! rlx-gguf's K-quant encoders are self-documented as "lower quality than
//! upstream's iterative search" (`rlx-gguf-0.2.14/src/quantize.rs:544`).
//! We carry byte-exact ports of llama.cpp's weighted encoders
//! (`gguf_quants::quantize_row_q*_k_weighted`) but they are reachable only
//! when an imatrix row exists — the sole gate is `cfg.imatrix.is_some()` in
//! `gguf_convert.rs`. So a K-quant method without `--imatrix` silently uses
//! the weaker encoder.
//!
//! Measured on YuE2-3B Q2_K (rel-L2, lower is better):
//! 0.32985 simplified (the default path) / 0.26936 weighted / 0.29840 for a
//! published reference.
//!
//! We make that visible instead of flipping the default, because flipping is
//! a parity event and the weighted encoders are golden-tested only WITH an
//! imatrix row — flipping first would leave the current default pinned by no
//! oracle at all.
//!
//! These tests cover the DETECTION predicate. The warning text itself is
//! verified end-to-end against the real model (see the commit message).

use quant_core::gguf_registry::{method_emits_k_quant, GgufScheme, METHODS};

// --------------------------------------------------------------------------
// The scheme-level predicate
// --------------------------------------------------------------------------

#[test]
fn k_quant_schemes_are_recognised() {
    for s in [
        GgufScheme::Q2K,
        GgufScheme::Q3K,
        GgufScheme::Q4K,
        GgufScheme::Q5K,
        GgufScheme::Q6K,
    ] {
        assert!(s.is_k_quant(), "{s:?} must be a K-quant");
    }
}

/// The legacy / IQ / float schemes must NOT be classified as K-quants —
/// they either go through a different encoder or are gated by
/// `requires_imatrix` instead.
#[test]
fn non_k_quant_schemes_are_rejected() {
    for s in [
        GgufScheme::Q4_0,
        GgufScheme::Q4_1,
        GgufScheme::Q5_0,
        GgufScheme::Q5_1,
        GgufScheme::Q8_0,
        // Q8_K is dead-but-mapped (an internal intermediate with no dispatch
        // arm) — it is emphatically not one of the five writable K-quants.
        GgufScheme::Q8K,
        GgufScheme::F16,
        GgufScheme::Bf16,
        GgufScheme::F32,
        GgufScheme::Tq1_0,
        GgufScheme::Tq2_0,
        GgufScheme::Iq2Xxs,
        GgufScheme::Iq4Nl,
    ] {
        assert!(!s.is_k_quant(), "{s:?} must NOT be a K-quant");
    }
}

// --------------------------------------------------------------------------
// The method-level predicate — this is what the CLI actually calls
// --------------------------------------------------------------------------

/// Every K-quant method must be detected, including the composites whose
/// default scheme alone would be misleading.
///
/// `q4_k_m` is the important case: its default is Q4_K but the llama.cpp
/// `use_more_bits` engine can promote `attn_v`/`ffn_down`, and `q2_k`
/// downgrades `attn_v` to Q3_K — so a default-only check would be a
/// partial test of the real behaviour.
#[test]
fn all_k_quant_methods_are_detected() {
    let expected = [
        "q2_k", "q2_k_l", "q3_k_m", "q3_k_l", "q3_k_s", "q3_k_xs", "q4_k_m", "q4_k_s", "q5_k_m",
        "q5_k_s", "q6_k",
    ];
    for id in expected {
        let entry = METHODS
            .iter()
            .find(|e| e.method.id == id)
            .unwrap_or_else(|| panic!("{id} must exist in the registry"));
        assert!(
            method_emits_k_quant(entry),
            "{id} must be reported as emitting K-quants"
        );
    }
}

/// A method that can only produce non-K schemes must not warn. `f16` and
/// `bf16` are lossless; the legacy `q4_0` family and the `iq*` family go
/// through other encoders (and the iq* ones are `requires_imatrix`-gated, so
/// they hard-error before the warning is even reachable).
#[test]
fn non_k_quant_methods_are_not_flagged() {
    let quiet = [
        "f16", "bf16", "f32", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "tq1_0", "tq2_0",
    ];
    for id in quiet {
        let entry = METHODS
            .iter()
            .find(|e| e.method.id == id)
            .unwrap_or_else(|| panic!("{id} must exist in the registry"));
        assert!(
            !method_emits_k_quant(entry),
            "{id} must NOT be reported as emitting K-quants"
        );
    }
}

/// The IQ family is gated by `requires_imatrix` upstream of this check, so it
/// must not be double-reported here — the two mechanisms are distinct and
/// this pins that.
#[test]
fn iq_methods_are_imatrix_gated_not_k_quant_flagged() {
    for entry in METHODS.iter() {
        if !entry.method.id.starts_with("iq") {
            continue;
        }
        assert!(
            entry.method.requires_imatrix,
            "{} is an iq* method and must stay requires_imatrix",
            entry.method.id
        );
        assert!(
            !method_emits_k_quant(entry),
            "{} must not be classified as K-quant — it is gated a different way",
            entry.method.id
        );
    }
}

/// Guards the mutation that would silently disable the whole feature: if the
/// predicate collapsed to "always true", the quiet-methods test goes red.
#[test]
fn predicate_is_not_vacuous() {
    // A spot-check that both answers really are reachable, so an
    // always-false or always-true implementation cannot pass the suite.
    assert!(GgufScheme::Q4K.is_k_quant());
    assert!(!GgufScheme::Q4_0.is_k_quant());

    let k = METHODS.iter().find(|e| e.method.id == "q4_k_m").unwrap();
    let legacy = METHODS.iter().find(|e| e.method.id == "q4_0").unwrap();
    assert_ne!(
        method_emits_k_quant(k),
        method_emits_k_quant(legacy),
        "the two must disagree — otherwise the predicate carries no signal"
    );
}

/// PINS THE INVARIANT that makes the one-line predicate correct.
///
/// `method_emits_k_quant` consults ONLY `policy.default`. That is sound
/// because every K-quant method has a K-quant default AND no non-K method
/// mentions a K-quant anywhere else. An earlier version also checked
/// `policy.rules` and the `KMoreBits { base, more }` engine "for safety" —
/// and a mutation disabling those checks kept the whole suite green,
/// because nothing exercised them. Dead code.
///
/// This test walks the WHOLE registry and fails if any method ever mixes
/// (K-quant reachable only through a rule or the engine), so the shortcut
/// cannot silently become wrong.
#[test]
fn registry_never_mixes_k_quant_into_a_non_k_default() {
    use quant_core::llama_policy::LlamaPolicy;

    for entry in METHODS {
        let id = entry.method.id;
        let by_default = entry.policy.default.is_k_quant();

        // Exhaustive: does a K-quant appear anywhere the default does not?
        let mut elsewhere = entry.policy.rules.iter().any(|(_, s)| s.is_k_quant())
            || entry.policy.embd_scheme.is_k_quant();
        if let LlamaPolicy::KMoreBits { base, more } = entry.policy.engine {
            elsewhere |= base.is_k_quant() || more.is_k_quant();
        }

        // The shortcut is wrong in exactly one direction: a method whose
        // default is NOT a K-quant but which can still reach one. That is
        // the only case `policy.default` alone would miss, so it is the only
        // one worth failing on.
        assert!(
            !(elsewhere && !by_default),
            "method '{id}' can emit a K-quant outside `policy.default` \
             (default={:?}) — `method_emits_k_quant` must be widened",
            entry.policy.default
        );
    }
}

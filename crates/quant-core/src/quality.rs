//! Opt-in quality refinements (Tier 2).
//!
//! # Why this exists
//!
//! Every knob in this module **changes output bytes by design**. The crate's
//! governing contract is byte-exact parity with the Python/torch reference and
//! `llama-quantize`, so a quality refinement may only ever be reached through
//! an explicit opt-in — never by changing what an existing format produces.
//! [`Quality::Exact`] is the byte-parity default and MUST remain the zero
//! value: `QuantConfig::default().config_hash()` has to keep hashing to
//! `56920c6553cfa241`, a vector captured from the Python reference.
//!
//! # The two enums are different things
//!
//! [`crate::manifest::Format`] is the core format family (`Int8`, `Fp8E4m3`,
//! `Mxfp8`, `Nvfp4`). `Quality` is orthogonal: it refines *how* one of those is
//! applied. That is why no `Format` variant is added here — the semver surface
//! stays at zero.
//!
//! # `id()` returning `None` is load-bearing
//!
//! [`Quality::id`] returns `None` for `Exact`, and
//! [`crate::manifest::QuantConfig::config_hash`] appends the id to its hash
//! payload **only when `id()` is `Some`**. The conditional append is the whole
//! design: the four committed hash vectors (INT8 `56920c6553cfa241` — captured
//! from Python, FP8 `5f14780b1bcf30f2`, MXFP8 `cff3b89365c9544d`, NVFP4
//! `95ede677cf402b53`) were all produced by a 9-key payload with no quality
//! field. Emitting the key unconditionally is the natural-looking
//! implementation and it **silently breaks every one of them**.
//!
//! The flip side is that the hash is the *only* resume-compatibility guard:
//! `StreamState::load_manifest` trusts a partial output if and only if the hash
//! matches. Without a distinct hash per quality variant, a plain NVFP4 run that
//! is interrupted and then re-run with `nvfp4_l2` would resume into the
//! byte-exact partial file and mix two different algorithms in one artifact —
//! with no error and no warning.
//!
//! # What the Tier 2 experiments actually found
//!
//! Three of the four refinements here are **measured negatives or no-ops**,
//! and the pattern is worth recording because it is a result, not a failure:
//!
//! -- `Mxfp8E8m0Compensated` — exactly 0.00% improvement. A proven no-op.
//! -- `Int8Clip09` — regresses 31x-14000x. At 127 levels the baseline error is
//!    already tiny, so clipping trades it for an unbounded saturation error.
//! -- `Nvfp4HessianScaleSearch` — unreachable by design (no calibration source).
//! -- `Nvfp4L2ScaleSearch` — **+17.7% aggregate**, the one that earned its place.
//!
//! The survivors have something in common: they run a real SEARCH over the
//! scale and let the data pick a point. The failures merely nudge a constant.
//! A Tier 2 proposal should be able to say which of those two it is before it
//! is written, because "tweak a constant and hope" has now failed three times
//! against a benchmark that already existed to catch it.

/// Opt-in quality refinements. `Exact` is the byte-parity default and MUST
/// remain the zero value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quality {
    /// Pure absmax / reference arithmetic. Byte-exact with the references.
    #[default]
    Exact,
    /// MXFP8: multiply `scale_needed` by 4/3 before the E8M0 `ceil(log2(...))`.
    ///
    /// Rationale (arXiv:2509.23202): E8M0 power-of-two scales are the largest
    /// single error source in the FP8 path — +40% `MSErel` versus +10% for
    /// E4M3. The factor buys clipping headroom, **not** finer scales: because
    /// `log2(4/3) = 0.415 < 1`, the resulting `e8m0` delta is provably in
    /// `{0, +1}` and can never be `-1`.
    Mxfp8E8m0Compensated,
    /// NVFP4: anchored alternating L2 search over the per-tensor / block scale
    /// pair (arXiv:2509.23202).
    ///
    /// "Anchored" is mandatory, not stylistic. The reconstruction is
    /// `X̂ = s_T · s_G · Q(X / (s_T·s_G))`, so rescaling `s_T → k·s_T` and
    /// `s_G → s_G/k` leaves the product invariant; because the E2M1 grid is
    /// `{2^j, 1.5·2^j}`, the codes are *also* unchanged for `k = 2^j`. The L2
    /// objective is therefore **exactly flat** along powers of two, and an
    /// unanchored search converges to an arbitrary — and eventually
    /// E4M3-unrepresentable — scale.
    Nvfp4L2ScaleSearch,
    /// INT8: derive the quant scale from `clip_ratio * row_max` instead of
    /// `row_max`, saturating the top of the range to buy finer resolution
    /// everywhere else (Tier 2 S07, `--format int8_clip09`).
    ///
    /// The ratio is FIXED at [`crate::quant::INT8_CLIP_RATIO`] and is not a
    /// knob. A ratio swept until the error benchmark improved would be a
    /// threshold fitted to the fixture, not a measurement of the idea — so the
    /// pre-registered decision rule is only meaningful with the configuration
    /// fixed in advance.
    ///
    /// The case AGAINST this mode is specific and worth recording: the
    /// published evidence for clipping is a perplexity gain, this crate cannot
    /// compute perplexity, and clipping's usual benefit is to ACTIVATION
    /// outliers — which this path never quantizes. MEASURED on the crate's own
    /// metric it regresses `rel_l2` by 31x-14000x depending on the
    /// distribution, because INT8's 127 levels leave no error for a finer step
    /// to recover while the saturated tail's error is unbounded. Retained as a
    /// documented negative result, not a live recommendation; see
    /// `tests/int8_clip09_negative_result.rs` and `benches/quality_error.rs`.
    Int8Clip09,
    /// Reserved for Hessian-guided search (H-Scale, arXiv:2608.28113).
    ///
    /// Not constructible from the CLI: it needs a diagonal second-order proxy
    /// and this crate has no calibration-data source (its "calibration" is
    /// synthetic MT19937 `randn`, which is a parity device rather than a
    /// statistics source). Optimising against a distribution that provably
    /// does not match the model would likely be *worse* than the data-free
    /// search, so the variant is reserved and unreachable until a calibration
    /// source exists.
    #[allow(dead_code)]
    Nvfp4HessianScaleSearch,
}

impl Quality {
    /// Every variant, in declaration order. Used by exhaustiveness tests so a
    /// future variant cannot be added without updating the id/label/hash
    /// tables.
    pub const ALL: [Quality; 5] = [
        Quality::Exact,
        Quality::Mxfp8E8m0Compensated,
        Quality::Nvfp4L2ScaleSearch,
        Quality::Nvfp4HessianScaleSearch,
        Quality::Int8Clip09,
    ];

    /// Stable id used in `config_hash` and in CLI output.
    ///
    /// `None` for [`Quality::Exact`] **by design** — see the module docs. The
    /// conditional hash append is what keeps the four committed vectors stable.
    pub fn id(&self) -> Option<&'static str> {
        match self {
            Quality::Exact => None,
            Quality::Mxfp8E8m0Compensated => Some("mxfp8_e8m0_43"),
            Quality::Nvfp4L2ScaleSearch => Some("nvfp4_l2"),
            Quality::Nvfp4HessianScaleSearch => Some("nvfp4_hessian"),
            Quality::Int8Clip09 => Some("int8_clip09"),
        }
    }

    /// Whether this variant is byte-exact against the references.
    pub fn is_parity_exact(&self) -> bool {
        matches!(self, Quality::Exact)
    }

    /// One-line human-readable reason, printed on the `parity:` marker line for
    /// non-parity formats. `None` for [`Quality::Exact`].
    pub fn reason(&self) -> Option<&'static str> {
        match self {
            Quality::Exact => None,
            Quality::Mxfp8E8m0Compensated => {
                Some("MXFP8 E8M0 scale compensation (x4/3 before the E8M0 ceil)")
            }
            Quality::Nvfp4L2ScaleSearch => Some("NVFP4 anchored alternating L2 scale search"),
            Quality::Int8Clip09 => {
                Some("INT8 absmax clipping (quant scale from 0.9x row_max; NOT byte-exact)")
            }
            Quality::Nvfp4HessianScaleSearch => {
                Some("NVFP4 Hessian-guided scale search (not reachable from the CLI)")
            }
        }
    }

    /// The core format family this variant refines, for the CLI preset table.
    pub fn applies_to(&self) -> crate::manifest::Format {
        use crate::manifest::Format;
        match self {
            Quality::Mxfp8E8m0Compensated => Format::Mxfp8,
            Quality::Nvfp4L2ScaleSearch | Quality::Nvfp4HessianScaleSearch => Format::Nvfp4,
            // `Exact` is format-agnostic; callers must not route on it.
            Quality::Exact | Quality::Int8Clip09 => Format::Int8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_is_the_zero_value() {
        assert_eq!(Quality::default(), Quality::Exact);
        // The derive order guarantees `Exact` is variant 0, so a
        // `#[derive(Default)]`-style zeroed config is the parity path.
        assert_eq!(Quality::Exact as u8, 0);
    }

    /// The load-bearing property: `Exact` contributes NO key, so the
    /// pre-Tier-2 payload is byte-identical. If this fails, every committed
    /// hash vector has moved.
    #[test]
    fn quality_id_is_none_exactly_for_exact() {
        assert_eq!(Quality::Exact.id(), None);
        for q in Quality::ALL {
            if q == Quality::Exact {
                continue;
            }
            assert!(q.id().is_some(), "{q:?} must contribute a hash key");
            assert!(q.reason().is_some(), "{q:?} must explain itself");
        }
    }

    #[test]
    fn every_non_exact_id_is_distinct() {
        let mut ids: Vec<&str> = Quality::ALL.iter().filter_map(|q| q.id()).collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before, "quality ids must be pairwise distinct");
    }

    #[test]
    fn only_exact_is_parity_exact() {
        for q in Quality::ALL {
            assert_eq!(q.is_parity_exact(), q == Quality::Exact, "{q:?}");
        }
    }

    #[test]
    fn applies_to_matches_the_naming_convention() {
        assert_eq!(
            Quality::Mxfp8E8m0Compensated.applies_to(),
            crate::manifest::Format::Mxfp8
        );
        assert_eq!(
            Quality::Nvfp4L2ScaleSearch.applies_to(),
            crate::manifest::Format::Nvfp4
        );
    }
}

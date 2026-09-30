//! T3 — a recipe whose rules match NOTHING must say so.
//!
//! # The failure mode
//!
//! `gguf_convert.rs` maps HF names to GGUF-side names FIRST, then matches
//! recipe rules against the MAPPED name. So a `--recipe-from` reference
//! written in a foreign name space (`model_weights/model.layers.0...`, as
//! the AudioCpp-produced GGUFs are) matches zero rules. Every tensor then
//! falls through to the method default, the run exits 0, and the tool
//! appears to have honoured the reference when it honoured nothing.
//!
//! That is the worst failure mode a tool can have, and it used to be silent.
//! `TensorRecipe::matching_rule` now records whether any rule actually
//! decided a tensor, and the driver reports it once at the end.
//!
//! # What is deliberately NOT an error
//!
//! A recipe carrying only a bare default (zero rules) is a legitimate
//! artifact, not a defect, so no warning fires for it. And the warning
//! never fails the run: exit codes 0/1/2/3/130 are all taken, and taking
//! 1 for a visible, fixable condition would be wrong.

use quant_core::gguf_recipe::TensorRecipe;

// --------------------------------------------------------------------------
// matching_rule — the primitive the whole fix rests on
// --------------------------------------------------------------------------

/// A rule matching a name we DO produce reports its index.
#[test]
fn matching_rule_finds_a_our_name_space_rule() {
    let r = TensorRecipe::parse("^blk\\.0\\.attn_q\\.weight=q8_0\n", "test").unwrap();
    assert_eq!(
        r.matching_rule("blk.0.attn_q.weight"),
        Some(0),
        "a rule written in our name space must match"
    );
}

/// THE CORE CASE: an AudioCpp-style reference addresses names we never
/// produce, so nothing matches — which is what the warning reports.
#[test]
fn foreign_name_space_rules_match_nothing() {
    let r = TensorRecipe::parse(
        "^model_weights/model\\.layers\\.0\\.mlp\\.down_proj\\.weight=q4_0\nq4_0\n",
        "test",
    )
    .unwrap();
    assert_eq!(r.rules.len(), 1, "the recipe does carry a rule");
    assert_eq!(
        r.matching_rule("blk.0.ffn_down.weight"),
        None,
        "a foreign-name-space rule must not match our GGUF names"
    );
    // …yet `scheme_for` still answers, which is exactly why the failure was
    // silent: the bare default filled the gap invisibly.
    assert_eq!(
        r.scheme_for("blk.0.ffn_down.weight"),
        Some(r.default.expect("bare default present")),
        "scheme_for falls through to the default — the silent part"
    );
}

/// The distinction the fix depends on: a rule that DOES fire must be
/// distinguishable from the bare default filling in for it.
#[test]
fn rule_and_bare_default_are_distinguishable() {
    let with_rule = TensorRecipe::parse("^blk\\.0=q4_0\nq8_0\n", "test").unwrap();
    let default_only = TensorRecipe::parse("q8_0\n", "test").unwrap();

    assert_eq!(
        with_rule.matching_rule("blk.0.weight"),
        Some(0),
        "the rule fires"
    );
    assert_eq!(
        with_rule.scheme_for("blk.0.weight"),
        Some(quant_core::gguf_registry::GgufScheme::Q4_0),
        "and it decides the scheme (manual mode)"
    );

    assert_eq!(
        default_only.matching_rule("blk.0.weight"),
        None,
        "a default-only recipe matches nothing BY DESIGN"
    );
    assert_eq!(
        default_only.scheme_for("blk.0.weight"),
        Some(quant_core::gguf_registry::GgufScheme::Q8_0),
        "the bare default still supplies a scheme"
    );
}

/// First-match-wins ordering, matching `scheme_for`'s loop.
///
/// Note the patterns are ANCHORED with `$` as well as `^`: recipe patterns are
/// unanchored substring searches (like upstream `regex_search`), so a bare
/// `^blk\.0` would also match `blk.0.ffn.weight` and legitimately win on
/// order. Anchoring both ends is what makes the ordering observable.
#[test]
fn first_matching_rule_wins() {
    let r = TensorRecipe::parse(
        "^blk\\.0\\.weight=q4_0\n^blk\\.0\\.ffn\\.weight=q5_0\nq8_0\n",
        "test",
    )
    .unwrap();
    assert_eq!(r.matching_rule("blk.0.weight"), Some(0));
    assert_eq!(r.matching_rule("blk.0.ffn.weight"), Some(1));
    // Both anchored patterns miss this one, so the bare default applies and
    // `matching_rule` reports None — the case the end-of-run warning counts.
    assert_eq!(r.matching_rule("blk.1.weight"), None);
}

// --------------------------------------------------------------------------
// Round-trip preservation — the change must not alter parsing
// --------------------------------------------------------------------------

/// `matching_rule` is purely additive: `to_text` still reproduces the input,
/// so `--emit-recipe` and the existing round-trip tests are unaffected.
#[test]
fn to_text_round_trip_is_unchanged() {
    let src = "^blk\\.0=q4_0\nq8_0\n";
    let r = TensorRecipe::parse(src, "test").unwrap();
    assert_eq!(r.to_text(), src, "round-trip must be byte-identical");
}

/// An empty recipe has zero rules — the legitimate default-only case that
/// must NOT warn.
#[test]
fn default_only_recipe_has_no_rules() {
    let r = TensorRecipe::parse("q4_0\n", "test").unwrap();
    assert!(
        r.rules.is_empty(),
        "a bare-default recipe must report zero rules so the driver stays quiet"
    );
}

// --------------------------------------------------------------------------
// The regression this file's own fix introduced (found by CI, T4)
// --------------------------------------------------------------------------

/// `matching_rule` is NOT a drop-in for `scheme_for`.
///
/// The silent-no-op guard above was built by swapping the driver's
/// `r.scheme_for(&name)` for `r.matching_rule(&name)` plus an index lookup.
/// That swap silently DROPPED the recipe's bare default: `scheme_for` returns
/// `self.default` when no rule matches, `matching_rule` returns `None`. So a
/// recipe of `ffn_down=q6_k` + a trailing `q8_0` sent every unmatched tensor
/// to the method's own policy engine (Q4K) instead of q8_0.
///
/// The unit tests in `gguf_recipe.rs` all passed, because they exercise
/// `scheme_for` directly. Only the end-to-end driver used the wrong function.
/// This test pins the DISTINCTION so the two can never be conflated again.
#[test]
fn matching_rule_is_not_scheme_for_the_bare_default_is_the_difference() {
    let r = TensorRecipe::parse("ffn_down\\.weight=q6_k\nq8_0\n", "test").unwrap();

    // Unmatched name: the rule primitive says "nothing fired"...
    assert_eq!(
        r.matching_rule("blk.0.attn_q.weight"),
        None,
        "matching_rule reports rule hits only, by design"
    );
    // ...while the scheme lookup still honours the bare default. This is the
    // exact pair the driver got wrong.
    assert_eq!(
        r.scheme_for("blk.0.attn_q.weight"),
        Some(quant_core::gguf_registry::GgufScheme::Q8_0),
        "the bare default must cover unmatched tensors"
    );
    // A matched name agrees either way.
    assert_eq!(r.matching_rule("blk.0.ffn_down.weight"), Some(0));
    assert_eq!(
        r.scheme_for("blk.0.ffn_down.weight"),
        Some(quant_core::gguf_registry::GgufScheme::Q6K)
    );
}

/// The warning must name the CONSEQUENCE that actually happened.
///
/// With no bare default, the method default won every tensor. WITH one, the
/// bare default still applies and only the RULES were ignored. A single
/// wording cannot describe both, and the wrong one sends the user to debug
/// the wrong layer — so the driver branches, and this pins the discriminator.
#[test]
fn warning_consequence_depends_on_whether_a_bare_default_exists() {
    // No default -> nothing to fall back on but the method default.
    let no_default = TensorRecipe::parse("ffn_down\\.weight=q6_k\n", "test").unwrap();
    assert!(no_default.default.is_none());
    assert_eq!(no_default.scheme_for("blk.0.attn_q.weight"), None);

    // Default present -> the driver warns, but names the bare default as the
    // thing that actually won, not "the method default".
    let with_default = TensorRecipe::parse("ffn_down\\.weight=q6_k\nq8_0\n", "test").unwrap();
    assert!(with_default.default.is_some());
    assert_eq!(
        with_default.scheme_for("blk.0.attn_q.weight"),
        Some(quant_core::gguf_registry::GgufScheme::Q8_0),
        "the warning path must not change what the bare default resolves to"
    );
    // Both still count as "no rule matched", so both still warn.
    assert_eq!(no_default.matching_rule("blk.0.attn_q.weight"), None);
    assert_eq!(with_default.matching_rule("blk.0.attn_q.weight"), None);
}

/// `scheme_id` is public so the driver can name the bare default in its
/// warning. Pin the spelling it emits, since that text is user-facing.
#[test]
fn scheme_id_spells_the_canonical_method_id() {
    use quant_core::gguf_recipe::scheme_id;
    use quant_core::gguf_registry::GgufScheme;
    assert_eq!(scheme_id(GgufScheme::Q8_0), "q8_0");
    assert_eq!(scheme_id(GgufScheme::Q6K), "q6_k");
    // And it is the exact inverse of parse for every scheme a recipe carries.
    // Single-line sources only: a bare default is only legal as the LAST
    // line, so a two-line "rule + default" text cannot be re-parsed as a
    // bare-default file (the rule line makes the second line malformed).
    for src in ["q8_0\n", "q6_k\n", "f16\n"] {
        let r = TensorRecipe::parse(src, "test").unwrap();
        if let Some(d) = r.default {
            let spelled = scheme_id(d);
            let round = TensorRecipe::parse(&format!("{spelled}\n"), "test").unwrap();
            assert_eq!(
                round.default,
                Some(d),
                "scheme_id({d:?}) = {spelled:?} must re-parse to the same scheme"
            );
        }
    }
}

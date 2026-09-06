//! WP6 / NTH-001 — property tests for the recipe parser (`gguf_recipe.rs`).
//!
//! The parser was hardened manually (WP1); these properties pin the two
//! behaviors no example test can cover:
//!   * round-trip: `parse(to_text(r)) == r` for any *valid* recipe —
//!     rules in order, patterns verbatim, schemes, bare default;
//!   * totality: `parse` returns a `Result` for ARBITRARY text — it may
//!     reject, but it must never panic (proptest shrinks a panic to the
//!     minimal reproducer).
//!
//! Patterns are generated from the hostile regex metacharacter alphabet
//! so the round-trip also proves nothing is mangled by quoting/escaping
//! inside `to_text`.
//!
//! Case count: proptest's default is 256, which is exactly the plan's
//! number (6.1); these files rely on the default so the config stays
//! boilerplate-free.

use proptest::prelude::*;

use quant_core::gguf_registry::usable_ids;

/// Regex metacharacters + recipe-hostile punctuation, expressed as a
/// bracketed class so `string_regex` can consume it. Generated patterns
/// include invalid regexes (unbalanced `(`, dangling `+`) — those
/// exercise the `BadRegex` error path, which the round-trip filters out
/// by only asserting on successfully parsed recipes.
const HOSTILE: &str = r"[a-z0-9_.\\^$*+?()\[\]{}|-]*";

fn pattern_strategy() -> impl Strategy<Value = String> {
    proptest::string::string_regex(HOSTILE)
        .expect("hostile alphabet is a valid regex character class")
        .prop_filter(
            "empty and whitespace-only patterns collapse to the same error",
            |s| !s.trim().is_empty(),
        )
}

fn qtype_strategy() -> impl Strategy<Value = String> {
    proptest::sample::select(usable_ids()).prop_map(|s| s.to_string())
}

fn recipe_text_strategy() -> impl Strategy<Value = String> {
    (
        proptest::collection::vec((pattern_strategy(), qtype_strategy()), 0..3),
        proptest::option::of(qtype_strategy()),
    )
        .prop_map(|(rules, default)| {
            let mut text = String::new();
            for (p, q) in &rules {
                text.push_str(&format!("{p}={q}\n"));
            }
            if let Some(d) = &default {
                text.push_str(&format!("{d}\n"));
            }
            text
        })
}

/// `parse(to_text(r)) == r` for any recipe the parser accepted: same
/// rule count, patterns and schemes in the same order, same default.
/// (Recipes the generator built from invalid regexes are rejected and
/// skipped — the round-trip property only speaks about valid files.)
#[test]
fn recipe_parse_print_round_trip() {
    proptest!(|(text in recipe_text_strategy())| {
        let parsed = quant_core::gguf_recipe::TensorRecipe::parse(&text, "prop");
        if let Ok(r) = parsed {
            let again = quant_core::gguf_recipe::TensorRecipe::parse(&r.to_text(), "prop")
                .expect("to_text output must re-parse");
            prop_assert_eq!(r.rules.len(), again.rules.len());
            for (a, b) in r.rules.iter().zip(&again.rules) {
                prop_assert_eq!(&a.pattern, &b.pattern);
                prop_assert_eq!(a.scheme, b.scheme);
            }
            prop_assert_eq!(r.default, again.default);
        }
    });
}

/// Arbitrary text (including truncations of valid recipes, line-number
/// traps, unknown qtypes, embedded NULs/unicode) → `Result`, never a
/// panic.
#[test]
fn recipe_never_panics_on_arbitrary_text() {
    proptest!(|(s in ".*")| {
        let _ = quant_core::gguf_recipe::TensorRecipe::parse(&s, "prop");
    });
}

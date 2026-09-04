//! Per-tensor recipe file (`--tensor-type-file`) — the open, reproducible
//! equivalent of Unsloth's proprietary `UD-*` dynamic recipes (plan §3-F,
//! Phase 6.1; OpenDynamicGGUF / GGUF-Tool-Suite `regex=qtype` semantics,
//! cross-checked against llama-quantize's `--tensor-type` /
//! `--tensor-type-file` plumbing: quantize.cpp:317-364, llama-quant.cpp:713-727).
//!
//! File format:
//! - `# comment` lines and blank lines are ignored;
//! - a line `regex=qtype` adds an override rule — the regex is applied with
//!   **search** semantics (unanchored substring match, exactly like
//!   `std::regex_search` upstream) against the GGUF-side tensor name, and the
//!   FIRST matching rule wins (llama-quant.cpp:716-726 `break`s on the first
//!   hit);
//! - the LAST line may be a bare `qtype` (no `=`) — the default for tensors
//!   no rule matches (llama-quantize keeps its method default instead; the
//!   bare-default is the GGUF-Tool-Suite recipe extension);
//! - every `qtype` must be a USABLE registry method id (`usable_ids()`),
//!   validated at load with the line number in the error.
//!
//! Resolution precedence (mirroring llama-quant.cpp:683-739):
//! 1. 1-D tensors → F32 (never overridden — the shared convention);
//! 2. `token_embd`/`output` category overrides (`--token-embedding-type` /
//!    `--output-tensor-type`) return EARLY (:688-706) — they win over any
//!    recipe rule for their tensors, except `per_layer_token_embd` when a
//!    recipe rule names it (upstream: it is a "large separate table");
//! 3. the recipe's first matching rule (manual mode — the method's own
//!    policy engine is SKIPPED entirely, including its counter
//!    advancement, upstream `manual = true` path, :713-731), or the
//!    recipe's bare default when no rule matches;
//! 4. the method's own policy (flat rules / llama_policy engine).
//!
//! Steps 3-4 only run when the method's default type is quantized (:711) —
//! an `f16`/`f32`/`bf16` method ignores per-tensor recipes entirely.

use crate::gguf_registry::{self, GgufScheme};

/// One parsed `regex=qtype` rule.
#[derive(Debug, Clone)]
pub struct RecipeRule {
    pub pattern: String,
    pub regex: regex::Regex,
    pub scheme: GgufScheme,
}

/// A parsed recipe file: ordered rules + optional bare default.
#[derive(Debug, Clone, Default)]
pub struct TensorRecipe {
    pub rules: Vec<RecipeRule>,
    /// Bare trailing `qtype` line, when present.
    pub default: Option<GgufScheme>,
}

/// Errors from parsing/loading a recipe file. Every variant carries the
/// line number so the user can fix the file without a debugger.
#[derive(Debug, thiserror::Error)]
pub enum RecipeError {
    #[error("recipe file '{path}' cannot be read: {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "recipe file '{path}' line {line}: expected 'regex=qtype' or a bare 'qtype', got '{text}'"
    )]
    MalformedLine {
        path: String,
        line: usize,
        text: String,
    },
    #[error("recipe file '{path}' line {line}: unknown or unusable GGUF method '{qtype}' (usable: {usable})")]
    UnknownQType {
        path: String,
        line: usize,
        qtype: String,
        usable: String,
    },
    #[error("recipe file '{path}' line {line}: invalid regex '{pattern}': {reason}")]
    BadRegex {
        path: String,
        line: usize,
        pattern: String,
        reason: String,
    },
}

impl TensorRecipe {
    /// Parse recipe text (already read from disk). `path` is only for error
    /// messages.
    pub fn parse(text: &str, path: &str) -> Result<Self, RecipeError> {
        let mut rules = Vec::new();
        let mut default = None;
        for (idx, raw) in text.lines().enumerate() {
            let line_no = idx + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            match line.split_once('=') {
                Some((pattern, qtype)) => {
                    if pattern.is_empty() {
                        return Err(RecipeError::MalformedLine {
                            path: path.to_string(),
                            line: line_no,
                            text: line.to_string(),
                        });
                    }
                    // A rule after the bare default would silently shadow it
                    // (the default is only reached when no rule matches) —
                    // malformed.
                    if default.is_some() {
                        return Err(RecipeError::MalformedLine {
                            path: path.to_string(),
                            line: line_no,
                            text: line.to_string(),
                        });
                    }
                    let scheme = parse_qtype(qtype, path, line_no)?;
                    let regex = regex::Regex::new(pattern).map_err(|e| RecipeError::BadRegex {
                        path: path.to_string(),
                        line: line_no,
                        pattern: pattern.to_string(),
                        reason: e.to_string(),
                    })?;
                    rules.push(RecipeRule {
                        pattern: pattern.to_string(),
                        regex,
                        scheme,
                    });
                }
                None => {
                    // Bare qtype — the file-level default. It must be the
                    // LAST meaningful line: a second default, or any rule
                    // after a default, is a malformed file (we never
                    // silently ignore an earlier default).
                    if default.is_some() {
                        return Err(RecipeError::MalformedLine {
                            path: path.to_string(),
                            line: line_no,
                            text: line.to_string(),
                        });
                    }
                    default = Some(parse_qtype(line, path, line_no)?);
                }
            }
        }
        Ok(Self { rules, default })
    }

    /// Load and parse a recipe file from disk.
    pub fn load(path: &std::path::Path) -> Result<Self, RecipeError> {
        let p = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|source| RecipeError::Open {
            path: p.clone(),
            source,
        })?;
        Self::parse(&text, &p)
    }

    /// First-match-wins lookup against a GGUF-side tensor name
    /// (search semantics, like upstream `regex_search`). Returns `None`
    /// when no rule matches and no bare default exists.
    pub fn scheme_for(&self, gguf_name: &str) -> Option<GgufScheme> {
        for r in &self.rules {
            if r.regex.is_match(gguf_name) {
                return Some(r.scheme);
            }
        }
        self.default
    }

    /// Serialize back to the file format (used by `--emit-recipe` and the
    /// round-trip test): rules in order, then the bare default.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for r in &self.rules {
            out.push_str(&format!("{}={}\n", r.pattern, scheme_id(r.scheme)));
        }
        if let Some(d) = self.default {
            out.push_str(&format!("{}\n", scheme_id(d)));
        }
        out
    }
}

/// Map a usable method id to its default scheme (the recipe's `qtype`
/// vocabulary is method ids, like OpenDynamicGGUF; the scheme is what the
/// conversion driver consumes). Validated against `usable_ids()` so
/// Dynamic 2.0 / no-encoder ids are rejected at parse time.
fn parse_qtype(qtype: &str, path: &str, line: usize) -> Result<GgufScheme, RecipeError> {
    let entry = gguf_registry::get_method(qtype).ok_or_else(|| RecipeError::UnknownQType {
        path: path.to_string(),
        line,
        qtype: qtype.to_string(),
        usable: gguf_registry::usable_ids().join(", "),
    })?;
    if entry.method.dynamic_v2
        || matches!(
            entry.method.support,
            gguf_registry::BackendSupport::NoEncoder(_)
        )
    {
        return Err(RecipeError::UnknownQType {
            path: path.to_string(),
            line,
            qtype: qtype.to_string(),
            usable: gguf_registry::usable_ids().join(", "),
        });
    }
    Ok(entry.policy.default)
}

/// Inverse of [`parse_qtype`] for a scheme: any usable method whose default
/// is this scheme round-trips (prefer the canonical plain method ids).
fn scheme_id(scheme: GgufScheme) -> String {
    // The canonical id for each scheme used in recipes: the plain method
    // whose policy.default is exactly this scheme (deterministic pick:
    // first usable id in registry order).
    for e in gguf_registry::METHODS {
        if !e.method.dynamic_v2
            && !matches!(
                e.method.support,
                gguf_registry::BackendSupport::NoEncoder(_)
            )
            && e.policy.default == scheme
        {
            return e.method.id.to_string();
        }
    }
    // Unreachable for every scheme a recipe can carry (all defaults come
    // from usable methods); fall back to a debug name rather than panic.
    format!("{scheme:?}")
}

/// Errors from extracting a recipe from a reference GGUF
/// (`--recipe-from`, task #8).
#[derive(Debug, thiserror::Error)]
pub enum RecipeFromError {
    #[error("recipe-from: cannot read reference GGUF: {0}")]
    Gguf(String),
    #[error("recipe-from: tensor '{name}' has GGML type {code} ({type_name}), which has no encoder in this build")]
    UnsupportedType {
        name: String,
        code: u32,
        type_name: String,
    },
}

/// Recipe file format version emitted by `--emit-recipe` and
/// `--recipe-from` dumps. Parsers ignore comment lines, so older versionless
/// files stay loadable — this only marks NEW outputs so a future format
/// change can be detected by reading the first line instead of guessing.
pub const RECIPE_FORMAT_VERSION: &str = "v1";

/// Extract the per-tensor dtype assignment from an existing GGUF
/// (unsloth, llama-quantize, or our own output) as a [`TensorRecipe`].
///
/// Rules are `^name$=qtype` lines — exact-anchored regexes reproduce the
/// reference assignment precisely (llama.cpp GGUF names can contain regex
/// metacharacters like `.`; anchoring makes every rule name-exact). The
/// ggml type → method-id mapping is the inverse of the registry's
/// `scheme_to_ggml`; types with no encoder in this build (e.g. Q8_1)
/// error with the offending tensor named.
///
/// F32 entries are skipped: 1-D norms/biases go to F32 by the shared
/// convention anyway (recipe step 1 runs before rules), so emitting rules
/// for them is noise. The caller is expected to combine this with the
/// method default (`--method`) so tensors absent from the reference fall
/// back cleanly.
pub fn recipe_from_gguf(path: &std::path::Path) -> Result<TensorRecipe, RecipeFromError> {
    use rlx_gguf::{GgmlType, GgufFile};

    let f = GgufFile::from_path(path)
        .map_err(|e| RecipeFromError::Gguf(format!("{}: {e}", path.display())))?;

    // Deterministic order: sorted by name.
    let mut names: Vec<&rlx_gguf::GgufTensor> = f.tensors.values().collect();
    names.sort_by(|a, b| a.name.cmp(&b.name));

    let mut text = format!(
        "# quantui-rs recipe format {RECIPE_FORMAT_VERSION}\n\
         # recipe extracted from reference GGUF (--recipe-from)\n\
         # one ^name$=qtype rule per tensor, sorted by name\n",
    );
    for t in names {
        if t.dtype == GgmlType::F32 {
            continue; // shared convention, no rule needed
        }
        let scheme = scheme_for_ggml(t.dtype).ok_or_else(|| RecipeFromError::UnsupportedType {
            name: t.name.clone(),
            code: t.dtype as u32,
            type_name: format!("{:?}", t.dtype),
        })?;
        // Anchor: exact-name match. Regex-escape the name first — GGUF
        // tensor names are dotted, and `.` is a regex wildcard.
        let pattern = regex::escape(&t.name);
        text.push_str(&format!("^{pattern}$={}\n", scheme_id(scheme)));
    }
    TensorRecipe::parse(&text, &path.display().to_string())
        .map_err(|e| RecipeFromError::Gguf(e.to_string()))
}

/// ggml type → scheme (inverse of `gguf_convert::scheme_to_ggml`).
/// `None` for types this build has no encoder for.
fn scheme_for_ggml(t: rlx_gguf::GgmlType) -> Option<crate::gguf_registry::GgufScheme> {
    use crate::gguf_registry::GgufScheme as S;
    use rlx_gguf::GgmlType as G;
    match t {
        G::F32 => Some(S::F32),
        G::F16 => Some(S::F16),
        G::BF16 => Some(S::Bf16),
        G::Q8_0 => Some(S::Q8_0),
        G::Q4_0 => Some(S::Q4_0),
        G::Q4_1 => Some(S::Q4_1),
        G::Q5_0 => Some(S::Q5_0),
        G::Q5_1 => Some(S::Q5_1),
        G::Q2K => Some(S::Q2K),
        G::Q3K => Some(S::Q3K),
        G::Q4K => Some(S::Q4K),
        G::Q5K => Some(S::Q5K),
        G::Q6K => Some(S::Q6K),
        G::Q8K => Some(S::Q8K),
        G::IQ2XXS => Some(S::Iq2Xxs),
        G::IQ2XS => Some(S::Iq2Xs),
        G::IQ3XXS => Some(S::Iq3Xxs),
        G::IQ4NL => Some(S::Iq4Nl),
        G::IQ1S => Some(S::Iq1S),
        G::IQ1M => Some(S::Iq1M),
        G::IQ2S => Some(S::Iq2S),
        G::IQ3S => Some(S::Iq3S),
        G::IQ4XS => Some(S::Iq4Xs),
        G::TQ1_0 => Some(S::Tq1_0),
        G::TQ2_0 => Some(S::Tq2_0),
        G::Q1_0 => Some(S::Q1_0),
        G::Q2_0 => Some(S::Q2_0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_registry::GgufScheme as S;

    fn parse_ok(text: &str) -> TensorRecipe {
        TensorRecipe::parse(text, "test.recipe").unwrap()
    }

    #[test]
    fn parses_rules_comments_and_default() {
        let r = parse_ok(
            "# FFN down layers get 6 bits\n\
             blk\\.\\d+\\.ffn_down\\.weight=q6_k\n\
             \n\
             # attn value gets 5\n\
             attn_v\\.weight=q5_k_s\n\
             # trailing default\n\
             q4_k_m\n",
        );
        assert_eq!(r.rules.len(), 2);
        assert_eq!(r.rules[0].scheme, S::Q6K);
        assert_eq!(r.rules[1].scheme, S::Q5K);
        assert_eq!(r.default, Some(S::Q4K));
        // scheme_for honors first-match-wins…
        assert_eq!(
            r.scheme_for("blk.3.ffn_down.weight"),
            Some(S::Q6K),
            "first rule matches ffn_down"
        );
        assert_eq!(r.scheme_for("blk.3.attn_v.weight"), Some(S::Q5K));
        // …and the bare default covers unmatched names.
        assert_eq!(r.scheme_for("blk.3.attn_q.weight"), Some(S::Q4K));
    }

    #[test]
    fn first_match_wins_with_overlapping_regexes() {
        // Two rules both matching the same name: the FIRST (in file order)
        // must win, mirroring llama-quant.cpp:716-726.
        let r = parse_ok("attn.*=q5_k_s\nattn_q\\.weight=q6_k\n");
        assert_eq!(r.scheme_for("blk.0.attn_q.weight"), Some(S::Q5K));
        // The second rule still applies to names the first doesn't match.
        assert_eq!(r.scheme_for("blk.0.attn_k.weight"), Some(S::Q5K));
    }

    #[test]
    fn search_semantics_unanchored() {
        // Upstream regex_search is unanchored: the pattern may match a
        // substring anywhere in the tensor name.
        let r = parse_ok("ffn_down=q6_k\n");
        assert_eq!(r.scheme_for("blk.7.ffn_down.weight"), Some(S::Q6K));
    }

    #[test]
    fn no_default_and_no_match_is_none() {
        let r = parse_ok("attn_v\\.weight=q5_k_s\n");
        assert_eq!(r.scheme_for("blk.0.attn_q.weight"), None);
    }

    #[test]
    fn unknown_qtype_names_line_number() {
        let err = TensorRecipe::parse("attn_v\\.weight=bogus\n", "r.txt").unwrap_err();
        match err {
            RecipeError::UnknownQType { line, qtype, .. } => {
                assert_eq!(line, 1);
                assert_eq!(qtype, "bogus");
            }
            other => panic!("expected UnknownQType, got {other}"),
        }
        // Dynamic 2.0 ids are rejected with the same error.
        let err = TensorRecipe::parse("attn_v\\.weight=q4_k_xl\n", "r.txt").unwrap_err();
        assert!(matches!(err, RecipeError::UnknownQType { line: 1, .. }));
    }

    #[test]
    fn malformed_line_names_line_number() {
        // A line with multiple '=' splits on the FIRST '=' (split_once
        // semantics): the qtype becomes "b=q6_k", unknown → a line-numbered
        // UnknownQType error.
        match TensorRecipe::parse("a=b=q6_k\n", "r.txt") {
            Err(RecipeError::UnknownQType { line, qtype, .. }) => {
                assert_eq!(line, 1);
                assert_eq!(qtype, "b=q6_k");
            }
            other => panic!("expected UnknownQType, got {other:?}"),
        }
        // Empty pattern is malformed.
        let err = TensorRecipe::parse("=q6_k\n", "r.txt").unwrap_err();
        assert!(matches!(err, RecipeError::MalformedLine { line: 1, .. }));
        // Bare default BEFORE rules is malformed (must be last).
        let err = TensorRecipe::parse("q4_k_m\nattn_v=q6_k\n", "r.txt").unwrap_err();
        assert!(matches!(err, RecipeError::MalformedLine { line: 2, .. }));
        // Two bare defaults are malformed.
        let err = TensorRecipe::parse("q4_k_m\nq5_k_s\n", "r.txt").unwrap_err();
        assert!(matches!(err, RecipeError::MalformedLine { line: 2, .. }));
    }

    #[test]
    fn invalid_regex_names_line_and_reason() {
        let err = TensorRecipe::parse("[unclosed=q6_k\n", "r.txt").unwrap_err();
        match err {
            RecipeError::BadRegex { line, pattern, .. } => {
                assert_eq!(line, 1);
                assert_eq!(pattern, "[unclosed");
            }
            other => panic!("expected BadRegex, got {other}"),
        }
    }

    #[test]
    fn serialize_round_trips() {
        let text = "# c\nattn_v\\.weight=q5_k_s\nblk\\.\\d+\\.ffn_down\\.weight=q6_k\nq4_k_m\n";
        let r = parse_ok(text);
        // Round-trip: parse(to_text(r)) is semantically identical.
        let r2 = parse_ok(&r.to_text());
        assert_eq!(r2.rules.len(), r.rules.len());
        for (a, b) in r.rules.iter().zip(r2.rules.iter()) {
            assert_eq!(a.pattern, b.pattern);
            assert_eq!(a.scheme, b.scheme);
        }
        assert_eq!(r2.default, r.default);
        // And on a concrete name both agree.
        let name = "blk.2.ffn_down.weight";
        assert_eq!(r.scheme_for(name), r2.scheme_for(name));
    }

    #[test]
    fn duplicate_regex_is_allowed_but_flagged_by_test() {
        // Upstream does not dedupe; first-match-wins makes duplicates
        // harmless. We pin that a duplicate simply keeps file order.
        let r = parse_ok("attn_v\\.weight=q5_k_s\nattn_v\\.weight=q6_k\n");
        assert_eq!(r.scheme_for("blk.0.attn_v.weight"), Some(S::Q5K));
    }

    #[test]
    fn f16_f32_bf16_methods_are_usable_qtypes() {
        // The non-quantized methods are valid recipe targets (a user can
        // force e.g. embeddings to F16 with a recipe).
        let r = parse_ok("token_embd\\.weight=f16\n");
        assert_eq!(r.scheme_for("token_embd.weight"), Some(S::F16));
    }
}

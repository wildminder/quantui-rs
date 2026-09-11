//! Per-tensor quantization policy engine — a port of llama.cpp's
//! `llama_model_quantize_impl` type-selection logic.
//!
//! Unsloth coverage plan Phase 3.0 (§3-I). Official Unsloth delegates all
//! quantization to `llama-quantize` (`ref/unsloth/unsloth/save.py:2340`), so
//! "following official Unsloth" means reproducing llama.cpp's per-tensor
//! policy exactly. The legacy [`crate::gguf_registry::scheme_for`]
//! (name-substring rules) remains the engine for the simple/default
//! methods; this module provides the richer engine the composite
//! `_M`/`_L`/`_XS`/`iq2_m`/`iq3_m`/`q2_k_l` methods require:
//!
//! - tensor categories (`tensor_get_category`, llama-quant.cpp:119),
//! - a running per-category index/count state (quantize_state_impl, :167),
//! - `use_more_bits(i, n)` (llama-quant.cpp:434),
//! - model facts the policy consults: `n_gqa`, `n_expert`, `LLM_TYPE_70B`.
//!
//! Source: llama.cpp (MIT, "Copyright (c) 2023-2026 The ggml
//! authors"). Line references below point into the vendored upstream tree.

use crate::gguf_registry::GgufScheme;

/// Broad tensor category (port of `tensor_category`, llama-quant.cpp:26).
///
/// Deliberately broader than per-arch tensor names: the policy tree matches
/// *roles* (attention-v, ffn-down, ...), not architecture-specific names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCategory {
    TokenEmbd,
    AttentionQ,
    AttentionV,
    AttentionK,
    AttentionQkv,
    AttentionKvB,
    AttentionOutput,
    FfnUp,
    FfnGate,
    FfnDown,
    Output,
    Other,
}

/// Port of `tensor_get_category` (llama-quant.cpp:119-153). Order matters
/// and mirrors upstream exactly: e.g. `attn_qkv.weight` must classify as
/// `AttentionQkv`, not as `AttentionQ`, and `output.weight`/`token_embd`
/// are exact-match before any substring test.
pub fn tensor_get_category(gguf_name: &str) -> TensorCategory {
    use TensorCategory::*;
    if gguf_name == "output.weight" {
        return Output;
    }
    if gguf_name == "token_embd.weight" || gguf_name == "per_layer_token_embd.weight" {
        return TokenEmbd;
    }
    // Substring order mirrors upstream's if-chain (llama-quant.cpp:127-151).
    if gguf_name.contains("attn_qkv.weight") {
        return AttentionQkv;
    }
    if gguf_name.contains("attn_kv_b.weight") {
        return AttentionKvB;
    }
    if gguf_name.contains("attn_v.weight") {
        return AttentionV;
    }
    if gguf_name.contains("attn_k.weight") {
        return AttentionK;
    }
    if gguf_name.contains("attn_q.weight") {
        return AttentionQ;
    }
    if gguf_name.contains("attn_output.weight") {
        return AttentionOutput;
    }
    if gguf_name.contains("ffn_up") {
        return FfnUp;
    }
    if gguf_name.contains("ffn_gate") {
        return FfnGate;
    }
    if gguf_name.contains("ffn_down") {
        return FfnDown;
    }
    Other
}

/// Port of `category_is_attn_v` (llama-quant.cpp:155-161): attention-v-like
/// tensors are more sensitive to quantization, so several methods bump them.
pub fn category_is_attn_v(cat: TensorCategory) -> bool {
    matches!(
        cat,
        TensorCategory::AttentionV | TensorCategory::AttentionQkv | TensorCategory::AttentionKvB
    )
}

/// Port of `use_more_bits` (llama-quant.cpp:434-436, upstream comment:
/// "on layers: 1st and last eighth, then every third of what remains").
pub fn use_more_bits(i_layer: i32, n_layers: i32) -> bool {
    i_layer < n_layers / 8 || i_layer >= 7 * n_layers / 8 || (i_layer - n_layers / 8) % 3 == 2
}

/// Model facts the policy tree consults (llama.cpp reads these from
/// `qs.model.hparams` / `qs.model.type`). Carried per conversion since our
/// registry is static data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModelFacts {
    /// Grouped-query-attention ratio (`n_gqa() >= 4` branches).
    pub n_gqa: i32,
    /// MoE expert count (`n_expert == 8` special cases; max(1, n) upstream).
    pub n_expert: i32,
    /// The 70B-type attention-sharing heuristic (LLM_TYPE_70B, :559).
    pub is_70b_type: bool,
}

impl ModelFacts {
    /// `n_expert` as upstream sees it: `max(1, n_expert)`.
    pub fn experts(&self) -> i32 {
        self.n_expert.max(1)
    }
}

/// Running per-conversion state (port of the counters in
/// `quantize_state_impl`, llama-quant.cpp:171-178). The driver creates one
/// `PolicyState` per conversion and advances it in tensor-file order,
/// exactly as llama.cpp's single pass does.
#[derive(Debug, Clone, Copy, Default)]
pub struct PolicyState {
    pub n_attention_wv: i32,
    pub n_ffn_down: i32,
    pub i_attention_wv: i32,
    pub i_ffn_down: i32,
    /// Whether an imatrix is active (affects iq3_xxs/q4_0/... branches).
    pub has_imatrix: bool,
}

/// The per-tensor policy decision context, combining everything the
/// llama.cpp tree consults for one tensor.
#[derive(Debug, Clone, Copy)]
pub struct PolicyCtx<'a> {
    /// GGUF-side tensor name (e.g. `blk.3.attn_v.weight`).
    pub name: &'a str,
    /// Tensor rank (1-D tensors are F32 by the shared convention).
    pub ndim: usize,
    pub category: TensorCategory,
    pub facts: ModelFacts,
    pub state: &'a PolicyState,
}

/// The method-side policy spec: what the llama.cpp ftype tree reduces to
/// for the methods this engine drives. Kept as data so the registry stays
/// declarative; `resolve` is the interpreter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaPolicy {
    /// Use the flat registry engine (simple/default methods) — the engine
    /// short-circuits to `gguf_registry::scheme_for` and never touches
    /// state. Everything reachable before Phase 3.0 stays here.
    Flat,
    /// `q4_k_m` / `q5_k_m`: attn_v-like AND ffn_down tensors get the
    /// "more bits" scheme when `use_more_bits(i_attn_wv, n_attn_wv)` /
    /// `use_more_bits(i_ffn_down, n_ffn_down)` (llama-quant.cpp:556-557,
    /// 617, 623).
    KMoreBits { base: GgufScheme, more: GgufScheme },
    /// `q2_k`: attn_v → Q4_K under n_gqa>=4 else Q3_K; ffn_down → Q3_K
    /// (:534-535, :593). (The pre-3.0 simplification applied Q4_K to
    /// every attn_v unconditionally.)
    Q2K,
    /// `q3_k_m`: attn_output → Q5_K; attn_q/k/v → Q4_K; ffn_down →
    /// Q6_K on use_more_bits(i_ffn_down, n_ffn_down) else Q4_K (:600-604,
    /// 647); base Q3_K.
    Q3KM,
    /// `q3_k_l`: attn_output → Q5_K; ffn_down → Q5_K (:609-610, 648).
    Q3KL,
    /// `q2_k_l` (Unsloth preset, save.py:377): Q2_K base + output AND
    /// token-embedding forced to Q8_0.
    Q2KL,
    /// `iq2_m` (ftype :29 → GGML_TYPE_IQ2_S, llama-quant.cpp:858).
    /// Rules (:509-531, :500, :472): attn_v → IQ3_S when n_gqa<4 (else
    /// Q4_K when n_gqa>=4 or n_expert>=4); ffn_down first n/8 → IQ3_S;
    /// tok_embd → IQ3_S; output → Q5_K; base IQ2_S.
    Iq2M,
    /// `iq3_m` (ftype :27 → GGML_TYPE_IQ3_S, llama-quant.cpp:870).
    /// Rules (:547, :605-608, :649): attn_v → Q4_K always; ffn_down first
    /// n/8 (or use_more_bits for 8-expert) → Q4_K; attn_output → Q4_K;
    /// tok_embd → base IQ3_S (falls through :495-508); output → Q6_K
    /// (via the generic :474-476 rule — IQ3_M is not in the Q5_K list);
    /// base IQ3_S.
    Iq3M,
}

/// Resolve the scheme for one tensor under a composite policy.
///
/// This is the interpreter for [`LlamaPolicy`]; the driver (gguf_convert)
/// calls it per tensor with the state advanced in file order, then
/// advances the counters via [`advance`] itself.
pub fn resolve(policy: &LlamaPolicy, ctx: &PolicyCtx<'_>) -> GgufScheme {
    use crate::gguf_registry::GgufScheme as S;
    use TensorCategory::*;

    // Shared conventions (identical to the flat engine, gguf_registry:
    // scheme_for): 1-D stays F32.
    if ctx.ndim < 2 {
        return S::F32;
    }

    match policy {
        LlamaPolicy::Flat => unreachable!("Flat policies route through scheme_for"),
        LlamaPolicy::KMoreBits { base, more } => {
            // output/tok_embd (incl. tied): llama-quant.cpp:456-476 —
            // divisibility check, then IQ-family → Q5_K, else Q6_K (unless
            // the method's base is already Q8_0). NOT F16: the flat
            // engine's F16 embd convention is a documented pre-3.0
            // simplification, kept only for Flat methods.
            if ctx.category == Output || ctx.category == TokenEmbd {
                if *base == S::Q8_0 {
                    return S::Q8_0;
                }
                return S::Q6K;
            }
            // attn_v-like: use_more_bits over the attention-wv counter (:556-557).
            if category_is_attn_v(ctx.category) {
                let i = ctx.state.i_attention_wv;
                let n = ctx.state.n_attention_wv;
                return if use_more_bits(i, n) { *more } else { *base };
            }
            // ffn_down: use_more_bits over the ffn-down counter (:617,623).
            if ctx.category == FfnDown {
                let i = ctx.state.i_ffn_down;
                let n = ctx.state.n_ffn_down;
                return if use_more_bits(i, n) { *more } else { *base };
            }
            *base
        }
        LlamaPolicy::Q2K => {
            // :456-476 — the embd/output rule runs before the ftype tree.
            if ctx.category == Output || ctx.category == TokenEmbd {
                return S::Q6K;
            }
            if category_is_attn_v(ctx.category) {
                // :534-535 — n_gqa >= 4 ? Q4_K : Q3_K
                return if ctx.facts.n_gqa >= 4 { S::Q4K } else { S::Q3K };
            }
            if ctx.category == FfnDown {
                return S::Q3K; // :593
            }
            S::Q2K
        }
        LlamaPolicy::Q3KM => {
            if ctx.category == Output || ctx.category == TokenEmbd {
                return S::Q6K;
            }
            match ctx.category {
                AttentionOutput => S::Q5K,                      // :647
                AttentionQ | AttentionK | AttentionV => S::Q4K, // :571-589 pass-through via ftype
                FfnDown => {
                    // :600-604 — use_more_bits(i_ffn_down) ? Q6_K : Q4_K
                    let i = ctx.state.i_ffn_down;
                    let n = ctx.state.n_ffn_down;
                    if use_more_bits(i, n) {
                        S::Q6K
                    } else {
                        S::Q4K
                    }
                }
                _ => S::Q3K,
            }
        }
        LlamaPolicy::Q3KL => {
            if ctx.category == Output || ctx.category == TokenEmbd {
                return S::Q6K;
            }
            match ctx.category {
                AttentionOutput => S::Q5K, // :648
                FfnDown => S::Q5K,         // :609-610 (non-Falcon)
                _ => S::Q3K,
            }
        }
        LlamaPolicy::Q2KL => {
            // Unsloth preset: output AND token embeddings forced to Q8_0;
            // everything else follows plain q2_k.
            match ctx.category {
                Output | TokenEmbd => S::Q8_0,
                _ if ctx.ndim < 2 => S::F32,
                _ => resolve(&LlamaPolicy::Q2K, ctx),
            }
        }
        LlamaPolicy::Iq2M => {
            // Base IQ2_S (ftype :29 → IQ2_S, :858).
            if ctx.ndim < 2 {
                return S::F32;
            }
            match ctx.category {
                // :469-472 — IQ2_M output → Q5_K.
                Output => S::Q5K,
                // :499-501 — IQ2_M tok_embd → IQ3_S.
                TokenEmbd => S::Iq3S,
                cat if category_is_attn_v(cat) => {
                    // :511-514 — n_gqa>=4 || n_expert>=4 → Q4_K, else IQ3_S.
                    if ctx.facts.n_gqa >= 4 || ctx.facts.n_expert >= 4 {
                        S::Q4K
                    } else {
                        S::Iq3S
                    }
                }
                FfnDown => {
                    // :519-522 — first n/8 of ffn_down → IQ3_S, else base.
                    if ctx.state.i_ffn_down < ctx.state.n_ffn_down / 8 {
                        S::Iq3S
                    } else {
                        S::Iq2S
                    }
                }
                _ => S::Iq2S,
            }
        }
        LlamaPolicy::Iq3M => {
            // Base IQ3_S (ftype :27 → IQ3_S, :870).
            if ctx.ndim < 2 {
                return S::F32;
            }
            match ctx.category {
                // :474-476 — IQ3_M is not in the Q5_K list → generic Q6_K.
                Output => S::Q6K,
                // :495-508 — IQ3_M is not in any tok_embd branch → base.
                TokenEmbd => S::Iq3S,
                cat if category_is_attn_v(cat) => S::Q4K, // :546-548
                FfnDown => {
                    // :605-608 — first n/8 (or 8-expert use_more_bits) → Q4_K.
                    let first8 = ctx.state.i_ffn_down < ctx.state.n_ffn_down / 8;
                    let expert8 = ctx.facts.n_expert == 8
                        && use_more_bits(ctx.state.i_ffn_down, ctx.state.n_ffn_down);
                    if first8 || expert8 {
                        S::Q4K
                    } else {
                        S::Iq3S
                    }
                }
                AttentionOutput => S::Q4K, // :649 (non-8-expert branch)
                _ => S::Iq3S,
            }
        }
    }
}

/// Advance the running counters for one processed tensor — port of the
/// `++qs.i_attention_wv` / `++qs.i_ffn_down` increments (llama-quant.cpp:570,
/// 634). Call AFTER resolving the tensor's scheme (llama.cpp resolves,
/// then increments).
pub fn advance(state: &mut PolicyState, ctx_category: TensorCategory) {
    if category_is_attn_v(ctx_category) {
        state.i_attention_wv += 1;
    }
    if ctx_category == TensorCategory::FfnDown {
        state.i_ffn_down += 1;
    }
}

/// Count the totals for a whole tensor-name list — port of
/// `init_quantize_state_counters` (llama-quant.cpp:882-896), which counts
/// attention-wv and ffn-down tensors before the main pass.
pub fn count_state(names: &[&str], has_imatrix: bool) -> PolicyState {
    let mut st = PolicyState {
        has_imatrix,
        ..Default::default()
    };
    for name in names {
        match tensor_get_category(name) {
            TensorCategory::Output => {
                // has_tied_embeddings is set false when output.weight is
                // present; it does not affect the counters we track here.
            }
            cat if category_is_attn_v(cat) => st.n_attention_wv += 1,
            TensorCategory::FfnDown => st.n_ffn_down += 1,
            _ => {}
        }
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── tensor_get_category: the classifier table ───────────────────

    #[test]
    fn classifier_matches_llama_cpp_order() {
        use TensorCategory::*;
        // Exact-match precedence (llama-quant.cpp:104-112).
        assert_eq!(tensor_get_category("output.weight"), Output);
        assert_eq!(tensor_get_category("token_embd.weight"), TokenEmbd);
        assert_eq!(
            tensor_get_category("per_layer_token_embd.weight"),
            TokenEmbd
        );
        // Substring precedence: qkv before v/k/q (:127-137).
        assert_eq!(tensor_get_category("blk.0.attn_qkv.weight"), AttentionQkv);
        assert_eq!(tensor_get_category("blk.0.attn_kv_b.weight"), AttentionKvB);
        assert_eq!(tensor_get_category("blk.0.attn_v.weight"), AttentionV);
        assert_eq!(tensor_get_category("blk.0.attn_k.weight"), AttentionK);
        assert_eq!(tensor_get_category("blk.0.attn_q.weight"), AttentionQ);
        // attn_output must not match "attn_q" first? No: upstream tests
        // attn_q.weight BEFORE attn_output.weight, but "attn_output" does
        // not contain "attn_q.weight" (it contains "attn_o..."). Both
        // orders agree because the needles include ".weight"/full words.
        assert_eq!(
            tensor_get_category("blk.0.attn_output.weight"),
            AttentionOutput
        );
        // FFN family (:141-150).
        assert_eq!(tensor_get_category("blk.0.ffn_up.weight"), FfnUp);
        assert_eq!(tensor_get_category("blk.0.ffn_gate.weight"), FfnGate);
        assert_eq!(tensor_get_category("blk.0.ffn_down.weight"), FfnDown);
        // Catch-all.
        assert_eq!(tensor_get_category("blk.0.attn_norm.weight"), Other);
        assert_eq!(tensor_get_category("rope.freqs"), Other);
    }

    #[test]
    fn attn_v_like_includes_qkv_and_kv_b() {
        // :155-161 — the "more sensitive" set is v, qkv, kv_b.
        for n in [
            "blk.0.attn_v.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_kv_b.weight",
        ] {
            assert!(category_is_attn_v(tensor_get_category(n)), "{n}");
        }
        for n in [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.ffn_down.weight",
        ] {
            assert!(!category_is_attn_v(tensor_get_category(n)), "{n}");
        }
    }

    // ─── use_more_bits: the layer heuristic ──────────────────────────

    #[test]
    fn use_more_bits_matches_llama_cpp_table() {
        // :434-436 — first and last eighth, then every third of what remains.
        // Upstream: i < n/8 || i >= 7n/8 || (i - n/8) % 3 == 2.
        // n = 32: first 4 (0-3), last 4 (28-31), then i in 4..28 with
        // (i-4) % 3 == 2 → i ∈ {6, 9, 12, 15, 18, 21, 24, 27}.
        let n = 32;
        let expect: Vec<bool> = (0..n)
            .map(|i| !(4..28).contains(&i) || (i - 4) % 3 == 2)
            .collect();
        for (i, want) in expect.iter().enumerate() {
            assert_eq!(use_more_bits(i as i32, n), *want, "i={i}");
        }
        // Spot values.
        assert!(use_more_bits(0, 32));
        assert!(use_more_bits(6, 32));
        assert!(!use_more_bits(4, 32));
        assert!(use_more_bits(31, 32));
        // Tiny model (n=1): everything is "first eighth".
        assert!(use_more_bits(0, 1));
    }

    // ─── KMoreBits: q4_k_m / q5_k_m ─────────────────────────────────

    fn mk_ctx<'a>(
        name: &'a str,
        ndim: usize,
        state: &'a PolicyState,
        facts: ModelFacts,
    ) -> PolicyCtx<'a> {
        PolicyCtx {
            name,
            ndim,
            category: tensor_get_category(name),
            facts,
            state,
        }
    }

    #[test]
    fn k_more_bits_attn_v_uses_counter() {
        // :556-557 — Q4_K_M attn_v gets Q6_K exactly when
        // use_more_bits(i_attention_wv, n_attention_wv).
        let policy = LlamaPolicy::KMoreBits {
            base: crate::gguf_registry::GgufScheme::Q4K,
            more: crate::gguf_registry::GgufScheme::Q6K,
        };
        let facts = ModelFacts::default();
        // State: 8 attn_v seen, we're on index 3 → use_more_bits(3,8)?
        // 3 < 1? no. 3 >= 7? no. (3-1)%3==2 → 2%3==2 → yes.
        let st = PolicyState {
            n_attention_wv: 8,
            i_attention_wv: 3,
            ..Default::default()
        };
        let ctx = mk_ctx("blk.3.attn_v.weight", 2, &st, facts);
        assert_eq!(
            resolve(&policy, &ctx),
            crate::gguf_registry::GgufScheme::Q6K
        );

        // Index 0 with n=8: 0 < 1 → true (first eighth).
        let st = PolicyState {
            n_attention_wv: 8,
            i_attention_wv: 0,
            ..Default::default()
        };
        let ctx = mk_ctx("blk.0.attn_v.weight", 2, &st, facts);
        assert_eq!(
            resolve(&policy, &ctx),
            crate::gguf_registry::GgufScheme::Q6K
        );

        // Index 2 with n=8: 2<1 no, 2>=7 no, (2-1)%3==1 no → base.
        let st = PolicyState {
            n_attention_wv: 8,
            i_attention_wv: 2,
            ..Default::default()
        };
        let ctx = mk_ctx("blk.2.attn_v.weight", 2, &st, facts);
        assert_eq!(
            resolve(&policy, &ctx),
            crate::gguf_registry::GgufScheme::Q4K
        );
    }

    #[test]
    fn k_more_bits_ffn_down_uses_its_own_counter() {
        // :617 — Q4_K_M ffn_down: Q6_K on use_more_bits(i_ffn_down).
        let policy = LlamaPolicy::KMoreBits {
            base: crate::gguf_registry::GgufScheme::Q4K,
            more: crate::gguf_registry::GgufScheme::Q6K,
        };
        let st = PolicyState {
            n_ffn_down: 32,
            i_ffn_down: 6, // (6-4)%3 == 2 → true
            n_attention_wv: 4,
            i_attention_wv: 3, // 3 with n=4: >= 7*4/8 → true — must NOT apply here
            ..Default::default()
        };
        let facts = ModelFacts::default();
        let ctx = mk_ctx("blk.6.ffn_down.weight", 2, &st, facts);
        assert_eq!(
            resolve(&policy, &ctx),
            crate::gguf_registry::GgufScheme::Q6K
        );
    }

    #[test]
    fn k_more_bits_conventions() {
        let policy = LlamaPolicy::KMoreBits {
            base: crate::gguf_registry::GgufScheme::Q4K,
            more: crate::gguf_registry::GgufScheme::Q6K,
        };
        let st = PolicyState::default();
        let facts = ModelFacts::default();
        // 1-D stays F32.
        assert_eq!(
            resolve(&policy, &mk_ctx("blk.0.attn_norm.weight", 1, &st, facts)),
            crate::gguf_registry::GgufScheme::F32
        );
        // embd/output take the llama.cpp convention (:456-476): Q6_K for
        // a K-quant base method (F16 belongs to the pre-3.0 flat engine).
        assert_eq!(
            resolve(&policy, &mk_ctx("token_embd.weight", 2, &st, facts)),
            crate::gguf_registry::GgufScheme::Q6K
        );
        assert_eq!(
            resolve(&policy, &mk_ctx("output.weight", 2, &st, facts)),
            crate::gguf_registry::GgufScheme::Q6K
        );
        // Non-special 2-D → base.
        assert_eq!(
            resolve(&policy, &mk_ctx("blk.0.attn_q.weight", 2, &st, facts)),
            crate::gguf_registry::GgufScheme::Q4K
        );
    }

    // ─── Q2K: the gqa branch ────────────────────────────────────────

    #[test]
    fn q2k_attn_v_depends_on_gqa() {
        let st = PolicyState::default();
        let lo = ModelFacts {
            n_gqa: 1,
            ..Default::default()
        };
        let hi = ModelFacts {
            n_gqa: 4,
            ..Default::default()
        };
        // n_gqa < 4 → Q3_K (:535).
        assert_eq!(
            resolve(
                &LlamaPolicy::Q2K,
                &mk_ctx("blk.0.attn_v.weight", 2, &st, lo)
            ),
            crate::gguf_registry::GgufScheme::Q3K
        );
        // n_gqa >= 4 → Q4_K (:534-535).
        assert_eq!(
            resolve(
                &LlamaPolicy::Q2K,
                &mk_ctx("blk.0.attn_v.weight", 2, &st, hi)
            ),
            crate::gguf_registry::GgufScheme::Q4K
        );
        // ffn_down → Q3_K (:593) regardless of gqa.
        assert_eq!(
            resolve(
                &LlamaPolicy::Q2K,
                &mk_ctx("blk.0.ffn_down.weight", 2, &st, hi)
            ),
            crate::gguf_registry::GgufScheme::Q3K
        );
    }

    // ─── Q3KM / Q3KL ────────────────────────────────────────────────

    #[test]
    fn q3km_categories() {
        let st = PolicyState {
            n_ffn_down: 8,
            i_ffn_down: 0, // 0 < 8/8=1 → true → Q6_K
            ..Default::default()
        };
        let facts = ModelFacts::default();
        assert_eq!(
            resolve(
                &LlamaPolicy::Q3KM,
                &mk_ctx("blk.0.ffn_down.weight", 2, &st, facts)
            ),
            crate::gguf_registry::GgufScheme::Q6K
        );
        // Same but i=2, n=8: 2<1 no; 2>=7 no; (2-1)%3=1 no → Q4_K (:602).
        let st = PolicyState {
            n_ffn_down: 8,
            i_ffn_down: 2,
            ..Default::default()
        };
        assert_eq!(
            resolve(
                &LlamaPolicy::Q3KM,
                &mk_ctx("blk.2.ffn_down.weight", 2, &st, facts)
            ),
            crate::gguf_registry::GgufScheme::Q4K
        );
        // attn_output → Q5_K (:647); attn_q → Q4_K; plain → Q3_K.
        let st = PolicyState::default();
        assert_eq!(
            resolve(
                &LlamaPolicy::Q3KM,
                &mk_ctx("blk.0.attn_output.weight", 2, &st, facts)
            ),
            crate::gguf_registry::GgufScheme::Q5K
        );
        assert_eq!(
            resolve(
                &LlamaPolicy::Q3KM,
                &mk_ctx("blk.0.attn_q.weight", 2, &st, facts)
            ),
            crate::gguf_registry::GgufScheme::Q4K
        );
        assert_eq!(
            resolve(
                &LlamaPolicy::Q3KM,
                &mk_ctx("blk.0.ffn_up.weight", 2, &st, facts)
            ),
            crate::gguf_registry::GgufScheme::Q3K
        );
    }

    #[test]
    fn q3kl_categories() {
        let st = PolicyState::default();
        let facts = ModelFacts::default();
        assert_eq!(
            resolve(
                &LlamaPolicy::Q3KL,
                &mk_ctx("blk.0.attn_output.weight", 2, &st, facts)
            ),
            crate::gguf_registry::GgufScheme::Q5K
        );
        assert_eq!(
            resolve(
                &LlamaPolicy::Q3KL,
                &mk_ctx("blk.0.ffn_down.weight", 2, &st, facts)
            ),
            crate::gguf_registry::GgufScheme::Q5K
        );
        assert_eq!(
            resolve(
                &LlamaPolicy::Q3KL,
                &mk_ctx("blk.0.attn_q.weight", 2, &st, facts)
            ),
            crate::gguf_registry::GgufScheme::Q3K
        );
    }

    // ─── Iq2M / Iq3M: the ftype-resolved policy variants ───────────

    #[test]
    fn iq2m_rules() {
        let facts_no_gqa = ModelFacts {
            n_gqa: 1,
            ..Default::default()
        };
        let facts_gqa = ModelFacts {
            n_gqa: 4,
            ..Default::default()
        };
        // 8 ffn_down: first n/8 = 1 gets IQ3_S, the rest base IQ2_S.
        let st8 = PolicyState {
            n_ffn_down: 8,
            ..Default::default()
        };
        // attn_v: n_gqa<4 → IQ3_S (:513); n_gqa>=4 → Q4_K (:512).
        assert_eq!(
            resolve(
                &LlamaPolicy::Iq2M,
                &mk_ctx("blk.0.attn_v.weight", 2, &st8, facts_no_gqa)
            ),
            crate::gguf_registry::GgufScheme::Iq3S
        );
        assert_eq!(
            resolve(
                &LlamaPolicy::Iq2M,
                &mk_ctx("blk.0.attn_v.weight", 2, &st8, facts_gqa)
            ),
            crate::gguf_registry::GgufScheme::Q4K
        );
        // output → Q5_K (:469-472); tok_embd → IQ3_S (:499-501).
        assert_eq!(
            resolve(
                &LlamaPolicy::Iq2M,
                &mk_ctx("output.weight", 2, &st8, facts_no_gqa)
            ),
            crate::gguf_registry::GgufScheme::Q5K
        );
        assert_eq!(
            resolve(
                &LlamaPolicy::Iq2M,
                &mk_ctx("token_embd.weight", 2, &st8, facts_no_gqa)
            ),
            crate::gguf_registry::GgufScheme::Iq3S
        );
        // ffn_down i=0 (<8/8=1) → IQ3_S; i=2 → base IQ2_S.
        let st_i0 = PolicyState {
            n_ffn_down: 8,
            i_ffn_down: 0,
            ..Default::default()
        };
        assert_eq!(
            resolve(
                &LlamaPolicy::Iq2M,
                &mk_ctx("blk.0.ffn_down.weight", 2, &st_i0, facts_no_gqa)
            ),
            crate::gguf_registry::GgufScheme::Iq3S
        );
        let st_i2 = PolicyState {
            n_ffn_down: 8,
            i_ffn_down: 2,
            ..Default::default()
        };
        assert_eq!(
            resolve(
                &LlamaPolicy::Iq2M,
                &mk_ctx("blk.2.ffn_down.weight", 2, &st_i2, facts_no_gqa)
            ),
            crate::gguf_registry::GgufScheme::Iq2S
        );
        // Plain 2-D → base IQ2_S.
        assert_eq!(
            resolve(
                &LlamaPolicy::Iq2M,
                &mk_ctx("blk.0.attn_q.weight", 2, &st8, facts_no_gqa)
            ),
            crate::gguf_registry::GgufScheme::Iq2S
        );
    }

    #[test]
    fn iq3m_rules() {
        let facts = ModelFacts::default();
        let st8 = PolicyState {
            n_ffn_down: 8,
            ..Default::default()
        };
        // attn_v → Q4_K always (:546-548).
        assert_eq!(
            resolve(
                &LlamaPolicy::Iq3M,
                &mk_ctx("blk.0.attn_v.weight", 2, &st8, facts)
            ),
            crate::gguf_registry::GgufScheme::Q4K
        );
        // attn_output → Q4_K (:649).
        assert_eq!(
            resolve(
                &LlamaPolicy::Iq3M,
                &mk_ctx("blk.0.attn_output.weight", 2, &st8, facts)
            ),
            crate::gguf_registry::GgufScheme::Q4K
        );
        // output → Q6_K (:474-476, IQ3_M not in the Q5_K list).
        assert_eq!(
            resolve(&LlamaPolicy::Iq3M, &mk_ctx("output.weight", 2, &st8, facts)),
            crate::gguf_registry::GgufScheme::Q6K
        );
        // tok_embd → base IQ3_S (no :499-508 branch matches IQ3_M).
        assert_eq!(
            resolve(
                &LlamaPolicy::Iq3M,
                &mk_ctx("token_embd.weight", 2, &st8, facts)
            ),
            crate::gguf_registry::GgufScheme::Iq3S
        );
        // ffn_down i=0 (<1) → Q4_K; i=2 → base IQ3_S.
        let st_i0 = PolicyState {
            n_ffn_down: 8,
            i_ffn_down: 0,
            ..Default::default()
        };
        assert_eq!(
            resolve(
                &LlamaPolicy::Iq3M,
                &mk_ctx("blk.0.ffn_down.weight", 2, &st_i0, facts)
            ),
            crate::gguf_registry::GgufScheme::Q4K
        );
        let st_i2 = PolicyState {
            n_ffn_down: 8,
            i_ffn_down: 2,
            ..Default::default()
        };
        assert_eq!(
            resolve(
                &LlamaPolicy::Iq3M,
                &mk_ctx("blk.2.ffn_down.weight", 2, &st_i2, facts)
            ),
            crate::gguf_registry::GgufScheme::Iq3S
        );
        // Plain 2-D → base IQ3_S.
        assert_eq!(
            resolve(
                &LlamaPolicy::Iq3M,
                &mk_ctx("blk.0.attn_q.weight", 2, &st8, facts)
            ),
            crate::gguf_registry::GgufScheme::Iq3S
        );
    }

    // ─── Q2KL: the Unsloth preset ──────────────────────────────────

    #[test]
    fn q2kl_forces_q8_embeddings() {
        // save.py:377 — q2_k + --output-tensor-type q8_0
        // --token-embedding-type q8_0. Both embeddings forced, unlike the
        // Q6_K convention the engine gives plain q2_k.
        let st = PolicyState::default();
        let facts = ModelFacts::default();
        assert_eq!(
            resolve(
                &LlamaPolicy::Q2KL,
                &mk_ctx("token_embd.weight", 2, &st, facts)
            ),
            crate::gguf_registry::GgufScheme::Q8_0
        );
        assert_eq!(
            resolve(&LlamaPolicy::Q2KL, &mk_ctx("output.weight", 2, &st, facts)),
            crate::gguf_registry::GgufScheme::Q8_0
        );
        // And the rest follows q2_k's own rules.
        assert_eq!(
            resolve(
                &LlamaPolicy::Q2KL,
                &mk_ctx("blk.0.attn_v.weight", 2, &st, facts)
            ),
            resolve(
                &LlamaPolicy::Q2K,
                &mk_ctx("blk.0.attn_v.weight", 2, &st, facts)
            )
        );
    }

    // ─── counters: count_state + advance ────────────────────────────

    #[test]
    fn count_state_and_advance_mirror_llama_counters() {
        // 3 attn_v + 1 qkv (also attn_v-like) + 2 ffn_down + noise.
        let names = [
            "blk.0.attn_v.weight",
            "blk.1.attn_v.weight",
            "blk.2.attn_v.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ffn_down.weight",
            "blk.1.ffn_down.weight",
            "blk.0.attn_norm.weight", // Other — not counted
            "output.weight",
            "token_embd.weight",
        ];
        let mut st = count_state(&names, false);
        assert_eq!(st.n_attention_wv, 4);
        assert_eq!(st.n_ffn_down, 2);
        assert_eq!(st.i_attention_wv, 0);
        assert_eq!(st.i_ffn_down, 0);
        assert!(!st.has_imatrix);

        // Process in order; advance must bump the matching counter only.
        advance(&mut st, tensor_get_category("blk.0.attn_v.weight"));
        assert_eq!(st.i_attention_wv, 1);
        advance(&mut st, tensor_get_category("blk.0.ffn_down.weight"));
        assert_eq!(st.i_ffn_down, 1);
        advance(&mut st, tensor_get_category("blk.0.attn_norm.weight"));
        assert_eq!(st.i_attention_wv, 1);
        assert_eq!(st.i_ffn_down, 1);
    }

    /// End-to-end mini-model: 8 layers, standard llama tensor names — the
    /// same shape llama.cpp's own tests exercise implicitly. Proves the
    /// driver protocol (count → resolve → advance) produces the exact
    /// use_more_bits pattern on attn_v for q4_k_m.
    #[test]
    fn q4_k_m_driver_protocol_end_to_end() {
        // Build 8-layer names: attn_v + ffn_down per layer, attn_q as filler.
        let mut names: Vec<String> = Vec::new();
        for l in 0..8 {
            names.push(format!("blk.{l}.attn_v.weight"));
            names.push(format!("blk.{l}.ffn_down.weight"));
        }
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let mut st = count_state(&refs, false);
        assert_eq!(st.n_attention_wv, 8);
        assert_eq!(st.n_ffn_down, 8);

        let policy = LlamaPolicy::KMoreBits {
            base: crate::gguf_registry::GgufScheme::Q4K,
            more: crate::gguf_registry::GgufScheme::Q6K,
        };
        let facts = ModelFacts::default();

        // Resolve every attn_v in order and record the scheme.
        let mut got = Vec::new();
        for name in &refs {
            let cat = tensor_get_category(name);
            let ctx = mk_ctx(name, 2, &st, facts);
            let s = resolve(&policy, &ctx);
            if cat == TensorCategory::AttentionV {
                got.push(s);
            }
            advance(&mut st, cat);
        }
        // Expected per use_more_bits(i, 8): i<1 → true (i=0);
        // i>=7 → true (i=7); (i-1)%3==2 → i ∈ {1+2=3? (3-1)%3=2 yes; (4-1)%3=0 no; (5-1)%3=1 no; (6-1)%3=2 yes}.
        // So true for i ∈ {0, 3, 6, 7}.
        use crate::gguf_registry::GgufScheme as S;
        let want = vec![
            S::Q6K,
            S::Q4K,
            S::Q4K,
            S::Q6K,
            S::Q4K,
            S::Q4K,
            S::Q6K,
            S::Q6K,
        ];
        assert_eq!(got, want);
    }
}

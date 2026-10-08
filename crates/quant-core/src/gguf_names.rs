//! GGUF-side tensor-name mapping + architecture detection (IMP-001).
//!
//! Extracted verbatim from `gguf_convert.rs` — pure functions with zero
//! coupling to the encoding pipeline. The naming table is sourced from
//! llama.cpp and must never drift.
//!
//! Provenance:
//! - tensor names: llama.cpp `gguf-py/gguf/tensor_mapping.py` (the
//!   SHORTCONV_*, w1/w2/w3, operator_norm, ffn_norm, out_proj,
//!   q/k_layernorm, embedding_norm arms) + `convert_hf_to_gguf.py` dense
//!   family;
//! - arch strings: llama.cpp `src/llama-arch.cpp` (e.g. `lfm2` at :126);
//! - nested-prefix stripping: generic rule for wrapped multimodal
//!   checkpoints (`model.language_model.*`, `model.model.*` — verified
//!   against real VibeVoice-1.5B and LFM2.5-VL-3B sources).

use std::path::Path;

use rlx_gguf::MetaValue;

// ─── HF → GGUF tensor name mapping ──────────────────────────────────

/// Map a HuggingFace tensor name to its GGUF (llama.cpp) equivalent for the
/// dense llama/qwen/mistral/gemma family. Returns `None` for names with no
/// known mapping (callers pass them through unchanged).
///
/// `arch` disambiguates the one HF core name that maps differently per
/// family: `post_attention_layernorm` is the gemma2/gemma3 post-attention
/// norm (`post_attention_norm`) but the llama-family FFN norm (`ffn_norm`)
/// — llama.cpp registers BOTH arms (tensor_mapping.py:371 ATTN_POST_NORM
/// `# gemma2`, :415 FFN_NORM `# llama-hf`) and the per-arch tensor lists
/// pick (GEMMA2/GEMMA3 carry ATTN_POST_NORM + FFN_PRE_NORM + FFN_POST_NORM
/// and NOT FFN_NORM; GEMMA keeps FFN_NORM, constants.py:3123-3135).
///
/// Generic nested-prefix handling: multimodal / wrapped checkpoints
/// (VibeVoice, LFM2-VL, ...) nest the language model one level below the
/// top-level `model.` module (`model.language_model.layers.3.…`,
/// `model.model.layers.3.…`). The wrapper is stripped and the inner name is
/// fed through the same dense mapper below, so wrapped dense cores still
/// land on `blk.*`. Arch-specific cores that are not in the dense table
/// (LFM2's `conv.*` / `feed_forward.w1`, vision towers, conv heads) return
/// `None` and keep their ORIGINAL full name — nothing is ever guessed.
///
/// Port of the naming used by llama.cpp `convert_hf_to_gguf.py` for the
/// architectures the reference Unsloth GGUF path supports.
pub fn hf_to_gguf_name(name: &str, arch: &str) -> Option<String> {
    // Strip the nested language-model wrapper before any matching, so both
    // the dense form and the wrapped form share one mapping path.
    let name = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model.model."))
        .unwrap_or(name);

    // Top-level (non-layer) tensors — both the dense `model.*` form and the
    // bare form left after the prefix strip above.
    match name {
        "model.embed_tokens.weight" | "embed_tokens.weight" => {
            return Some("token_embd.weight".into())
        }
        "lm_head.weight" => return Some("output.weight".into()),
        "model.norm.weight" | "norm.weight" => return Some("output_norm.weight".into()),
        // LFM2's pre-embedding norm (tensor_mapping.py:65 → TOKEN_EMBD_NORM).
        "model.embedding_norm.weight" | "embedding_norm.weight" => {
            return Some("token_embd_norm.weight".into())
        }
        _ => {}
    }

    // Layer tensors: `model.layers.{i}.<rest>` (dense) or `layers.{i}.<rest>`
    // (after nested-prefix stripping).
    let rest = name
        .strip_prefix("model.layers.")
        .or_else(|| name.strip_prefix("layers."))?;
    let (idx_str, tail) = rest.split_once('.')?;
    let idx: u32 = idx_str.parse().ok()?;
    let blk = format!("blk.{idx}");

    // Split off the trailing .weight / .bias (if any) to map the core path.
    let (core, suffix) = if let Some(c) = tail.strip_suffix(".weight") {
        (c, ".weight")
    } else if let Some(c) = tail.strip_suffix(".bias") {
        (c, ".bias")
    } else {
        (tail, "")
    };

    let mapped = match core {
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        // LFM2.5's full-attention layers (out_proj / q_layernorm /
        // k_layernorm; tensor_mapping.py:330, :698, :715).
        "self_attn.out_proj" => "attn_output",
        "self_attn.q_layernorm" => "attn_q_norm",
        "self_attn.k_layernorm" => "attn_k_norm",
        "self_attn.q_norm" => "attn_q_norm",
        "self_attn.k_norm" => "attn_k_norm",
        "self_attn.rotary_emb.inv_freq" => return None, // skip rotary cache
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        "mlp.gate" => "ffn_gate_inp",
        "input_layernorm" => "attn_norm",
        // The arch split documented on the function: gemma2/gemma3 have a
        // SEPARATE post-attention norm tensor; llama-family and gemma v1
        // fold it into ffn_norm. gemma3n registers the same norm set as
        // gemma3 (constants.py GEMMA3N list).
        "post_attention_layernorm" => match arch {
            "gemma2" | "gemma3" | "gemma3n" => "post_attention_norm",
            _ => "ffn_norm",
        },
        // FFN_PRE_NORM → blk.{bid}.ffn_norm (constants.py:1428). Only the
        // gemma family (gemma2+: HF GemmaRMSNorm pre-FFN) produces this HF
        // name, so it needs no arch split.
        "pre_feedforward_layernorm" => "ffn_norm",
        // FFN_POST_NORM → blk.{bid}.post_ffw_norm (constants.py:1429).
        "post_feedforward_layernorm" => "post_ffw_norm",
        // LFM2 layer cores (tensor_mapping.py:1464-1468, :214 + the
        // llama-pth w1/w2/w3 entries; arch `lfm2`, llama-arch.cpp:126).
        "conv.conv" => "shortconv.conv",
        "conv.in_proj" => "shortconv.in_proj",
        "conv.out_proj" => "shortconv.out_proj",
        "feed_forward.w1" => "ffn_gate",
        "feed_forward.w2" => "ffn_down",
        "feed_forward.w3" => "ffn_up",
        "operator_norm" => "attn_norm", // lfm2's pre-attention norm
        // `ffn_norm` (LFM2 uses the internlm2-style pattern,
        // tensor_mapping.py:407) — same GGUF name as llama's
        // post_attention_layernorm.
        "ffn_norm" => "ffn_norm",
        _ => return None,
    };
    Some(format!("{blk}.{mapped}{suffix}"))
}

/// Gemma-family norm pre-bake: every gemma `*norm.weight` tensor is stored
/// as `1 + w`. The HF runtime multiplies by `(1 + w)` inside RMSNorm
/// (transformers modeling_gemma.py `GemmaRMSNorm`), so the llama.cpp
/// converter folds the offset into the stored weight (conversion/gemma.py:
/// `if name.endswith("norm.weight"): data_torch = data_torch + 1` — the
/// Gemma/Gemma2/Gemma3 classes all do it, name-based, no shape gate).
///
/// Ground-truthed against the unsloth gemma3 reference: the stored payload
/// is the EXACT f32 sum `f32(w) + 1.0` (probe: 640/640 elements exact for
/// all seven norm kinds; the bf16-rounded sum matched only partially), so
/// decode-to-f32 then `+= 1.0` reproduces the reference bit-for-bit.
pub fn gemma_norm_plus_one(arch: &str, hf_name: &str) -> bool {
    arch.starts_with("gemma") && hf_name.ends_with("norm.weight")
}

// ─── config.json arch detection ─────────────────────────────────────

/// Architecture + numeric parameters parsed from `config.json` (best-effort).
#[derive(Debug, Clone, Default)]
pub struct ArchInfo {
    /// GGUF architecture string (e.g. `"llama"`, `"qwen2"`).
    pub arch: String,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub block_count: Option<u64>,
    pub feed_forward_length: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub rms_norm_eps: Option<f64>,
    pub vocab_size: Option<u64>,
}

impl ArchInfo {
    /// Emit llama.cpp-compatible `{arch}.*` metadata pairs for the values we
    /// managed to read.
    pub(crate) fn meta_pairs(&self, arch: &str) -> Vec<(String, MetaValue)> {
        let mut out = Vec::new();
        let a = arch.to_string();
        if let Some(v) = self.context_length {
            out.push((format!("{a}.context_length"), MetaValue::U64(v)));
        }
        if let Some(v) = self.embedding_length {
            out.push((format!("{a}.embedding_length"), MetaValue::U64(v)));
        }
        if let Some(v) = self.block_count {
            out.push((format!("{a}.block_count"), MetaValue::U64(v)));
        }
        if let Some(v) = self.feed_forward_length {
            out.push((format!("{a}.feed_forward_length"), MetaValue::U64(v)));
        }
        if let Some(v) = self.head_count {
            out.push((format!("{a}.attention.head_count"), MetaValue::U64(v)));
        }
        if let Some(v) = self.head_count_kv {
            out.push((format!("{a}.attention.head_count_kv"), MetaValue::U64(v)));
        }
        if let Some(v) = self.rms_norm_eps {
            out.push((
                format!("{a}.attention.layer_norm_rms_epsilon"),
                MetaValue::F32(v as f32),
            ));
        }
        out
    }
}

/// Map a HF `architectures[0]` class name to a GGUF arch string.
pub(crate) fn arch_from_classname(cls: &str) -> String {
    match cls {
        "LlamaForCausalLM" | "MistralForCausalLM" | "MixtralForCausalLM" => "llama".into(),
        "Qwen2ForCausalLM" | "Qwen3ForCausalLM" => "qwen2".into(),
        "GemmaForCausalLM" => "gemma".into(),
        "Gemma2ForCausalLM" => "gemma2".into(),
        "Gemma3ForCausalLM" => "gemma3".into(),
        "PhiForCausalLM" | "Phi3ForCausalLM" => "phi".into(),
        "GPT2LMHeadModel" => "gpt2".into(),
        "BloomForCausalLM" => "bloom".into(),
        "FalconForCausalLM" => "falcon".into(),
        "StableLmForCausalLM" => "stablelm".into(),
        // LFM2 family (LiquidAI): llama.cpp registers `lfm2`
        // (llama-arch.cpp:126). The VL class wraps the same language model.
        "LFM2ForCausalLM" | "Lfm2ForCausalLM" => "lfm2".into(),
        "LFM2VLForConditionalGeneration" | "Lfm2VLForConditionalGeneration" => "lfm2".into(),
        other => {
            // Fallback: lowercase the leading word before "For".
            let stem = other.split("For").next().unwrap_or(other);
            stem.to_lowercase()
        }
    }
}

/// Read `<dir>/config.json` and extract arch + numeric params. Returns `None`
/// if the file is absent or unparseable (conversion still proceeds with a
/// default arch).
pub(crate) fn load_arch_info(dir: &Path) -> Option<ArchInfo> {
    let path = dir.join("config.json");
    let raw = std::fs::read_to_string(&path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;

    let cls = v
        .get("architectures")
        .and_then(|a| a.get(0))
        .and_then(|s| s.as_str())
        .unwrap_or("LlamaForCausalLM");
    let arch = v
        .get("model_type")
        .and_then(|s| s.as_str())
        .map(|mt| {
            // Prefer model_type when it's a known GGUF arch; else class map.
            match mt {
                "llama" | "mistral" | "mixtral" => "llama".to_string(),
                "qwen2" | "qwen3" => "qwen2".to_string(),
                "gemma" => "gemma".to_string(),
                "gemma2" => "gemma2".to_string(),
                "gemma3" | "gemma3_text" => "gemma3".to_string(),
                "phi" | "phi3" => "phi".to_string(),
                "gpt2" => "gpt2".to_string(),
                "lfm2" => "lfm2".to_string(),
                _ => arch_from_classname(cls),
            }
        })
        .unwrap_or_else(|| arch_from_classname(cls));

    let u64 = |k: &str| v.get(k).and_then(|x| x.as_u64());
    let f64 = |k: &str| v.get(k).and_then(|x| x.as_f64());

    Some(ArchInfo {
        arch,
        context_length: u64("max_position_embeddings"),
        embedding_length: u64("hidden_size"),
        block_count: u64("num_hidden_layers"),
        feed_forward_length: u64("intermediate_size"),
        head_count: u64("num_attention_heads"),
        head_count_kv: u64("num_key_value_heads"),
        rms_norm_eps: f64("rms_norm_eps"),
        vocab_size: u64("vocab_size"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_mapping_dense_llama_family() {
        let f = |n| hf_to_gguf_name(n, "llama");
        assert_eq!(
            f("model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(f("lm_head.weight").as_deref(), Some("output.weight"));
        assert_eq!(
            f("model.norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        assert_eq!(
            f("model.layers.3.self_attn.q_proj.weight").as_deref(),
            Some("blk.3.attn_q.weight")
        );
        assert_eq!(
            f("model.layers.0.mlp.down_proj.weight").as_deref(),
            Some("blk.0.ffn_down.weight")
        );
        assert_eq!(
            f("model.layers.11.input_layernorm.weight").as_deref(),
            Some("blk.11.attn_norm.weight")
        );
        assert_eq!(
            f("model.layers.2.self_attn.k_proj.bias").as_deref(),
            Some("blk.2.attn_k.bias")
        );
        // Rotary cache is skipped.
        assert_eq!(f("model.layers.0.self_attn.rotary_emb.inv_freq"), None);
        // Unknown passes through as None (caller keeps original name).
        assert_eq!(
            f("model.layers.0.block_sparse_moe.experts.0.w1.weight"),
            None
        );
    }

    #[test]
    fn name_mapping_nested_language_model_prefix() {
        // VibeVoice / LFM2-VL wrap the dense LM one level down. The wrapper
        // is stripped and the inner dense name maps exactly like the bare
        // form — same results, one generic rule.
        let f = |n| hf_to_gguf_name(n, "llama");
        assert_eq!(
            f("model.language_model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            f("model.language_model.norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        assert_eq!(
            f("model.language_model.layers.3.self_attn.q_proj.weight").as_deref(),
            Some("blk.3.attn_q.weight")
        );
        assert_eq!(
            f("model.language_model.layers.0.mlp.down_proj.weight").as_deref(),
            Some("blk.0.ffn_down.weight")
        );
        assert_eq!(
            f("model.language_model.layers.11.input_layernorm.weight").as_deref(),
            Some("blk.11.attn_norm.weight")
        );
        // lm_head sits ABOVE the wrapper in real checkpoints and must keep
        // mapping (it is untouched by the strip).
        assert_eq!(f("lm_head.weight").as_deref(), Some("output.weight"));
        // Rotary inside the wrapper is still skipped.
        assert_eq!(
            f("model.language_model.layers.0.self_attn.rotary_emb.inv_freq"),
            None
        );
        // The generic `model.model.` wrapper (other multimodal families).
        assert_eq!(
            f("model.model.layers.1.self_attn.o_proj.weight").as_deref(),
            Some("blk.1.attn_output.weight")
        );
    }

    #[test]
    fn name_mapping_lfm2_cores() {
        // LFM2 layer cores map per the pinned llama.cpp reference
        // (tensor_mapping.py SHORTCONV_* :1464-1468, w1/w2/w3 llama-pth
        // arms, operator_norm :214; arch `lfm2`, llama-arch.cpp:126).
        // The wrapper strip feeds the same table, and the mapping matches
        // what real unsloth LFM2 GGUFs use.
        let f = |n| hf_to_gguf_name(n, "lfm2");
        assert_eq!(
            f("model.language_model.layers.0.conv.conv.weight").as_deref(),
            Some("blk.0.shortconv.conv.weight")
        );
        assert_eq!(
            f("model.language_model.layers.0.conv.in_proj.weight").as_deref(),
            Some("blk.0.shortconv.in_proj.weight")
        );
        assert_eq!(
            f("model.language_model.layers.0.conv.out_proj.weight").as_deref(),
            Some("blk.0.shortconv.out_proj.weight")
        );
        assert_eq!(
            f("model.language_model.layers.0.feed_forward.w1.weight").as_deref(),
            Some("blk.0.ffn_gate.weight")
        );
        assert_eq!(
            f("model.language_model.layers.0.feed_forward.w2.weight").as_deref(),
            Some("blk.0.ffn_down.weight")
        );
        assert_eq!(
            f("model.language_model.layers.0.feed_forward.w3.weight").as_deref(),
            Some("blk.0.ffn_up.weight")
        );
        assert_eq!(
            f("model.language_model.layers.0.operator_norm.weight").as_deref(),
            Some("blk.0.attn_norm.weight")
        );
        assert_eq!(
            f("model.language_model.layers.0.ffn_norm.weight").as_deref(),
            Some("blk.0.ffn_norm.weight")
        );
        // LFM2's pre-embedding norm → token_embd_norm (tensor_mapping.py:65).
        assert_eq!(
            f("model.language_model.embedding_norm.weight").as_deref(),
            Some("token_embd_norm.weight")
        );
        // Bare (post-strip) forms map identically.
        assert_eq!(
            f("embedding_norm.weight").as_deref(),
            Some("token_embd_norm.weight")
        );
        assert_eq!(
            f("layers.0.operator_norm.weight").as_deref(),
            Some("blk.0.attn_norm.weight")
        );
        // LFM2.5 full-attention layer arms (tensor_mapping.py:330, :698,
        // :715 — layers 2/5/9/13/17/21/24/27 of LFM2.5-VL-3B).
        assert_eq!(
            f("model.language_model.layers.2.self_attn.out_proj.weight").as_deref(),
            Some("blk.2.attn_output.weight")
        );
        assert_eq!(
            f("model.language_model.layers.2.self_attn.q_layernorm.weight").as_deref(),
            Some("blk.2.attn_q_norm.weight")
        );
        assert_eq!(
            f("model.language_model.layers.2.self_attn.k_layernorm.weight").as_deref(),
            Some("blk.2.attn_k_norm.weight")
        );
        assert_eq!(
            f("model.language_model.layers.2.self_attn.q_proj.weight").as_deref(),
            Some("blk.2.attn_q.weight")
        );
    }

    #[test]
    fn name_mapping_gemma_norms() {
        // Ground-truthed against the unsloth gemma3 reference: gemma3 has
        // FOUR per-layer norms. gemma2 has three (no post_ffw); gemma v1
        // keeps the llama-style fold (constants.py:3123-3135 GEMMA list
        // carries FFN_NORM, not ATTN_POST_NORM).
        let g3 = |n| hf_to_gguf_name(n, "gemma3");
        assert_eq!(
            g3("model.layers.0.input_layernorm.weight").as_deref(),
            Some("blk.0.attn_norm.weight")
        );
        assert_eq!(
            g3("model.layers.0.post_attention_layernorm.weight").as_deref(),
            Some("blk.0.post_attention_norm.weight")
        );
        assert_eq!(
            g3("model.layers.0.pre_feedforward_layernorm.weight").as_deref(),
            Some("blk.0.ffn_norm.weight")
        );
        assert_eq!(
            g3("model.layers.0.post_feedforward_layernorm.weight").as_deref(),
            Some("blk.0.post_ffw_norm.weight")
        );
        // q/k norms keep the generic arms.
        assert_eq!(
            g3("model.layers.0.self_attn.q_norm.weight").as_deref(),
            Some("blk.0.attn_q_norm.weight")
        );
        assert_eq!(
            g3("model.layers.0.self_attn.k_norm.weight").as_deref(),
            Some("blk.0.attn_k_norm.weight")
        );
        assert_eq!(
            g3("model.norm.weight").as_deref(),
            Some("output_norm.weight")
        );

        // gemma2: post-attention norm splits the same way; no post_ffw arm
        // produces that name in a real gemma2 checkpoint.
        let g2 = |n| hf_to_gguf_name(n, "gemma2");
        assert_eq!(
            g2("model.layers.0.post_attention_layernorm.weight").as_deref(),
            Some("blk.0.post_attention_norm.weight")
        );
        assert_eq!(
            g2("model.layers.0.pre_feedforward_layernorm.weight").as_deref(),
            Some("blk.0.ffn_norm.weight")
        );

        // gemma v1 + llama-family: the fold must stay `ffn_norm` (the
        // pre-split behavior, correct for GEMMA/LLAMA per constants.py).
        let g1 = |n| hf_to_gguf_name(n, "gemma");
        assert_eq!(
            g1("model.layers.0.post_attention_layernorm.weight").as_deref(),
            Some("blk.0.ffn_norm.weight")
        );
        let ll = |n| hf_to_gguf_name(n, "llama");
        assert_eq!(
            ll("model.layers.0.post_attention_layernorm.weight").as_deref(),
            Some("blk.0.ffn_norm.weight")
        );
    }

    #[test]
    fn gemma_norm_plus_one_scope() {
        // Applies to every gemma `*norm.weight` (all seven kinds present in
        // the unsloth gemma3 reference: attn / post_attention / ffn /
        // post_ffw / output / q / k), whatever the wrapper prefix.
        for n in [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.pre_feedforward_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.norm.weight",
            "model.language_model.layers.0.input_layernorm.weight",
        ] {
            assert!(gemma_norm_plus_one("gemma", n), "{n} must get +1");
            assert!(gemma_norm_plus_one("gemma3", n), "{n} must get +1");
        }
        // Never for non-gemma archs, non-norm names, or norm BIASES.
        assert!(!gemma_norm_plus_one("llama", "model.norm.weight"));
        assert!(!gemma_norm_plus_one(
            "lfm2",
            "model.layers.0.operator_norm.weight"
        ));
        assert!(!gemma_norm_plus_one("gemma3", "model.embed_tokens.weight"));
        assert!(!gemma_norm_plus_one("gemma3", "lm_head.weight"));
        assert!(!gemma_norm_plus_one(
            "gemma3",
            "model.layers.0.input_layernorm.bias"
        ));
    }

    #[test]
    fn name_mapping_arch_specific_cores_pass_through() {
        // Vision tower / projector / conv heads are NOT in any mapping
        // table — they must keep their ORIGINAL full name (including the
        // wrapper), never be guessed.
        let f = |n| hf_to_gguf_name(n, "llama");
        assert_eq!(
            f("model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"),
            None
        );
        assert_eq!(f("model.multi_modal_projector.linear_1.weight"), None);
        // VibeVoice conv/tokenizer heads: pass-through (regression guard —
        // this was the pre-task behavior and must stay).
        assert_eq!(
            f("model.acoustic_tokenizer.encoder.block.0.conv.weight"),
            None
        );
        assert_eq!(f("model.semantic_connector.fc1.weight"), None);
        // A dense `model.layers.*` name must NOT be eaten by the prefix
        // strip: `model.layers.5.…` does not start with either wrapper.
        assert_eq!(
            f("model.layers.5.self_attn.v_proj.weight").as_deref(),
            Some("blk.5.attn_v.weight")
        );
    }

    #[test]
    fn arch_classname_mapping() {
        assert_eq!(arch_from_classname("LlamaForCausalLM"), "llama");
        assert_eq!(arch_from_classname("Qwen2ForCausalLM"), "qwen2");
        assert_eq!(arch_from_classname("GemmaForCausalLM"), "gemma");
        assert_eq!(arch_from_classname("Gemma2ForCausalLM"), "gemma2");
        assert_eq!(arch_from_classname("Gemma3ForCausalLM"), "gemma3");
        assert_eq!(arch_from_classname("SomethingForCausalLM"), "something");
        // LFM2 family → `lfm2` (llama.cpp LLM_ARCH_LFM2, llama-arch.cpp:126).
        assert_eq!(arch_from_classname("LFM2ForCausalLM"), "lfm2");
        assert_eq!(arch_from_classname("Lfm2ForCausalLM"), "lfm2");
        assert_eq!(
            arch_from_classname("LFM2VLForConditionalGeneration"),
            "lfm2"
        );
        assert_eq!(
            arch_from_classname("Lfm2VLForConditionalGeneration"),
            "lfm2"
        );
        // lfm2moe stays on the generic lowercase-stem fallback for now.
        assert_eq!(arch_from_classname("Lfm2MoeForCausalLM"), "lfm2moe");
    }
}

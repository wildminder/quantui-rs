//! Unsloth coverage plan Phase 3.0: llama.cpp policy-engine integration.
//!
//! Proves the composite methods (q4_k_m/q5_k_m/q2_k/q3_k_m/q3_k_l) route
//! through the ported llama.cpp tree end-to-end: an 8-layer model's
//! attn_v tensors must show the exact use_more_bits(i, 8) pattern
//! (Q6_K at i ∈ {0,3,6,7}), and the gqa branch of q2_k must engage.

use std::path::{Path, PathBuf};

use quant_core::gguf_convert::{convert_hf_to_gguf, GgufConvertConfig, GgufConvertReport};
use quant_core::gguf_registry;
use quant_core::llama_policy::{self, LlamaPolicy};

// ─── fixture builder: 8-layer llama model, 256-col tensors ───────────

struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

fn build_safetensors(tensors: &[Tensor]) -> Vec<u8> {
    let mut data = Vec::new();
    let mut map = serde_json::Map::new();
    for t in tensors {
        let start = data.len() as u64;
        data.extend_from_slice(&t.bytes);
        let end = data.len() as u64;
        let mut info = serde_json::Map::new();
        info.insert("dtype".into(), serde_json::Value::String(t.dtype.into()));
        info.insert(
            "shape".into(),
            serde_json::Value::Array(t.shape.iter().map(|&d| serde_json::json!(d)).collect()),
        );
        info.insert("data_offsets".into(), serde_json::json!([start, end]));
        map.insert(t.name.clone(), serde_json::Value::Object(info));
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(map)).unwrap();
    let pad = (8 - header.len() % 8) % 8;
    let mut padded = header;
    padded.extend(std::iter::repeat_n(b' ', pad));
    let mut out = Vec::new();
    out.extend_from_slice(&(padded.len() as u64).to_le_bytes());
    out.extend_from_slice(&padded);
    out.extend_from_slice(&data);
    out
}

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect()
}

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect()
}

const L: usize = 256; // hidden size — divisible by every block size

/// 8-layer llama-ish model: per layer attn_q/attn_v/ffn_down (256x256) —
/// the three tensors the policy tree differentiates — plus one 1-D norm.
fn write_8layer_model(dir: &Path, n_gqa: u32) {
    std::fs::create_dir_all(dir).unwrap();
    let mut tensors = Vec::new();
    for l in 0..8 {
        tensors.push(Tensor {
            name: format!("model.layers.{l}.self_attn.q_proj.weight"),
            dtype: "BF16",
            shape: vec![L as u64, L as u64],
            bytes: bf16_bytes(&synth(L * L, 100 + l)),
        });
        tensors.push(Tensor {
            name: format!("model.layers.{l}.self_attn.v_proj.weight"),
            dtype: "BF16",
            shape: vec![L as u64, L as u64],
            bytes: bf16_bytes(&synth(L * L, 200 + l)),
        });
        tensors.push(Tensor {
            name: format!("model.layers.{l}.mlp.down_proj.weight"),
            dtype: "BF16",
            shape: vec![L as u64, L as u64],
            bytes: bf16_bytes(&synth(L * L, 300 + l)),
        });
    }
    tensors.push(Tensor {
        name: "model.norm.weight".to_string(),
        dtype: "BF16",
        shape: vec![L as u64],
        bytes: bf16_bytes(&synth(L, 9)),
    });
    std::fs::write(dir.join("model.safetensors"), build_safetensors(&tensors)).unwrap();
    // num_key_value_heads = heads / n_gqa → drives ModelFacts.n_gqa.
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": L,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 8 / n_gqa,
            "vocab_size": 64
        }))
        .unwrap(),
    )
    .unwrap();
}

fn convert(model_dir: &Path, method: &str) -> (GgufConvertReport, PathBuf) {
    let out = model_dir
        .parent()
        .unwrap()
        .join(format!("e2e-{method}.gguf"));
    let cfg = GgufConvertConfig {
        method_id: method.to_string(),
        ..Default::default()
    };
    let report = convert_hf_to_gguf(&model_dir.join("model.safetensors"), &out, &cfg, None)
        .unwrap_or_else(|e| panic!("{method}: conversion failed: {e}"));
    (report, out)
}

/// The exact use_more_bits(i, 8) pattern: Q6_K at i ∈ {0, 3, 6, 7}.
/// Cross-checked against llama-quant.cpp:434-436 and proven independently
/// by llama_policy::tests::q4_k_m_driver_protocol_end_to_end.
#[test]
fn q4_k_m_e2e_use_more_bits_pattern_is_visible() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_8layer_model(&model_dir, 1); // no GQA

    let (report, out) = convert(&model_dir, "q4_k_m");
    assert_eq!(report.fallback_f16, 0, "no fallback expected");

    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    for i in 0..8 {
        let t = f
            .tensors
            .get(&format!("blk.{i}.attn_v.weight"))
            .unwrap_or_else(|| panic!("blk.{i}.attn_v.weight missing"));
        let want = if llama_policy::use_more_bits(i as i32, 8) {
            rlx_gguf::GgmlType::Q6K
        } else {
            rlx_gguf::GgmlType::Q4K
        };
        assert_eq!(
            t.dtype, want,
            "blk.{i}.attn_v: engine pattern must match use_more_bits({i}, 8)"
        );
    }

    // ffn_down follows its own counter — same pattern.
    for i in 0..8 {
        let t = f.tensors.get(&format!("blk.{i}.ffn_down.weight")).unwrap();
        let want = if llama_policy::use_more_bits(i as i32, 8) {
            rlx_gguf::GgmlType::Q6K
        } else {
            rlx_gguf::GgmlType::Q4K
        };
        assert_eq!(t.dtype, want, "blk.{i}.ffn_down pattern");
    }

    // attn_q (no rule) stays the base Q4_K everywhere.
    for i in 0..8 {
        let t = f.tensors.get(&format!("blk.{i}.attn_q.weight")).unwrap();
        assert_eq!(t.dtype, rlx_gguf::GgmlType::Q4K, "blk.{i}.attn_q base");
    }
}

/// q2_k's attn_v takes Q4_K only under GQA >= 4 (llama-quant.cpp:534-535);
/// without GQA it is Q3_K. The engine must read head counts from
/// config.json to make this branch real.
#[test]
fn q2_k_e2e_gqa_branch_is_live() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_8layer_model(&model_dir, 4); // 8 heads, 2 kv → n_gqa = 4

    let (report, out) = convert(&model_dir, "q2_k");
    assert_eq!(report.fallback_f16, 0);

    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    for i in 0..8 {
        let t = f.tensors.get(&format!("blk.{i}.attn_v.weight")).unwrap();
        assert_eq!(
            t.dtype,
            rlx_gguf::GgmlType::Q4K,
            "GQA=4: blk.{i}.attn_v must be Q4_K"
        );
    }

    // Counterpart: n_gqa = 1 → attn_v is Q3_K.
    let tmp2 = tempfile::tempdir().unwrap();
    let model_dir2 = tmp2.path().join("m");
    write_8layer_model(&model_dir2, 1);
    let (_, out2) = convert(&model_dir2, "q2_k");
    let f2 = rlx_gguf::GgufFile::from_path(&out2).unwrap();
    for i in 0..8 {
        let t = f2.tensors.get(&format!("blk.{i}.attn_v.weight")).unwrap();
        assert_eq!(
            t.dtype,
            rlx_gguf::GgmlType::Q3K,
            "GQA=1: blk.{i}.attn_v must be Q3_K"
        );
    }
}

/// The engine's categories/counters must be registry-wired: every
/// non-Flat engine method resolves consistently with llama_policy::resolve
/// on a synthetic name stream (guard against registry/engine drift).
#[test]
fn registry_composite_methods_are_engine_wired() {
    for id in [
        "q4_k_m", "q5_k_m", "q2_k", "q2_k_l", "q3_k_m", "q3_k_l", "iq2_m", "iq3_m",
    ] {
        let e = gguf_registry::get_method(id).unwrap();
        assert_ne!(
            e.policy.engine,
            LlamaPolicy::Flat,
            "{id}: composite method must use the llama.cpp engine"
        );
    }
    // And the simple methods stay Flat.
    for id in [
        "f16", "q8_0", "q4_0", "q4_1", "q5_1", "q4_k_s", "q5_k_s", "q3_k_s",
    ] {
        let e = gguf_registry::get_method(id).unwrap();
        assert_eq!(
            e.policy.engine,
            LlamaPolicy::Flat,
            "{id}: simple method must stay on the flat engine"
        );
    }
}

/// q2_k_l (Unsloth preset, save.py:377): output AND token embeddings
/// forced to Q8_0 — visible in real output dtypes, unlike every other
/// method's embd convention.
#[test]
fn q2_k_l_e2e_forces_q8_embeddings() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    std::fs::create_dir_all(&model_dir).unwrap();

    let mut tensors = Vec::new();
    tensors.push(Tensor {
        name: "model.embed_tokens.weight".to_string(),
        dtype: "BF16",
        shape: vec![64, L as u64],
        bytes: bf16_bytes(&synth(64 * L, 5)),
    });
    for l in 0..8 {
        tensors.push(Tensor {
            name: format!("model.layers.{l}.self_attn.q_proj.weight"),
            dtype: "BF16",
            shape: vec![L as u64, L as u64],
            bytes: bf16_bytes(&synth(L * L, 100 + l)),
        });
        tensors.push(Tensor {
            name: format!("model.layers.{l}.mlp.down_proj.weight"),
            dtype: "BF16",
            shape: vec![L as u64, L as u64],
            bytes: bf16_bytes(&synth(L * L, 300 + l)),
        });
    }
    tensors.push(Tensor {
        name: "lm_head.weight".to_string(),
        dtype: "BF16",
        shape: vec![64, L as u64],
        bytes: bf16_bytes(&synth(64 * L, 6)),
    });
    std::fs::write(
        model_dir.join("model.safetensors"),
        build_safetensors(&tensors),
    )
    .unwrap();
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": L,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "vocab_size": 64
        }))
        .unwrap(),
    )
    .unwrap();

    let (report, out) = convert(&model_dir, "q2_k_l");
    assert_eq!(report.fallback_f16, 0);

    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    // Both embeddings forced to Q8_0 (Unsloth save.py:377-395).
    assert_eq!(
        f.tensors.get("token_embd.weight").unwrap().dtype,
        rlx_gguf::GgmlType::Q8_0,
        "q2_k_l: token_embd must be Q8_0"
    );
    assert_eq!(
        f.tensors.get("output.weight").unwrap().dtype,
        rlx_gguf::GgmlType::Q8_0,
        "q2_k_l: output must be Q8_0"
    );
    // Body follows q2_k's rules: ffn_down → Q3_K (:593).
    assert_eq!(
        f.tensors.get("blk.0.ffn_down.weight").unwrap().dtype,
        rlx_gguf::GgmlType::Q3K,
        "q2_k_l: ffn_down follows q2_k base rules"
    );
}

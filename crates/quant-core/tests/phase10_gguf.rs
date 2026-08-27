//! Phase 10.2 E2E: HF safetensors → GGUF conversion.
//!
//! Builds a tiny synthetic HF-style model (single file + config.json), runs
//! [`quant_core::gguf_convert::convert_hf_to_gguf`], then reads the output
//! back with rlx-gguf's own spec-compliant parser to verify:
//! - HF → GGUF tensor-name mapping,
//! - GGUF reversed-dim shape storage,
//! - arch + `{arch}.*` metadata from config.json,
//! - per-tensor scheme selection (1-D → F32, embd → F16, rules, default),
//! - dequant round-trip stays close to the source floats.
//!
//! Parity note (plan 10.1): no llama.cpp binary is available in this env, so
//! "loadable GGUF" is proxied by round-tripping through rlx-gguf's parser.

use std::path::Path;

use quant_core::gguf_convert::{convert_hf_to_gguf, GgufConvertConfig};

// ─── tiny safetensors fixture builder (8-aligned header, our reader format) ─

struct Tensor {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

/// Serialize tensors into the reference safetensors layout our reader accepts:
/// `u64 LE header_len (8-aligned) | JSON header (space-padded) | data`.
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
        map.insert(t.name.into(), serde_json::Value::Object(info));
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(map)).unwrap();
    // Space-pad to an 8-byte boundary (reference `_align_header_to_8`).
    let pad = (8 - header.len() % 8) % 8;
    let mut padded = header;
    padded.extend(std::iter::repeat_n(b' ', pad));
    let len = (padded.len() as u64).to_le_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&len);
    out.extend_from_slice(&padded);
    out.extend_from_slice(&data);
    out
}

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect()
}

/// Deterministic pseudo-random floats in [-1, 1].
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

/// Write a tiny HF llama-style model dir: one safetensors + config.json.
fn write_tiny_model(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let h = 32usize; // hidden
    let v = 64usize; // vocab / intermediate

    let mk2d = |name: &'static str, rows: usize, cols: usize, seed: u64| Tensor {
        name,
        dtype: "BF16",
        shape: vec![rows as u64, cols as u64],
        bytes: bf16_bytes(&synth(rows * cols, seed)),
    };
    let mk1d = |name: &'static str, n: usize, seed: u64| Tensor {
        name,
        dtype: "BF16",
        shape: vec![n as u64],
        bytes: bf16_bytes(&synth(n, seed)),
    };

    let tensors = vec![
        mk2d("model.embed_tokens.weight", v, h, 1),
        mk2d("model.layers.0.self_attn.q_proj.weight", h, h, 2),
        mk2d("model.layers.0.self_attn.k_proj.weight", h, h, 3),
        mk2d("model.layers.0.self_attn.v_proj.weight", h, h, 4),
        mk2d("model.layers.0.self_attn.o_proj.weight", h, h, 5),
        mk2d("model.layers.0.mlp.gate_proj.weight", v, h, 6),
        mk2d("model.layers.0.mlp.up_proj.weight", v, h, 7),
        mk2d("model.layers.0.mlp.down_proj.weight", h, v, 8),
        mk1d("model.layers.0.input_layernorm.weight", h, 9),
        mk1d("model.layers.0.post_attention_layernorm.weight", h, 10),
        mk1d("model.norm.weight", h, 11),
        mk2d("lm_head.weight", v, h, 12),
    ];
    std::fs::write(dir.join("model.safetensors"), build_safetensors(&tensors)).unwrap();

    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": h,
        "intermediate_size": v,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 2048,
        "vocab_size": v,
        "rms_norm_eps": 1e-5
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
}

// ─── tests ──────────────────────────────────────────────────────────

#[test]
fn convert_q4_k_m_single_file() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("tinymodel");
    write_tiny_model(&model_dir);
    let out = tmp.path().join("tinymodel-q4_k_m.gguf");

    let cfg = GgufConvertConfig {
        method_id: "q4_k_m".into(),
        arch: None,
        name: None,
    };
    let report = convert_hf_to_gguf(&model_dir.join("model.safetensors"), &out, &cfg, None)
        .expect("conversion succeeds");

    assert_eq!(report.tensors, 12);
    assert_eq!(report.arch, "llama");
    assert!(report.output_bytes > 0);
    // 1-D norms (3) stay F32; embed + lm_head (2) → F16; rest quantized.
    assert_eq!(report.kept_f32, 3);
    assert!(
        report.quantized >= 7,
        "expected >=7 quantized, got {}",
        report.quantized
    );

    // Read back with rlx-gguf's parser.
    let f = rlx_gguf::GgufFile::from_path(&out).expect("parse output");
    // Name mapping applied.
    let names: Vec<&str> = f.tensors.keys().map(|s| s.as_str()).collect();
    assert!(
        names.contains(&"token_embd.weight"),
        "embed mapped: {names:?}"
    );
    assert!(names.contains(&"output.weight"), "lm_head mapped");
    assert!(names.contains(&"output_norm.weight"), "model.norm mapped");
    assert!(names.contains(&"blk.0.attn_q.weight"), "q_proj mapped");
    assert!(names.contains(&"blk.0.ffn_down.weight"), "down_proj mapped");
    assert!(
        names.contains(&"blk.0.attn_norm.weight"),
        "input_layernorm mapped"
    );
    // No HF-style names leak through for the mapped set.
    assert!(!names.contains(&"model.embed_tokens.weight"));
    assert!(!names.contains(&"lm_head.weight"));

    // Arch metadata present.
    let arch = f
        .metadata
        .get("general.architecture")
        .and_then(rlx_gguf::MetaValue::as_str);
    assert_eq!(arch, Some("llama"));
    assert!(f.metadata.contains_key("llama.block_count"));
    assert!(f.metadata.contains_key("llama.embedding_length"));

    // Shapes are GGUF-reversed: HF [v,h]=[64,32] → GGUF [32,64].
    let embd = f.tensors.get("token_embd.weight").unwrap();
    assert_eq!(embd.shape, vec![32, 64], "dims reversed");

    // Dequant round-trip: q_proj (Q4K) stays close to source.
    let (deq, shape) = f.dequant_f32("blk.0.attn_q.weight").unwrap();
    assert_eq!(shape, vec![32, 32]);
    let src = synth(32 * 32, 2);
    let cos = cosine(&src, &deq);
    assert!(cos > 0.98, "Q4K round-trip cosine too low: {cos}");

    // F32 norm round-trips exactly.
    let (norm, _) = f.dequant_f32("blk.0.attn_norm.weight").unwrap();
    let src_norm = synth(32, 9)
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_f32())
        .collect::<Vec<_>>();
    // 1-D kept as F32 from the bf16-decoded floats → exact vs decoded source.
    for (a, b) in norm.iter().zip(&src_norm) {
        assert!((a - b).abs() < 1e-6, "F32 norm not exact");
    }
}

#[test]
fn convert_f16_is_lossless_for_floats() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("tinymodel");
    write_tiny_model(&model_dir);
    let out = tmp.path().join("tinymodel-f16.gguf");

    let cfg = GgufConvertConfig {
        method_id: "f16".into(),
        arch: None,
        name: None,
    };
    let report =
        convert_hf_to_gguf(&model_dir.join("model.safetensors"), &out, &cfg, None).unwrap();
    assert_eq!(report.method_id, "f16");

    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    // q_proj stored as F16 → dequant ≈ bf16 source (both are 16-bit-ish).
    let (deq, _) = f.dequant_f32("blk.0.attn_q.weight").unwrap();
    let src = synth(32 * 32, 2);
    let cos = cosine(&src, &deq);
    assert!(cos > 0.999, "F16 round-trip cosine too low: {cos}");
}

#[test]
fn rejects_dynamic_and_unknown_methods() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("tinymodel");
    write_tiny_model(&model_dir);
    let input = model_dir.join("model.safetensors");
    let out = tmp.path().join("x.gguf");

    let dyn_cfg = GgufConvertConfig {
        method_id: "q4_k_xl".into(),
        arch: None,
        name: None,
    };
    let err = convert_hf_to_gguf(&input, &out, &dyn_cfg, None).unwrap_err();
    assert!(matches!(
        err,
        quant_core::gguf_convert::GgufError::DynamicMethod(_)
    ));

    let bad_cfg = GgufConvertConfig {
        method_id: "nope".into(),
        arch: None,
        name: None,
    };
    let err = convert_hf_to_gguf(&input, &out, &bad_cfg, None).unwrap_err();
    assert!(matches!(
        err,
        quant_core::gguf_convert::GgufError::UnknownMethod(_, _)
    ));
}

#[test]
fn progress_callback_fires_per_tensor() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("tinymodel");
    write_tiny_model(&model_dir);
    let out = tmp.path().join("p.gguf");

    let cfg = GgufConvertConfig {
        method_id: "q8_0".into(),
        arch: None,
        name: None,
    };
    let mut calls: Vec<(usize, usize)> = Vec::new();
    let mut cb = |done: usize, total: usize| calls.push((done, total));
    convert_hf_to_gguf(
        &model_dir.join("model.safetensors"),
        &out,
        &cfg,
        Some(&mut cb),
    )
    .unwrap();
    assert_eq!(calls.len(), 12);
    assert_eq!(calls.last(), Some(&(12, 12)));
    // Monotonic done counter.
    for (i, (d, t)) in calls.iter().enumerate() {
        assert_eq!(*d, i + 1);
        assert_eq!(*t, 12);
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

// ─── sharded-folder input ───────────────────────────────────────────

/// Write a 2-shard HF model dir (index.json + two safetensors) and convert it.
#[test]
fn convert_sharded_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("shardedmodel");
    std::fs::create_dir_all(&dir).unwrap();
    let h = 32usize;

    let mk2d = |name: &'static str, rows: usize, cols: usize, seed: u64| Tensor {
        name,
        dtype: "BF16",
        shape: vec![rows as u64, cols as u64],
        bytes: bf16_bytes(&synth(rows * cols, seed)),
    };
    let mk1d = |name: &'static str, n: usize, seed: u64| Tensor {
        name,
        dtype: "BF16",
        shape: vec![n as u64],
        bytes: bf16_bytes(&synth(n, seed)),
    };

    // Shard 1: embeddings + layer 0 attn.
    let shard1 = vec![
        mk2d("model.embed_tokens.weight", 64, h, 1),
        mk2d("model.layers.0.self_attn.q_proj.weight", h, h, 2),
        mk2d("model.layers.0.self_attn.v_proj.weight", h, h, 4),
        mk1d("model.layers.0.input_layernorm.weight", h, 9),
    ];
    // Shard 2: mlp + norms + head.
    let shard2 = vec![
        mk2d("model.layers.0.mlp.down_proj.weight", h, 64, 8),
        mk1d("model.norm.weight", h, 11),
        mk2d("lm_head.weight", 64, h, 12),
    ];
    std::fs::write(
        dir.join("model-00001-of-00002.safetensors"),
        build_safetensors(&shard1),
    )
    .unwrap();
    std::fs::write(
        dir.join("model-00002-of-00002.safetensors"),
        build_safetensors(&shard2),
    )
    .unwrap();

    let index = serde_json::json!({
        "metadata": {"total_size": 0},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors"
        }
    });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index).unwrap(),
    )
    .unwrap();
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": h,
        "num_hidden_layers": 1,
        "vocab_size": 64
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let out = tmp.path().join("shardedmodel-q8_0.gguf");
    let cfg = GgufConvertConfig {
        method_id: "q8_0".into(),
        arch: None,
        name: None,
    };
    let report = convert_hf_to_gguf(&dir, &out, &cfg, None).expect("sharded conversion succeeds");

    assert_eq!(report.tensors, 7, "all tensors from both shards");
    assert_eq!(report.arch, "llama");

    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    let names: Vec<&str> = f.tensors.keys().map(|s| s.as_str()).collect();
    // Tensors from BOTH shards present + mapped.
    assert!(
        names.contains(&"token_embd.weight"),
        "shard1 embed: {names:?}"
    );
    assert!(names.contains(&"blk.0.attn_q.weight"), "shard1 q_proj");
    assert!(names.contains(&"blk.0.ffn_down.weight"), "shard2 down_proj");
    assert!(names.contains(&"output.weight"), "shard2 lm_head");
    assert!(names.contains(&"output_norm.weight"), "shard2 model.norm");

    // Round-trip a shard-2 tensor.
    let (deq, _) = f.dequant_f32("blk.0.ffn_down.weight").unwrap();
    let src = synth(32 * 64, 8);
    assert!(cosine(&src, &deq) > 0.999, "Q8_0 round-trip");
}

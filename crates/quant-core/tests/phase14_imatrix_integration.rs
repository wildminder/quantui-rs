//! Phase 4.4 integration: imatrix consumption through the full
//! HF -> GGUF conversion.
//!
//! Proves the driver wiring end-to-end: given the same f32 source and the
//! same per-tensor weights that produced tests/golden/llamacpp/, a
//! conversion with `imatrix: Some(...)` and a flat-policy method whose
//! body scheme is one of the weighted K-quants (q4_k_s → Q4K, and the
//! slice-2 additions q3_k_s → Q3K, q5_k_s → Q5K, q6_k → Q6K) must emit
//! tensor bytes byte-identical to llama-quantize's --imatrix output.
//! Also proves the negative: WITHOUT the imatrix the bytes differ
//! (rlx-gguf's unweighted path).

use std::path::PathBuf;

use quant_core::gguf_convert::{convert_hf_to_gguf, GgufConvertConfig};
use quant_core::imatrix::Imatrix;

fn golden_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.join("tests/golden/llamacpp")
}

fn load_f32(name: &str, expect: usize) -> Vec<f32> {
    let path = golden_dir().join(name);
    let raw = std::fs::read(&path).unwrap();
    assert_eq!(raw.len(), expect * 4);
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

// ─── tiny safetensors fixture (one 2x256 F32 tensor) ─────────────────

fn build_safetensors(name: &str, shape: Vec<u64>, floats: &[f32]) -> Vec<u8> {
    let data: Vec<u8> = floats.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut map = serde_json::Map::new();
    let mut info = serde_json::Map::new();
    info.insert("dtype".into(), "F32".into());
    info.insert(
        "shape".into(),
        serde_json::Value::Array(shape.iter().map(|&d| serde_json::json!(d)).collect()),
    );
    info.insert(
        "data_offsets".into(),
        serde_json::json!([0u64, data.len() as u64]),
    );
    map.insert(name.into(), serde_json::Value::Object(info));
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

fn write_model(dir: &std::path::Path, src: &[f32]) {
    std::fs::create_dir_all(dir).unwrap();
    let tensors = build_safetensors("model.layers.0.self_attn.q_proj.weight", vec![2, 256], src);
    std::fs::write(dir.join("model.safetensors"), tensors).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "vocab_size": 64
        }))
        .unwrap(),
    )
    .unwrap();
}

/// Build an Imatrix whose single entry for `blk.0.attn_q.weight` carries
/// the golden weights (derivation: sums/count with count=1 would give
/// exact passthrough, but count=8 exercises the real division path — use
/// the golden weights directly via a 1-count entry of sums=weights).
fn build_imatrix(weights: &[f32]) -> Imatrix {
    // Write a legacy-format imatrix: one entry, ncall=1 -> sums/1 = weights.
    let mut out = Vec::new();
    out.extend_from_slice(&1i32.to_le_bytes()); // n_entries
    let name = b"blk.0.attn_q.weight";
    out.extend_from_slice(&(name.len() as i32).to_le_bytes());
    out.extend_from_slice(name);
    out.extend_from_slice(&1i32.to_le_bytes()); // ncall = 1
    out.extend_from_slice(&(weights.len() as i32).to_le_bytes());
    for v in weights {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("imatrix.dat");
    std::fs::write(&path, out).unwrap();
    Imatrix::load(&path).unwrap()
}

/// One flat-policy method whose body scheme hits a weighted encoder,
/// with the matching golden payload and expected ggml dtype.
fn run_weighted_conversion_case(
    method_id: &str,
    golden_name: &str,
    expect_dtype: rlx_gguf::GgmlType,
) {
    let src = load_f32("src.f32.bin", 512);
    let weights = load_f32("weights.f32.bin", 256);
    let raw = std::fs::read(golden_dir().join(golden_name)).unwrap();
    // Goldens may carry the writer's 32-byte alignment padding after the
    // true payload; the in-memory rlx read is exactly N_ROWS blocks. The
    // slice below requires the golden to cover the full payload.

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_model(&model_dir, &src);

    let out = tmp.path().join("m-weighted.gguf");
    let cfg = GgufConvertConfig {
        method_id: method_id.into(),
        arch: None,
        name: None,
        imatrix: Some(build_imatrix(&weights)),
    };
    let report = convert_hf_to_gguf(&model_dir.join("model.safetensors"), &out, &cfg, None)
        .expect("conversion succeeds");
    assert_eq!(report.fallback_f16, 0);

    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    let t = f
        .tensors
        .get("blk.0.attn_q.weight")
        .expect("tensor present");
    assert_eq!(t.dtype, expect_dtype);
    let got = f.tensor_bytes(t).unwrap();
    // Exactly the quantized payload — no alignment padding in rlx's
    // in-memory read. The golden may be LONGER (padding), so slice it.
    let golden = &raw[..got.len()];
    assert_eq!(
        got, golden,
        "{method_id}: conversion with imatrix must reproduce llama-quantize bytes"
    );
}

#[test]
fn conversion_with_imatrix_matches_llama_quantize_bytes() {
    run_weighted_conversion_case("q4_k_s", "weighted.q4_k.bin", rlx_gguf::GgmlType::Q4K);
}

#[test]
fn conversion_with_imatrix_matches_llama_quantize_bytes_q3_k() {
    run_weighted_conversion_case("q3_k_s", "weighted.q3_k.bin", rlx_gguf::GgmlType::Q3K);
}

#[test]
fn conversion_with_imatrix_matches_llama_quantize_bytes_q5_k() {
    run_weighted_conversion_case("q5_k_s", "weighted.q5_k.bin", rlx_gguf::GgmlType::Q5K);
}

#[test]
fn conversion_with_imatrix_matches_llama_quantize_bytes_q6_k() {
    // q6_k is itself the flat method (no _s variant).
    run_weighted_conversion_case("q6_k", "weighted.q6_k.bin", rlx_gguf::GgmlType::Q6K);
}

#[test]
fn conversion_without_imatrix_differs() {
    // The negative control: rlx-gguf's unweighted Q4K is a different
    // (documented simpler) search — so the bytes must NOT match the
    // weighted golden. If they ever match, the weighted path is broken.
    let src = load_f32("src.f32.bin", 512);
    let golden = std::fs::read(golden_dir().join("weighted.q4_k.bin")).unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_model(&model_dir, &src);

    let out = tmp.path().join("m-q4_k_s.gguf");
    let cfg = GgufConvertConfig {
        method_id: "q4_k_s".into(),
        arch: None,
        name: None,
        imatrix: None,
    };
    convert_hf_to_gguf(&model_dir.join("model.safetensors"), &out, &cfg, None).unwrap();

    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    let t = f.tensors.get("blk.0.attn_q.weight").unwrap();
    let got = f.tensor_bytes(t).unwrap();
    assert_ne!(
        got,
        &golden[..],
        "unweighted path must differ from the weighted golden"
    );
}

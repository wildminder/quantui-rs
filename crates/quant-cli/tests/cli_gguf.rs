//! Phase 10.2 CLI integration tests: `gguf` subcommand.
//!
//! Exit-code contract: 0 ok, 1 runtime failure, 2 usage error.
//! Verifies the CLI drives the core conversion engine and produces a GGUF
//! file that rlx-gguf's parser accepts.

use std::path::Path;
use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_quantui-rs"))
}

// ─── tiny HF model fixture (same layout as the core E2E test) ─────────

struct Tensor {
    name: &'static str,
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
        map.insert(t.name.into(), serde_json::Value::Object(info));
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

fn write_tiny_model(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let h = 32usize;
    let v = 64usize;
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
        mk2d("model.layers.0.mlp.down_proj.weight", h, v, 8),
        mk1d("model.norm.weight", h, 11),
        mk2d("lm_head.weight", v, h, 12),
    ];
    std::fs::write(dir.join("model.safetensors"), build_safetensors(&tensors)).unwrap();
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": h,
        "num_hidden_layers": 1,
        "vocab_size": v
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();
}

// ─── tests ──────────────────────────────────────────────────────────

#[test]
fn gguf_list_methods_exit_0() {
    let out = bin().args(["gguf", "--list-methods"]).output().unwrap();
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("q4_k_m"));
    assert!(s.contains("f16"));
    assert!(s.contains("[DYNAMIC 2.0]"));
}

#[test]
fn gguf_convert_single_file_exit_0() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);
    let out = tmp.path().join("m-q8_0.gguf");

    let status = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "q8_0",
            "--no-progress",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(out.exists());

    // Output is a valid GGUF readable by rlx-gguf.
    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    assert!(f.tensors.contains_key("token_embd.weight"));
    assert!(f.tensors.contains_key("blk.0.attn_q.weight"));
    assert!(f.tensors.contains_key("output.weight"));
}

#[test]
fn gguf_auto_naming() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("mymodel");
    write_tiny_model(&model_dir);

    // No OUTPUT arg → auto-name `<base>-<method>.gguf` next to the input.
    let status = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            "--method",
            "f16",
            "--no-progress",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    // base name = file stem "model" (not a shard marker).
    assert!(model_dir.join("model-f16.gguf").exists());
}

#[test]
fn gguf_unknown_method_exit_2() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);
    let out = tmp.path().join("x.gguf");
    let status = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "bogus",
            "--no-progress",
        ])
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(2));
}

#[test]
fn gguf_dynamic_method_exit_2() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);
    let out = tmp.path().join("x.gguf");
    let status = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "q4_k_xl",
            "--no-progress",
        ])
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(2));
}

#[test]
fn gguf_missing_input_exit_2() {
    let status = bin().args(["gguf", "--method", "q4_k_m"]).status().unwrap();
    assert_eq!(status.code(), Some(2));
}

#[test]
fn gguf_bad_input_path_exit_2() {
    let tmp = tempfile::tempdir().unwrap();
    let bogus = tmp.path().join("does_not_exist.safetensors");
    let out = tmp.path().join("x.gguf");
    let status = bin()
        .args([
            "gguf",
            bogus.to_str().unwrap(),
            out.to_str().unwrap(),
            "--no-progress",
        ])
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(2));
}

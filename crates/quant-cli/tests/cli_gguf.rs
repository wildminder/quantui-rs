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

// ─── Phase 0.3 / 2: capability markers + honest fallback ────────────

#[test]
fn gguf_list_methods_marks_imatrix_and_rejected() {
    let out = bin().args(["gguf", "--list-methods"]).output().unwrap();
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);

    // The four iq* methods are in official Unsloth IMATRIX_QUANTS — they
    // must be visibly marked so a user knows an imatrix is expected.
    // Phase 3 extends the set with the five new IMATRIX_QUANTS members.
    for id in [
        "iq4_nl", "iq3_xxs", "iq2_xxs", "iq2_xs", "iq1_s", "iq1_m", "iq2_s", "iq3_s", "iq4_xs",
    ] {
        let line = s
            .lines()
            .find(|l| l.starts_with(id))
            .unwrap_or_else(|| panic!("no listing line for {id}"));
        assert!(
            line.contains("[IMATRIX]"),
            "{id} line lacks [IMATRIX]: {line}"
        );
    }

    // Plain methods must NOT be marked. Phase 3's f32/bf16/ternary/q1_0/
    // q2_0 are plain list entries per decision Q4 (no gating flag).
    for id in [
        "f16", "f32", "bf16", "q8_0", "q4_k_m", "q4_1", "q5_1", "tq1_0", "tq2_0", "q1_0", "q2_0",
    ] {
        let line = s
            .lines()
            .find(|l| l.starts_with(id))
            .unwrap_or_else(|| panic!("no listing line for {id}"));
        assert!(
            !line.contains("[IMATRIX]"),
            "{id} wrongly marked [IMATRIX]: {line}"
        );
        assert!(
            !line.contains("unsupported natively"),
            "{id} wrongly listed as unsupported: {line}"
        );
    }

    // The wrongly-rejected-then-fixed pair is now advertised as runnable.
    for id in ["q4_1", "q5_1"] {
        let line = s
            .lines()
            .find(|l| l.starts_with(id))
            .unwrap_or_else(|| panic!("no listing line for {id}"));
        assert!(
            !line.contains("unsupported natively"),
            "{id} still listed as unsupported: {line}"
        );
    }
}

/// Phase 2.3: an indivisible tensor must produce exactly one per-tensor
/// warning naming the tensor, plus the summary warning — and still exit 0
/// (resume semantics; a hard fail would break the contract).
#[test]
fn gguf_f16_fallback_is_loud() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    std::fs::create_dir_all(&model_dir).unwrap();

    // 100 columns: not divisible by Q8_0's block size 32 → fallback.
    // 2-D weight (10 rows x 100 cols = 1000 elements) + a divisible norm
    // so the fixture is otherwise clean.
    let vals = synth(10 * 100, 42);
    let tensors = vec![
        Tensor {
            name: "model.layers.0.self_attn.q_proj.weight",
            dtype: "BF16",
            shape: vec![10, 100],
            bytes: bf16_bytes(&vals),
        },
        Tensor {
            name: "model.norm.weight",
            dtype: "BF16",
            shape: vec![32],
            bytes: bf16_bytes(&synth(32, 7)),
        },
    ];
    std::fs::write(
        model_dir.join("model.safetensors"),
        build_safetensors(&tensors),
    )
    .unwrap();
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 100,
            "num_hidden_layers": 1,
            "vocab_size": 8
        }))
        .unwrap(),
    )
    .unwrap();

    let out = tmp.path().join("m-q8_0.gguf");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "q8_0",
            "--no-progress",
        ])
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(0),
        "fallback must not fail the run"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Exactly one per-tensor warning line, naming the GGUF-side tensor.
    let per_tensor: Vec<&str> = stderr
        .lines()
        .filter(|l| l.contains("fell back to F16"))
        .collect();
    assert_eq!(
        per_tensor.len(),
        1,
        "expected 1 per-tensor warning, got: {stderr}"
    );
    assert!(
        per_tensor[0].contains("blk.0.attn_q.weight"),
        "warning must name the tensor: {}",
        per_tensor[0]
    );

    // And the summary line names the method and the tensor list.
    let summary: Vec<&str> = stderr
        .lines()
        .filter(|l| l.contains("were NOT quantized with 'q8_0'"))
        .collect();
    assert_eq!(
        summary.len(),
        1,
        "expected 1 summary warning, got: {stderr}"
    );
    assert!(summary[0].contains("blk.0.attn_q.weight"));
}

/// The counterpart: a clean fixture produces NO fallback warnings.
#[test]
fn gguf_clean_convert_has_no_fallback_warnings() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);
    let out = tmp.path().join("m-q8_0.gguf");

    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "q8_0",
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("fell back to F16"),
        "clean fixture must not warn, stderr: {stderr}"
    );
    assert!(
        !stderr.contains("were NOT quantized"),
        "clean fixture must not summarize a fallback, stderr: {stderr}"
    );
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

// ─── Phase 1.2: q4_1 / q5_1 end-to-end parity lock ──────────────────
//
// Phase 1 of the 2026-08-31 Unsloth coverage plan reclassified q4_1/q5_1
// as usable (official Unsloth ALLOWED_QUANTS save.py:163,170). The
// encoder-level goldens already pass (gguf_parity.rs); this test locks the
// FULL CLI conversion path: every 2-D tensor must come out byte-identical
// to quantizing the same f32 source with the same encoder, and no tensor
// may silently fall back to F16.

fn golden_dir() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // quant-cli
    p.pop(); // crates/
    p.join("tests/golden/gguf_quants")
}

/// Deterministic case from the gguf-py golden generator (randn_256): the
/// exact 256-element f32 vector whose Q4_1/Q5_1 encodings are committed
/// as goldens. Re-deriving it here would duplicate the generator, so we
/// read the committed f32 golden instead.
fn load_golden_f32(case: &str) -> Vec<f32> {
    let raw = std::fs::read(golden_dir().join(format!("{case}.f32.bin"))).unwrap();
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[test]
fn gguf_q4_1_q5_1_end_to_end_byte_parity() {
    // 256 elements: divisible by every legacy block size (32), so no F16
    // fallback is legitimate for these 2-D tensors.
    let case = load_golden_f32("randn_256");
    assert_eq!(case.len(), 256);
    let w_bytes: Vec<u8> = case.iter().flat_map(|v| v.to_le_bytes()).collect();
    // Direct encodings of the same source — the byte-parity expectations.
    let f16_want = rlx_gguf::quantize(&case, rlx_gguf::GgmlType::F16).unwrap();

    for (method, ggml) in [
        ("q4_1", rlx_gguf::GgmlType::Q4_1),
        ("q5_1", rlx_gguf::GgmlType::Q5_1),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("m");
        std::fs::create_dir_all(&model_dir).unwrap();

        // Two 2-D tensors carrying the SAME 256-element golden vector:
        //  - embed_tokens → token_embd.weight → embd_scheme F16 (policy)
        //  - q_proj      → blk.0.attn_q.weight → default scheme (the method)
        // plus one 1-D norm, which must stay F32.
        let tensors = vec![
            Tensor {
                name: "model.embed_tokens.weight",
                dtype: "F32",
                shape: vec![4, 64],
                bytes: w_bytes.clone(),
            },
            Tensor {
                name: "model.layers.0.self_attn.q_proj.weight",
                dtype: "F32",
                shape: vec![4, 64],
                bytes: w_bytes.clone(),
            },
            Tensor {
                name: "model.norm.weight",
                dtype: "F32",
                shape: vec![8],
                bytes: case[..8].iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        ];
        std::fs::write(
            model_dir.join("model.safetensors"),
            build_safetensors(&tensors),
        )
        .unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            serde_json::to_string(&serde_json::json!({
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "vocab_size": 4
            }))
            .unwrap(),
        )
        .unwrap();

        let out = tmp.path().join(format!("m-{method}.gguf"));
        let status = bin()
            .args([
                "gguf",
                model_dir.join("model.safetensors").to_str().unwrap(),
                out.to_str().unwrap(),
                "--method",
                method,
                "--no-progress",
            ])
            .status()
            .unwrap();
        assert!(status.success(), "{method}: CLI exited non-zero");
        assert!(out.exists(), "{method}: output missing");

        let f = rlx_gguf::GgufFile::from_path(&out).unwrap();

        // The regular 2-D weight must carry the method's own dtype with
        // bytes identical to a direct encode of the same f32 source.
        let attn_q = f.tensors.get("blk.0.attn_q.weight").unwrap();
        assert_eq!(attn_q.dtype, ggml, "{method}: attn_q dtype");
        let got = f.tensor_bytes(attn_q).unwrap();
        let want = rlx_gguf::quantize(&case, ggml).unwrap();
        assert_eq!(
            got,
            &want[..],
            "{method}: attn_q bytes differ from direct encode"
        );

        // Embeddings use the embd_scheme (F16) — also byte-checked, so a
        // future policy change cannot silently alter embedding bytes either.
        let embd = f.tensors.get("token_embd.weight").unwrap();
        assert_eq!(embd.dtype, rlx_gguf::GgmlType::F16, "{method}: embd dtype");
        let got_embd = f.tensor_bytes(embd).unwrap();
        assert_eq!(got_embd, &f16_want[..], "{method}: embd bytes");

        // 1-D norm stays F32 untouched.
        let norm = f
            .tensors
            .iter()
            .find(|(k, _)| k.ends_with("norm.weight"))
            .map(|(_, v)| v)
            .unwrap_or_else(|| panic!("{method}: no norm tensor found"));
        assert_eq!(norm.dtype, rlx_gguf::GgmlType::F32);
    }
}

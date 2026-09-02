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

// ─── Phase 4.2: --imatrix flag + iq* gate (Unsloth save.py:2162) ──────
//
// Every iq* method is in official Unsloth IMATRIX_QUANTS: without an
// importance matrix the quantized weights would be garbage, so the CLI
// rejects the combination with exit 2 (same contract as llama-quantize's
// "this quantization requires an imatrix!", llama-quant.cpp:1084-1090).

use quant_core::imatrix::Imatrix;

/// All eleven iq* method ids — must all be gated.
const IQ_METHODS_ALL: [&str; 11] = [
    "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs",
    "iq2_m", "iq3_m",
];

fn write_legacy_imatrix(dir: &Path, name: &str, weights: &[f32]) -> std::path::PathBuf {
    let mut out = Vec::new();
    out.extend_from_slice(&1i32.to_le_bytes()); // n_entries
    out.extend_from_slice(&(name.len() as i32).to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(&1i32.to_le_bytes()); // ncall = 1 → sums/1 = weights
    out.extend_from_slice(&(weights.len() as i32).to_le_bytes());
    for v in weights {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let p = dir.join("imatrix.dat");
    std::fs::write(&p, out).unwrap();
    p
}

#[test]
fn gguf_iq_method_without_imatrix_flag_exit_2() {
    // Every iq* id, no --imatrix → exit 2, message names method + flag.
    // Reuses one fixture dir; the method check happens before any I/O.
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);
    let out = tmp.path().join("x.gguf");
    for method in IQ_METHODS_ALL {
        let output = bin()
            .args([
                "gguf",
                model_dir.join("model.safetensors").to_str().unwrap(),
                out.to_str().unwrap(),
                "--method",
                method,
                "--no-progress",
            ])
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{method}: iq* without --imatrix must exit 2"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(method),
            "{method}: message must name the method: {stderr}"
        );
        assert!(
            stderr.contains("--imatrix"),
            "{method}: message must name the flag: {stderr}"
        );
        assert!(
            !out.exists(),
            "{method}: no output may be written on rejection"
        );
    }
}

#[test]
fn gguf_iq_method_with_missing_imatrix_file_exit_2() {
    // --imatrix pointing at a non-existent file → exit 2 with the loader's
    // message naming the path.
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);
    let out = tmp.path().join("x.gguf");
    let bogus = tmp.path().join("no_such_imatrix.dat");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "iq2_xxs",
            "--imatrix",
            bogus.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no_such_imatrix.dat"),
        "must name the path: {stderr}"
    );
}

#[test]
fn gguf_iq_method_with_malformed_imatrix_exit_2() {
    // A file that is neither GGUF nor legacy → exit 2 with a specific cause.
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);
    let out = tmp.path().join("x.gguf");
    let junk = tmp.path().join("junk.dat");
    std::fs::write(&junk, b"definitely not an imatrix").unwrap();
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "iq2_xxs",
            "--imatrix",
            junk.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not a GGUF and not a valid legacy"),
        "must carry the loader's specific cause: {stderr}"
    );
}

#[test]
fn gguf_iq_method_with_imatrix_proceeds() {
    // With a valid imatrix covering every quantized tensor, iq2_xxs runs
    // to completion (exit 0) and produces byte-identical attn_q bytes to
    // the golden llama-quantize --imatrix output for the same source.
    // The tiny model's tensors: h=32, v=64 — attn_q is 32×32, so the
    // weight vector needs 32 entries... but the tiny model's shapes are
    // too small for IQ2XXS's QK_K=256 blocks. Use the phase14 layout
    // instead (256-col attn_q) and its committed golden.
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    std::fs::create_dir_all(&model_dir).unwrap();

    // Read the shared golden source + weights (same files the phase14
    // core-level tests consume).
    let mut gd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    gd.pop();
    gd.pop();
    let gd = gd.join("tests/golden/llamacpp");
    let read_f32 = |name: &str, n: usize| {
        let raw = std::fs::read(gd.join(name)).unwrap();
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .take(n)
            .collect::<Vec<f32>>()
    };
    let src = read_f32("src.f32.bin", 512);
    let weights = read_f32("weights.f32.bin", 256);

    let w_bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensors = vec![Tensor {
        name: "model.layers.0.self_attn.q_proj.weight",
        dtype: "F32",
        shape: vec![2, 256],
        bytes: w_bytes,
    }];
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
            "hidden_size": 256,
            "num_hidden_layers": 1,
            "vocab_size": 64
        }))
        .unwrap(),
    )
    .unwrap();
    let imatrix = write_legacy_imatrix(tmp.path(), "blk.0.attn_q.weight", &weights);

    let out = tmp.path().join("m-iq2_xxs.gguf");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "iq2_xxs",
            "--imatrix",
            imatrix.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "iq2_xxs with a valid imatrix must proceed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("loaded 1 importance matrix entries"),
        "must log the load: {stderr}"
    );

    // Byte-exact vs the llama-quantize golden (same contract as the
    // phase14 core test, now through the CLI flag).
    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    let t = f.tensors.get("blk.0.attn_q.weight").unwrap();
    assert_eq!(t.dtype, rlx_gguf::GgmlType::IQ2XXS);
    let got = f.tensor_bytes(t).unwrap();
    let golden = std::fs::read(gd.join("weighted.iq2_xxs.bin")).unwrap();
    assert_eq!(
        got,
        &golden[..got.len()],
        "CLI path must reproduce llama-quantize bytes"
    );
}

#[test]
fn gguf_iq_tensor_missing_imatrix_entry_hard_error() {
    // imatrix loaded but has NO entry for the IQ tensor being quantized →
    // the conversion must hard-fail (llama-quant.cpp:1245-1251), not
    // silently produce garbage. Exit 1 (runtime failure mid-conversion).
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);
    let out = tmp.path().join("x.gguf");

    // An imatrix with an entry for a DIFFERENT tensor name.
    let imatrix = write_legacy_imatrix(tmp.path(), "blk.5.attn_q.weight", &[1.0f32; 32]);

    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "iq4_nl",
            "--imatrix",
            imatrix.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(1),
        "missing entry on an iq* tensor must hard-fail (not garbage)"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Missing importance matrix for tensor"),
        "must name the missing tensor: {stderr}"
    );
}

#[test]
fn gguf_k_method_with_imatrix_is_optional_and_consumed() {
    // K-quant methods accept --imatrix as OPTIONAL (no gate): a file
    // covering attn_q makes q4_k_s take the weighted path and produce
    // the llama-quantize golden bytes for that tensor; other 2-D
    // tensors (embed/down/lm_head) get the "did not find weights" note
    // and stay unweighted — exactly llama-quantize's behaviour.
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    std::fs::create_dir_all(&model_dir).unwrap();
    let mut gd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    gd.pop();
    gd.pop();
    let gd = gd.join("tests/golden/llamacpp");
    let read_f32 = |name: &str, n: usize| {
        let raw = std::fs::read(gd.join(name)).unwrap();
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .take(n)
            .collect::<Vec<f32>>()
    };
    let src = read_f32("src.f32.bin", 512);
    let weights = read_f32("weights.f32.bin", 256);

    let w_bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensors = vec![Tensor {
        name: "model.layers.0.self_attn.q_proj.weight",
        dtype: "F32",
        shape: vec![2, 256],
        bytes: w_bytes,
    }];
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
            "hidden_size": 256,
            "num_hidden_layers": 1,
            "vocab_size": 64
        }))
        .unwrap(),
    )
    .unwrap();
    let imatrix = write_legacy_imatrix(tmp.path(), "blk.0.attn_q.weight", &weights);

    let out = tmp.path().join("m-q4_k_s.gguf");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "q4_k_s",
            "--imatrix",
            imatrix.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "K-quant with imatrix must proceed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    let t = f.tensors.get("blk.0.attn_q.weight").unwrap();
    assert_eq!(t.dtype, rlx_gguf::GgmlType::Q4K);
    let got = f.tensor_bytes(t).unwrap();
    let golden = std::fs::read(gd.join("weighted.q4_k.bin")).unwrap();
    assert_eq!(
        got,
        &golden[..got.len()],
        "q4_k_s with imatrix must reproduce llama-quantize weighted bytes"
    );
}

// Keep the Imatrix import used (loader exercised indirectly through the
// CLI in tests above; this anchors the direct load path too).
#[test]
fn imatrix_legacy_load_direct() {
    let tmp = tempfile::tempdir().unwrap();
    let p = write_legacy_imatrix(tmp.path(), "blk.0.attn_q.weight", &[0.5f32, 2.0]);
    let im = Imatrix::load(&p).unwrap();
    assert_eq!(im.len(), 1);
    assert_eq!(im.weights_for("blk.0.attn_q.weight").unwrap(), &[0.5, 2.0]);
    assert!(im.is_legacy);
}

// ─── Phase 6: --tensor-type-file recipes + overrides + --emit-recipe ──
//
// The open equivalent of Unsloth's proprietary UD-* dynamic recipes
// (plan §3-F/Phase 6). llama-quant.cpp:713-727 semantics: first matching
// regex wins (search), the method policy is skipped for matched tensors,
// --token-embedding-type/--output-tensor-type override their categories.

#[test]
fn gguf_recipe_assigns_per_tensor_qtypes() {
    // A recipe routing ffn_down to q6_k under q4_k_m must yield Q6K bytes
    // for that tensor and the base Q4K elsewhere.
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);

    let recipe = tmp.path().join("recipe.txt");
    std::fs::write(&recipe, "ffn_down\\.weight=q6_k\n").unwrap();

    let out = tmp.path().join("m-recipe.gguf");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "q4_k_m",
            "--tensor-type-file",
            recipe.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "recipe run must succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    // ffn_down got the recipe's Q6K…
    let ffn_down = f.tensors.get("blk.0.ffn_down.weight").unwrap();
    assert_eq!(ffn_down.dtype, rlx_gguf::GgmlType::Q6K, "recipe must win");
    // …while attn_q keeps the method base Q4K.
    let attn_q = f.tensors.get("blk.0.attn_q.weight").unwrap();
    assert_eq!(attn_q.dtype, rlx_gguf::GgmlType::Q4K);
}

#[test]
fn gguf_recipe_skips_composite_policy_engine() {
    // For a composite method (q4_k_m routes through the llama_policy
    // engine, which assigns Q6K to attn_v via use_more_bits), a recipe
    // rule hitting the SAME tensor must win with the recipe's scheme —
    // proving the manual path bypasses the engine (llama-quant.cpp:730).
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);

    let recipe = tmp.path().join("recipe.txt");
    // Route ffn_down to q5_k_s — WITHOUT the recipe the engine gives Q6K.
    std::fs::write(&recipe, "ffn_down\\.weight=q5_k_s\n").unwrap();

    let out = tmp.path().join("m-recipe-engine.gguf");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "q4_k_m",
            "--tensor-type-file",
            recipe.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    let ffn_down = f.tensors.get("blk.0.ffn_down.weight").unwrap();
    assert_eq!(
        ffn_down.dtype,
        rlx_gguf::GgmlType::Q5K,
        "recipe must override the engine's Q6K"
    );
}

#[test]
fn gguf_recipe_bare_default_applies_to_unmatched() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);

    // Only ffn_down has a rule; the bare default q8_0 covers everything
    // else that the method would otherwise quantize.
    let recipe = tmp.path().join("recipe.txt");
    std::fs::write(&recipe, "ffn_down\\.weight=q6_k\nq8_0\n").unwrap();

    let out = tmp.path().join("m-default.gguf");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "q4_k_m",
            "--tensor-type-file",
            recipe.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    // ffn_down → rule's Q6K; attn_q → default's Q8_0 (not the method's Q4K).
    assert_eq!(
        f.tensors.get("blk.0.ffn_down.weight").unwrap().dtype,
        rlx_gguf::GgmlType::Q6K
    );
    assert_eq!(
        f.tensors.get("blk.0.attn_q.weight").unwrap().dtype,
        rlx_gguf::GgmlType::Q8_0,
        "bare default must cover unmatched tensors"
    );
}

#[test]
fn gguf_recipe_bad_qtype_exit_2_names_line() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);

    let recipe = tmp.path().join("recipe.txt");
    std::fs::write(&recipe, "attn_v\\.weight=bogus\n").unwrap();

    let out = tmp.path().join("x.gguf");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "q4_k_m",
            "--tensor-type-file",
            recipe.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("line 1") && stderr.contains("bogus"),
        "must name line and qtype: {stderr}"
    );
    assert!(
        stderr.contains("recipe"),
        "must say which file is bad: {stderr}"
    );
}

#[test]
fn gguf_recipe_missing_file_exit_2() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);
    let out = tmp.path().join("x.gguf");
    let bogus = tmp.path().join("no_recipe.txt");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "q4_k_m",
            "--tensor-type-file",
            bogus.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn gguf_recipe_with_dynamic_method_still_exit_2() {
    // A recipe cannot smuggle a UD-* method through: the method itself is
    // rejected before the recipe is even loaded.
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);
    let recipe = tmp.path().join("recipe.txt");
    std::fs::write(&recipe, "attn_v\\.weight=q6_k\n").unwrap();
    let out = tmp.path().join("x.gguf");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "q4_k_xl",
            "--tensor-type-file",
            recipe.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Dynamic 2.0"),
        "UD-* stays rejected: {stderr}"
    );
}

#[test]
fn gguf_token_embedding_and_output_type_overrides() {
    // --token-embedding-type / --output-tensor-type win over the method's
    // embd_scheme and any recipe rule for their tensors.
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);

    let recipe = tmp.path().join("recipe.txt");
    // A recipe that would ALSO hit token_embd (broad pattern) — the
    // override must still win for token_embd, and the recipe must apply to
    // per-layer embd (upstream `named` exception only covers
    // per_layer_token_embd when the recipe names IT specifically).
    std::fs::write(&recipe, "token_embd\\.weight=q6_k\n").unwrap();

    let out = tmp.path().join("m-over.gguf");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "q4_k_m",
            "--token-embedding-type",
            "q8_0",
            "--output-tensor-type",
            "q6_k",
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
    assert_eq!(
        f.tensors.get("token_embd.weight").unwrap().dtype,
        rlx_gguf::GgmlType::Q8_0,
        "--token-embedding-type must win over the F16 embd convention"
    );
    assert_eq!(
        f.tensors.get("output.weight").unwrap().dtype,
        rlx_gguf::GgmlType::Q6K,
        "--output-tensor-type must win"
    );

    // The overrides must also beat the recipe for their own tensors.
    let out2 = tmp.path().join("m-over-recipe.gguf");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out2.to_str().unwrap(),
            "--method",
            "q4_k_m",
            "--token-embedding-type",
            "q8_0",
            "--tensor-type-file",
            recipe.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let f = rlx_gguf::GgufFile::from_path(&out2).unwrap();
    assert_eq!(
        f.tensors.get("token_embd.weight").unwrap().dtype,
        rlx_gguf::GgmlType::Q8_0,
        "category override must beat the recipe for token_embd"
    );
}

#[test]
fn gguf_bad_category_override_exit_2() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);
    let out = tmp.path().join("x.gguf");
    for flag in ["--token-embedding-type", "--output-tensor-type"] {
        let output = bin()
            .args([
                "gguf",
                model_dir.join("model.safetensors").to_str().unwrap(),
                out.to_str().unwrap(),
                "--method",
                "q4_k_m",
                flag,
                "bogus",
                "--no-progress",
            ])
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{flag} with bogus method must exit 2"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(flag),
            "{flag}: error must name the flag: {stderr}"
        );
    }
}

#[test]
fn gguf_emit_recipe_round_trip() {
    // --emit-recipe dumps the effective assignment; feeding the dump back
    // through --tensor-type-file reproduces the same per-tensor dtypes.
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_tiny_model(&model_dir);

    // Run 1: q4_k_m + emit.
    let out1 = tmp.path().join("m-a.gguf");
    let emitted = tmp.path().join("effective.txt");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out1.to_str().unwrap(),
            "--method",
            "q4_k_m",
            "--emit-recipe",
            emitted.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(emitted.exists());
    let text = std::fs::read_to_string(&emitted).unwrap();
    assert!(
        text.contains("^blk.0.attn_q.weight$="),
        "dump must contain per-tensor lines: {text}"
    );

    // Run 2: same method, dump fed back via --tensor-type-file.
    let out2 = tmp.path().join("m-b.gguf");
    let output = bin()
        .args([
            "gguf",
            model_dir.join("model.safetensors").to_str().unwrap(),
            out2.to_str().unwrap(),
            "--method",
            "q4_k_m",
            "--tensor-type-file",
            emitted.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Every quantized tensor dtype must match run 1.
    let f1 = rlx_gguf::GgufFile::from_path(&out1).unwrap();
    let f2 = rlx_gguf::GgufFile::from_path(&out2).unwrap();
    for (name, t1) in &f1.tensors {
        let t2 = f2.tensors.get(name).expect("same tensor set");
        assert_eq!(
            t1.dtype, t2.dtype,
            "{name}: round-tripped recipe changed the dtype"
        );
    }
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

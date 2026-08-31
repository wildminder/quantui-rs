//! Unsloth coverage plan Phase 0.1: the method-encodes-without-fallback
//! sweep.
//!
//! For EVERY usable registry method, convert a fixture model and assert:
//! 1. `fallback_f16 == 0` — no tensor silently degraded to F16;
//! 2. the method's default scheme actually appears in the output for a
//!    regular 2-D weight (the method did something, not just exited);
//! 3. the output is a loadable GGUF per rlx-gguf's parser;
//! 4. 1-D tensors still stay F32 (shared convention, must not drift).
//!
//! This is the regression net for every later phase: adding a registry
//! entry without an encoder, or breaking a policy, fails here first.

use std::path::{Path, PathBuf};

use quant_core::gguf_convert::{convert_hf_to_gguf, GgufConvertConfig, GgufConvertReport};
use quant_core::gguf_registry::{self, BackendSupport, GgufScheme};

// ─── fixture builder ────────────────────────────────────────────────

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

/// K-quant block size (llama.cpp QK_K) — every 2-D row length must be a
/// multiple of this so the K/IQ methods never hit divisibility fallback.
const K_COLS: usize = 256;

/// Write a model whose every 2-D weight row length is a multiple of 256:
/// the `q_proj` (K_COLS x K_COLS) is the probe tensor for the default
/// scheme, plus an ffn_down for rule coverage and a 1-D norm.
fn write_fixture(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let mk2d = |name: &'static str, rows: usize, cols: usize, seed: u64| Tensor {
        name,
        dtype: "BF16",
        shape: vec![rows as u64, cols as u64],
        bytes: bf16_bytes(&synth(rows * cols, seed)),
    };
    let tensors = vec![
        mk2d("model.layers.0.self_attn.q_proj.weight", K_COLS, K_COLS, 2),
        mk2d("model.layers.0.mlp.down_proj.weight", K_COLS, K_COLS, 8),
        mk2d("model.layers.0.self_attn.v_proj.weight", K_COLS, K_COLS, 4),
        Tensor {
            name: "model.layers.0.input_layernorm.weight",
            dtype: "BF16",
            shape: vec![K_COLS as u64],
            bytes: bf16_bytes(&synth(K_COLS, 9)),
        },
    ];
    std::fs::write(dir.join("model.safetensors"), build_safetensors(&tensors)).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": K_COLS,
            "num_hidden_layers": 1,
            "vocab_size": 32
        }))
        .unwrap(),
    )
    .unwrap();
}

/// The GgmlType the sweep expects for the default scheme — mirrors
/// `gguf_convert::scheme_to_ggml` (kept private there, so duplicated
/// here as an independent check the mapping didn't drift).
fn expected_ggml(s: GgufScheme) -> rlx_gguf::GgmlType {
    use rlx_gguf::GgmlType;
    match s {
        GgufScheme::F32 => GgmlType::F32,
        GgufScheme::F16 => GgmlType::F16,
        GgufScheme::Bf16 => GgmlType::BF16,
        GgufScheme::Q8_0 => GgmlType::Q8_0,
        GgufScheme::Q4_0 => GgmlType::Q4_0,
        GgufScheme::Q4_1 => GgmlType::Q4_1,
        GgufScheme::Q5_0 => GgmlType::Q5_0,
        GgufScheme::Q5_1 => GgmlType::Q5_1,
        GgufScheme::Q2K => GgmlType::Q2K,
        GgufScheme::Q3K => GgmlType::Q3K,
        GgufScheme::Q4K => GgmlType::Q4K,
        GgufScheme::Q5K => GgmlType::Q5K,
        GgufScheme::Q6K => GgmlType::Q6K,
        GgufScheme::Q8K => GgmlType::Q8K,
        GgufScheme::Iq2Xxs => GgmlType::IQ2XXS,
        GgufScheme::Iq2Xs => GgmlType::IQ2XS,
        GgufScheme::Iq3Xxs => GgmlType::IQ3XXS,
        GgufScheme::Iq4Nl => GgmlType::IQ4NL,
    }
}

fn run_conversion(method: &str, dir: &Path) -> (GgufConvertReport, PathBuf) {
    let out = dir.parent().unwrap().join(format!("sweep-{}.gguf", method));
    let cfg = GgufConvertConfig {
        method_id: method.to_string(),
        arch: None,
        name: None,
    };
    let report = convert_hf_to_gguf(dir.join("model.safetensors").as_path(), &out, &cfg, None)
        .unwrap_or_else(|e| panic!("{method}: conversion failed: {e}"));
    (report, out)
}

/// One probe per usable method. Not a `#[test]` per se — see
/// [`sweep_all_methods`] which drives this for the full list, keeping the
/// failure message anchored on the offending method id.
fn probe_method(e: &gguf_registry::RegistryEntry) {
    let id = e.method.id;
    assert!(
        matches!(e.method.support, BackendSupport::Encodable),
        "{id}: sweep only covers Encodable methods"
    );

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_fixture(&model_dir);

    let (report, out) = run_conversion(id, &model_dir);

    // 1. No silent F16 fallback.
    assert_eq!(
        report.fallback_f16,
        0,
        "{id}: {n} tensor(s) silently fell back to F16",
        n = report.fallback_f16
    );

    // 2. Output loads and the probe tensor carries the scheme the POLICY
    //    selects for it (default, or a rule override like q3_k_m's
    //    attn_q→Q4K). Asserting the resolved scheme, not policy.default,
    //    is what makes this a real policy regression net.
    let want_scheme = gguf_registry::scheme_for(e, "blk.0.attn_q.weight", 2);
    let f = rlx_gguf::GgufFile::from_path(&out)
        .unwrap_or_else(|e| panic!("{id}: output not loadable GGUF: {e}"));
    let probe = f
        .tensors
        .get("blk.0.attn_q.weight")
        .unwrap_or_else(|| panic!("{id}: probe tensor blk.0.attn_q.weight missing"));
    assert_eq!(
        probe.dtype,
        expected_ggml(want_scheme),
        "{id}: probe tensor dtype != policy-resolved scheme"
    );

    // 3. 1-D norms stay F32.
    let norm = f
        .tensors
        .get("blk.0.attn_norm.weight")
        .unwrap_or_else(|| panic!("{id}: norm tensor missing"));
    assert_eq!(
        norm.dtype,
        rlx_gguf::GgmlType::F32,
        "{id}: 1-D norm not F32"
    );

    // 4. Everything quantized (no F16-fallback path taken, no kept F32 2-D).
    assert!(
        report.quantized >= 3,
        "{id}: expected >=3 quantized 2-D tensors"
    );
}

#[test]
fn sweep_all_methods() {
    let usable = gguf_registry::usable_ids();
    assert!(
        usable.len() >= 18,
        "sweep expects at least the 18 pre-plan methods, got {}",
        usable.len()
    );
    for id in &usable {
        let e = gguf_registry::get_method(id)
            .unwrap_or_else(|| panic!("usable_ids contains {id} but get_method misses it"));
        probe_method(e);
    }
}

/// The usable list is exactly the registry snapshot this plan expects:
/// 20 usable (18 original + q4_1 + q5_1 from Phase 1), 4 rejected.
/// When Phase 3 adds methods, this snapshot is updated atomically with
/// the registry change — a method cannot appear without its sweep coverage.
#[test]
fn usable_ids_snapshot() {
    assert_eq!(
        gguf_registry::usable_ids(),
        vec![
            "f16", "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q5_0", "q5_1", "q4_k_m", "q4_k_s", "q4_0",
            "q4_1", "q3_k_m", "q3_k_l", "q3_k_s", "q3_k_xs", "q2_k", "iq4_nl", "iq3_xxs",
            "iq2_xxs", "iq2_xs",
        ]
    );
}

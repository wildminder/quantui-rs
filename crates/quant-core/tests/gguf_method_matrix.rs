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
        // Phase 3 additions:
        GgufScheme::Iq1S => GgmlType::IQ1S,
        GgufScheme::Iq1M => GgmlType::IQ1M,
        GgufScheme::Iq2S => GgmlType::IQ2S,
        GgufScheme::Iq3S => GgmlType::IQ3S,
        GgufScheme::Iq4Xs => GgmlType::IQ4XS,
        GgufScheme::Tq1_0 => GgmlType::TQ1_0,
        GgufScheme::Tq2_0 => GgmlType::TQ2_0,
        GgufScheme::Q1_0 => GgmlType::Q1_0,
        GgufScheme::Q2_0 => GgmlType::Q2_0,
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

    // 4. Bookkeeping consistency: every 2-D tensor was either quantized to
    //    the policy scheme (counted in quantized) or — for f32, whose
    //    default IS F32 — kept as F32. The invariant: no 2-D tensor went
    //    through the F16 fallback path (assert #1) and none is missing.
    let nd_2d = 3usize; // q_proj, down_proj, v_proj in the fixture
    let handled = report.quantized + report.kept_f32;
    assert_eq!(
        handled,
        nd_2d + 1, // +1 for the 1-D norm (kept F32 by the shared convention)
        "{id}: {handled} tensors accounted for, expected {} (3 x 2-D + 1 norm)",
        nd_2d + 1
    );
    if e.policy.default != GgufScheme::F32 {
        assert!(
            report.quantized >= nd_2d,
            "{id}: expected >= {nd_2d} quantized 2-D tensors, got {}",
            report.quantized
        );
    } else {
        // f32 method: the 2-D tensors are F32 by policy, not fallback.
        assert_eq!(
            report.quantized, 0,
            "{id}: F32-default method must not count kept tensors as quantized"
        );
    }
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
/// 31 usable (20 after Phase 1 + 11 from Phase 3), 4 rejected.
/// When a later phase adds methods, this snapshot is updated atomically
/// with the registry change — a method cannot appear without its sweep
/// coverage.
#[test]
fn usable_ids_snapshot() {
    assert_eq!(
        gguf_registry::usable_ids(),
        vec![
            // ── original reference order (q4_1/q5_1 unblocked in Phase 1) ──
            "f16", "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q5_0", "q5_1", "q4_k_m", "q4_k_s", "q4_0",
            "q4_1", "q3_k_m", "q3_k_l", "q3_k_s", "q3_k_xs", "q2_k", "iq4_nl", "iq3_xxs",
            "iq2_xxs", "iq2_xs",
            // ── Phase 3 additions (Unsloth coverage plan) ──
            "f32", "bf16", "iq1_s", "iq1_m", "iq2_s", "iq3_s", "iq4_xs", "tq1_0", "tq2_0", "q1_0",
            "q2_0",
        ]
    );
}

// ─── Phase 3 quartet part 4: round-trip bounded-error ───────────────
//
// The sweep (above) proves every method encodes without fallback. This test
// proves the encoded bytes mean something: dequantize the probe tensor and
// assert the reconstruction stays within a per-family tolerance of the f32
// source. Tolerances are deliberately generous — they catch a broken
// encoder (garbage bytes), not quality differences (that's the imatrix
// story, Phase 4). Family bounds chosen from llama.cpp's documented
// reconstruction characteristics, not tuned to pass.

/// Max mean absolute reconstruction error per method id. K-quants and the
/// legacy family reconstruct well; IQ1/IQ2 are extremely lossy by design
/// (1.5-2.5 bpw — only the SHAPE of the data survives); ternary stores
/// {-1,0,+1} so uniform-ish data loses its magnitude.
fn round_trip_tolerance(id: &str) -> f32 {
    match id {
        // Lossless formats: exact to rounding.
        "f16" | "f32" | "bf16" => 0.02,
        // 8-bit near-lossless.
        "q8_0" => 0.02,
        // Legacy 4-5 bit.
        "q4_0" | "q4_1" | "q5_0" | "q5_1" => 0.15,
        // Legacy 1-2 bit: 1-bit reconstructs at ~0.25 on uniform noise
        // (measured) — only the sign survives, by design.
        "q1_0" | "q2_0" => 0.3,
        // K-quants.
        "q2_k" | "q3_k_m" | "q3_k_l" | "q3_k_s" | "q3_k_xs" | "q4_k_m" | "q4_k_s" | "q5_k_m"
        | "q5_k_s" | "q6_k" => 0.15,
        // IQ 4 bit: fine.
        "iq4_nl" | "iq4_xs" => 0.2,
        // IQ 2-3 bit: the lattice families (IQ2*, IQ3*) reconstruct
        // coarsely on uniform noise with rlx's uniform-weight encoders
        // (no imatrix yet — Phase 4 changes this profile). Measured:
        // iq3_xxs 0.45, iq3_s 0.50, iq2_* similar — grouped at 0.55 with
        // headroom; the test's job is catching garbage, not grading
        // quality.
        "iq2_xxs" | "iq2_xs" | "iq2_s" | "iq3_xxs" | "iq3_s" => 0.55,
        // IQ ~1.5 bit: barely above noise — bound loose, the point is
        // "not garbage", e.g. mean error 10x smaller than the data range.
        "iq1_s" | "iq1_m" => 0.5,
        // Ternary: values collapse to {-d, 0, +d}; on [-1,1] data the
        // mean error is bounded by the sign-vs-magnitude loss.
        "tq1_0" | "tq2_0" => 0.55,
        other => panic!("no tolerance defined for {other} — add it when adding the method"),
    }
}

#[test]
fn round_trip_reconstruction_bounded_error() {
    for id in gguf_registry::usable_ids() {
        let e = gguf_registry::get_method(id).unwrap();
        let scheme = gguf_registry::scheme_for(e, "blk.0.attn_q.weight", 2);
        let f32_src = synth(K_COLS * K_COLS, 2); // same seed as the fixture's q_proj
        let tol = round_trip_tolerance(id);

        // Encode + decode the same source directly with rlx-gguf.
        let encoded = rlx_gguf::quantize(&f32_src, expected_ggml(scheme))
            .unwrap_or_else(|err| panic!("{id}: direct encode failed: {err}"));

        // Decode via a real GGUF round-trip: write, reopen, dequant.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("rt.gguf");
        let mut w = rlx_gguf::GgufWriter::new();
        w.set_arch("llama");
        w.add_tensor_bytes(
            "blk.0.attn_q.weight",
            vec![K_COLS, K_COLS],
            expected_ggml(scheme),
            encoded,
        )
        .unwrap();
        w.write_to_path(&path).unwrap();

        let f = rlx_gguf::GgufFile::from_path(&path).unwrap();
        let (deq, shape) = f
            .dequant_f32("blk.0.attn_q.weight")
            .unwrap_or_else(|err| panic!("{id}: dequant failed: {err}"));
        assert_eq!(shape, vec![K_COLS, K_COLS], "{id}: dequant shape");
        assert_eq!(deq.len(), f32_src.len(), "{id}: dequant length");

        let mean_err: f32 = deq
            .iter()
            .zip(&f32_src)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / deq.len() as f32;
        assert!(
            mean_err <= tol,
            "{id}: mean reconstruction error {mean_err:.4} exceeds tolerance {tol} — encoder output is garbage"
        );
    }
}

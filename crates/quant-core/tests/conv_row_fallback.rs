//! BugFix: per-row GGUF block-size fallback.
//!
//! GGUF requires every quantized tensor's row width `ne[0]` to be a
//! multiple of the type's block size — `gguf.cpp:724` treats a violation
//! as an error and `:1409` asserts it. The pinned `rlx-gguf` 0.2.14
//! encoder only checks the FLAT element count (`quantize.rs:199`
//! `check_div(name, n, blk)`), so Conv1d / ConvTranspose1d weights — whose
//! `ne[0]` is the kernel size — were quantized into blocks that straddle
//! row boundaries. `VibeVoice-1.5B` has 102 such tensors (kernel sizes
//! 4/7/8/10/16), none divisible by Q8_0's block size 32.
//!
//! The fix ports llama-quantize's `tensor_type_fallback`
//! (`docs/ref/llama.cpp/src/llama-quant.cpp:372-425`), which upstream calls
//! from `llama_tensor_get_type` (:309) at TYPE-SELECTION time.
//!
//! These tests pin the observable contract:
//!   * a row width that is not block-aligned NEVER reaches the encoder —
//!     the tensor is demoted, and to F16 when nothing narrower fits;
//!   * a block-aligned row is untouched and produces NO diagnostic;
//!   * 1-D tensors (F32 by convention) are never demoted;
//!   * the demoted scheme is what `effective_schemes` records, so
//!     `--emit-recipe` round-trips.

use std::path::Path;

use quant_core::gguf_convert::{convert_hf_to_gguf, ggml_blck_size, GgufConvertConfig};
use rlx_gguf::GgmlType;

// ─── safetensors fixture ────────────────────────────────────────────

struct Tensor {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

/// Serialize tensors into the reference safetensors layout our reader
/// accepts: `u64 LE header_len (8-aligned) | JSON header (padded) | data`.
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

// ─── tensor names ───────────────────────────────────────────────────
//
// None of these start with `model.layers.`, so `hf_to_gguf_name` returns
// `None` for all of them and the GGUF-side name equals the HF-side name.
// They are also invisible to the registry's name rules (q8_0 / q6_k /
// q4_k all use `rules: &[]`), so every 2-D+ tensor here gets the method's
// default scheme — exactly what we need to isolate the row-width check.

/// Conv1d weight `(out, in/groups, kernel)`: `ne[0]` = kernel size 7.
const CONV_K7: &str = "model.decoder.layers.0.conv1d.weight";
/// ConvTranspose1d weight: `ne[0]` = kernel size 10.
const CONV_K10: &str = "model.decoder.layers.0.conv_transpose.weight";
/// 1-D norm: F32 by the shared convention, never a demotion candidate.
const CONV_NORM: &str = "model.decoder.layers.0.norm.weight";
/// Block-aligned 2-D control: `ne[0]` = 256 (divisible by 32 and 256).
const WIDE_2D: &str = "model.decoder.layers.0.proj.weight";
/// Block-aligned 3-D control: `ne[0]` = 256.
const WIDE_3D: &str = "model.decoder.layers.0.big.weight";

fn conv_stack() -> Vec<Tensor> {
    let mk = |name: &'static str, shape: Vec<u64>, seed: u64| {
        let n = shape.iter().product::<u64>() as usize;
        Tensor {
            name,
            dtype: "BF16",
            shape,
            bytes: bf16_bytes(&synth(n, seed)),
        }
    };
    vec![
        mk(CONV_K7, vec![32, 1, 7], 1),
        mk(CONV_K10, vec![16, 256, 10], 2),
        mk(CONV_NORM, vec![64], 3),
        mk(WIDE_2D, vec![256, 256], 4),
        mk(WIDE_3D, vec![512, 2, 256], 5),
    ]
}

fn write_model(dir: &Path, tensors: &[Tensor]) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("model.safetensors"), build_safetensors(tensors)).unwrap();
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 256,
        "num_hidden_layers": 1,
        "vocab_size": 256,
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();
}

struct Outcome {
    report: quant_core::gguf_convert::GgufConvertReport,
    /// `(gguf_name, dtype)` as actually written to the file.
    dtypes: Vec<(String, GgmlType)>,
}

impl Outcome {
    fn dtype(&self, name: &str) -> GgmlType {
        self.dtypes
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("tensor {name} missing from output: {:?}", self.dtypes))
            .1
    }
}

/// Convert `tensors` with `method`, then read the result back through
/// rlx-gguf's own spec-compliant parser.
fn run(tensors: &[Tensor], method: &str) -> Outcome {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("m");
    write_model(&dir, tensors);
    let out = tmp.path().join("o.gguf");
    let cfg = GgufConvertConfig {
        method_id: method.into(),
        ..Default::default()
    };
    let report =
        convert_hf_to_gguf(&dir.join("model.safetensors"), &out, &cfg, None).expect("conversion");
    let f = rlx_gguf::GgufFile::from_path(&out).expect("parse output");
    let dtypes = f
        .tensors
        .iter()
        .map(|(n, t)| (n.clone(), t.dtype))
        .collect();
    Outcome { report, dtypes }
}

// ─── tests ──────────────────────────────────────────────────────────

/// The block-size table itself: the values the demotion chain is built on.
#[test]
fn blck_size_matches_the_pinned_encoder() {
    let cases: &[(GgmlType, usize)] = &[
        (GgmlType::F32, 1),
        (GgmlType::F16, 1),
        (GgmlType::BF16, 1),
        (GgmlType::Q4_0, 32),
        (GgmlType::Q4_1, 32),
        (GgmlType::Q5_0, 32),
        (GgmlType::Q5_1, 32),
        (GgmlType::Q8_0, 32),
        (GgmlType::Q8_1, 32),
        (GgmlType::IQ4NL, 32),
        (GgmlType::Q2K, 256),
        (GgmlType::Q3K, 256),
        (GgmlType::Q4K, 256),
        (GgmlType::Q5K, 256),
        (GgmlType::Q6K, 256),
        (GgmlType::Q8K, 256),
        (GgmlType::IQ1S, 256),
        (GgmlType::IQ1M, 256),
        (GgmlType::IQ2XXS, 256),
        (GgmlType::IQ2XS, 256),
        (GgmlType::IQ2S, 256),
        (GgmlType::IQ3XXS, 256),
        (GgmlType::IQ3S, 256),
        (GgmlType::IQ4XS, 256),
        (GgmlType::TQ1_0, 256),
        (GgmlType::TQ2_0, 256),
        // rlx-gguf custom formats: 128, NOT llama.cpp's 64 / upstream 32.
        (GgmlType::Q1_0, 128),
        (GgmlType::Q2_0, 128),
    ];
    for (t, want) in cases {
        assert_eq!(ggml_blck_size(*t), *want, "blck_size({t:?})");
    }
}

/// Q8_0 is already the narrowest 32-block type: a row width of 7 or 10
/// cannot be described by any quantized block, so both conv tensors land
/// as F16 — which is legal GGUF, unlike the block-straddling Q8_0 the
/// bug used to write.
#[test]
fn q8_0_conv_rows_fall_back_to_f16() {
    let o = run(&conv_stack(), "q8_0");

    assert_eq!(o.dtype(CONV_K7), GgmlType::F16, "ne[0]=7 must be F16");
    assert_eq!(o.dtype(CONV_K10), GgmlType::F16, "ne[0]=10 must be F16");

    // Everything block-aligned keeps the requested scheme.
    assert_eq!(o.dtype(WIDE_2D), GgmlType::Q8_0);
    assert_eq!(o.dtype(WIDE_3D), GgmlType::Q8_0);
    // 1-D stays F32 and is not a demotion candidate.
    assert_eq!(o.dtype(CONV_NORM), GgmlType::F32);

    // Exactly the two conv tensors were demoted.
    assert_eq!(
        o.report.row_fallback, 2,
        "{:?}",
        o.report.row_fallback_tensors
    );
    assert_eq!(
        o.report.row_fallback_tensors,
        vec![CONV_K7.to_string(), CONV_K10.to_string()]
    );
    // The demotion landed on F16, which has the SAME user-visible outcome as
    // the Phase 2.3 (flat element count) fallback: the tensor is NOT
    // q8_0-quantized. It is therefore reported through the same channel
    // (`fallback_f16`, and the CLI's "were NOT quantized with 'q8_0'"
    // summary); `row_fallback` above is what records the CAUSE. What must
    // NOT happen is the encoder rejecting it — the demotion preempts that.
    assert_eq!(o.report.fallback_f16, 2, "{:?}", o.report.fallback_tensors);
    assert_eq!(o.report.kept_f32, 1);

    // `--emit-recipe` records the POST-demotion scheme, so a recipe dumped
    // from this run reproduces the same file.
    for name in [CONV_K7, CONV_K10] {
        let (_, scheme) = o
            .report
            .effective_schemes
            .iter()
            .find(|(n, _)| n == name)
            .expect(name);
        assert_eq!(
            *scheme,
            quant_core::gguf_registry::GgufScheme::F16,
            "{name}"
        );
    }
}

/// Q6_K (256) → Q8_0 (32) → still not divisible → F16
/// (llama-quant.cpp:400 then :411-421).
#[test]
fn q6_k_conv_rows_demote_q8_0_then_f16() {
    let o = run(&conv_stack(), "q6_k");

    assert_eq!(
        o.dtype(CONV_K7),
        GgmlType::F16,
        "Q6_K -> Q8_0, and 7 is not divisible by 32 either"
    );
    assert_eq!(o.dtype(CONV_K10), GgmlType::F16);

    assert_eq!(o.dtype(WIDE_2D), GgmlType::Q6K, "ne[0]=256 fits Q6_K");
    assert_eq!(o.dtype(WIDE_3D), GgmlType::Q6K);
    assert_eq!(o.dtype(CONV_NORM), GgmlType::F32);

    assert_eq!(o.report.row_fallback, 2);
    // Both conv tensors ended as F16, so both are reported as "not
    // q6_k-quantized" — see the note in `q8_0_conv_rows_fall_back_to_f16`.
    assert_eq!(o.report.fallback_f16, 2);
}

/// Q4_K (256) → Q5_0 (32) → still not divisible → F16
/// (llama-quant.cpp:398 then :411-421).
#[test]
fn q4_k_conv_rows_demote_q5_0_then_f16() {
    let o = run(&conv_stack(), "q4_k_s");

    assert_eq!(
        o.dtype(CONV_K7),
        GgmlType::F16,
        "Q4_K -> Q5_0, and 7 is not divisible by 32 either"
    );
    assert_eq!(o.dtype(CONV_K10), GgmlType::F16);

    assert_eq!(o.dtype(WIDE_2D), GgmlType::Q4K, "ne[0]=256 fits Q4_K");
    assert_eq!(o.dtype(WIDE_3D), GgmlType::Q4K);
    assert_eq!(o.dtype(CONV_NORM), GgmlType::F32);

    assert_eq!(o.report.row_fallback, 2);
    // Both conv tensors ended as F16, so both are reported as "not
    // q4_k-quantized" — see the note in `q8_0_conv_rows_fall_back_to_f16`.
    assert_eq!(o.report.fallback_f16, 2);
}

/// The fast path: every row is block-aligned, so nothing is demoted and
/// the conversion is byte-for-byte what it was before the fix.
#[test]
fn block_aligned_rows_are_untouched() {
    let mut tensors = conv_stack();
    tensors.retain(|t| t.name != CONV_K7 && t.name != CONV_K10);

    for (method, want) in [
        ("q8_0", GgmlType::Q8_0),
        ("q6_k", GgmlType::Q6K),
        ("q4_k_s", GgmlType::Q4K),
    ] {
        let o = run(&tensors, method);
        assert_eq!(o.dtype(WIDE_2D), want, "{method}");
        assert_eq!(o.dtype(WIDE_3D), want, "{method}");
        assert_eq!(o.dtype(CONV_NORM), GgmlType::F32, "{method}");
        assert_eq!(o.report.row_fallback, 0, "{method} must not demote");
        assert!(
            o.report.row_fallback_tensors.is_empty(),
            "{method}: {:?}",
            o.report.row_fallback_tensors
        );
        assert_eq!(o.report.fallback_f16, 0, "{method}");
    }
}

/// A row width that divides 256 but not 32 is impossible (32 | 256), but a
/// K-quant row of exactly 32 or 64 must demote to a 32-block type and stay
/// there — the second check must NOT force it to F16.
#[test]
fn k_quant_narrow_rows_demote_to_32_block_type_not_f16() {
    let mk = |name: &'static str, shape: Vec<u64>, seed: u64| {
        let n = shape.iter().product::<u64>() as usize;
        Tensor {
            name,
            dtype: "BF16",
            shape,
            bytes: bf16_bytes(&synth(n, seed)),
        }
    };
    // ne[0] = 64: not divisible by 256, divisible by 32.
    let tensors = vec![
        mk("model.decoder.layers.0.narrow.weight", vec![8, 64], 1),
        mk("model.decoder.layers.0.wide.weight", vec![8, 256], 2),
    ];
    let o = run(&tensors, "q6_k");

    // Q6_K (256) → Q8_0 (32); 64 % 32 == 0, so it stays Q8_0.
    assert_eq!(
        o.dtype("model.decoder.layers.0.narrow.weight"),
        GgmlType::Q8_0,
        "a legal 32-block fallback must not be pushed to F16"
    );
    assert_eq!(
        o.dtype("model.decoder.layers.0.wide.weight"),
        GgmlType::Q6K,
        "ne[0]=256 is legal for Q6_K"
    );
    assert_eq!(o.report.row_fallback, 1);
    assert_eq!(o.report.fallback_f16, 0);
}

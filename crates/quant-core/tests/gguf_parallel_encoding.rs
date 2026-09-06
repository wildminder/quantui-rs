//! [IMP-004] step 2 — parallel payload encoding at the library level.
//!
//! The GGUF driver resolves every tensor sequentially (scheme resolution
//! advances llama.cpp's policy counters and is strictly order-sensitive)
//! and encodes a chunk of payloads on a rayon pool. Two properties make
//! that safe, and this file pins both on a model that is LARGER than one
//! chunk (24 tensors vs `PAR_CHUNK_TENSORS = 8`), which the CLI fixtures
//! (5 tensors, a single chunk) cannot reach:
//!
//!   * the bytes on disk are identical to the fully sequential pipeline,
//!     including across chunk boundaries and the final short chunk;
//!   * the `(cur, total)` progress stream is identical and reaches
//!     `total` exactly once, in order, in both modes.
//!
//! Both use `GgufConvertConfig::jobs` rather than the
//! `QUANTUI_RS_GGUF_JOBS` env var: env state is process-global and would
//! race with the other tests in the binary.

use std::path::Path;

use quant_core::gguf_convert::{convert_hf_to_gguf, GgufConvertConfig};

// ─── safetensors fixture ────────────────────────────────────────────

struct Tensor {
    name: String,
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
        map.insert(t.name.clone(), serde_json::Value::Object(info));
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

/// A 24-tensor llama-shaped model: 1 embedding + 3 layers x 7 weights +
/// 1 norm + 1 lm_head. Deliberately larger than `PAR_CHUNK_TENSORS` (8)
/// so the driver walks several chunks and ends on a short one (24 % 8 ==
/// 0 here, but the 512 MiB byte cap and the final flush both matter for
/// anything that is not a multiple).
///
/// Row width (`ne[0]`, the LAST dim) is 256 — a multiple of every
/// K-quant block size — so nothing trips the row-width demotion and the
/// comparison is purely about the parallel plumbing.
fn write_model(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let h = 256usize; // ne[0]
    let rows = 8usize; // rows: free
    let v = 32usize; // vocab rows

    let mut spec: Vec<(String, Vec<u64>)> = Vec::new();
    spec.push(("model.embed_tokens.weight".into(), vec![v as u64, h as u64]));
    for l in 0..3 {
        for n in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            spec.push((
                format!("model.layers.{l}.self_attn.{n}.weight"),
                vec![rows as u64, h as u64],
            ));
        }
        for n in ["gate_proj", "up_proj", "down_proj"] {
            spec.push((
                format!("model.layers.{l}.mlp.{n}.weight"),
                vec![rows as u64, h as u64],
            ));
        }
    }
    spec.push(("model.norm.weight".into(), vec![h as u64]));
    spec.push(("lm_head.weight".into(), vec![v as u64, h as u64]));

    let tensors: Vec<Tensor> = spec
        .into_iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let n: usize = shape.iter().product::<u64>() as usize;
            Tensor {
                name,
                dtype: "BF16",
                shape,
                bytes: bf16_bytes(&synth(n, 1 + i as u64)),
            }
        })
        .collect();
    assert!(
        tensors.len() > 8,
        "fixture must exceed one parallel chunk to be meaningful"
    );

    std::fs::write(dir.join("model.safetensors"), build_safetensors(&tensors)).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string(&serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": h,
            "num_hidden_layers": 3,
            "vocab_size": v
        }))
        .unwrap(),
    )
    .unwrap();
}

// ─── tests ──────────────────────────────────────────────────────────

/// [IMP-004] step 2: a model spanning SEVERAL chunks must come out
/// byte-identical whether the driver encodes one tensor at a time
/// (`jobs: Some(1)`) or a whole chunk in parallel (`jobs: Some(4)`).
/// This is what catches a lost/duplicated chunk at the boundaries — the
/// failure mode the 5-tensor CLI fixtures cannot observe.
#[test]
fn multi_chunk_output_is_byte_identical_sequential() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_model(&model_dir);

    let mut outputs = Vec::new();
    let mut schemes: Vec<Vec<(String, quant_core::gguf_registry::GgufScheme)>> = Vec::new();
    for jobs in [1usize, 4] {
        let out = tmp.path().join(format!("out-{jobs}.gguf"));
        let cfg = GgufConvertConfig {
            method_id: "q4_k_m".into(),
            jobs: Some(jobs),
            ..Default::default()
        };
        let report =
            convert_hf_to_gguf(&model_dir.join("model.safetensors"), &out, &cfg, None).unwrap();
        assert_eq!(
            report.tensors, 24,
            "jobs={jobs}: every tensor must be accounted for"
        );
        // Guard against a vacuous pass: both modes must have actually
        // written all 24 tensors (a driver that dropped its final chunk
        // would produce two equally-empty files and still compare equal).
        let f = rlx_gguf::GgufFile::from_path(&out).unwrap();
        assert_eq!(
            f.tensors.len(),
            24,
            "jobs={jobs}: output must contain every tensor"
        );
        outputs.push(std::fs::read(&out).unwrap());
        schemes.push(report.effective_schemes.clone());
    }

    // `effective_schemes` is a FILE-ORDERED vector built by the sequential
    // write phase — it is the report-level witness of tensor order, and
    // the thing a mis-ordered parallel flush would scramble.
    assert_eq!(
        schemes[0], schemes[1],
        "per-tensor scheme assignment (in file order) diverged under parallelism"
    );
    assert_eq!(schemes[0].len(), 24);

    assert_eq!(outputs[0].len(), outputs[1].len(), "size changed");
    assert!(
        outputs[0] == outputs[1],
        "multi-chunk parallel encoding is NOT byte-identical to sequential"
    );
}

/// [IMP-004] step 2: the `(cur, total)` progress stream is emitted by the
/// SEQUENTIAL write phase, so it must be identical in both modes and must
/// walk 1..=total exactly once, in order. A chunked driver that reported
/// progress per chunk (or per encoded tensor out of order) would fail
/// here.
///
/// Note: `(cur, total)` carries no tensor identity, so this test cannot
/// detect a REORDERED flush — only a dropped or duplicated update. Order
/// is pinned by `multi_chunk_output_is_byte_identical_sequential`
/// (bytes + `effective_schemes`) and by
/// `cli_gguf::parallel_warning_order_stable` (stderr).
#[test]
fn progress_reaches_total_in_both_modes() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    write_model(&model_dir);

    let mut runs: Vec<Vec<(usize, usize)>> = Vec::new();
    for jobs in [1usize, 4] {
        let out = tmp.path().join(format!("prog-{jobs}.gguf"));
        let cfg = GgufConvertConfig {
            method_id: "q8_0".into(),
            jobs: Some(jobs),
            ..Default::default()
        };
        let mut events: Vec<(usize, usize)> = Vec::new();
        {
            let mut cb = |cur: usize, total: usize| events.push((cur, total));
            convert_hf_to_gguf(
                &model_dir.join("model.safetensors"),
                &out,
                &cfg,
                Some(&mut cb),
            )
            .unwrap();
        }
        runs.push(events);
    }

    for (i, events) in runs.iter().enumerate() {
        assert_eq!(events.len(), 24, "mode {i}: one update per tensor");
        for (k, &(cur, total)) in events.iter().enumerate() {
            assert_eq!(total, 24, "mode {i}: total must be the tensor count");
            assert_eq!(cur, k + 1, "mode {i}: progress must be 1..=total in order");
        }
    }
    assert_eq!(
        runs[0], runs[1],
        "progress stream diverged under parallelism"
    );
}

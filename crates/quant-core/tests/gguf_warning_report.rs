//! Warning-report contract (warning-spam fix): quant-core no longer
//! PRINTS per-tensor warnings — it collects them into
//! [`GgufConvertReport::warnings`] in file order (per-tensor pairs stay
//! adjacent: the resolve-phase "not divisible by" line precedes the
//! encode-phase "fell back to F16" line for the same tensor), fires the
//! optional `on_warning` callback with the SAME stream, and produces a
//! byte-identical list for every `jobs` setting.
//!
//! The old behaviour had two defects this pins shut:
//!  1. Warnings went straight to stderr, re-breaking the CLI's indicatif
//!     progress bar (the user-visible bug).
//!  2. The interleaved stderr stream DIFFERED between jobs modes: chunked
//!     Phase A printed all resolve-warns of a chunk before Phase C printed
//!     its encode-warns, so JOBS=1 alternated r0,e0,r1,e1 while JOBS=8
//!     batched r0..r7,e0..e7. The report list is per-tensor-paired in file
//!     order regardless of chunking.

use quant_core::gguf_convert::{convert_hf_to_gguf, GgufConvertConfig};

/// Deterministic BF16 ramp bytes for `n` elements.
fn bf16_ramp(n: usize, seed: u32) -> Vec<u8> {
    let mut b = Vec::with_capacity(n * 2);
    for j in 0..n {
        let v = ((j + seed as usize) as f32 * 0.25) - 8.0;
        let bf = ((v.to_bits() >> 16) as u16).to_le_bytes(); // truncate
        b.extend_from_slice(&bf);
    }
    b
}

/// Build a single-file model with one odd-shaped conv tensor per entry of
/// `specs` (name, kernel) plus one aligned 1-D norm. Hand-built
/// safetensors header keeps INSERTION order (serde_json preserve_order —
/// see project memory: a `save_file` fixture cannot discriminate orders).
fn write_model(dir: &std::path::Path, specs: &[(String, u64)]) {
    std::fs::create_dir_all(dir).unwrap();
    let mut entries: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, k) in specs {
        entries.push((
            name.clone(),
            vec![32, 1, *k as usize],
            bf16_ramp((32 * *k) as usize, 1),
        ));
    }
    entries.push(("model.norm.weight".into(), vec![32], bf16_ramp(32, 7)));

    // Header JSON: insertion order, bf16 dtype.
    let mut json = String::from("{");
    let mut offset: u64 = 0;
    for (i, (name, shape, bytes)) in entries.iter().enumerate() {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        json.push_str(&format!(
            "{}\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{offset},{}]}}",
            if i == 0 { "" } else { "," },
            dims.join(","),
            offset + bytes.len() as u64
        ));
        offset += bytes.len() as u64;
    }
    json.push('}');
    let json_bytes = json.as_bytes();
    let pad = (8 - (json_bytes.len() as u64) % 8) % 8;
    let mut file = Vec::new();
    file.extend_from_slice(&((json_bytes.len() as u64) + pad).to_le_bytes());
    file.extend_from_slice(json_bytes);
    file.extend(std::iter::repeat_n(b' ', pad as usize));
    for (_, _, bytes) in &entries {
        file.extend_from_slice(bytes);
    }
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string(&serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "vocab_size": 32
        }))
        .unwrap(),
    )
    .unwrap();
}

/// Standard llama projections (mapping pinned by the existing CLI stderr
/// tests): layer i's q/k/v map to blk.i.attn_q/attn_k/attn_v.
fn conv(layer: u64, proj: &str, k: u64) -> (String, u64) {
    (format!("model.layers.{layer}.self_attn.{proj}.weight"), k)
}

/// Report lists ALL warnings in file order: per-tensor pairs
/// (resolve-warn then encode-warn), adjacent, tensor by tensor.
#[test]
fn warnings_report_lists_all_warnings_in_file_order() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("m");
    write_model(
        &dir,
        &[
            conv(0, "q_proj", 7),
            conv(0, "k_proj", 4),
            conv(0, "v_proj", 10),
        ],
    );

    let out = tmp.path().join("o.gguf");
    let cfg = GgufConvertConfig {
        method_id: "q8_0".into(),
        ..Default::default()
    };
    let report =
        convert_hf_to_gguf(&dir.join("model.safetensors"), &out, &cfg, None, None).unwrap();

    // 3 tensors × (1 resolve-warn + 1 encode-warn) = 6 lines.
    assert_eq!(report.warnings.len(), 6, "{:?}", report.warnings);
    let row: Vec<&String> = report
        .warnings
        .iter()
        .filter(|w| w.contains("not divisible by"))
        .collect();
    let f16: Vec<&String> = report
        .warnings
        .iter()
        .filter(|w| w.contains("fell back to F16"))
        .collect();
    assert_eq!(row.len(), 3, "{:?}", report.warnings);
    assert_eq!(f16.len(), 3, "{:?}", report.warnings);
    // File order within each kind.
    for (i, name) in ["attn_q", "attn_k", "attn_v"].iter().enumerate() {
        assert!(
            row[i].contains(&format!("blk.0.{name}.weight")),
            "{}",
            row[i]
        );
        assert!(
            f16[i].contains(&format!("blk.0.{name}.weight")),
            "{}",
            f16[i]
        );
    }
    // Pair adjacency: each tensor's resolve-warn comes IMMEDIATELY before
    // its encode-warn (positions 0/1, 2/3, 4/5).
    for i in 0..3 {
        let r = &report.warnings[2 * i];
        let e = &report.warnings[2 * i + 1];
        assert!(r.contains("not divisible by"), "{r}");
        assert!(e.contains("fell back to F16"), "{e}");
        let name = ["attn_q", "attn_k", "attn_v"][i];
        assert!(
            r.contains(&format!("blk.0.{name}.weight"))
                && e.contains(&format!("blk.0.{name}.weight")),
            "pair {i} not adjacent: {r} / {e}"
        );
    }
}

/// The report stream is IDENTICAL across jobs settings — the fix for the
/// pre-existing chunked-stderr divergence (JOBS=1 alternated per-tensor,
/// JOBS=8 batched per-chunk).
#[test]
fn warnings_report_identical_across_jobs() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("m");
    // 10 odd-kernel convs across 10 layers + norm = 11 tensors: with
    // JOBS=8 that is 1 full chunk + a 3-tensor flush chunk.
    let specs: Vec<(String, u64)> = (0..10u64)
        .map(|i| conv(i, "q_proj", [7, 4, 10, 9, 11, 5, 13, 6, 17, 19][i as usize]))
        .collect();
    write_model(&dir, &specs);

    let mut runs: Vec<Vec<String>> = Vec::new();
    for jobs in [1usize, 4, 8] {
        let out = tmp.path().join(format!("w-{jobs}.gguf"));
        let cfg = GgufConvertConfig {
            method_id: "q8_0".into(),
            jobs: Some(jobs),
            ..Default::default()
        };
        let report =
            convert_hf_to_gguf(&dir.join("model.safetensors"), &out, &cfg, None, None).unwrap();
        runs.push(report.warnings.clone());
    }
    assert_eq!(runs[0].len(), 20, "10 tensors × 2 warnings");
    assert_eq!(runs[0], runs[1], "JOBS=1 vs 4 warning streams diverged");
    assert_eq!(runs[0], runs[2], "JOBS=1 vs 8 warning streams diverged");
}

/// The `on_warning` callback receives the SAME lines as the report, in
/// the same order, without stderr printing.
#[test]
fn on_warning_callback_mirrors_report_stream() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("m");
    write_model(
        &dir,
        &[
            conv(0, "q_proj", 7),
            conv(0, "k_proj", 4),
            conv(0, "v_proj", 10),
        ],
    );

    let out = tmp.path().join("o.gguf");
    let cfg = GgufConvertConfig {
        method_id: "q8_0".into(),
        ..Default::default()
    };
    let mut seen: Vec<String> = Vec::new();
    let report = {
        let mut wb = |line: &str| seen.push(line.to_string());
        convert_hf_to_gguf(
            &dir.join("model.safetensors"),
            &out,
            &cfg,
            None,
            Some(&mut wb),
        )
        .unwrap()
    };
    assert_eq!(seen, report.warnings, "callback stream != report stream");
}

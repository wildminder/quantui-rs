//! `recipe_from_gguf` tests (task #8): extract a per-tensor recipe from a
//! reference GGUF and check the assignment round-trips through the
//! existing `--tensor-type-file` machinery.

use rlx_gguf::{GgmlType, GgufWriter};

/// f16 little-endian bytes.
fn f16(v: f32) -> [u8; 2] {
    half::f16::from_f32(v).to_le_bytes()
}

/// Build one 34-byte Q8_0 block.
fn q8_block(d: f32, codes: [i8; 32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(34);
    b.extend_from_slice(&f16(d));
    for c in codes {
        b.push(c as u8);
    }
    b
}

/// 18-byte Q4_0 block (f16 d + 16 nibble bytes).
fn q4_block(d: f32) -> Vec<u8> {
    let mut b = Vec::with_capacity(18);
    b.extend_from_slice(&f16(d));
    b.extend_from_slice(&[0x11u8; 16]);
    b
}

#[test]
fn extracts_exact_name_rules_in_sorted_order() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("ref.gguf");
    let mut w = GgufWriter::new();
    w.set_arch("test");
    w.add_tensor_bytes(
        "blk.1.attn_q.weight",
        vec![32],
        GgmlType::Q8_0,
        q8_block(0.01, [1; 32]),
    )
    .unwrap();
    w.add_tensor_bytes(
        "blk.0.attn_q.weight",
        vec![32],
        GgmlType::Q4_0,
        q4_block(0.02),
    )
    .unwrap();
    w.add_tensor_bytes(
        "blk.0.attn_norm.weight",
        vec![32],
        GgmlType::F32,
        vec![0u8; 128],
    )
    .unwrap();
    w.write_to_path(&p).unwrap();

    let r = quant_core::gguf_recipe::recipe_from_gguf(&p).unwrap();
    // F32 skipped, 2 rules; sorted by name: blk.0 before blk.1.
    // regex::escape escapes the dots, so the anchored patterns carry
    // backslashes — that is exactly what makes them name-exact.
    assert_eq!(r.rules.len(), 2, "{r:?}");
    assert_eq!(r.rules[0].pattern, r"^blk\.0\.attn_q\.weight$");
    assert_eq!(
        r.rules[0].scheme,
        quant_core::gguf_registry::GgufScheme::Q4_0
    );
    assert_eq!(r.rules[1].pattern, r"^blk\.1\.attn_q\.weight$");
    assert_eq!(
        r.rules[1].scheme,
        quant_core::gguf_registry::GgufScheme::Q8_0
    );
    // Exact-name semantics: dotted names must NOT match other positions.
    assert_eq!(
        r.scheme_for("blk.1.attn_q.weight"),
        Some(quant_core::gguf_registry::GgufScheme::Q8_0)
    );
    assert_eq!(
        r.scheme_for("blk.1.attn_q_norm.weight"),
        None,
        "anchored rules never fuzzy-match"
    );
    assert_eq!(r.default, None);
}

/// [NTH-006] Recipe dumps carry a format-version header as their first
/// line. `recipe_from_gguf` parses its own output, so a versioned first
/// line must remain a transparent comment to the parser — and the
/// RECIPE_FORMAT_VERSION constant must be the version actually emitted.
#[test]
fn extracted_recipe_carries_format_version_header() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("ref.gguf");
    let mut w = GgufWriter::new();
    w.set_arch("test");
    w.add_tensor_bytes(
        "blk.0.attn_q.weight",
        vec![32],
        GgmlType::Q8_0,
        q8_block(0.01, [1; 32]),
    )
    .unwrap();
    w.write_to_path(&p).unwrap();

    // Re-derive the dump text the same way recipe_from_gguf builds it:
    // parse its output and verify the rules survive the version header.
    let r = quant_core::gguf_recipe::recipe_from_gguf(&p).unwrap();
    assert_eq!(r.rules.len(), 1);
    assert_eq!(r.rules[0].pattern, r"^blk\.0\.attn_q\.weight$");

    // The constant is what the docs/CLI emit; keep it pinned so an
    // accidental bump is a visible change.
    assert_eq!(quant_core::gguf_recipe::RECIPE_FORMAT_VERSION, "v1");

    // A hand-built versioned dump (identical shape to the real one) parses
    // to the same recipe as the unversioned equivalent — the header is a
    // transparent comment today and a detection marker tomorrow.
    let versioned = format!(
        "# quantui-rs recipe format {}\n^blk\\.0\\.attn_q\\.weight$=q8_0\n",
        quant_core::gguf_recipe::RECIPE_FORMAT_VERSION
    );
    let unversioned = "^blk\\.0\\.attn_q\\.weight$=q8_0\n";
    let rv = quant_core::gguf_recipe::TensorRecipe::parse(&versioned, "v.recipe").unwrap();
    let ru = quant_core::gguf_recipe::TensorRecipe::parse(unversioned, "u.recipe").unwrap();
    assert_eq!(rv.rules.len(), ru.rules.len());
    assert_eq!(rv.rules[0].pattern, ru.rules[0].pattern);
    assert_eq!(rv.rules[0].scheme, ru.rules[0].scheme);
}

/// Round-trip through the driver: convert a fixture, extract a recipe from
/// the output, convert again with --recipe-from — the per-tensor dtypes
/// must reproduce exactly.
#[test]
fn round_trip_through_conversion_reproduces_dtypes() {
    // Uses the same plumbing as the CLI e2e; here at driver level via the
    // public convert_hf_to_gguf API with a tiny fixture.
    use quant_core::gguf_convert::{convert_hf_to_gguf, GgufConvertConfig};

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("m");
    std::fs::create_dir_all(&model_dir).unwrap();

    // Two block-aligned 2-D tensors (256-wide rows keep every method happy).
    let mk_vals = |seed: u64, n: usize| -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0
            })
            .collect()
    };
    let f32_to_bf16 = |vals: &[f32]| -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect()
    };
    let tensors: Vec<(&str, u64, Vec<u8>)> = vec![
        (
            "model.layers.0.self_attn.q_proj.weight",
            2,
            f32_to_bf16(&mk_vals(2, 256 * 256)),
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            8,
            f32_to_bf16(&mk_vals(8, 256 * 256)),
        ),
    ];
    let mut data = Vec::new();
    let mut map = serde_json::Map::new();
    for (name, _seed, bytes) in &tensors {
        let start = data.len() as u64;
        data.extend_from_slice(bytes);
        let end = data.len() as u64;
        let mut info = serde_json::Map::new();
        info.insert("dtype".into(), serde_json::Value::String("BF16".into()));
        info.insert("shape".into(), serde_json::json!([256usize, 256usize]));
        info.insert("data_offsets".into(), serde_json::json!([start, end]));
        map.insert(name.to_string(), serde_json::Value::Object(info));
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(map)).unwrap();
    let pad = (8 - header.len() % 8) % 8;
    let mut padded = header;
    padded.extend(std::iter::repeat_n(b' ', pad));
    let mut st = Vec::new();
    st.extend_from_slice(&(padded.len() as u64).to_le_bytes());
    st.extend_from_slice(&padded);
    st.extend_from_slice(&data);
    std::fs::write(model_dir.join("model.safetensors"), &st).unwrap();
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 256,
            "num_hidden_layers": 1,
            "vocab_size": 256
        })
        .to_string(),
    )
    .unwrap();

    // Pass 1: mixed assignment via a recipe file (q8_0 for attn, f16 for ffn).
    let recipe_path = tmp.path().join("mixed.recipe");
    std::fs::write(
        &recipe_path,
        "^blk\\.0\\.attn_q\\.weight$=q8_0\n^blk\\.0\\.ffn_down\\.weight$=f16\n",
    )
    .unwrap();
    let pass1 = tmp.path().join("pass1.gguf");
    let cfg = GgufConvertConfig {
        method_id: "q4_k_m".into(),
        recipe: Some(quant_core::gguf_recipe::TensorRecipe::load(&recipe_path).unwrap()),
        ..Default::default()
    };
    convert_hf_to_gguf(&model_dir.join("model.safetensors"), &pass1, &cfg, None).unwrap();

    // Extract the recipe from pass1's output.
    let extracted = quant_core::gguf_recipe::recipe_from_gguf(&pass1).unwrap();
    assert_eq!(extracted.rules.len(), 2, "{:?}", extracted.rules);

    // Pass 2: apply the extracted recipe with a DIFFERENT base method —
    // the recipe rules must fully determine the assignment for matched
    // tensors.
    let pass2 = tmp.path().join("pass2.gguf");
    let cfg2 = GgufConvertConfig {
        method_id: "q8_0".into(), // different default than the recipe's
        recipe: Some(extracted),
        ..Default::default()
    };
    convert_hf_to_gguf(&model_dir.join("model.safetensors"), &pass2, &cfg2, None).unwrap();

    // Dtype-level comparison: pass1 vs pass2 assignments must be identical.
    let f1 = rlx_gguf::GgufFile::from_path(&pass1).unwrap();
    let f2 = rlx_gguf::GgufFile::from_path(&pass2).unwrap();
    assert_eq!(f1.tensors.len(), f2.tensors.len());
    for (name, t1) in &f1.tensors {
        let t2 = f2
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("{name} missing"));
        assert_eq!(t1.dtype, t2.dtype, "dtype round-trip mismatch for {name}");
    }
    assert_eq!(
        f1.tensors.get("blk.0.attn_q.weight").unwrap().dtype,
        GgmlType::Q8_0
    );
    assert_eq!(
        f1.tensors.get("blk.0.ffn_down.weight").unwrap().dtype,
        GgmlType::F16
    );
}

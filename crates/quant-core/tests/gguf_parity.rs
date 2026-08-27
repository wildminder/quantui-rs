//! Phase 10.4: parity spot-checks vs the Python gguf-py writer.
//!
//! Goldens in `tests/golden/gguf_quants/` were produced by
//! `tools/gen_golden_gguf.py` (gguf-py `gguf.quantize`) from deterministic
//! f32 vectors. This test re-quantizes the SAME f32 inputs with rlx-gguf and
//! asserts the encoded bytes are byte-identical.
//!
//! Parity contract (plan 10.1 spike): the legacy schemes F16/BF16/Q8_0/Q4_0/
//! Q4_1/Q5_0/Q5_1 are byte-identical between gguf-py and rlx-gguf (both mirror
//! llama.cpp `quantize_row_*`). K-quants/IQ* are deliberately excluded — rlx
//! uses a simpler per-sub-block min/max search than upstream's iterative
//! `make_qx_quants`, so those are valid-but-not-byte-identical.

use std::path::PathBuf;

use rlx_gguf::{quantize, GgmlType};

fn golden_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p.join("tests/golden/gguf_quants")
}

/// (case name, element count) — mirrors tools/gen_golden_gguf.py make_cases().
const CASES: &[(&str, usize)] = &[
    ("randn_256", 256),
    ("randn_1024", 1024),
    ("wide_range", 512),
    ("zeros", 256),
    ("tiny", 256),
    ("constant", 256),
    ("linspace", 512),
];

/// (format name in golden filename, GgmlType) — the legacy byte-parity set.
const FORMATS: &[(&str, GgmlType)] = &[
    ("F16", GgmlType::F16),
    ("BF16", GgmlType::BF16),
    ("Q8_0", GgmlType::Q8_0),
    ("Q4_0", GgmlType::Q4_0),
    ("Q4_1", GgmlType::Q4_1),
    ("Q5_0", GgmlType::Q5_0),
    ("Q5_1", GgmlType::Q5_1),
];

fn load_f32(case: &str) -> Vec<f32> {
    let path = golden_dir().join(format!("{case}.f32.bin"));
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(raw.len() % 4, 0);
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[test]
fn legacy_formats_byte_identical_to_gguf_py() {
    let dir = golden_dir();
    assert!(dir.is_dir(), "golden dir missing: {}", dir.display());

    let mut checked = 0usize;
    for (case, n) in CASES {
        let floats = load_f32(case);
        assert_eq!(floats.len(), *n, "{case}: element count");

        for (fmt, ggml) in FORMATS {
            let golden_path = dir.join(format!("{case}.{fmt}.bin"));
            let golden = std::fs::read(&golden_path)
                .unwrap_or_else(|e| panic!("read {}: {e}", golden_path.display()));

            let ours =
                quantize(&floats, *ggml).unwrap_or_else(|e| panic!("quantize {case}/{fmt}: {e}"));

            assert_eq!(
                ours.len(),
                golden.len(),
                "{case}/{fmt}: encoded size mismatch (ours {} vs gguf-py {})",
                ours.len(),
                golden.len()
            );
            assert_eq!(
                ours, golden,
                "{case}/{fmt}: encoded bytes differ from gguf-py"
            );
            checked += 1;
        }
    }
    // 7 cases x 7 formats = 49 byte-parity assertions.
    assert_eq!(checked, CASES.len() * FORMATS.len());
}

#[test]
fn manifest_matches_golden_files() {
    let dir = golden_dir();
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let cases = manifest["cases"].as_object().unwrap();
    assert_eq!(cases.len(), CASES.len());
    for (case, n) in CASES {
        let entry = &cases[*case];
        assert_eq!(entry["n"].as_u64().unwrap() as usize, *n, "{case}: n");
        for (fmt, _) in FORMATS {
            let golden = std::fs::read(dir.join(format!("{case}.{fmt}.bin"))).unwrap();
            assert_eq!(
                entry[*fmt]["bytes"].as_u64().unwrap() as usize,
                golden.len(),
                "{case}/{fmt}: manifest byte count"
            );
        }
    }
}

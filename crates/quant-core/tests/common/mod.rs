//! Shared helpers for the Phase 12 streaming payload-parity tests
//! (plan Phase C.2/C.4).
//!
//! The new-format goldens (`output_<fmt>.safetensors`) were produced by ctq's
//! WHOLE-FILE path, which writes tensors in its own processing order — not the
//! input file order our streaming orchestrator uses. Parity is therefore
//! defined per-tensor (plan §3.3 / §4.3): for every tensor in the golden we
//! byte-compare the payload and check `(dtype_raw, shape)` from the headers.
//! `__metadata__` is excluded here (it is emitted in Phase D).
//!
//! Each integration test binary compiles this module separately, so not every
//! helper is used by every consumer — allow dead code at the module level.
#![allow(dead_code)]

use std::path::PathBuf;

use quant_core::dtype::DType;
use quant_core::st_io::reader::SafetensorsReader;
use quant_core::st_io::writer::IncrementalWriter;

/// Locate `tests/golden/<name>` from a crate test.
pub fn golden(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.join("tests/golden").join(name)
}

/// Open a safetensors file for reading.
pub fn load(path: &std::path::Path) -> SafetensorsReader {
    SafetensorsReader::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
}

/// Assert that `ours` reproduces every tensor of `golden` byte-for-byte with
/// identical `(dtype_raw, shape)`, and that the two files carry the same set
/// of tensor names (ignoring `__metadata__`, which Phase D adds).
///
/// `label` is used only in panic messages.
pub fn assert_payload_parity(ours: &std::path::Path, golden: &std::path::Path, label: &str) {
    let g = load(golden);
    let o = load(ours);

    // Every golden tensor must be present in ours with identical dtype,
    // shape, and payload bytes.
    let mut golden_names: Vec<&String> = Vec::new();
    for (name, info) in g.header().iter() {
        if name == "__metadata__" {
            continue; // Phase D concern; not emitted by Phase C.
        }
        golden_names.push(name);

        let o_info = o
            .header()
            .get(name)
            .unwrap_or_else(|| panic!("{label}: missing tensor {name} in streamed output"));
        assert_eq!(
            o_info.dtype_raw, info.dtype_raw,
            "{label}/{name}: dtype mismatch (ours {} vs golden {})",
            o_info.dtype_raw, info.dtype_raw
        );
        assert_eq!(
            o_info.shape, info.shape,
            "{label}/{name}: shape mismatch (ours {:?} vs golden {:?})",
            o_info.shape, info.shape
        );

        let g_bytes = g.tensor_bytes(name).unwrap();
        let o_bytes = o
            .tensor_bytes(name)
            .unwrap_or_else(|_| panic!("{label}: cannot read {name} from streamed output"));
        assert_eq!(
            o_bytes.len(),
            g_bytes.len(),
            "{label}/{name}: payload length mismatch (ours {} vs golden {})",
            o_bytes.len(),
            g_bytes.len()
        );
        assert!(
            o_bytes == g_bytes,
            "{label}/{name}: payload bytes differ (first diff at index {:?})",
            o_bytes.iter().zip(g_bytes.iter()).position(|(a, b)| a != b)
        );
    }

    // No extra tensors in ours (set equality, ignoring __metadata__).
    let mut our_names: Vec<&String> = o
        .header()
        .iter()
        .filter(|(n, _)| n.as_str() != "__metadata__")
        .map(|(n, _)| n)
        .collect();
    let mut golden_sorted = golden_names.clone();
    golden_sorted.sort();
    our_names.sort();
    assert_eq!(
        our_names, golden_sorted,
        "{label}: tensor name sets differ (ours vs golden)"
    );
}

/// Write a minimal single-file safetensors input from `(name, dtype, shape,
/// bytes)` tuples, in the given order. Used to build synthetic inputs for the
/// skip/heuristic/divisibility policy tests (plan C.3).
pub fn write_input(path: &std::path::Path, tensors: &[(&str, DType, Vec<u64>, Vec<u8>)]) {
    let mut w = IncrementalWriter::open_new(path).expect("open writer");
    for (name, dtype, shape, bytes) in tensors {
        w.add_tensor(name, *dtype, None, shape, bytes)
            .expect("add tensor");
    }
    w.finalize().expect("finalize");
}

/// Deterministic non-zero bf16 payload for an `[m, n]` weight.
pub fn bf16_weight_bytes(m: usize, n: usize) -> Vec<u8> {
    (0..m * n)
        .flat_map(|i| {
            let v = ((i as f32) * 0.0013).sin() * 0.5;
            quant_core::dtype::f32_to_bf16_bits(v).to_le_bytes()
        })
        .collect()
}

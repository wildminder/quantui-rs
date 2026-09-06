//! WP6 / NTH-001 — adversarial property tests for the GGUF reader, the
//! imatrix loader, and the writer→reader round-trip (`rlx-gguf` + our
//! `Imatrix`).
//!
//!   * totality: arbitrary bytes / truncated valid files → `Result`,
//!     never a panic (a fuzzed parser that panics is an attacker's DoS);
//!   * round-trip: any tensor set the writer accepts comes back with
//!     identical names / dtypes / shapes through `GgufFile::from_path`.

use proptest::prelude::*;
use rlx_gguf::GgmlType;
use std::io::Write as _;

/// Write bytes to a unique temp file and return its path (dropped when
/// the TempDir goes out of scope).
fn temp_file(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let p = dir.path().join(name);
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(bytes).unwrap();
    p
}

/// Arbitrary bytes → `GgufFile::from_reader` must return a Result, not
/// panic. Random garbage exercises the header paths; the truncated-valid
/// property below covers structured near-misses.
#[test]
fn gguf_parse_never_panics_on_arbitrary_bytes() {
    proptest!(|(blob in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..512))| {
        let mut cursor = std::io::Cursor::new(blob);
        let _ = rlx_gguf::GgufFile::from_reader(&mut cursor);
    });
}

/// Every prefix of a VALID minimal GGUF file — the structured adversarial
/// input random bytes rarely approximate. All prefixes parse to `Ok` or
/// `Err`; none may panic.
#[test]
fn gguf_truncated_valid_prefixes_never_panic() {
    // Build one valid 1-tensor F32 GGUF.
    let tmp = tempfile::TempDir::new().unwrap();
    let path = temp_file(&tmp, "valid.gguf", &[]);
    let mut w = rlx_gguf::GgufWriter::new();
    w.set_arch("llama");
    let elems = 32usize;
    let bytes: Vec<u8> = [0f32; 32].iter().flat_map(|f| f.to_le_bytes()).collect();
    w.add_tensor_bytes("blk.0.attn_q.weight", vec![elems], GgmlType::F32, bytes)
        .unwrap();
    w.write_to_path(&path).unwrap();
    let valid = std::fs::read(&path).unwrap();
    assert!(valid.len() > 100, "fixture must be non-trivial");

    proptest!(|(cut in 0usize..(valid.len() + 8))| {
        let mut cursor = std::io::Cursor::new(valid[..cut.min(valid.len())].to_vec());
        let _ = rlx_gguf::GgufFile::from_reader(&mut cursor);
    });
}

/// Arbitrary bytes → `Imatrix::load` → Result, never panic. The loader
/// must auto-detect GGUF vs legacy binary and reject everything else
/// cleanly.
#[test]
fn imatrix_load_never_panics_on_arbitrary_bytes() {
    proptest!(|(blob in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..512))| {
        let tmp = tempfile::TempDir::new().unwrap();
        let p = temp_file(&tmp, "im.dat", &blob);
        let _ = quant_core::imatrix::Imatrix::load(&p);
    });
}

/// Names from the GGUF-safe alphabet (printable ASCII, no NUL); shapes
/// block-aligned where the dtype needs it; byte counts computed with the
/// same rule the writer validates against (`bytes_for_public`).
fn tensor_spec_strategy() -> impl Strategy<Value = (String, GgmlType, Vec<usize>)> {
    (
        "[A-Za-z0-9_.]{1,64}",
        proptest::sample::select(vec![GgmlType::F32, GgmlType::F16, GgmlType::Q8_0]),
        proptest::collection::vec(1usize..=64, 1..=3),
    )
        .prop_filter(
            "name must be unique-candidate (dedupe below)",
            |(n, _, _)| !n.is_empty(),
        )
        .prop_map(|(name, dtype, mut shape)| {
            // Element count must satisfy the dtype's block rule: pad the
            // FLAT count up to the block size by growing the LAST dim
            // (GGML order innermost-first is what the writer sees).
            let blck = match dtype {
                GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => 1,
                GgmlType::Q8_0 => 32,
                _ => 256,
            };
            let n: usize = shape.iter().product();
            let rem = n % blck;
            if rem != 0 {
                *shape.last_mut().unwrap() += blck - rem;
            }
            (name, dtype, shape)
        })
        .prop_filter(
            "shape must satisfy the dtype block rule",
            |(_, dtype, shape)| {
                let n: usize = shape.iter().product();
                let blck = match dtype {
                    GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => 1,
                    GgmlType::Q8_0 => 32,
                    _ => 256,
                };
                // The writer rejects a FLAT count that is not a multiple of
                // the block size (`bytes_for`: `n % qk != 0 -> None`) — the
                // generator's pad-the-last-dim trick fixes the count but the
                // filter below guarantees it before bytes_for is consulted.
                n > 0 && n <= 64 * 64 * 64 && n % blck == 0
            },
        )
}

fn bytes_for(dtype: GgmlType, n: usize) -> usize {
    match dtype {
        GgmlType::F32 => n * 4,
        GgmlType::F16 | GgmlType::BF16 => n * 2,
        // Q8_0: f16 scale + 32 i8 per block (rlx-gguf bytes_for).
        GgmlType::Q8_0 => (n / 32) * (2 + 32),
        _ => unreachable!("strategy only generates F32/F16/Q8_0"),
    }
}

/// Writer → reader round-trip: names, dtypes and shapes survive
/// byte-identically for ARBITRARY valid tensor sets (not just the fixed
/// fixtures the phase tests use).
#[test]
fn writer_reader_round_trip() {
    proptest!(|(specs in proptest::collection::vec(tensor_spec_strategy(), 1..=8))| {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("rt.gguf");
        let mut w = rlx_gguf::GgufWriter::new();
        w.set_arch("llama");
        // Dedupe names (a GGUF table cannot hold duplicates).
        let mut seen = std::collections::HashSet::new();
        let mut payload_seed = 7u8;
        for (name, dtype, shape) in &specs {
            if !seen.insert(name.clone()) {
                continue;
            }
            let n: usize = shape.iter().product();
            let bytes = vec![payload_seed; bytes_for(*dtype, n)];
            payload_seed = payload_seed.wrapping_add(31);
            w.add_tensor_bytes(name.clone(), shape.clone(), *dtype, bytes).unwrap();
        }
        w.write_to_path(&path).unwrap();

        let f = rlx_gguf::GgufFile::from_path(&path).expect("own output must re-parse");
        prop_assert_eq!(seen.len(), f.tensors.len());
        for (name, dtype, shape) in &specs {
            if !seen.contains(name) {
                continue;
            }
            let t = f.tensors.get(name).unwrap_or_else(|| panic!("{name} missing"));
            prop_assert_eq!(*dtype, t.dtype, "{} dtype", name);
            // rlx-gguf keeps shape VERBATIM (GGML order innermost first)
            // and our writer wrote it in the same order.
            prop_assert_eq!(shape.as_slice(), t.shape.as_slice(), "{} shape", name);
        }
    });
}

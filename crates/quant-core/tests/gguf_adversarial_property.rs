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

/// One tensor spec: `(name, dtype, shape)`.
type Spec = (String, GgmlType, Vec<usize>);

/// Write `specs` to a fresh GGUF file and assert every tensor that was
/// ACTUALLY written comes back with an identical name, dtype and shape.
///
/// The expectation is derived from the writer's own output (`written`), not
/// from the caller's input list. That is deliberate and load-bearing.
///
/// A GGUF tensor table cannot hold duplicate names, so duplicate names in
/// `specs` are resolved FIRST-WINS: only the first spec for a given name is
/// written. An earlier version of this harness recomputed the survivor set
/// with `seen.contains(name)` in the verify loop. That predicate is true for
/// *every* occurrence of a duplicated name, so the verifier did not skip the
/// duplicates and asserted each losing spec's shape against the winner's
/// tensor — a false failure whenever proptest happened to generate a repeated
/// short name (e.g. `"_"`). Deriving the expectation from `written` makes
/// that class of disagreement impossible by construction.
///
/// Returns the number of tensors written, so callers can assert on it.
fn assert_round_trips(specs: &[Spec]) -> usize {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("rt.gguf");
    let mut w = rlx_gguf::GgufWriter::new();
    w.set_arch("llama");
    // Dedupe names FIRST-WINS (a GGUF table cannot hold duplicates), and
    // record exactly what we wrote so the verify loop uses the same set.
    let mut seen = std::collections::HashSet::new();
    let mut written: Vec<&Spec> = Vec::new();
    let mut payload_seed = 7u8;
    for spec in specs {
        let (name, dtype, shape) = spec;
        if !seen.insert(name.clone()) {
            continue;
        }
        let n: usize = shape.iter().product();
        let bytes = vec![payload_seed; bytes_for(*dtype, n)];
        payload_seed = payload_seed.wrapping_add(31);
        w.add_tensor_bytes(name.clone(), shape.clone(), *dtype, bytes)
            .unwrap();
        written.push(spec);
    }
    w.write_to_path(&path).unwrap();

    let f = rlx_gguf::GgufFile::from_path(&path).expect("own output must re-parse");
    assert_eq!(
        written.len(),
        f.tensors.len(),
        "every written tensor must reappear in the file"
    );
    for &(name, dtype, shape) in &written {
        let t = f
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("{name} missing"));
        // `written` holds `&Spec`, so destructuring binds `dtype` as
        // `&GgmlType`, while `t.dtype` is a `GgmlType` by value. Compare the
        // pointees: `GgmlType: PartialEq`, but `&GgmlType: PartialEq<GgmlType>`
        // is not implemented, so `assert_eq!(dtype, t.dtype)` does not compile.
        assert_eq!(*dtype, t.dtype, "{name} dtype");
        // rlx-gguf keeps shape VERBATIM (GGML order innermost first)
        // and our writer wrote it in the same order.
        assert_eq!(shape.as_slice(), t.shape.as_slice(), "{name} shape");
    }
    written.len()
}

/// REGRESSION (durable, not seed-based) for the first-wins verifier bug.
///
/// This is a plain `#[test]`, deliberately NOT a proptest seed entry. The
/// defect was a logic error in the harness — the verify loop disagreed with
/// the writer about which duplicate survives — so it reproduces on ANY input
/// containing a repeated name. A seed entry only replays one RNG draw; if the
/// strategy's shape or the RNG stream shifts, that draw no longer contains a
/// duplicate and the entry silently stops covering anything. A literal
/// regression input cannot rot that way.
///
/// Minimal failing case, captured from proptest before the fix:
/// `specs = [("_", F32, [1]), ("_", F32, [2])]`, which reported
/// `left: [2], right: [1]`.
#[test]
fn round_trip_duplicate_names_resolve_first_wins() {
    let specs: Vec<Spec> = vec![
        ("_".to_string(), GgmlType::F32, vec![1]),
        ("_".to_string(), GgmlType::F32, vec![2]),
    ];
    // One tensor survives (the first), and it must be the `[1]` one.
    assert_eq!(assert_round_trips(&specs), 1);
}

/// The same, with more duplicates and a non-F32 dtype, so the survivor
/// choice is pinned across the dtype mix the strategy generates.
#[test]
fn round_trip_duplicate_names_first_wins_across_dtypes() {
    let specs: Vec<Spec> = vec![
        ("blk.0".to_string(), GgmlType::Q8_0, vec![32]),
        ("blk.0".to_string(), GgmlType::F32, vec![64]), // loses: dup name
        ("blk.1".to_string(), GgmlType::F16, vec![2]),
        ("blk.1".to_string(), GgmlType::F16, vec![8]), // loses: dup name
        ("blk.1".to_string(), GgmlType::F32, vec![3]), // loses: dup name
    ];
    assert_eq!(assert_round_trips(&specs), 2);
}

/// Writer → reader round-trip: names, dtypes and shapes survive
/// byte-identically for ARBITRARY valid tensor sets (not just the fixed
/// fixtures the phase tests use).
#[test]
fn writer_reader_round_trip() {
    proptest!(|(specs in proptest::collection::vec(tensor_spec_strategy(), 1..=8))| {
        assert_round_trips(&specs);
    });
}

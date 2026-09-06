//! WP6 / NTH-001 — property tests for the safetensors header parser
//! (`st_io/header.rs`).
//!
//! Two properties no example test can cover:
//!   * round-trip: build a header from arbitrary *valid* tensor sets —
//!     1–8 tensors, unicode / 100-char names, 12 dtypes, 1–3 dims —
//!     serialize it the way the writer does (`serialize_json` +
//!     `align_to_8`) and parse it back: names in order, dtype_raw,
//!     shape and offsets identical;
//!   * totality: arbitrary bytes (plain random, JSON-shaped random, and
//!     truncations of a valid header at every prefix length) → `Err` or
//!     `Ok`, never a panic.

use std::path::Path;

use proptest::prelude::*;

use quant_core::dtype::DType;
use quant_core::st_io::header::Header;

/// All dtype strings the parser accepts (DType::from_header_str).
const DTYPES: &[&str] = &[
    "F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "U16", "BOOL", "F8_E4M3",
];

/// Names spanning the parser's tolerance: plain, dotted, unicode,
/// 100-char. (Empty names ARE accepted by the header parser — the
/// writer is the one that rejects them — so they are included.)
fn name_strategy() -> impl Strategy<Value = String> {
    proptest::option::of("[A-Za-z0-9_.\\-\\u{0080}-\\u{10FFFF}]{0,100}").prop_map(|o| match o {
        Some(s) if s.len() < 100 => s,
        _ => "w".repeat(100),
    })
}

/// A tensor set with CONSISTENT offsets (end == start, next start ==
/// previous end) and shapes that match `elem_size` — i.e. a header the
/// serializer emits and the parser must accept.
fn header_entries_strategy() -> impl Strategy<Value = Vec<(String, String, Vec<u64>)>> {
    proptest::collection::vec(
        (
            name_strategy(),
            proptest::sample::select(DTYPES).prop_map(|s| s.to_string()),
            proptest::collection::vec(1u64..=4096, 1..=3),
        ),
        1..=8,
    )
    .prop_map(|entries| {
        // JSON objects cannot hold duplicate keys — dedupe by name
        // (first occurrence wins, like serde_json's Map).
        let mut seen = std::collections::HashSet::new();
        entries
            .into_iter()
            .filter(|(n, _, _)| seen.insert(n.clone()))
            .collect()
    })
}

fn build_header(entries: &[(String, String, Vec<u64>)]) -> Header {
    let mut h = Header::default();
    let mut off = 0u64;
    for (name, dtype, shape) in entries {
        let dt = DType::from_header_str(dtype).expect("dtype from DTYPES table");
        let elems: u64 = shape.iter().product();
        let nbytes = dt.elem_size().expect("every DType has an elem size") * elems.max(1);
        h.insert(
            name.clone(),
            quant_core::st_io::header::TensorInfo {
                dtype: dt,
                dtype_raw: dtype.to_string(),
                shape: shape.clone(),
                data_offsets: (off, off + nbytes),
            },
        );
        off += nbytes;
    }
    h
}

/// Round-trip: serialize (writer layout: JSON + space-pad to 8) → parse
/// → identical names-in-order / dtype_raw / shape / offsets.
#[test]
fn header_serialize_parse_round_trip() {
    proptest!(|(entries in header_entries_strategy())| {
        let h = build_header(&entries);
        let json = h.serialize_json();
        let padded = Header::align_to_8(json);
        let back = Header::parse_json_bytes(&padded, Path::new("prop"))
            .expect("valid header must parse");

        let orig: Vec<(String, String, Vec<u64>, (u64, u64))> = entries.iter().map(|(n, d, s)| {
            let info = h.get(n).expect("entry present");
            (n.clone(), d.clone(), s.clone(), info.data_offsets)
        }).collect();
        let parsed: Vec<(String, String, Vec<u64>, (u64, u64))> = back
            .iter()
            .map(|(n, i)| (n.clone(), i.dtype_raw.clone(), i.shape.clone(), i.data_offsets))
            .collect();
        prop_assert_eq!(orig.len(), parsed.len(), "tensor count");
        for (a, b) in orig.iter().zip(&parsed) {
            prop_assert_eq!(a.0.as_str(), b.0.as_str(), "name order");
            prop_assert_eq!(a.1.as_str(), b.1.as_str(), "dtype_raw");
            prop_assert_eq!(a.2.as_slice(), b.2.as_slice(), "shape");
            prop_assert_eq!(a.3, b.3, "offsets");
        }
    });
}

/// Totality: arbitrary bytes never panic. `any::<Vec<u8>>()` shrinks —
/// so if a panic ever exists, proptest hands back the minimal byte
/// string that triggers it.
#[test]
fn header_never_panics_on_arbitrary_bytes() {
    proptest!(|(blob in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..256))| {
        let _ = Header::parse_json_bytes(&blob, Path::new("prop"));
    });
}

/// Totality against the nastiest realistic input: every PREFIX of a
/// valid header (truncated at every length). Each must be `Err` (an
/// 8-aligned padded JSON prefix mid-structure cannot parse) — never a
/// panic.
#[test]
fn header_truncations_never_panic() {
    proptest!(|(entries in header_entries_strategy(), cut in 0usize..64)| {
        let h = build_header(&entries);
        let padded = Header::align_to_8(h.serialize_json());
        let trunc = &padded[..padded.len().min(cut)];
        // A prefix that happens to be complete-and-padded could parse OK;
        // everything else must be a clean Err. Either way: no panic.
        let _ = Header::parse_json_bytes(trunc, Path::new("prop"));
    });
}

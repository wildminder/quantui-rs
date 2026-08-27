//! Phase 4.3 parity tests: `align_header_to_8` round-trips vs golden headers.
//!
//! Two writer conventions exist in the goldens:
//!   * **ctq one-shot** (`safetensors.torch.save_file` → reference
//!     `write_safetensors`): minimal space padding — stripping trailing spaces
//!     and re-applying `align_header_to_8` reproduces the header byte-exactly.
//!   * **Reference incremental writer** (INT8 `output.safetensors`): fixed
//!     64 KiB pre-aligned slot — the minimal-padded header is a prefix of the
//!     slot, followed by spaces to the slot length.
//!
//! Both conventions 8-align the data section, and both round-trip through
//! `align_header_to_8` on the trimmed JSON.

use std::path::PathBuf;

use quant_core::st_io::{align_header_to_8, HEADER_ALIGN};

fn golden(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.join("tests/golden").join(name)
}

const CASES: [&str; 4] = ["linear_basic_bf16", "odd_shapes", "conv_net", "zero_blocks"];
const SUFFIXES: [&str; 6] = ["", "_fp8", "_fp8_tensor", "_fp8_row", "_mxfp8", "_nvfp4"];

/// Read `(header_len, raw_header_bytes)` from a safetensors file.
fn read_header(path: &std::path::Path) -> (u64, Vec<u8>) {
    use std::io::Read;
    let mut f = std::fs::File::open(path).expect("open golden");
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).expect("read len");
    let n = u64::from_le_bytes(len_buf);
    let mut raw = vec![0u8; n as usize];
    f.read_exact(&mut raw).expect("read header");
    (n, raw)
}

fn trim_trailing_spaces(raw: &[u8]) -> &[u8] {
    let mut end = raw.len();
    while end > 0 && raw[end - 1] == b' ' {
        end -= 1;
    }
    &raw[..end]
}

#[test]
fn golden_headers_are_8_aligned() {
    for case in CASES {
        let dir = golden(case);
        for suffix in SUFFIXES {
            let p = dir.join(format!("output{suffix}.safetensors"));
            if !p.exists() {
                continue;
            }
            let (n, _) = read_header(&p);
            assert_eq!(
                n % HEADER_ALIGN as u64,
                0,
                "{case}/output{suffix}: header length {n} not 8-aligned"
            );
        }
    }
}

#[test]
fn align_round_trip_vs_golden_headers() {
    for case in CASES {
        let dir = golden(case);
        for suffix in SUFFIXES {
            let p = dir.join(format!("output{suffix}.safetensors"));
            if !p.exists() {
                continue;
            }
            let (_, raw) = read_header(&p);
            let trimmed = trim_trailing_spaces(&raw);
            let aligned = align_header_to_8(trimmed);
            assert_eq!(
                aligned.len() % HEADER_ALIGN,
                0,
                "{case}/output{suffix}: aligned length not 8-aligned"
            );
            if raw.len() == aligned.len() {
                // ctq one-shot convention: minimal padding, byte-exact match.
                assert_eq!(
                    aligned, raw,
                    "{case}/output{suffix}: minimal-padded header mismatch"
                );
            } else {
                // Fixed-slot convention (reference incremental writer): the
                // minimal-padded header is a prefix of the slot.
                assert!(
                    raw.starts_with(&aligned),
                    "{case}/output{suffix}: slot does not start with minimal-padded header"
                );
                assert!(
                    raw[aligned.len()..].iter().all(|&b| b == b' '),
                    "{case}/output{suffix}: slot tail is not space padding"
                );
            }
        }
    }
}

#[test]
fn align_header_to_8_unit_semantics() {
    // Verbatim reference semantics: pad = (8 - len % 8) % 8, spaces only.
    for len in 0..=16usize {
        let hdr = vec![b'x'; len];
        let out = align_header_to_8(&hdr);
        assert_eq!(out.len() % 8, 0);
        assert!(out.starts_with(&hdr));
        assert!(out[len..].iter().all(|&b| b == b' '));
        // Already-aligned input is unchanged (no extra pad block).
        if len % 8 == 0 {
            assert_eq!(out.len(), len);
        }
    }
}

#[test]
fn aligned_header_still_parses() {
    // The aligned (space-padded) header must still parse via the reader.
    use quant_core::st_io::reader::SafetensorsReader;
    let p = golden("linear_basic_bf16").join("output_fp8.safetensors");
    let r = SafetensorsReader::open(&p).expect("open");
    assert!(r.header().iter().any(|(n, _)| n.ends_with(".comfy_quant")));
}

//! Phase 3.1 parity test: Rust E4M3 cast vs torch `.to(float8_e4m3fn)`.
//!
//! Fixture produced by `tools/probe_fp8_cast.py` (torch 2.13.0+cpu):
//!   tests/golden/fp8_cast/fp8_cast_inputs.bin   f32 LE cast inputs
//!   tests/golden/fp8_cast/fp8_cast_outputs.bin  u8 torch output bytes
//!
//! Coverage: 1793 full 128-vectors (randn / uniform / log-uniform spanning
//! 2^-150..2^10) + subnormal-grid + explicit edges (±448/±464/±480, ±inf,
//! ±NaN, ±0, 2^-6, 2^-9, RNE ties). All 256 output codes appear in the
//! fixture. Plan §3.1 criterion: bit-exact on ≥1000 random vectors incl.
//! subnormals/infs — satisfied (229,614 values).

use std::fs;
use std::path::PathBuf;

use quant_core::dtype::{f32_to_fp8_e4m3_bits, fp8_e4m3_bits_to_f32};

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.join("tests/golden/fp8_cast").join(name)
}

#[test]
fn e4m3_cast_bit_exact_vs_torch() {
    let in_bytes = fs::read(fixture("fp8_cast_inputs.bin")).expect("read inputs fixture");
    let out_bytes = fs::read(fixture("fp8_cast_outputs.bin")).expect("read outputs fixture");

    assert_eq!(in_bytes.len() % 4, 0);
    let n = in_bytes.len() / 4;
    assert_eq!(out_bytes.len(), n, "fixture size mismatch");
    assert!(n / 128 >= 1000, "plan requires >=1000 random vectors");

    let mut mismatches = 0usize;
    let mut first_bad: Option<(usize, f32, u8, u8)> = None;
    for i in 0..n {
        let v = f32::from_le_bytes(in_bytes[i * 4..i * 4 + 4].try_into().unwrap());
        let got = f32_to_fp8_e4m3_bits(v);
        let want = out_bytes[i];
        if got != want {
            mismatches += 1;
            if first_bad.is_none() {
                first_bad = Some((i, v, got, want));
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "E4M3 cast mismatch on {mismatches}/{n} values; first: idx={:?} value={:?} rust={:?} torch={:?}",
        first_bad.map(|(i, _, _, _)| i),
        first_bad.map(|(_, v, _, _)| v),
        first_bad.map(|(_, _, g, _)| format!("{g:#04x}")),
        first_bad.map(|(_, _, _, w)| format!("{w:#04x}")),
    );
}

#[test]
fn e4m3_decode_roundtrip_on_fixture() {
    // Every torch-produced byte must decode to an f32 that re-encodes to the
    // same byte (cast stability — guards the decode direction too).
    let out_bytes = fs::read(fixture("fp8_cast_outputs.bin")).expect("read outputs fixture");
    for &code in &out_bytes {
        let v = fp8_e4m3_bits_to_f32(code);
        if code & 0x7F == 0x7F {
            assert!(v.is_nan());
        } else {
            assert_eq!(f32_to_fp8_e4m3_bits(v), code, "code {code:#04x}");
        }
    }
}

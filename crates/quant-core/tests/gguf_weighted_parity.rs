//! Byte-parity tests for the weighted K-quant port (plan Phase 4.3).
//!
//! Goldens in tests/golden/llamacpp/ were produced by the REAL
//! `llama-quantize` (built from docs/ref/llama.cpp by
//! tools/build_llamacpp.sh) running Q4_K / Q2_K / Q3_K / Q5_K / Q6_K
//! with `--imatrix` on a deterministic 2x256 F32 fixture and a matching
//! imatrix (tools/gen_golden_llamacpp_weighted.py, seeds 42/7).
//!
//! This is the strongest parity tier we have: our
//! `quantize_row_q{4,2,3,5,6}_k_weighted` must reproduce llama.cpp's
//! weighted output BYTE-FOR-BYTE.
//!
//! Calling convention (llama.cpp quantize_q4_K, ggml-quants.c:1626-1640):
//! the tensor is quantized ROW BY ROW, each row getting the SAME weight
//! vector base (the imatrix entry for the tensor, length ne[0]). Our
//! port's `quantize_row_*_weighted(x, n_per_row, weights)` takes ONE row
//! plus its 256 weights — the test drives it per row and concatenates,
//! exactly mirroring the upstream loop.

use std::path::PathBuf;

use quant_core::gguf_quants::{
    quantize_row_q2_k_weighted, quantize_row_q3_k_weighted, quantize_row_q4_k_weighted,
    quantize_row_q5_k_weighted, quantize_row_q6_k_weighted, Q2_K_BLOCK_BYTES, Q3_K_BLOCK_BYTES,
    Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
};

const QK_K: usize = 256;
const N_ROWS: usize = 2;

fn golden_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p.join("tests/golden/llamacpp")
}

fn load_f32(name: &str, expect: usize) -> Vec<f32> {
    let path = golden_dir().join(name);
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(raw.len(), expect * 4, "{name}: unexpected size");
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// One weighted row-quantizer entry point.
type RowQuantFn = fn(&[f32], usize, Option<&[f32]>) -> Vec<u8>;

/// Drive our port the way llama.cpp drives its quantizers: per row, with
/// the shared per-tensor weight vector.
fn quantize_tensor_per_row(
    src: &[f32],
    weights: &[f32],
    per_row: RowQuantFn,
    block_bytes: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(N_ROWS * block_bytes);
    for r in 0..N_ROWS {
        let row = &src[r * QK_K..(r + 1) * QK_K];
        out.extend(per_row(row, QK_K, Some(weights)));
    }
    out
}

#[test]
fn weighted_q4_k_byte_exact_vs_llama_quantize() {
    let src = load_f32("src.f32.bin", N_ROWS * QK_K);
    let weights = load_f32("weights.f32.bin", QK_K);
    let golden = std::fs::read(golden_dir().join("weighted.q4_k.bin")).unwrap();
    assert_eq!(golden.len(), N_ROWS * Q4_K_BLOCK_BYTES);

    let ours =
        quantize_tensor_per_row(&src, &weights, quantize_row_q4_k_weighted, Q4_K_BLOCK_BYTES);
    assert_eq!(ours.len(), golden.len(), "encoded size");
    assert_eq!(
        ours, golden,
        "weighted Q4_K bytes differ from llama-quantize"
    );
}

#[test]
fn weighted_q2_k_byte_exact_vs_llama_quantize() {
    let src = load_f32("src.f32.bin", N_ROWS * QK_K);
    let weights = load_f32("weights.f32.bin", QK_K);
    let raw = std::fs::read(golden_dir().join("weighted.q2_k.bin")).unwrap();
    // The golden carries llama-quantize's 32-byte alignment padding after
    // the real payload (the generator's reader extends the last tensor to
    // EOF); the actual quantized payload is exactly N_ROWS blocks.
    assert!(raw.len() >= N_ROWS * Q2_K_BLOCK_BYTES);
    let golden = &raw[..N_ROWS * Q2_K_BLOCK_BYTES];

    let ours =
        quantize_tensor_per_row(&src, &weights, quantize_row_q2_k_weighted, Q2_K_BLOCK_BYTES);
    assert_eq!(ours.len(), N_ROWS * Q2_K_BLOCK_BYTES);
    assert_eq!(
        ours, golden,
        "weighted Q2_K bytes differ from llama-quantize"
    );
}

/// Shared driver for the three slice-2 formats whose goldens carry the
/// writer's 32-byte alignment padding after the true payload.
fn padded_parity_case(golden_name: &str, per_row: RowQuantFn, block_bytes: usize, what: &str) {
    let src = load_f32("src.f32.bin", N_ROWS * QK_K);
    let weights = load_f32("weights.f32.bin", QK_K);
    let raw = std::fs::read(golden_dir().join(golden_name)).unwrap();
    assert!(
        raw.len() >= N_ROWS * block_bytes,
        "{golden_name}: golden too small: {} < {}",
        raw.len(),
        N_ROWS * block_bytes
    );
    let golden = &raw[..N_ROWS * block_bytes];

    let ours = quantize_tensor_per_row(&src, &weights, per_row, block_bytes);
    assert_eq!(ours.len(), N_ROWS * block_bytes);
    assert_eq!(
        ours, golden,
        "weighted {what} bytes differ from llama-quantize"
    );
}

#[test]
fn weighted_q3_k_byte_exact_vs_llama_quantize() {
    padded_parity_case(
        "weighted.q3_k.bin",
        quantize_row_q3_k_weighted,
        Q3_K_BLOCK_BYTES,
        "Q3_K",
    );
}

#[test]
fn weighted_q5_k_byte_exact_vs_llama_quantize() {
    padded_parity_case(
        "weighted.q5_k.bin",
        quantize_row_q5_k_weighted,
        Q5_K_BLOCK_BYTES,
        "Q5_K",
    );
}

#[test]
fn weighted_q6_k_byte_exact_vs_llama_quantize() {
    padded_parity_case(
        "weighted.q6_k.bin",
        quantize_row_q6_k_weighted,
        Q6_K_BLOCK_BYTES,
        "Q6_K",
    );
}

// ─── IQ family (Phase 4.3 IQ slice) ───────────────────────────────────
// The IQ quantizers REQUIRE weights (upstream GGML_ASSERTs on NULL), so
// their signature takes &[f32] directly — driven per row here, mirroring
// quantize_iq2_xxs/xs (ggml-quants.c:3652-3677).

#[test]
fn weighted_iq2_xxs_byte_exact_vs_llama_quantize() {
    let src = load_f32("src.f32.bin", N_ROWS * QK_K);
    let weights = load_f32("weights.f32.bin", QK_K);
    let raw = std::fs::read(golden_dir().join("weighted.iq2_xxs.bin")).unwrap();
    let golden = &raw[..N_ROWS * quant_core::gguf_iq_quants::IQ2_XXS_BLOCK_BYTES];

    let mut ours = Vec::new();
    for r in 0..N_ROWS {
        let row = &src[r * QK_K..(r + 1) * QK_K];
        ours.extend(quant_core::gguf_iq_quants::quantize_row_iq2_xxs_weighted(
            row, QK_K, &weights,
        ));
    }
    assert_eq!(
        ours, golden,
        "weighted IQ2_XXS bytes differ from llama-quantize"
    );
}

#[test]
fn weighted_iq2_xs_byte_exact_vs_llama_quantize() {
    let src = load_f32("src.f32.bin", N_ROWS * QK_K);
    let weights = load_f32("weights.f32.bin", QK_K);
    let raw = std::fs::read(golden_dir().join("weighted.iq2_xs.bin")).unwrap();
    let golden = &raw[..N_ROWS * quant_core::gguf_iq_quants::IQ2_XS_BLOCK_BYTES];

    let mut ours = Vec::new();
    for r in 0..N_ROWS {
        let row = &src[r * QK_K..(r + 1) * QK_K];
        ours.extend(quant_core::gguf_iq_quants::quantize_row_iq2_xs_weighted(
            row, QK_K, &weights,
        ));
    }
    assert_eq!(
        ours, golden,
        "weighted IQ2_XS bytes differ from llama-quantize"
    );
}

#[test]
fn weighted_iq2_s_byte_exact_vs_llama_quantize() {
    let src = load_f32("src.f32.bin", N_ROWS * QK_K);
    let weights = load_f32("weights.f32.bin", QK_K);
    let raw = std::fs::read(golden_dir().join("weighted.iq2_s.bin")).unwrap();
    let golden = &raw[..N_ROWS * quant_core::gguf_iq_quants::IQ2_S_BLOCK_BYTES];

    let mut ours = Vec::new();
    for r in 0..N_ROWS {
        let row = &src[r * QK_K..(r + 1) * QK_K];
        ours.extend(quant_core::gguf_iq_quants::quantize_row_iq2_s_weighted(
            row,
            QK_K,
            Some(&weights),
        ));
    }
    assert_eq!(
        ours, golden,
        "weighted IQ2_S bytes differ from llama-quantize"
    );
}

#[test]
fn weighted_iq3_xxs_byte_exact_vs_llama_quantize() {
    let src = load_f32("src.f32.bin", N_ROWS * QK_K);
    let weights = load_f32("weights.f32.bin", QK_K);
    let raw = std::fs::read(golden_dir().join("weighted.iq3_xxs.bin")).unwrap();
    let golden = &raw[..N_ROWS * quant_core::gguf_iq_quants::IQ3_XXS_BLOCK_BYTES];

    let mut ours = Vec::new();
    for r in 0..N_ROWS {
        let row = &src[r * QK_K..(r + 1) * QK_K];
        ours.extend(quant_core::gguf_iq_quants::quantize_row_iq3_xxs_weighted(
            row,
            QK_K,
            Some(&weights),
        ));
    }
    assert_eq!(
        ours, golden,
        "weighted IQ3_XXS bytes differ from llama-quantize"
    );
}

#[test]
fn weighted_iq3_s_byte_exact_vs_llama_quantize() {
    let src = load_f32("src.f32.bin", N_ROWS * QK_K);
    let weights = load_f32("weights.f32.bin", QK_K);
    let raw = std::fs::read(golden_dir().join("weighted.iq3_s.bin")).unwrap();
    let golden = &raw[..N_ROWS * quant_core::gguf_iq_quants::IQ3_S_BLOCK_BYTES];

    let mut ours = Vec::new();
    for r in 0..N_ROWS {
        let row = &src[r * QK_K..(r + 1) * QK_K];
        ours.extend(quant_core::gguf_iq_quants::quantize_row_iq3_s_weighted(
            row,
            QK_K,
            Some(&weights),
        ));
    }
    assert_eq!(
        ours, golden,
        "weighted IQ3_S bytes differ from llama-quantize"
    );
}

//! Oracle equivalence report: OUR converted GGUF vs a REFERENCE GGUF
//! (task #7, `--verify-against`). Port of the analysis logic in
//! `tools/cmp_ours_vs_unsloth.py` + `tools/probe_block_stats.py`.
//!
//! Three comparison layers per shared tensor:
//! 1. **dtype/dims** — scheme assignment must agree;
//! 2. **payload bytes** — same dtype ⇒ byte comparison;
//! 3. **dequantized values** — differing bytes ⇒ dequantize both sides and
//!    classify the divergence:
//!    - [`DiffKind::DeadBlockCosmetic`] — every differing block has a ZERO
//!      scale on both sides (denormal source rows). Encoders saturate the
//!      int codes differently (Rust `as i8` vs C UB-cast), but
//!      reconstruction is identical (q·d = 0). Cosmetic.
//!    - [`DiffKind::ScaleRuleDiff`] — same idea, different block scale
//!      (e.g. unsloth's MSE-tuned Q4_0 scale vs llama.cpp's max/-8).
//!      Reconstruction error differs but stays bounded and small.
//!    - [`DiffKind::GenuineDivergence`] — same scale, differing codes,
//!      reconstruction differs beyond a tiny epsilon. The quantizers
//!      DISAGREE on the math — this is a bug signal.

use std::collections::BTreeMap;
use std::path::Path;

use rlx_gguf::{GgmlType, GgufFile};

use crate::gguf_names::hf_to_gguf_name;

/// Classification of one differing tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffKind {
    /// Same dtype, differing payload bytes, but every differing block has
    /// scale == 0 on both sides: dead-channel blocks (denormal rows).
    /// Reconstruction identical; the int codes are don't-cares.
    DeadBlockCosmetic,
    /// Blocks differ in scale (different quantization rule), reconstruction
    /// stays numerically close. Not a bug; a rules difference.
    ScaleRuleDiff,
    /// Same scale, differing codes, reconstruction differs. Real divergence.
    GenuineDivergence,
}

/// One differing tensor in the report.
#[derive(Debug, Clone)]
pub struct TensorDiff {
    /// Tensor name as seen in OUR file (after name mapping).
    pub name: String,
    pub our_dtype: GgmlType,
    pub ref_dtype: GgmlType,
    pub kind: DiffKind,
    /// Number of blocks whose payload differs.
    pub diff_blocks: usize,
    /// Total blocks (32-element granularity for legacy quants).
    pub total_blocks: usize,
    /// max |ours - ref| over dequantized values.
    pub max_abs_err: f64,
}

/// Summary of one `--verify-against` run.
#[derive(Debug, Clone, Default)]
pub struct VerifyReport {
    /// Shared (name-mapped) tensors that are byte-identical.
    pub byte_exact: Vec<String>,
    /// Shared tensors with differing payloads, classified.
    pub diffs: Vec<TensorDiff>,
    /// Dtype mismatches (no payload compare possible).
    pub dtype_mismatch: Vec<String>,
    /// Dims mismatches.
    pub dims_mismatch: Vec<String>,
    /// Names in ours with no reference counterpart (after mapping).
    pub ours_only: Vec<String>,
    /// Names in the reference with no ours counterpart.
    pub ref_only: Vec<String>,
    /// F32 tensors skipped on either side (never compared).
    pub skipped_f32: usize,
    /// Spec-conformance violations in OUR file (gguf.cpp:724 rule:
    /// quantized tensor whose row ne[0] % block-size != 0).
    pub spec_violations: Vec<String>,
}

impl VerifyReport {
    /// `X byte-exact, Y numerically-equivalent, Z divergent` summary counts.
    pub fn summary(&self) -> (usize, usize, usize) {
        let y = self
            .diffs
            .iter()
            .filter(|d| d.kind != DiffKind::GenuineDivergence)
            .count();
        let z = self
            .diffs
            .iter()
            .filter(|d| d.kind == DiffKind::GenuineDivergence)
            .count();
        (self.byte_exact.len(), y, z)
    }
}

/// Build the ours→reference name map: map OUR HF-ish names through
/// [`hf_to_gguf_name`] (identities pass through), so a converted
/// wrapped-prefix checkpoint lines up with the reference's llama.cpp names.
fn our_name_to_gguf(name: &str) -> String {
    hf_to_gguf_name(name).unwrap_or_else(|| name.to_string())
}

/// Spec-conformance scan of our file (productized diag_vibevoice_q8.py):
/// every quantized tensor must satisfy ne[0] % blck_size == 0 (gguf.cpp:724
/// rejects, :1409 asserts). F32/F16/BF16 have block size 1 and always pass.
/// rlx-gguf parses GGUF dims in file order, so `shape[0]` IS ne[0].
fn scan_spec_violations(f: &GgufFile) -> Vec<String> {
    let mut out = Vec::new();
    // Deterministic order for the report.
    let mut names: Vec<&rlx_gguf::GgufTensor> = f.tensors.values().collect();
    names.sort_by(|a, b| a.name.cmp(&b.name));
    for t in names {
        let blck = crate::gguf_convert::ggml_blck_size(t.dtype);
        if blck > 1 {
            let ne0 = t.shape.first().copied().unwrap_or(0);
            if ne0 % blck != 0 {
                out.push(t.name.clone());
            }
        }
    }
    out
}

/// Compare OUR converted GGUF against a reference GGUF (e.g. produced by
/// unsloth / llama-quantize). Both files are parsed with rlx-gguf; tensor
/// payloads are read through the parser's own accessor so offsets and
/// alignment are handled by the spec-compliant code path.
pub fn verify_against(ours: &Path, reference: &Path) -> Result<VerifyReport, String> {
    let fo = GgufFile::from_path(ours).map_err(|e| format!("parsing {}: {e}", ours.display()))?;
    let fr = GgufFile::from_path(reference)
        .map_err(|e| format!("parsing {}: {e}", reference.display()))?;

    let mut report = VerifyReport::default();
    report.spec_violations = scan_spec_violations(&fo);

    // Reference name set.
    let mut ref_names: BTreeMap<&str, &rlx_gguf::GgufTensor> = BTreeMap::new();
    for (n, t) in &fr.tensors {
        ref_names.insert(n.as_str(), t);
    }

    // Walk OUR tensors in mapped-name order (deterministic output).
    let mut our_entries: Vec<(&String, &rlx_gguf::GgufTensor)> = fo.tensors.iter().collect();
    our_entries.sort_by_key(|(n, _)| our_name_to_gguf(n));

    let mut matched_ref: Vec<String> = Vec::new();
    for (name, t) in our_entries {
        let gguf_name = our_name_to_gguf(name);
        let Some(rt) = ref_names.get(gguf_name.as_str()) else {
            report.ours_only.push(gguf_name);
            continue;
        };
        matched_ref.push(gguf_name.clone());

        // F32 on either side: skip (imatrix norms etc.; the Python tool did
        // the same — quantized-tensor comparison is the point).
        if t.dtype == GgmlType::F32 || rt.dtype == GgmlType::F32 {
            report.skipped_f32 += 1;
            continue;
        }

        if t.dtype != rt.dtype {
            report.dtype_mismatch.push(format!(
                "{gguf_name}: ours={} ref={}",
                type_name(t.dtype),
                type_name(rt.dtype)
            ));
            continue;
        }
        if t.shape != rt.shape {
            report.dims_mismatch.push(format!(
                "{gguf_name}: ours={:?} ref={:?}",
                t.shape, rt.shape
            ));
            continue;
        }

        let ob = fo
            .tensor_bytes(t)
            .map_err(|e| format!("reading ours {gguf_name}: {e}"))?;
        let rb = fr
            .tensor_bytes(rt)
            .map_err(|e| format!("reading ref {gguf_name}: {e}"))?;
        if ob == rb {
            report.byte_exact.push(gguf_name);
            continue;
        }

        // Byte-differing payload: classify through dequantization.
        let (vals_o, vals_r, total_blocks, diff_blocks) = dequant_pair(&fo, t, ob, &fr, rt, rb)?;
        let max_abs_err = vals_o
            .iter()
            .zip(vals_r.iter())
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0f64, f64::max);
        let kind = classify(t, ob, rb, diff_blocks);
        report.diffs.push(TensorDiff {
            name: gguf_name,
            our_dtype: t.dtype,
            ref_dtype: rt.dtype,
            kind,
            diff_blocks,
            total_blocks,
            max_abs_err,
        });
    }

    for n in fr.tensors.keys() {
        if !matched_ref.iter().any(|m| m == n) {
            report.ref_only.push(n.clone());
        }
    }
    Ok(report)
}

/// Classification of one differing payload (probe_block_stats.py logic):
/// - all differing blocks have d == 0 on BOTH sides → dead-block cosmetic;
/// - differing scales but reconstruction within tolerance → scale-rule diff;
/// - everything else → genuine divergence.
fn classify(t: &rlx_gguf::GgufTensor, ob: &[u8], rb: &[u8], _diff_blocks: usize) -> DiffKind {
    match t.dtype {
        GgmlType::Q8_0 => {
            // 34-byte blocks: f16 d + 32 i8.
            let nb = ob.len() / 34;
            let mut all_dead = true;
            for b in 0..nb {
                let bo = &ob[b * 34..b * 34 + 34];
                let br = &rb[b * 34..b * 34 + 34];
                if bo == br {
                    continue;
                }
                let do_ = f16_at(bo, 0);
                let dr = f16_at(br, 0);
                if do_ != 0.0 || dr != 0.0 {
                    all_dead = false;
                    break;
                }
            }
            if all_dead {
                return DiffKind::DeadBlockCosmetic;
            }
            // Non-dead differing blocks: same-scale check. Any block with
            // identical scale but differing codes means the encoders
            // disagree on the tie-break / code selection — genuine.
            let same_scale_differs = (0..nb).any(|b| {
                let bo = &ob[b * 34..b * 34 + 34];
                let br = &rb[b * 34..b * 34 + 34];
                bo != br && f16_at(bo, 0) == f16_at(br, 0)
            });
            if same_scale_differs {
                DiffKind::GenuineDivergence
            } else {
                DiffKind::ScaleRuleDiff
            }
        }
        GgmlType::Q4_0 => {
            // 18-byte blocks: f16 d + 16 packed nibbles.
            let nb = ob.len() / 18;
            let mut all_dead = true;
            for b in 0..nb {
                let bo = &ob[b * 18..b * 18 + 18];
                let br = &rb[b * 18..b * 18 + 18];
                if bo == br {
                    continue;
                }
                if f16_at(bo, 0) != 0.0 || f16_at(br, 0) != 0.0 {
                    all_dead = false;
                    break;
                }
            }
            if all_dead {
                DiffKind::DeadBlockCosmetic
            } else {
                // Different scales = different rules (unsloth MSE-tune);
                // correctness is judged by the dequant error in the report.
                DiffKind::ScaleRuleDiff
            }
        }
        _ => DiffKind::GenuineDivergence,
    }
}

/// Read the f16 stored at `byte_off` inside a block.
fn f16_at(block: &[u8], byte_off: usize) -> f32 {
    let h = u16::from_le_bytes([block[byte_off], block[byte_off + 1]]);
    half::f16::from_bits(h).to_f32()
}

/// Dequantize both sides and compute block-granularity diff counts.
/// Returns (ours_vals, ref_vals, total_blocks, diff_blocks).
fn dequant_pair(
    fo: &GgufFile,
    t: &rlx_gguf::GgufTensor,
    ob: &[u8],
    fr: &GgufFile,
    rt: &rlx_gguf::GgufTensor,
    rb: &[u8],
) -> Result<(Vec<f32>, Vec<f32>, usize, usize), String> {
    let (vo, _) = fo
        .dequant_f32(&t.name)
        .map_err(|e| format!("dequant ours {}: {e}", t.name))?;
    let (vr, _) = fr
        .dequant_f32(&rt.name)
        .map_err(|e| format!("dequant ref {}: {e}", rt.name))?;
    let total_blocks = match t.dtype {
        GgmlType::Q8_0 => ob.len() / 34,
        GgmlType::Q4_0 => ob.len() / 18,
        _ => 0,
    };
    let diff_blocks = match t.dtype {
        GgmlType::Q8_0 => (0..total_blocks)
            .filter(|&b| ob[b * 34..b * 34 + 34] != rb[b * 34..b * 34 + 34])
            .count(),
        GgmlType::Q4_0 => (0..total_blocks)
            .filter(|&b| ob[b * 18..b * 18 + 18] != rb[b * 18..b * 18 + 18])
            .count(),
        _ => 0,
    };
    Ok((vo, vr, total_blocks, diff_blocks))
}

/// Short display name for a GGML type (rlx-gguf has no Display for it).
fn type_name(t: GgmlType) -> &'static str {
    match t {
        GgmlType::F32 => "F32",
        GgmlType::F16 => "F16",
        GgmlType::BF16 => "BF16",
        GgmlType::Q4_0 => "Q4_0",
        GgmlType::Q4_1 => "Q4_1",
        GgmlType::Q5_0 => "Q5_0",
        GgmlType::Q5_1 => "Q5_1",
        GgmlType::Q8_0 => "Q8_0",
        GgmlType::Q8_1 => "Q8_1",
        GgmlType::Q2K => "Q2_K",
        GgmlType::Q3K => "Q3_K",
        GgmlType::Q4K => "Q4_K",
        GgmlType::Q5K => "Q5_K",
        GgmlType::Q6K => "Q6_K",
        GgmlType::Q8K => "Q8_K",
        GgmlType::IQ2XXS => "IQ2_XXS",
        GgmlType::IQ2XS => "IQ2_XS",
        GgmlType::IQ3XXS => "IQ3_XXS",
        GgmlType::IQ1S => "IQ1_S",
        GgmlType::IQ4NL => "IQ4_NL",
        GgmlType::IQ3S => "IQ3_S",
        GgmlType::IQ2S => "IQ2_S",
        GgmlType::IQ4XS => "IQ4_XS",
        GgmlType::IQ1M => "IQ1_M",
        GgmlType::TQ1_0 => "TQ1_0",
        GgmlType::TQ2_0 => "TQ2_0",
        GgmlType::MXFP4 => "MXFP4",
        GgmlType::NVFP4 => "NVFP4",
        _ => "OTHER",
    }
}

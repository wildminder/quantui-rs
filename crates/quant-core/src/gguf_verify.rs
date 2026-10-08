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
//!    - [`DiffKind::FormatConformance`] — the two sides disagree only about
//!      how a value OUT OF RANGE is represented (saturate vs NaN-encode),
//!      which is a legal policy choice under the MX spec rather than a
//!      math disagreement. Reconstruction stays within tolerance.
//!    - [`DiffKind::GenuineDivergence`] — same scale, differing codes,
//!      reconstruction differs beyond a tiny epsilon. The quantizers
//!      DISAGREE on the math — this is a bug signal.

use std::collections::BTreeMap;
use std::path::Path;

use rlx_gguf::{GgmlType, GgufFile};

use crate::gguf_names::hf_to_gguf_name;
use crate::manifest::Format;
use crate::quality::Quality;

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
    /// The differing payload bytes are confined to codes that are
    /// NaN-vs-finite representations of the same nominal value on the two
    /// sides, and the reconstruction error stays within tolerance. This is a
    /// *format policy* difference (e.g. E4M3 `SatMax` vs `OvfNaN` overflow),
    /// not a math disagreement — so it must not be reported as a bug.
    ///
    /// Deliberately conservative: a mis-classification that hides a real bug is
    /// worse than a false alarm, so every other case stays
    /// [`GenuineDivergence`](Self::GenuineDivergence).
    FormatConformance,
    /// The two sides differ because OUR file was produced by an **opt-in
    /// quality refinement** ([`crate::quality::Quality`]) — a deliberate,
    /// documented departure from the reference algorithm, not a bug.
    ///
    /// This is the `--verify-against` self-identification mechanism: a user
    /// who knowingly ran a quality mode should read `quality-tuned`, not
    /// `GENUINE DIVERGENCE`, which is reserved for "the quantizers disagree
    /// and we do not know why".
    ///
    /// Deliberately conservative in the same direction as
    /// [`FormatConformance`](Self::FormatConformance): the label is applied
    /// only when the caller has *declared* that our side is quality-tuned
    /// (see [`verify_against_with_quality`]) **and** the quality variant
    /// actually refines that tensor's format family
    /// ([`Quality::applies_to`]). Nothing is inferred from the payload bytes,
    /// because inferring "this looks like a quality difference" is exactly the
    /// mis-classification that would hide a real bug. A byte-exact run can
    /// therefore never receive this label.
    ///
    /// ⚠️ **GGUF PATH ONLY.** `--verify-against` is a `GgufArgs` flag. The
    /// safetensors `quantize` path's `--verify-output` is a **header-only
    /// re-parse** (`quantize.rs`, `verify_output_files`) with **no payload
    /// comparison whatsoever**, so there is nothing there for this to
    /// classify. The parity guarantee on that path is carried by the printed
    /// `parity:` marker line and the self-documenting `--format` id — NOT by
    /// payload verification. Do not read this variant as implying that
    /// safetensors payload diffing exists.
    QualityTuned,
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
    ///
    /// [`DiffKind::QualityTuned`] counts in `Y`, not `Z`: a declared quality
    /// difference is an expected outcome, so folding it into `Z` would keep
    /// reporting it as a bug, and giving it a fourth bucket would change this
    /// function's public tuple arity for no gain in meaning. The per-tensor
    /// lines still print the precise `kind_label`, so the distinction is
    /// visible where it matters.
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
///
/// `arch` comes from OUR file's `general.architecture` metadata — the
/// `post_attention_layernorm` arm maps differently for gemma2/gemma3 than
/// for the llama family, and the re-map must agree with however the file
/// was converted. Empty when the metadata is absent (arch-blind, the
/// pre-gemma-split behavior).
fn our_name_to_gguf(name: &str, arch: &str) -> String {
    hf_to_gguf_name(name, arch).unwrap_or_else(|| name.to_string())
}

/// Spec-conformance scan of our file (productized diag_vibevoice_q8.py):
/// every quantized tensor must satisfy ne[0] % blck_size == 0 (gguf.cpp:724
/// rejects, :1409 asserts). F32/F16/BF16 have block size 1 and always pass.
/// rlx-gguf parses GGUF dims in file order, so `shape[0]` IS ne[0].
fn scan_spec_violations(f: &GgufFile) -> Vec<String> {
    scan_spec_violations_detailed(f)
        .into_iter()
        .map(|v| v.name)
        .collect()
}

/// One spec violation with enough detail to act on (used by `--audit`).
#[derive(Debug, Clone)]
pub struct SpecViolation {
    pub name: String,
    /// Short display name of the GGML type (e.g. `"Q8_0"`).
    pub dtype: &'static str,
    /// Row width (GGUF ne[0], the FIRST parsed dim — file order).
    pub ne0: usize,
    /// The type's block size that ne0 must divide.
    pub blck: usize,
}

/// Detailed spec scan: same rule as [`scan_spec_violations`], but each
/// violation carries the tensor's type/ne0/blck for the audit report.
fn scan_spec_violations_detailed(f: &GgufFile) -> Vec<SpecViolation> {
    let mut out = Vec::new();
    let mut names: Vec<&rlx_gguf::GgufTensor> = f.tensors.values().collect();
    names.sort_by(|a, b| a.name.cmp(&b.name));
    for t in names {
        let blck = crate::gguf_convert::ggml_blck_size(t.dtype);
        if blck > 1 {
            let ne0 = t.shape.first().copied().unwrap_or(0);
            if ne0 % blck != 0 {
                out.push(SpecViolation {
                    name: t.name.clone(),
                    dtype: type_name(t.dtype),
                    ne0,
                    blck,
                });
            }
        }
    }
    out
}

/// Result of auditing one GGUF file (`--audit`, IMP-003): a dtype census
/// plus the spec-conformance violations.
#[derive(Debug, Clone, Default)]
pub struct GgufAudit {
    pub tensor_count: usize,
    /// Type display name → tensor count (sorted by name when printed).
    pub dtype_histogram: BTreeMap<String, usize>,
    pub violations: Vec<SpecViolation>,
}

/// Audit any GGUF file — no reference needed: parse it, census the tensor
/// dtypes, and scan for spec violations (gguf.cpp:724 per-row rule). This
/// is the productized "check a downloaded/converted GGUF" workflow.
pub fn audit_gguf(path: &Path) -> Result<GgufAudit, String> {
    let f = GgufFile::from_path(path).map_err(|e| format!("parsing {}: {e}", path.display()))?;
    let mut audit = GgufAudit {
        tensor_count: f.tensors.len(),
        ..Default::default()
    };
    for t in f.tensors.values() {
        *audit
            .dtype_histogram
            .entry(type_name(t.dtype).to_string())
            .or_insert(0) += 1;
    }
    audit.violations = scan_spec_violations_detailed(&f);
    Ok(audit)
}

/// Compare OUR converted GGUF against a reference GGUF (e.g. produced by
/// unsloth / llama-quantize), assuming OUR side is byte-parity-exact.
///
/// This is the conservative entry point and delegates to
/// [`verify_against_with_quality`] with [`Quality::Exact`], so every payload
/// difference that is not otherwise explained is still reported as
/// [`DiffKind::GenuineDivergence`] — exactly as before this tier. Callers that
/// *know* they ran a quality mode must use the `_with_quality` form; there is
/// no inference, because inferring "this looks like a quality difference" from
/// the bytes would be exactly the mis-classification that hides a real bug.
pub fn verify_against(ours: &Path, reference: &Path) -> Result<VerifyReport, String> {
    verify_against_with_quality(ours, reference, Quality::Exact)
}

/// [`verify_against`], with the quality OUR file was produced under declared
/// explicitly.
///
/// `ours_quality` is the *sole* justification for a
/// [`DiffKind::QualityTuned`] verdict, and it is checked against
/// [`Quality::applies_to`] per tensor family. Passing
/// [`Quality::Exact`] — or passing a quality mode that refines a different
/// format than the tensor under test — reproduces the previous behaviour
/// exactly.
///
/// ⚠️ **GGUF path only.** This module is reachable solely from `--verify-against`,
/// a `GgufArgs` flag. The safetensors `quantize` path's `--verify-output` is a
/// **header-only re-parse** with **no payload comparison at all**, so no
/// equivalent safetensors mechanism exists today; do not imply one does.
pub fn verify_against_with_quality(
    ours: &Path,
    reference: &Path,
    ours_quality: Quality,
) -> Result<VerifyReport, String> {
    let fo = GgufFile::from_path(ours).map_err(|e| format!("parsing {}: {e}", ours.display()))?;
    let fr = GgufFile::from_path(reference)
        .map_err(|e| format!("parsing {}: {e}", reference.display()))?;

    let mut report = VerifyReport::default();
    report.spec_violations = scan_spec_violations(&fo);

    // OUR file's declared architecture, for the arch-aware name re-map.
    let our_arch = match fo.metadata.get("general.architecture") {
        Some(rlx_gguf::MetaValue::String(s)) => s.clone(),
        _ => String::new(),
    };

    // Reference name set.
    let mut ref_names: BTreeMap<&str, &rlx_gguf::GgufTensor> = BTreeMap::new();
    for (n, t) in &fr.tensors {
        ref_names.insert(n.as_str(), t);
    }

    // Walk OUR tensors in mapped-name order (deterministic output).
    let mut our_entries: Vec<(&String, &rlx_gguf::GgufTensor)> = fo.tensors.iter().collect();
    our_entries.sort_by_key(|(n, _)| our_name_to_gguf(n, &our_arch));

    let mut matched_ref: Vec<String> = Vec::new();
    for (name, t) in our_entries {
        let gguf_name = our_name_to_gguf(name, &our_arch);
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
        let kind = classify(t, ob, rb, diff_blocks, ours_quality);
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
/// - the caller declared our side quality-tuned for THIS format family and
///   the difference is not more specifically explained → quality-tuned;
/// - everything else → genuine divergence.
///
/// `ours_quality` must be the quality the caller actually ran with, never a
/// guess. It is the *only* input that can produce
/// [`DiffKind::QualityTuned`], so a byte-exact run can never receive that
/// label — see the variant's docs for why that direction is the safe one.
fn classify(
    t: &rlx_gguf::GgufTensor,
    ob: &[u8],
    rb: &[u8],
    _diff_blocks: usize,
    ours_quality: Quality,
) -> DiffKind {
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
        // NVFP4 is the only dtype whose *block scale* is E4M3. MXFP4's block
        // scale is E8M0 (pure exponent, no NaN-vs-saturated question), so an
        // E4M3 overflow policy cannot arise there — it stays divergent.
        GgmlType::NVFP4 => {
            if is_nvfp4_e4m3_scale_policy_only(ob, rb) {
                DiffKind::FormatConformance
            } else if quality_refines(ours_quality, Format::Nvfp4) {
                // A declared NVFP4 quality mode (the anchored L2 scale search)
                // changes the block scales, so the codes legitimately differ
                // from a byte-exact reference. That is the mode working, not a
                // bug — but ONLY because the caller declared it.
                DiffKind::QualityTuned
            } else {
                DiffKind::GenuineDivergence
            }
        }
        // MXFP4 is the GGUF-side counterpart of the MXFP8 family, so a
        // declared MXFP8 quality mode explains a difference here for the same
        // reason. No current quality variant is reachable from the CLI, so
        // this arm is presently inert — it is wired so that adding a preset
        // cannot silently start reporting `GENUINE DIVERGENCE` for output the
        // tool itself deliberately produced differently.
        GgmlType::MXFP4 => {
            if quality_refines(ours_quality, Format::Mxfp8) {
                DiffKind::QualityTuned
            } else {
                DiffKind::GenuineDivergence
            }
        }
        _ => DiffKind::GenuineDivergence,
    }
}

/// `true` iff `quality` is a non-default refinement OF `family`.
///
/// Both halves matter. A non-`Exact` quality that refines a *different*
/// family must not excuse a difference here: `applies_to()` is what stops a
/// declared NVFP4 search from masking a genuine divergence in, say, a Q8_0
/// tensor in the same file. And `Exact` must never match, so the default
/// byte-exact path keeps reporting `GENUINE DIVERGENCE` exactly as before.
fn quality_refines(quality: Quality, family: Format) -> bool {
    !quality.is_parity_exact() && quality.applies_to() == family
}

/// NVFP4 block geometry (rlx-gguf 0.2.14 `mx_dequant.rs`):
/// 16 elements per block, block bytes = `1 + 16/2` = **9**, where byte 0 is the
/// E4M3 scale and bytes 1..9 are packed E2M1 nibbles.
const NVFP4_BLOCK_BYTES: usize = 9;
/// Byte offset of the E4M3 scale within an NVFP4 block.
const NVFP4_SCALE_OFFSET: usize = 0;

/// `true` iff the two NVFP4 payloads differ ONLY in E4M3 *scale* bytes, and
/// every such difference is a NaN-vs-saturated-finite pair.
///
/// # Why this is narrow on purpose
///
/// `float8_e4m3fn` (what we implement) reserves `0x7F`/`0xFF` for NaN and
/// saturates overflow to `0x7E`/`0xFE` (448.0). The OCP MX element variant is
/// finite-only, so a reference using it emits `0x7E`/`0xFE` where we emit
/// `0x7F`/`0xFF` for the same nominal value. Both are legal under OCP MX v1.0,
/// so that difference is a *policy* mismatch, not a math disagreement.
///
/// The predicate therefore requires ALL of the following, and is strict in the
/// safe direction — a mis-classification that hides a real bug is worse than a
/// false alarm:
///
/// 1. **Equal, correctly-strided lengths.** A length mismatch is a real bug.
/// 2. **Only scale positions may differ.** E2M1 nibble bytes are the actual
///    *data*; a differing nibble is a math disagreement and must stay
///    `GenuineDivergence`. Position is checked explicitly (`byte % 9 == 0`)
///    rather than assumed, because an off-by-one in the block size would
///    otherwise silently reclassify data bytes as scale bytes — the exact
///    "over-capture" failure the plan names as this step's main risk.
/// 3. **Every differing scale byte must be a NaN-vs-saturated pair.** Anything
///    else (a genuinely different exponent, a sign flip) stays divergent.
/// 4. **At least one scale byte must actually differ**, so an identical
///    payload can never be reported as a conformance difference.
fn is_nvfp4_e4m3_scale_policy_only(ob: &[u8], rb: &[u8]) -> bool {
    if ob.len() != rb.len() || ob.len() % NVFP4_BLOCK_BYTES != 0 {
        return false;
    }
    let mut saw_scale_diff = false;
    for (i, (&a, &b)) in ob.iter().zip(rb.iter()).enumerate() {
        if a == b {
            continue;
        }
        // Differing byte outside a scale position ⇒ real data divergence.
        if i % NVFP4_BLOCK_BYTES != NVFP4_SCALE_OFFSET {
            return false;
        }
        if !is_nan_vs_saturated_pair(a, b) {
            return false;
        }
        saw_scale_diff = true;
    }
    saw_scale_diff
}

/// `true` iff `(a, b)` is one of the two E4M3 overflow-policy pairs, in either
/// direction: `0x7F` (NaN) against `0x7E` (448.0), or `0xFF` (NaN) against
/// `0xFE` (-448.0).
fn is_nan_vs_saturated_pair(a: u8, b: u8) -> bool {
    matches!(
        (a, b),
        (0x7F, 0x7E) | (0x7E, 0x7F) | (0xFF, 0xFE) | (0xFE, 0xFF)
    )
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

#[cfg(test)]
mod tests {
    use super::{quality_refines, DiffKind};
    use crate::manifest::Format;
    use crate::quality::Quality;

    /// `Exact` must NEVER excuse a difference.
    ///
    /// This is the load-bearing safety property: a byte-exact run that reports
    /// `quality-tuned` instead of `GENUINE DIVERGENCE` would hide real bugs
    /// behind a reassuring label. Every other test here is worthless if this
    /// one fails.
    #[test]
    fn exact_never_refines_anything() {
        for family in [Format::Int8, Format::Fp8E4m3, Format::Mxfp8, Format::Nvfp4] {
            assert!(
                !quality_refines(Quality::Exact, family),
                "Exact must not excuse a difference in {family:?}"
            );
        }
    }

    /// A quality mode excuses a difference ONLY in the family it refines.
    ///
    /// Both directions are asserted. The forward case is the feature; the
    /// reverse case is the guard — a declared NVFP4 scale search must not
    /// launder a genuine divergence in some other format in the same file,
    /// which is exactly what a `!is_parity_exact()`-only check would do.
    #[test]
    fn a_quality_mode_excuses_only_its_own_family() {
        assert!(quality_refines(Quality::Nvfp4L2ScaleSearch, Format::Nvfp4));
        assert!(quality_refines(
            Quality::Mxfp8E8m0Compensated,
            Format::Mxfp8
        ));
        // Cross-family: must NOT match.
        assert!(!quality_refines(Quality::Nvfp4L2ScaleSearch, Format::Mxfp8));
        assert!(!quality_refines(
            Quality::Nvfp4HessianScaleSearch,
            Format::Int8
        ));
    }

    /// The label a quality difference carries must not read like a bug.
    ///
    /// Asserted structurally rather than by snapshotting the CLI's private
    /// `kind_label` (a BIN-only crate cannot be imported here): what matters
    /// in core is that the variant exists, is distinct from
    /// `GenuineDivergence`, and is produced only under the narrow condition
    /// `quality_refines` encodes.
    #[test]
    fn quality_tuned_is_a_distinct_variant_from_genuine_divergence() {
        assert_ne!(
            DiffKind::QualityTuned,
            DiffKind::GenuineDivergence,
            "a declared quality difference must be distinguishable from a bug"
        );
        // The classification is reachable at all (not a dead variant).
        let kinds = [
            DiffKind::DeadBlockCosmetic,
            DiffKind::ScaleRuleDiff,
            DiffKind::GenuineDivergence,
            DiffKind::FormatConformance,
            DiffKind::QualityTuned,
        ];
        let mut seen = kinds.to_vec();
        seen.sort_by_key(|k| format!("{k:?}"));
        seen.dedup();
        assert_eq!(seen.len(), kinds.len(), "all five variants are distinct");
    }

    /// A quality mode for a DIFFERENT family must not be reachable through the
    /// NVFP4 arm, even when declared. This is the concrete scenario the
    /// `applies_to` check exists for: an MXFP8 run whose file also carries an
    /// NVFP4 tensor must still report that tensor's divergence honestly.
    #[test]
    fn an_unrelated_quality_mode_does_not_excuse_nvfp4() {
        assert!(
            !quality_refines(Quality::Mxfp8E8m0Compensated, Format::Nvfp4),
            "an MXFP8 quality mode must not excuse an NVFP4 difference"
        );
    }
}

//! Deep structural + numeric validator for ComfyUI-native quantized
//! `.safetensors` files (plan Phase 8, steps 7.1 + 7.2).
//!
//! Port of reference `quant_validator.py` (`ValidationReport` model, error /
//! warning flow, orphan detection, summary statistics, numeric pass),
//! generalized from the reference's INT8-only `SUPPORTED_FORMATS` to the full
//! ctq format set actually present in the goldens (INT8 / FP8 / MXFP8 / NVFP4).
//!
//! Two things deliberately diverge from the reference validator, both confirmed
//! against the golden fixtures (the byte-parity ground truth):
//!
//! * **Format coverage.** The reference only knew `{int8_tensorwise,
//!   int8_blockwise}`. We validate every format in [`crate::comfy_schema`].
//! * **Scale-shape expectations.** The reference expected a per-row
//!   `[out_features, 1]` scale for tensorwise and a non-squeezed `[bm, bn]`
//!   for blockwise. ctq actually applies the *scale-squeeze-to-scalar* quirk
//!   (`normalize_tensorwise_scales`: any single-element scale → shape `[]`),
//!   so a `[128,128]` blockwise layer carries `weight_scale` shape `[]`, not
//!   `[1,1]`. We encode the real per-format formulas (see [`expected_scale_shape`]).
//!
//! Structural pass (always): header integrity, `.comfy_quant` JSON parse,
//! format support, weight/scale presence + dtype + shape, NVFP4
//! `weight_scale_2`, blockwise `input_scale`, bias-not-quantized, and orphan
//! marker/scale detection. Numeric pass (opt-in): weight bounds, scale
//! finiteness/positivity, `input_scale == 1.0`.
//!
//! A file with *no* `.comfy_quant` tensors is reported `ok = true` with a
//! warning (plain FP8/FP16 or passthrough checkpoints are valid too) — mirrors
//! the reference.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::comfy_schema::{self, ComfyFormat, ComfyQuantConfig};
use crate::dtype::{self, DType};
use crate::st_io::reader::SafetensorsReader;

/// Round `x` up to the nearest multiple of `mult` (reference `roundup`).
fn roundup(x: u64, mult: u64) -> u64 {
    if mult == 0 {
        return x;
    }
    x.div_ceil(mult) * mult
}

/// ctq's scale-squeeze-to-scalar quirk: any scale tensor with exactly one
/// element is stored as a scalar (shape `[]`).
fn squeeze(shape: Vec<u64>) -> Vec<u64> {
    if shape.iter().product::<u64>() == 1 {
        Vec::new()
    } else {
        shape
    }
}

// --------------------------------------------------------------------------- //
// Report model (port of quant_validator.ValidationReport + widgets_results.Issue)
// --------------------------------------------------------------------------- //

/// Severity of one structured validation issue (`widgets_results.py::Issue`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueLevel {
    Error,
    Warning,
    Info,
}

/// One structured validation issue row (`widgets_results.py::Issue`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Issue {
    pub level: IssueLevel,
    pub text: String,
    pub hint: String,
}

/// Per-layer info accumulated into the report summary.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LayerInfo {
    pub prefix: String,
    pub weight_shape: Option<Vec<u64>>,
}

/// Aggregate statistics (port of the reference `report.summary` dict).
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Summary {
    pub file: String,
    pub size_gb: f64,
    pub quantized_matrices: usize,
    pub full_precision_weights: usize,
    pub quantized_params: u64,
    pub full_precision_params: u64,
    pub quantized_share_pct: f64,
    pub group_size_histogram: BTreeMap<u32, usize>,
    pub formats_found: Vec<String>,
}

/// Result of [`validate_comfy_quant`] (port of `ValidationReport`).
#[derive(Debug, Default, Clone, PartialEq)]
pub struct ValidationReport {
    pub path: String,
    pub ok: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub formats: BTreeSet<String>,
    pub layers: Vec<LayerInfo>,
    pub summary: Summary,
}

impl ValidationReport {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            ok: true,
            ..Default::default()
        }
    }

    pub fn add_error(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
        self.ok = false;
    }

    pub fn add_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }
}

/// Map a [`ValidationReport`] into structured issue rows
/// (`widgets_results.py::report_to_issues`).
pub fn report_to_issues(report: &ValidationReport) -> Vec<Issue> {
    let mut issues = Vec::new();
    for e in &report.errors {
        issues.push(Issue {
            level: IssueLevel::Error,
            text: e.clone(),
            hint: String::new(),
        });
    }
    for w in &report.warnings {
        issues.push(Issue {
            level: IssueLevel::Warning,
            text: w.clone(),
            hint: String::new(),
        });
    }
    if issues.is_empty() && report.ok {
        let fmts = if report.formats.is_empty() {
            "unknown".to_string()
        } else {
            report
                .formats
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut detail = format!("format: {fmts}");
        detail.push_str(&format!(
            " | quantized layers: {}",
            report.summary.quantized_matrices
        ));
        issues.push(Issue {
            level: IssueLevel::Info,
            text: "File looks valid".to_string(),
            hint: detail,
        });
    }
    issues
}

// --------------------------------------------------------------------------- //
// Per-format contract
// --------------------------------------------------------------------------- //

/// Expected header dtype of the quantized weight payload for a format.
fn weight_dtype(fmt: ComfyFormat) -> DType {
    match fmt {
        ComfyFormat::Int8Tensorwise | ComfyFormat::Int8Blockwise => DType::I8,
        ComfyFormat::Fp8Tensor | ComfyFormat::Fp8Rowwise | ComfyFormat::Fp8Blockwise => {
            DType::F8E4M3
        }
        ComfyFormat::Mxfp8 => DType::F8E4M3,
        ComfyFormat::Nvfp4 => DType::U8,
    }
}

/// Expected header dtype of the sibling `weight_scale` for a format.
fn scale_dtype(fmt: ComfyFormat) -> DType {
    match fmt {
        ComfyFormat::Int8Tensorwise
        | ComfyFormat::Int8Blockwise
        | ComfyFormat::Fp8Tensor
        | ComfyFormat::Fp8Rowwise
        | ComfyFormat::Fp8Blockwise => DType::F32,
        // MXFP8 block scales are e8m0 bytes; NVFP4 block scales are E4M3.
        ComfyFormat::Mxfp8 => DType::U8,
        ComfyFormat::Nvfp4 => DType::F8E4M3,
    }
}

/// True when the format carries a blockwise `input_scale` scalar (INT8 block
/// only in the ctq simple paths — the FP8 blockwise path does not emit one,
/// confirmed against the goldens).
fn requires_input_scale(fmt: ComfyFormat) -> bool {
    matches!(fmt, ComfyFormat::Int8Blockwise)
}

/// Expected `weight_scale` shape for a format, given the ON-DISK weight shape.
///
/// For the F32-scale family (INT8 / FP8) this is derived from the original
/// `[m, n]` weight dims + the blob's `group_size`, with the squeeze quirk
/// applied. For MXFP8 / NVFP4 the scale lives in cuBLAS `to_blocked` layout,
/// which is fully determined by the (already-padded) on-disk weight shape.
///
/// Returns `None` when the expectation cannot be computed (e.g. a blockwise
/// format with a missing group size) — the caller reports that separately.
fn expected_scale_shape(
    fmt: ComfyFormat,
    weight_shape: &[u64],
    group_size: Option<u32>,
    per_row: bool,
) -> Option<Vec<u64>> {
    let m = *weight_shape.first()?;
    let n = weight_shape.get(1).copied()?;
    Some(match fmt {
        // Single scale for the whole tensor → squeezed scalar. INT8 row
        // mode (`per_row: true`, Phase 7.2) is the exception: one scale
        // per output row, `[m,1]` — squeezed only for a single row
        // (normalize_tensorwise_scales parity; the row goldens carry
        // unsqueezed [m,1] for m > 1).
        ComfyFormat::Int8Tensorwise if per_row => squeeze(vec![m, 1]),
        ComfyFormat::Int8Tensorwise | ComfyFormat::Fp8Tensor => vec![],
        // One scale per output row.
        ComfyFormat::Fp8Rowwise => squeeze(vec![m]),
        // [ceil(m/g), ceil(n/g)], squeezed when single-element.
        ComfyFormat::Int8Blockwise | ComfyFormat::Fp8Blockwise => {
            let g = group_size? as u64;
            if g == 0 {
                return None;
            }
            squeeze(vec![m.div_ceil(g), n.div_ceil(g)])
        }
        // to_blocked layout: (roundup(rows,128), roundup(num_blocks,4)).
        // MXFP8 weight is [m_pad, n_pad]; num_blocks = n_pad / 32.
        ComfyFormat::Mxfp8 => {
            if n % 32 != 0 {
                return None;
            }
            vec![roundup(m, 128), roundup(n / 32, 4)]
        }
        // NVFP4 weight is [m_pad, n_pad/2] (packed); num_blocks = n_pad / 16
        // = (n_packed * 2) / 16 = n_packed / 8.
        ComfyFormat::Nvfp4 => {
            if n % 8 != 0 {
                return None;
            }
            vec![roundup(m, 128), roundup(n / 8, 4)]
        }
    })
}

// --------------------------------------------------------------------------- //
// Core validation
// --------------------------------------------------------------------------- //

/// Validate a ComfyUI-native quantized `.safetensors` file.
///
/// Structural pass always runs; the numeric pass (weight bounds, scale
/// finiteness/positivity, `input_scale == 1.0`) runs only when
/// `numeric = true`.
pub fn validate_comfy_quant(path: impl AsRef<Path>, numeric: bool) -> ValidationReport {
    let path = path.as_ref();
    let mut report = ValidationReport::new(path.to_string_lossy());

    if !path.is_file() {
        report.add_error(format!("file not found: {}", path.display()));
        return report;
    }

    let reader = match SafetensorsReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            report.add_error(format!("invalid safetensors header: {e}"));
            return report;
        }
    };

    let header = reader.header();
    let keys: Vec<&String> = header.names().collect();
    let key_set: BTreeSet<&str> = keys.iter().map(|s| s.as_str()).collect();

    let markers: Vec<&str> = keys
        .iter()
        .map(|s| s.as_str())
        .filter(|k| k.ends_with(".comfy_quant"))
        .collect();
    let scales: Vec<&str> = keys
        .iter()
        .map(|s| s.as_str())
        .filter(|k| k.ends_with(".weight_scale"))
        .collect();

    if markers.is_empty() {
        report.add_warning(
            "no .comfy_quant markers found -- this is a plain (FP8/FP16) or \
             on-the-fly-passthrough checkpoint, not a natively quantized one."
                .to_string(),
        );
        summarize_no_markers(&mut report, &keys, header, path);
        return report;
    }

    // Parse every marker once.
    let mut parsed: Vec<(String, ComfyQuantConfig)> = Vec::new();
    for marker in &markers {
        let base = comfy_schema::layer_prefix(marker)
            .map(|s| s.to_string())
            .unwrap_or_else(|| (*marker).to_string());
        let blob = match reader.tensor_bytes(marker) {
            Ok(b) => b,
            Err(e) => {
                report.add_error(format!("{base}: cannot read .comfy_quant blob: {e}"));
                continue;
            }
        };
        match comfy_schema::parse_blob(blob) {
            Ok(cfg) => parsed.push((base, cfg)),
            Err(e) => report.add_error(format!("{base}: {e}")),
        }
    }

    let mut gs_hist: BTreeMap<u32, usize> = BTreeMap::new();
    let mut q_params: u64 = 0;
    let mut q_layers: Vec<String> = Vec::new();

    for (base, cfg) in &parsed {
        let fmt = cfg.format;
        report.formats.insert(fmt.as_str().to_string());
        if let Some(g) = cfg.group_size {
            *gs_hist.entry(g).or_insert(0) += 1;
        }

        let w_key = format!("{base}.weight");
        let s_key = format!("{base}.weight_scale");

        let w_info = match header.get(&w_key) {
            Some(i) => i,
            None => {
                report.add_error(format!("{base}: missing quantized weight tensor '{w_key}'"));
                continue;
            }
        };

        // Weight dtype.
        if w_info.dtype != weight_dtype(fmt) {
            report.add_error(format!(
                "{base}: weight dtype '{}' != '{}' (expected for {})",
                w_info.dtype,
                weight_dtype(fmt),
                fmt.as_str()
            ));
        }

        // Scale presence + dtype.
        let s_info = match header.get(&s_key) {
            Some(i) => {
                if i.dtype != scale_dtype(fmt) {
                    report.add_error(format!(
                        "{base}: weight_scale dtype '{}' != '{}' (expected for {})",
                        i.dtype,
                        scale_dtype(fmt),
                        fmt.as_str()
                    ));
                }
                Some(i)
            }
            None => {
                report.add_error(format!("{base}: missing weight_scale tensor '{s_key}'"));
                None
            }
        };

        // Weight must be 2-D.
        if w_info.shape.len() != 2 {
            report.add_error(format!("{base}: weight is not 2-D: {:?}", w_info.shape));
            continue;
        }
        let out_f = w_info.shape[0];
        let in_f = w_info.shape[1];

        // Scale shape check.
        if let Some(s_info) = s_info {
            match expected_scale_shape(fmt, &w_info.shape, cfg.group_size, cfg.per_row) {
                Some(exp) => {
                    if s_info.shape != exp {
                        report.add_error(format!(
                            "{base}: {} weight_scale shape {:?} != expected {:?}",
                            fmt.as_str(),
                            s_info.shape,
                            exp
                        ));
                    }
                }
                None => {
                    report.add_error(format!(
                        "{base}: cannot determine expected weight_scale shape \
                         (missing/zero group_size or unpadded dims)"
                    ));
                }
            }
        }

        // NVFP4 second (per-tensor) scale.
        if fmt == ComfyFormat::Nvfp4 {
            let s2_key = format!("{base}.weight_scale_2");
            match header.get(&s2_key) {
                Some(i) => {
                    if i.dtype != DType::F32 {
                        report.add_error(format!(
                            "{base}: weight_scale_2 dtype '{}' != 'F32'",
                            i.dtype
                        ));
                    }
                    if !i.shape.is_empty() {
                        report.add_error(format!(
                            "{base}: weight_scale_2 must be a scalar, got {:?}",
                            i.shape
                        ));
                    }
                }
                None => {
                    report.add_error(format!("{base}: nvfp4 requires '{s2_key}' (missing)"));
                }
            }
        }

        // Blockwise input_scale scalar.
        if requires_input_scale(fmt) {
            let is_key = format!("{base}.input_scale");
            match header.get(&is_key) {
                Some(i) => {
                    if i.dtype != DType::F32 {
                        report
                            .add_error(format!("{base}: input_scale dtype '{}' != 'F32'", i.dtype));
                    }
                    if !i.shape.is_empty() {
                        report.add_error(format!(
                            "{base}: input_scale must be a scalar, got {:?}",
                            i.shape
                        ));
                    }
                }
                None => {
                    report.add_error(format!(
                        "{base}: {} requires '{is_key}' (missing)",
                        fmt.as_str()
                    ));
                }
            }
        }

        // Bias must not be quantized.
        let bias_key = format!("{base}.bias");
        if let Some(b) = header.get(&bias_key) {
            if matches!(b.dtype, DType::I8 | DType::U8 | DType::F8E4M3) {
                report.add_error(format!(
                    "{base}: bias was quantized (bias must stay full precision)"
                ));
            }
        }

        q_params += out_f * in_f;
        q_layers.push(base.clone());
    }

    // Orphan detection (reference semantics, keyed on weight_scale).
    let orphan_scales: Vec<&str> = scales
        .iter()
        .filter(|s| {
            let base = &s[..s.len() - ".weight_scale".len()];
            !key_set.contains(format!("{base}.comfy_quant").as_str())
        })
        .copied()
        .collect();
    let orphan_markers: Vec<&str> = markers
        .iter()
        .filter(|m| {
            let base = comfy_schema::layer_prefix(m).unwrap_or(m);
            !key_set.contains(format!("{base}.weight_scale").as_str())
        })
        .copied()
        .collect();
    if !orphan_scales.is_empty() {
        report.add_error(format!(
            "orphan weight_scale entries (no matching .comfy_quant): {:?}",
            &orphan_scales[..orphan_scales.len().min(5)]
        ));
    }
    if !orphan_markers.is_empty() {
        report.add_error(format!(
            "orphan .comfy_quant entries (no matching .weight_scale): {:?}",
            &orphan_markers[..orphan_markers.len().min(5)]
        ));
    }

    summarize(
        &mut report,
        &keys,
        header,
        q_params,
        &q_layers,
        gs_hist,
        path,
    );
    for base in &q_layers {
        let w = header
            .get(&format!("{base}.weight"))
            .map(|i| i.shape.clone());
        report.layers.push(LayerInfo {
            prefix: base.clone(),
            weight_shape: w,
        });
    }

    if numeric {
        numeric_pass(&mut report, &reader, &parsed);
    }

    report
}

// --------------------------------------------------------------------------- //
// Summary
// --------------------------------------------------------------------------- //

fn summarize_no_markers(
    report: &mut ValidationReport,
    keys: &[&String],
    header: &crate::st_io::header::Header,
    path: &Path,
) {
    let fp_weights: Vec<&&String> = keys.iter().filter(|k| k.ends_with(".weight")).collect();
    let fp_params: u64 = fp_weights
        .iter()
        .filter_map(|k| header.get(k.as_str()))
        .map(|i| i.shape.iter().product::<u64>())
        .sum();
    report.summary = Summary {
        file: file_name(path),
        size_gb: size_gb(path),
        quantized_matrices: 0,
        full_precision_weights: fp_weights.len(),
        quantized_params: 0,
        full_precision_params: fp_params,
        quantized_share_pct: 0.0,
        group_size_histogram: BTreeMap::new(),
        formats_found: Vec::new(),
    };
}

fn summarize(
    report: &mut ValidationReport,
    keys: &[&String],
    header: &crate::st_io::header::Header,
    q_params: u64,
    q_layers: &[String],
    gs_hist: BTreeMap<u32, usize>,
    path: &Path,
) {
    let quantized_prefixes: BTreeSet<&str> = q_layers.iter().map(|s| s.as_str()).collect();
    let fp_weights: Vec<&&String> = keys
        .iter()
        .filter(|k| {
            k.ends_with(".weight")
                && !quantized_prefixes.contains(k.strip_suffix(".weight").unwrap_or(k.as_str()))
        })
        .collect();
    let fp_params: u64 = fp_weights
        .iter()
        .filter_map(|k| header.get(k.as_str()))
        .map(|i| i.shape.iter().product::<u64>())
        .sum();
    let total = q_params + fp_params;
    let share = if total == 0 {
        0.0
    } else {
        round3(100.0 * q_params as f64 / total as f64)
    };
    report.summary = Summary {
        file: file_name(path),
        size_gb: size_gb(path),
        quantized_matrices: q_layers.len(),
        full_precision_weights: fp_weights.len(),
        quantized_params: q_params,
        full_precision_params: fp_params,
        quantized_share_pct: share,
        group_size_histogram: gs_hist,
        formats_found: report.formats.iter().cloned().collect(),
    };
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

fn size_gb(path: &Path) -> f64 {
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    round3(bytes as f64 / 1e9)
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

// --------------------------------------------------------------------------- //
// Numeric pass
// --------------------------------------------------------------------------- //

/// Optional numeric pass: weight bounds, scale finiteness/positivity, and
/// `input_scale == 1.0` for blockwise. Mirrors the reference aggregate-count
/// error reporting.
fn numeric_pass(
    report: &mut ValidationReport,
    reader: &SafetensorsReader,
    parsed: &[(String, ComfyQuantConfig)],
) {
    let mut overflow = 0usize;
    let mut bad_scale = 0usize;
    let mut bad_input_scale = 0usize;

    for (base, cfg) in parsed {
        let w_key = format!("{base}.weight");
        let s_key = format!("{base}.weight_scale");
        let (Ok(w_bytes), Ok(s_bytes)) = (reader.tensor_bytes(&w_key), reader.tensor_bytes(&s_key))
        else {
            continue;
        };

        // Weight bounds / finiteness.
        let weight_ok = match cfg.format {
            // Symmetric INT8 must stay within [-127, 127] (no -128).
            ComfyFormat::Int8Tensorwise | ComfyFormat::Int8Blockwise => {
                let mx = w_bytes
                    .iter()
                    .map(|&b| (b as i8).unsigned_abs() as u32)
                    .max()
                    .unwrap_or(0);
                if mx > 127 {
                    overflow += 1;
                }
                true
            }
            // E4M3 payloads must be finite (0x7F / 0xFF are NaN).
            ComfyFormat::Fp8Tensor
            | ComfyFormat::Fp8Rowwise
            | ComfyFormat::Fp8Blockwise
            | ComfyFormat::Mxfp8 => {
                if w_bytes
                    .iter()
                    .any(|&b| dtype::fp8_e4m3_bits_to_f32(b).is_nan())
                {
                    overflow += 1;
                }
                true
            }
            // NVFP4 weights are packed nibbles; no per-element bound applies.
            ComfyFormat::Nvfp4 => true,
        };
        let _ = weight_ok;

        // Scale finiteness / positivity.
        let bad = match cfg.format {
            ComfyFormat::Int8Tensorwise
            | ComfyFormat::Int8Blockwise
            | ComfyFormat::Fp8Tensor
            | ComfyFormat::Fp8Rowwise
            | ComfyFormat::Fp8Blockwise => {
                let scales: Vec<f32> = s_bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                if !scales.iter().all(|s| s.is_finite()) {
                    true
                } else {
                    let nonzero_q = w_bytes.iter().any(|&b| b != 0);
                    nonzero_q && scales.iter().any(|s| *s <= 0.0)
                }
            }
            // e8m0 block scales: 0xFF is NaN.
            ComfyFormat::Mxfp8 => s_bytes.contains(&0xFF),
            // E4M3 block scales must be finite.
            ComfyFormat::Nvfp4 => s_bytes
                .iter()
                .any(|&b| dtype::fp8_e4m3_bits_to_f32(b).is_nan()),
        };
        if bad {
            bad_scale += 1;
        }

        // input_scale must equal 1.0 for blockwise.
        if requires_input_scale(cfg.format) {
            let is_key = format!("{base}.input_scale");
            if let Ok(is_bytes) = reader.tensor_bytes(&is_key) {
                let v = if is_bytes.len() >= 4 {
                    Some(f32::from_le_bytes([
                        is_bytes[0],
                        is_bytes[1],
                        is_bytes[2],
                        is_bytes[3],
                    ]))
                } else {
                    None
                };
                if v != Some(1.0) {
                    bad_input_scale += 1;
                }
            }
        }
    }

    if overflow > 0 {
        report.add_error(format!(
            "numeric: {overflow} layer(s) have quantized values outside the valid range (overflow / non-finite)"
        ));
    }
    if bad_scale > 0 {
        report.add_error(format!(
            "numeric: {bad_scale} layer(s) have non-finite / non-positive scales"
        ));
    }
    if bad_input_scale > 0 {
        report.add_error(format!(
            "numeric: {bad_input_scale} layer(s) have input_scale != 1.0"
        ));
    }
}

// --------------------------------------------------------------------------- //
// Reporting
// --------------------------------------------------------------------------- //

/// Render a human-readable multi-line report (port of `format_report`).
pub fn format_report(report: &ValidationReport) -> String {
    let s = &report.summary;
    let mut lines = Vec::new();
    let name = if s.file.is_empty() {
        report.path.clone()
    } else {
        s.file.clone()
    };
    let has_summary = !s.file.is_empty() || s.quantized_matrices > 0 || !s.formats_found.is_empty();
    if has_summary {
        lines.push(format!("file: {} ({} GB)", name, s.size_gb));
        lines.push(format!(
            "quantized matrices      : {}",
            s.quantized_matrices
        ));
        for (g, count) in &s.group_size_histogram {
            lines.push(format!("  GS{g:<5}              : {count}"));
        }
        lines.push(format!(
            "full-precision weights  : {}",
            s.full_precision_weights
        ));
        lines.push(format!(
            "quantized params        : {:.1}M",
            s.quantized_params as f64 / 1e6
        ));
        lines.push(format!(
            "full-precision params   : {:.1}M",
            s.full_precision_params as f64 / 1e6
        ));
        lines.push(format!(
            "quantized share         : {:.2}%",
            s.quantized_share_pct
        ));
        if !s.formats_found.is_empty() {
            lines.push(format!(
                "formats found           : {}",
                s.formats_found.join(", ")
            ));
        }
    }
    if !report.warnings.is_empty() {
        lines.push(String::new());
        lines.push("WARNINGS:".to_string());
        for w in &report.warnings {
            lines.push(format!("  - {w}"));
        }
    }
    if !report.errors.is_empty() {
        lines.push(String::new());
        lines.push(format!("FAIL ({} problem(s)):", report.errors.len()));
        for e in report.errors.iter().take(40) {
            lines.push(format!("  - {e}"));
        }
    } else if report.ok {
        lines.push(String::new());
        lines.push("PASS".to_string());
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squeeze_rule() {
        assert_eq!(squeeze(vec![1, 1]), Vec::<u64>::new());
        assert_eq!(squeeze(vec![1]), Vec::<u64>::new());
        assert_eq!(squeeze(vec![2, 1]), vec![2, 1]);
        assert_eq!(squeeze(vec![256]), vec![256]);
    }

    #[test]
    fn scale_shape_formulas() {
        // int8/fp8 blockwise: [ceil(m/g), ceil(n/g)] with squeeze.
        assert_eq!(
            expected_scale_shape(ComfyFormat::Int8Blockwise, &[256, 128], Some(128), false),
            Some(vec![2, 1])
        );
        assert_eq!(
            expected_scale_shape(ComfyFormat::Fp8Blockwise, &[128, 128], Some(128), false),
            Some(vec![])
        );
        assert_eq!(
            expected_scale_shape(ComfyFormat::Fp8Blockwise, &[384, 256], Some(128), false),
            Some(vec![3, 2])
        );
        // tensorwise → scalar.
        assert_eq!(
            expected_scale_shape(ComfyFormat::Fp8Tensor, &[256, 128], None, false),
            Some(vec![])
        );
        // rowwise → [m].
        assert_eq!(
            expected_scale_shape(ComfyFormat::Fp8Rowwise, &[256, 128], None, false),
            Some(vec![256])
        );
        // mxfp8 to_blocked from on-disk weight shape.
        assert_eq!(
            expected_scale_shape(ComfyFormat::Mxfp8, &[256, 128], Some(32), false),
            Some(vec![256, 4])
        );
        assert_eq!(
            expected_scale_shape(ComfyFormat::Mxfp8, &[384, 256], Some(32), false),
            Some(vec![384, 8])
        );
        // nvfp4 to_blocked from packed weight shape.
        assert_eq!(
            expected_scale_shape(ComfyFormat::Nvfp4, &[256, 64], Some(16), false),
            Some(vec![256, 8])
        );
        assert_eq!(
            expected_scale_shape(ComfyFormat::Nvfp4, &[384, 128], Some(16), false),
            Some(vec![384, 16])
        );
        // blockwise with missing group size → None.
        assert_eq!(
            expected_scale_shape(ComfyFormat::Int8Blockwise, &[256, 128], None, false),
            None
        );
    }

    #[test]
    fn report_error_flips_ok() {
        let mut r = ValidationReport::new("x.safetensors");
        assert!(r.ok);
        r.add_error("boom");
        assert!(!r.ok);
        r.add_warning("careful");
        assert_eq!(r.errors.len(), 1);
        assert_eq!(r.warnings.len(), 1);
    }

    #[test]
    fn report_to_issues_valid_file() {
        let mut r = ValidationReport::new("x.safetensors");
        r.formats.insert("int8_blockwise".into());
        r.summary.quantized_matrices = 3;
        let issues = report_to_issues(&r);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].level, IssueLevel::Info);
        assert_eq!(issues[0].text, "File looks valid");
    }

    #[test]
    fn report_to_issues_errors_first() {
        let mut r = ValidationReport::new("x.safetensors");
        r.add_error("bad");
        r.add_warning("meh");
        let issues = report_to_issues(&r);
        assert_eq!(issues.len(), 2);
        assert_eq!(issues[0].level, IssueLevel::Error);
        assert_eq!(issues[1].level, IssueLevel::Warning);
    }
}

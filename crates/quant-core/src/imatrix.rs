//! Importance-matrix (imatrix) loading and weight derivation — a port of
//! llama.cpp's `common/imatrix-loader.cpp` + the derivation half of
//! `tools/quantize/quantize.cpp::load_imatrix`.
//!
//! Unsloth coverage plan Phase 4.1 (spec §3-B2). Source: llama.cpp
//! (MIT, "Copyright (c) 2023-2026 The ggml authors"). Line references point
//! into the vendored upstream tree.
//!
//! Two file formats, auto-detected (loader order per imatrix-loader.cpp:82):
//!
//! 1. **GGUF** (modern) — a GGUF container with tensor pairs named
//!    `<tensor>.in_sum2` (F32) and `<tensor>.counts` (F32, values are
//!    integer counts stored as floats). Metadata keys `imatrix.datasets`
//!    (string array), `imatrix.chunk_count`, `imatrix.chunk_size`
//!    (imatrix-loader.h:12-14). A GGUF imatrix without the metadata keys
//!    is rejected (quantize.cpp:190-193).
//!
//! 2. **Legacy** — raw binary: `i32 n_entries`, then per entry
//!    `i32 len` + name bytes (NOT NUL-terminated), `i32 ncall`,
//!    `i32 nval`, `nval × f32` sums; optional trailing
//!    `i32 n_calls` + `i32 len` + dataset name (imatrix-loader.cpp:10-80).
//!
//! **Weight derivation** (quantize.cpp:195-246) — the part that must be
//! reproduced exactly:
//! - GGUF: the sums array is divided into `ncounts` expert blocks of
//!   `ne0 = sums.len() / ncounts`. For each block `w[j*ne0+i] =
//!   sums[j*ne0+i] / count[j]`, and **if `count[j] <= 0` the whole block
//!   becomes 1.0** (not 0, not skipped).
//! - Legacy: `w[i] = sums[i] / ncall`, or `w[i] = sums[i]` when
//!   `ncall <= 0`.
//!
//! Phase 4.3 consumes these weights via `weights_for(gguf_name)`; the
//! per-tensor size check (`ne[0] * ne[2]`) and the 3-D `i03` slicing live
//! in the conversion driver (plan Phase 4.4), not here.

use std::collections::HashMap;
use std::path::Path;

use rlx_gguf::{GgmlType, GgufFile};

/// Metadata key names (imatrix-loader.h:12-14).
const KV_DATASETS: &str = "imatrix.datasets";
const KV_CHUNK_COUNT: &str = "imatrix.chunk_count";
const KV_CHUNK_SIZE: &str = "imatrix.chunk_size";

/// Errors from imatrix loading. Every variant names the file and the
/// specific cause — an imatrix is auxiliary input, so failures must be
/// actionable, not mysterious.
#[derive(Debug, thiserror::Error)]
pub enum ImatrixError {
    #[error("failed to open imatrix file '{path}': {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("imatrix file '{0}' is not a GGUF and not a valid legacy file: {1}")]
    NotImatrix(String, String),
    #[error("imatrix file '{0}' has no data (zero entries)")]
    NoEntries(String),
    #[error("imatrix file '{0}': missing required metadata ({1}) — a GGUF imatrix needs imatrix.datasets, imatrix.chunk_count and imatrix.chunk_size")]
    MissingMetadata(String, String),
    #[error(
        "imatrix file '{0}': mismatched .in_sum2/.counts pair for tensor '{1}' (one side missing)"
    )]
    MismatchedPair(String, String),
    #[error("imatrix file '{0}': tensor '{1}' sums/counts must be F32, got {2:?}/{3:?}")]
    BadDtype(String, String, GgmlType, GgmlType),
    #[error(
        "imatrix file '{0}': tensor '{1}' has {2} sums not evenly divisible into {3} counts blocks"
    )]
    UnevenBlocks(String, String, usize, usize),
    #[error("imatrix file '{0}': legacy entry '{1}' is truncated ({2})")]
    TruncatedLegacy(String, String, String),
    #[error("imatrix file '{0}': non-finite weight derived for tensor '{1}' (NaN/Inf in sums or counts)")]
    NonFinite(String, String),
}

/// Raw entry as stored in the file, before derivation.
#[derive(Debug, Clone)]
struct RawEntry {
    sums: Vec<f32>,
    /// GGUF: per-expert-block counts. Legacy: one element = ncall.
    counts: Vec<i64>,
}

/// One `.in_sum2` + `.counts` pairing under construction. Each side's
/// `*_dtype` is `Some(..)` iff the tensor exists in the file — that
/// distinguishes "missing partner" from "partner of the wrong dtype",
/// which upstream checks separately (:140-152).
#[derive(Debug, Clone, Default)]
struct Pair {
    sums: Option<Vec<f32>>,
    counts: Option<Vec<i64>>,
    sum_dtype: Option<GgmlType>,
    count_dtype: Option<GgmlType>,
}

/// A loaded, fully-derived importance matrix.
///
/// `weights` maps the GGUF tensor name (e.g. `blk.0.attn_q.weight`) to its
/// derived weight vector — `sums/count` per expert block, blocks with
/// `count <= 0` → all `1.0` (quantize.cpp:210-214).
#[derive(Debug, Clone, Default)]
pub struct Imatrix {
    /// Derived weight vectors, by GGUF tensor name.
    weights: HashMap<String, Vec<f32>>,
    /// Dataset names from the file (informational).
    pub datasets: Vec<String>,
    /// True when the file was the legacy raw-binary format.
    pub is_legacy: bool,
    /// GGUF metadata: number of chunks used to build the matrix.
    pub chunk_count: u32,
    /// GGUF metadata: chunk size.
    pub chunk_size: u32,
}

impl Imatrix {
    /// Derived weights for one GGUF tensor, or `None` if the file carries
    /// no entry for it. Callers apply the plan-§3-B2 consumption rules
    /// (size check, i03 slicing) to the returned slice.
    pub fn weights_for(&self, gguf_name: &str) -> Option<&[f32]> {
        self.weights.get(gguf_name).map(|v| v.as_slice())
    }

    /// Number of tensor entries in the file.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Load and fully derive an imatrix from `path`. GGUF first; when the
    /// file is not GGUF, falls back to the legacy binary format
    /// (imatrix-loader.cpp:82-90).
    pub fn load(path: &Path) -> Result<Self, ImatrixError> {
        match GgufFile::from_path(path) {
            Ok(f) => Self::load_gguf(path, f),
            Err(_) => Self::load_legacy(path),
        }
    }

    /// GGUF-format loader (imatrix-loader.cpp:82-173).
    fn load_gguf(path: &Path, f: GgufFile) -> Result<Self, ImatrixError> {
        let p = path.display().to_string();

        // Pair the .in_sum2/.counts tensors by base name (:118-133). The
        // HashMap iteration order does not matter — llama.cpp uses a
        // std::map (sorted) but only for deterministic logging; the data
        // result is order-independent.
        let mut pairs: HashMap<String, Pair> = HashMap::new();
        for (name, t) in &f.tensors {
            if let Some(base) = name.strip_suffix(".in_sum2") {
                // :147-152 — both tensors must be F32.
                let e = pairs.entry(base.to_string()).or_default();
                e.sum_dtype = Some(t.dtype);
                if t.dtype == GgmlType::F32 {
                    let bytes = f.tensor_bytes(t).map_err(|err| {
                        ImatrixError::NotImatrix(p.clone(), format!("reading tensor {name}: {err}"))
                    })?;
                    e.sums = Some(read_f32s(bytes));
                }
            } else if let Some(base) = name.strip_suffix(".counts") {
                let e = pairs.entry(base.to_string()).or_default();
                e.count_dtype = Some(t.dtype);
                if t.dtype == GgmlType::F32 {
                    let bytes = f.tensor_bytes(t).map_err(|err| {
                        ImatrixError::NotImatrix(p.clone(), format!("reading tensor {name}: {err}"))
                    })?;
                    // counts are stored as F32 floats holding integer
                    // values, lround'ed on load (:166).
                    e.counts = Some(
                        read_f32s(bytes)
                            .into_iter()
                            .map(|c| c.round() as i64)
                            .collect(),
                    );
                }
            }
        }

        if pairs.is_empty() {
            return Err(ImatrixError::NoEntries(p));
        }

        // Metadata (:101-116). All three keys required (quantize.cpp:190).
        let datasets = match f.metadata.get(KV_DATASETS) {
            Some(rlx_gguf::MetaValue::Array(items)) => items
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        let chunk_count = f
            .metadata
            .get(KV_CHUNK_COUNT)
            .and_then(rlx_gguf::MetaValue::as_u32)
            .unwrap_or(0);
        let chunk_size = f
            .metadata
            .get(KV_CHUNK_SIZE)
            .and_then(rlx_gguf::MetaValue::as_u32)
            .unwrap_or(0);
        if datasets.is_empty() || chunk_count == 0 || chunk_size == 0 {
            return Err(ImatrixError::MissingMetadata(
                p,
                format!("{KV_DATASETS}, {KV_CHUNK_COUNT}, {KV_CHUNK_SIZE}"),
            ));
        }

        // Build the raw entries with the F32/mismatch checks (:135-167).
        // Per side: dtype is Some(..) iff the tensor exists. So:
        //  - a side with dtype Some(non-F32) → BadDtype (:147-152);
        //  - a side with dtype None → the partner tensor is missing →
        //    MismatchedPair (:140-145).
        // Upstream reports mismatch first (its map lookup finds one null
        // pointer), but a dtype error is more actionable when both exist;
        // when a side is missing, only MismatchedPair can fire anyway.
        let mut raw: HashMap<String, RawEntry> = HashMap::new();
        for (name, pair) in pairs {
            let Pair {
                sums,
                counts,
                sum_dtype,
                count_dtype,
            } = pair;
            let bad_sums = matches!(sum_dtype, Some(dt) if dt != GgmlType::F32);
            let bad_counts = matches!(count_dtype, Some(dt) if dt != GgmlType::F32);
            if bad_sums || bad_counts {
                return Err(ImatrixError::BadDtype(
                    p.clone(),
                    name,
                    sum_dtype.unwrap_or(GgmlType::F32),
                    count_dtype.unwrap_or(GgmlType::F32),
                ));
            }
            let (Some(sums), Some(counts)) = (sums, counts) else {
                return Err(ImatrixError::MismatchedPair(p.clone(), name));
            };
            if sums.is_empty() || counts.is_empty() {
                return Err(ImatrixError::MismatchedPair(p.clone(), name));
            }
            raw.insert(name, RawEntry { sums, counts });
        }

        let mut im = Self {
            datasets,
            is_legacy: false,
            chunk_count,
            chunk_size,
            weights: HashMap::new(),
        };
        im.derive(raw, &p)?;
        Ok(im)
    }

    /// Legacy raw-binary loader (imatrix-loader.cpp:10-80).
    fn load_legacy(path: &Path) -> Result<Self, ImatrixError> {
        let p = path.display().to_string();
        let mut data = std::fs::read(path).map_err(|source| ImatrixError::Open {
            path: p.clone(),
            source,
        })?;

        // Reader helpers over the trailing bytes; each read consumes from
        // the front (mirrors the sequential ifstream reads upstream).
        fn take_i32(data: &mut Vec<u8>, what: &str) -> Result<i32, ImatrixError> {
            if data.len() < 4 {
                return Err(ImatrixError::TruncatedLegacy(
                    String::new(),
                    String::new(),
                    what.to_string(),
                ));
            }
            let v = i32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            data.drain(..4);
            Ok(v)
        }

        let n_entries = take_i32(&mut data, "n_entries").map_err(|e| match e {
            ImatrixError::TruncatedLegacy(.., what) => {
                ImatrixError::NotImatrix(p.clone(), format!("reading {what}: file too short"))
            }
            other => other,
        })?;
        if n_entries < 1 {
            return Err(ImatrixError::NoEntries(p));
        }

        let mut raw: HashMap<String, RawEntry> = HashMap::new();
        for i in 0..n_entries {
            let idx = i as usize + 1;
            // i32 len + name bytes (not NUL-terminated; :25-34).
            let len =
                take_i32(&mut data, "name length").map_err(|e| not_imatrix(p.clone(), e, idx))?;
            if len < 0 || data.len() < len as usize {
                return Err(ImatrixError::NotImatrix(
                    p.clone(),
                    format!("entry {idx}: name length {len} exceeds remaining file"),
                ));
            }
            let name = String::from_utf8_lossy(&data[..len as usize]).into_owned();
            data.drain(..len as usize);

            let ncall = take_i32(&mut data, "ncall").map_err(|e| not_imatrix(p.clone(), e, idx))?;
            let nval = take_i32(&mut data, "nval").map_err(|e| not_imatrix(p.clone(), e, idx))?;
            if nval < 1 || data.len() < nval as usize * 4 {
                return Err(ImatrixError::NotImatrix(
                    p.clone(),
                    format!("entry {idx} ({name}): {nval} values exceed remaining file"),
                ));
            }
            let sums: Vec<f32> = data[..nval as usize * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            data.drain(..nval as usize * 4);

            raw.insert(
                name,
                RawEntry {
                    sums,
                    counts: vec![ncall as i64],
                },
            );
        }

        // Optional trailing chunk-count + dataset name (:57-74).
        let mut datasets = Vec::new();
        let mut chunk_count = 0u32;
        if !data.is_empty() {
            if let Ok(n_calls) = take_i32(&mut data, "trailing n_calls") {
                chunk_count = n_calls.max(0) as u32;
                if let Ok(len) = take_i32(&mut data, "trailing dataset len") {
                    if len > 0 && data.len() >= len as usize {
                        datasets.push(String::from_utf8_lossy(&data[..len as usize]).into_owned());
                    }
                }
            }
        }

        let mut im = Self {
            weights: HashMap::new(),
            datasets,
            is_legacy: true,
            chunk_count,
            chunk_size: 0, // :76 — legacy has no chunk size
        };
        im.derive(raw, &p)?;
        Ok(im)
    }

    /// Weight derivation, both formats (quantize.cpp:195-246).
    fn derive(&mut self, raw: HashMap<String, RawEntry>, path: &str) -> Result<(), ImatrixError> {
        for (name, entry) in raw {
            let w: Vec<f32> = if !self.is_legacy {
                // GGUF: per-expert-block division. ne0 = sums.len() /
                // counts.len() (:201-215). count<=0 → the whole block is 1.0.
                let ncounts = entry.counts.len();
                if entry.sums.len() % ncounts != 0 {
                    return Err(ImatrixError::UnevenBlocks(
                        path.to_string(),
                        name,
                        entry.sums.len(),
                        ncounts,
                    ));
                }
                let ne0 = entry.sums.len() / ncounts;
                let mut out = vec![0f32; entry.sums.len()];
                for (j, &count) in entry.counts.iter().enumerate() {
                    let block = &mut out[j * ne0..(j + 1) * ne0];
                    if count > 0 {
                        for (i, &s) in entry.sums[j * ne0..(j + 1) * ne0].iter().enumerate() {
                            block[i] = s / count as f32;
                        }
                    } else {
                        block.fill(1.0);
                    }
                }
                out
            } else {
                // Legacy: divide by ncall, or passthrough when ncall<=0
                // (:228-239).
                let ncall = entry.counts.first().copied().unwrap_or(0);
                if ncall > 0 {
                    entry.sums.iter().map(|&s| s / ncall as f32).collect()
                } else {
                    entry.sums
                }
            };

            // Non-finite values anywhere → hard error before any work
            // (quantize.cpp:949-951 checks this before quantizing; we
            // check at load so the failure happens before partial output).
            if w.iter().any(|v| !v.is_finite()) {
                return Err(ImatrixError::NonFinite(path.to_string(), name));
            }
            self.weights.insert(name, w);
        }
        Ok(())
    }
}

/// F32 payload decode for an `.in_sum2`/`.counts` tensor (both are F32).
fn read_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Normalize the internal truncated-legacy errors (which carry no context)
/// into a NotImatrix with the entry index.
fn not_imatrix(path: String, e: ImatrixError, idx: usize) -> ImatrixError {
    match e {
        ImatrixError::TruncatedLegacy(.., what) => {
            ImatrixError::NotImatrix(path, format!("entry {idx}: reading {what}: file too short"))
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_gguf::{GgufWriter, MetaValue};

    // ─── GGUF format ────────────────────────────────────────────────

    /// Build a GGUF imatrix file with two tensors (one with 2 expert
    /// blocks) exactly as llama.cpp writes them.
    fn write_gguf_imatrix(path: &Path) {
        let mut w = GgufWriter::new();
        w.set_arch("llama");
        w.set_meta(
            KV_DATASETS,
            MetaValue::Array(vec![MetaValue::String("wiki".into())]),
        );
        w.set_meta(KV_CHUNK_COUNT, MetaValue::U32(64));
        w.set_meta(KV_CHUNK_SIZE, MetaValue::U32(512));

        // t1: 8 sums, 2 counts → 2 blocks of 4.
        let sums1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let counts1: Vec<f32> = vec![2.0, 0.0]; // block 2: count 0 → all 1.0
        let b: Vec<u8> = sums1.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor_bytes("blk.0.attn_q.in_sum2", vec![8], GgmlType::F32, b)
            .unwrap();
        let b: Vec<u8> = counts1.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor_bytes("blk.0.attn_q.counts", vec![2], GgmlType::F32, b)
            .unwrap();

        // t2: single block, count 4.
        let sums2: Vec<f32> = vec![8.0, 12.0, 16.0, 20.0];
        let b: Vec<u8> = sums2.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor_bytes("blk.0.attn_v.in_sum2", vec![4], GgmlType::F32, b)
            .unwrap();
        let b: Vec<u8> = vec![4.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor_bytes("blk.0.attn_v.counts", vec![1], GgmlType::F32, b)
            .unwrap();

        w.write_to_path(path).unwrap();
    }

    #[test]
    fn gguf_format_round_trip_and_derivation() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("imatrix.gguf");
        write_gguf_imatrix(&path);

        let im = Imatrix::load(&path).unwrap();
        assert!(!im.is_legacy);
        assert_eq!(im.datasets, vec!["wiki"]);
        assert_eq!(im.chunk_count, 64);
        assert_eq!(im.chunk_size, 512);
        assert_eq!(im.len(), 2);

        // Block 1: sums/count → [0.5, 1.0, 1.5, 2.0].
        // Block 2: count==0 → all 1.0 (quantize.cpp:210-214).
        assert_eq!(
            im.weights_for("blk.0.attn_q"),
            Some([0.5, 1.0, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0].as_slice())
        );
        // Single block: sums/4 → [2, 3, 4, 5].
        assert_eq!(
            im.weights_for("blk.0.attn_v"),
            Some([2.0, 3.0, 4.0, 5.0].as_slice())
        );
        // Missing tensor → None (the caller decides policy).
        assert_eq!(im.weights_for("blk.9.ffn_down"), None);
    }

    #[test]
    fn gguf_missing_metadata_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("imatrix.gguf");
        // Same tensors, no metadata keys.
        let mut w = GgufWriter::new();
        w.set_arch("llama");
        let b: Vec<u8> = vec![1.0f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.add_tensor_bytes("t.in_sum2", vec![2], GgmlType::F32, b)
            .unwrap();
        let b: Vec<u8> = vec![1.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor_bytes("t.counts", vec![1], GgmlType::F32, b)
            .unwrap();
        w.write_to_path(&path).unwrap();

        let err = Imatrix::load(&path).unwrap_err();
        assert!(
            matches!(err, ImatrixError::MissingMetadata(..)),
            "got: {err}"
        );
    }

    #[test]
    fn gguf_mismatched_pair_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("imatrix.gguf");
        let mut w = GgufWriter::new();
        w.set_arch("llama");
        w.set_meta(
            KV_DATASETS,
            MetaValue::Array(vec![MetaValue::String("d".into())]),
        );
        w.set_meta(KV_CHUNK_COUNT, MetaValue::U32(1));
        w.set_meta(KV_CHUNK_SIZE, MetaValue::U32(1));
        // in_sum2 without its counts partner.
        let b: Vec<u8> = vec![1.0f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.add_tensor_bytes("t.in_sum2", vec![2], GgmlType::F32, b)
            .unwrap();
        w.write_to_path(&path).unwrap();

        let err = Imatrix::load(&path).unwrap_err();
        assert!(
            matches!(err, ImatrixError::MismatchedPair(..)),
            "got: {err}"
        );
    }

    #[test]
    fn gguf_non_f32_pair_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("imatrix.gguf");
        let mut w = GgufWriter::new();
        w.set_arch("llama");
        w.set_meta(
            KV_DATASETS,
            MetaValue::Array(vec![MetaValue::String("d".into())]),
        );
        w.set_meta(KV_CHUNK_COUNT, MetaValue::U32(1));
        w.set_meta(KV_CHUNK_SIZE, MetaValue::U32(1));
        // in_sum2 as F16 instead of F32 → llama.cpp :147-152 rejects.
        let b: Vec<u8> = vec![half::f16::from_f32(1.0); 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.add_tensor_bytes("t.in_sum2", vec![2], GgmlType::F16, b)
            .unwrap();
        let b: Vec<u8> = vec![1.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor_bytes("t.counts", vec![1], GgmlType::F32, b)
            .unwrap();
        w.write_to_path(&path).unwrap();

        let err = Imatrix::load(&path).unwrap_err();
        match err {
            ImatrixError::BadDtype(_, name, sums, counts) => {
                assert_eq!(name, "t");
                assert_eq!(sums, GgmlType::F16, "sums dtype must be reported");
                assert_eq!(counts, GgmlType::F32);
            }
            other => panic!("expected BadDtype, got {other:?}"),
        }
    }

    // ─── Legacy format ──────────────────────────────────────────────

    /// Serialize the legacy raw format byte-for-byte as
    /// imatrix-loader.cpp:10-80 reads it.
    fn write_legacy_imatrix(path: &Path, with_trailer: bool) {
        let mut out = Vec::new();
        out.extend_from_slice(&2i32.to_le_bytes()); // n_entries
                                                    // Entry 1: "blk.0.attn_q", ncall=4, 4 sums.
        let name = b"blk.0.attn_q";
        out.extend_from_slice(&(name.len() as i32).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&4i32.to_le_bytes()); // ncall
        out.extend_from_slice(&4i32.to_le_bytes()); // nval
        for v in [4.0f32, 8.0, 12.0, 16.0] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        // Entry 2: "blk.1.attn_v", ncall=0 → passthrough.
        let name = b"blk.1.attn_v";
        out.extend_from_slice(&(name.len() as i32).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&0i32.to_le_bytes()); // ncall
        out.extend_from_slice(&2i32.to_le_bytes()); // nval
        for v in [7.0f32, 9.0] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        if with_trailer {
            out.extend_from_slice(&32i32.to_le_bytes()); // n_calls
            let ds = b"group1/texts";
            out.extend_from_slice(&(ds.len() as i32).to_le_bytes());
            out.extend_from_slice(ds);
        }
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn legacy_round_trip_and_derivation() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("imatrix.dat");
        write_legacy_imatrix(&path, true);

        let im = Imatrix::load(&path).unwrap();
        assert!(im.is_legacy);
        assert_eq!(im.chunk_count, 32);
        assert_eq!(im.chunk_size, 0); // :76
        assert_eq!(im.datasets, vec!["group1/texts"]);
        // ncall=4: sums/ncall → [1, 2, 3, 4].
        assert_eq!(
            im.weights_for("blk.0.attn_q"),
            Some([1.0, 2.0, 3.0, 4.0].as_slice())
        );
        // ncall=0: passthrough (quantize.cpp:235-238).
        assert_eq!(im.weights_for("blk.1.attn_v"), Some([7.0, 9.0].as_slice()));
    }

    #[test]
    fn legacy_without_trailer_loads() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("imatrix.dat");
        write_legacy_imatrix(&path, false);
        let im = Imatrix::load(&path).unwrap();
        assert!(im.is_legacy);
        assert!(im.datasets.is_empty());
        assert_eq!(im.chunk_count, 0);
        assert_eq!(im.len(), 2);
    }

    #[test]
    fn legacy_truncated_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("imatrix.dat");
        // Valid header, then truncated mid-name.
        std::fs::write(
            &path,
            (1i32.to_le_bytes())
                .iter()
                .chain([8, 0, 0, 0].iter())
                .cloned()
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        let err = Imatrix::load(&path).unwrap_err();
        assert!(matches!(err, ImatrixError::NotImatrix(..)), "got: {err}");
    }

    #[test]
    fn empty_file_and_garbage_are_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        // Zero-byte file → not GGUF, not legacy.
        let path = tmp.path().join("empty.dat");
        std::fs::write(&path, b"").unwrap();
        assert!(Imatrix::load(&path).is_err());

        // n_entries = 0 → NoEntries (imatrix-loader.cpp:19).
        let path = tmp.path().join("zero_entries.dat");
        std::fs::write(&path, 0i32.to_le_bytes()).unwrap();
        let err = Imatrix::load(&path).unwrap_err();
        assert!(matches!(err, ImatrixError::NoEntries(..)), "got: {err}");
    }

    #[test]
    fn missing_file_names_the_path() {
        let err = Imatrix::load(Path::new("does_not_exist.dat")).unwrap_err();
        match err {
            ImatrixError::Open { path, .. } => assert!(path.contains("does_not_exist")),
            other => panic!("expected Open, got {other:?}"),
        }
    }

    #[test]
    fn non_finite_sums_are_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("imatrix.dat");
        let mut out = Vec::new();
        out.extend_from_slice(&1i32.to_le_bytes());
        let name = b"t";
        out.extend_from_slice(&(name.len() as i32).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&1i32.to_le_bytes()); // ncall
        out.extend_from_slice(&2i32.to_le_bytes()); // nval
        out.extend_from_slice(&f32::NAN.to_le_bytes());
        out.extend_from_slice(&1.0f32.to_le_bytes());
        std::fs::write(&path, out).unwrap();

        let err = Imatrix::load(&path).unwrap_err();
        assert!(matches!(err, ImatrixError::NonFinite(..)), "got: {err}");
    }

    /// Golden-style regression: a REAL llama.cpp-written GGUF imatrix
    /// (chunked runs) round-trips through our loader with identical
    /// weights. The fixture is generated by hand following the writer
    /// side (llama-imatrix writes `<name>.in_sum2` + `<name>.counts`
    /// with counts per expert block); when a genuine llama.cpp-produced
    /// file becomes available, replace this fixture with it.
    #[test]
    fn multi_expert_blocks_derive_per_block() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("imatrix.gguf");
        let mut w = GgufWriter::new();
        w.set_arch("llama");
        w.set_meta(
            KV_DATASETS,
            MetaValue::Array(vec![MetaValue::String("d".into())]),
        );
        w.set_meta(KV_CHUNK_COUNT, MetaValue::U32(2));
        w.set_meta(KV_CHUNK_SIZE, MetaValue::U32(8));
        // 8 sums across 4 expert blocks (ne0=2), counts [2,1,0,5].
        let sums: Vec<f32> = vec![2.0, 4.0, 3.0, 6.0, 9.0, 9.0, 10.0, 20.0];
        let counts: Vec<f32> = vec![2.0, 1.0, 0.0, 5.0];
        let b: Vec<u8> = sums.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor_bytes("blk.0.ffn_down.in_sum2", vec![8], GgmlType::F32, b)
            .unwrap();
        let b: Vec<u8> = counts.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor_bytes("blk.0.ffn_down.counts", vec![4], GgmlType::F32, b)
            .unwrap();
        w.write_to_path(&path).unwrap();

        let im = Imatrix::load(&path).unwrap();
        assert_eq!(
            im.weights_for("blk.0.ffn_down"),
            // block0 /2 → [1,2]; block1 /1 → [3,6]; block2 count 0 → [1,1];
            // block3 /5 → [2,4].
            Some([1.0, 2.0, 3.0, 6.0, 1.0, 1.0, 2.0, 4.0].as_slice())
        );
    }
}

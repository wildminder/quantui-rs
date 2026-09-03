//! Resumable streaming quantization orchestrator (plan Phase 5, step 4.3;
//! Phase 7.3 bias correction; Phase 6.2 sharded input/output).
//!
//! Port of reference `stream_quant.py`:
//! - `stream_quantize` — single file → single file
//! - union streaming — several shards → ONE output file
//!   (`_resolve_union_header` + single `stream_quantize` body)
//! - `stream_quantize_sharded` — sharded model → sharded OUTPUT directory
//!   (one output shard per input shard, index json + sidecars copied verbatim,
//!   global `.quant-manifest.json`)
//!
//! Semantics (all mirrored from the reference):
//! - iterate tensors in file order (union: first-appearance order across shards)
//! - 2D `.weight` tensors that are quantizable (not excluded, dims divisible by
//!   block size when heur is on) are INT8-quantized into
//!   `<name>` + `<base>.weight_scale` + `<base>.comfy_quant` + `<base>.input_scale`
//!   AND trigger correction of their sibling `.bias` via simulated-calibration
//!   mean output-error subtraction (`bias_correction` module)
//! - skipped 2D `.weight` tensors are cast to the output dtype (bf16/f16)
//!   when they're a castable float dtype and differ from it
//! - everything else passes through verbatim
//! - biases reached BEFORE their weight defer the weight processing so the
//!   corrected value is available at the bias's file position (works across
//!   shards in union mode)
//! - manifest saved after EVERY tensor; resume skips names in `done`
//! - config-hash mismatch → clean restart from zero
//! - `__metadata__` policy (plan Phase D, §3.3): INT8/FP8 outputs NEVER carry
//!   metadata (the INT8 reference rebuilds the header via `_resolve_union_header`,
//!   which drops `__metadata__`; the ctq FP8 unified path only adds metadata
//!   behind the off-by-default `--save-quant-metadata` flag — golden-verified
//!   absent). MXFP8/NVFP4 outputs carry
//!   `__metadata__._quantization_metadata` — a byte-exact JSON string listing
//!   exactly the layers quantized in that file (per-shard in sharded mode).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde_json::Map;

use crate::bias_correction::{correct_bias, CalibCache};
use crate::comfy_schema::{
    encode_block_format, encode_comfy_quant, encode_comfy_quant_int8_convrot, ComfyFormat,
};
use crate::discover::{ShardedModel, INDEX_NAME};
use crate::dtype::{bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits, DType};
use crate::manifest::{Format, QuantConfig, ScalingMode, StreamState, MANIFEST_VERSION};
use crate::quant::{dequantize_int8, quantize_int8_weight, should_skip_shape};
use crate::quant_fp8::{dequantize_fp8, quantize_fp8_weight, Fp8ScalingMode};
use crate::quant_mxfp8::{dequantize_mxfp8, quantize_mxfp8_weight};
use crate::quant_nvfp4::{dequantize_nvfp4, quantize_nvfp4_weight};
use crate::st_io::header::TensorInfo;
use crate::st_io::reader::SafetensorsReader;
use crate::st_io::writer::IncrementalWriter;

/// Output result of one full streaming run (mirrors reference return dict).
#[derive(Debug)]
pub struct StreamResult {
    pub config_hash: String,
    pub order: Vec<String>,
    pub done: Vec<String>,
}

/// Output result of a sharded streaming run (mirrors the reference global
/// manifest dict).
#[derive(Debug)]
pub struct ShardedStreamResult {
    pub config_hash: String,
    pub output_dir: PathBuf,
    /// shard filename → sorted done list, in shard order.
    pub shards: Vec<(String, Vec<String>)>,
}

/// Global manifest filename inside a sharded output directory
/// (`stream_quant.py::SHARDED_MANIFEST_NAME`).
pub const SHARDED_MANIFEST_NAME: &str = ".quant-manifest.json";

/// Progress callback `(cur, total)` invoked after each completed tensor
/// (single/union runs) or shard (sharded runs) — mirrors the reference
/// `on_progress: Callable[[int, int], None]`. For a resumed run `cur` counts
/// already-done tensors too (`len(state.done)`), so it starts above zero.
pub type ProgressFn<'a> = dyn FnMut(usize, usize) + 'a;

/// Errors from the orchestrator.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("input not found: {0}")]
    InputNotFound(std::path::PathBuf),
    #[error("no input shards given")]
    NoInputShards,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    StIo(#[from] crate::st_io::Error),
    #[error(
        "INT8 block-wise requires dims divisible by block_size={block_size}, got ({m}, {n}); \
         enable skip_inefficient to copy such layers instead"
    )]
    NotDivisible { m: u64, n: u64, block_size: u32 },
    /// Cooperative cancellation requested (e.g. Ctrl-C). The partial output and
    /// manifest are already flushed to a valid, resumable state at the last
    /// completed tensor.
    #[error("cancelled after {done} of {total} tensors (partial output is resumable)")]
    Cancelled { done: usize, total: usize },
    /// Bias correction needs a float bias (f32/f16/bf16); got something else.
    #[error("bias `{name}` has unsupported dtype {dtype} for bias correction (need f32/f16/bf16)")]
    UnsupportedBiasDtype { name: String, dtype: DType },
    /// ConvRot requested with a format/scaling-mode combination the reference
    /// never rotates (learned_rounding.py:869: the rotation is gated on
    /// `self.convrot and self.scaling_mode == "row"`). Any other combination
    /// would silently emit a plain, unrotated layer under a ConvRot name —
    /// the one thing this CLI must never do (plan §3-G, decision Q3).
    #[error(
        "ConvRot requires --format int8 with --scaling-mode row (got format {format}, \
         scaling mode {scaling_mode}): the reference applies the Hadamard rotation only \
         in INT8 row-wise mode"
    )]
    ConvRotRequiresInt8Row {
        format: &'static str,
        scaling_mode: &'static str,
    },
    /// ConvRot group size is not on the power-of-4 ladder (4/16/64/256/1024),
    /// so no regular-Hadamard rotation exists for it.
    #[error(
        "ConvRot group size {0} is invalid: must be a power of 4 (4, 16, 64, 256, 1024) \
         (the scipy Sylvester fallback for other powers of two is not ported)"
    )]
    ConvRotBadGroupSize(u32),
    #[error(transparent)]
    ConvRot(#[from] crate::convrot::ConvRotError),
}

pub type Result<T> = std::result::Result<T, StreamError>;

// --------------------------------------------------------------------------- //
// Tensor source: one file OR several shards as one logical tensor sequence.
// --------------------------------------------------------------------------- //

/// One or more safetensors files presented as a single logical tensor sequence.
pub trait TensorSource {
    /// Ordered tensor names: file order for a single file; first-appearance
    /// union order across shards (each shard in its header order).
    fn names(&self) -> &[String];
    fn info(&self, name: &str) -> Option<&TensorInfo>;
    fn tensor_bytes(&self, name: &str) -> Result<&[u8]>;
}

/// Single-file source.
pub struct SingleFileSource {
    reader: SafetensorsReader,
    names: Vec<String>,
}

impl SingleFileSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(StreamError::InputNotFound(path.to_path_buf()));
        }
        let reader = SafetensorsReader::open(path)?;
        let names = reader.header().names().cloned().collect();
        Ok(Self { reader, names })
    }
}

impl TensorSource for SingleFileSource {
    fn names(&self) -> &[String] {
        &self.names
    }
    fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.reader.header().get(name)
    }
    fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        Ok(self.reader.tensor_bytes(name)?)
    }
}

/// Union of several shards (mirrors `_resolve_union_header` + per-shard
/// `safe_open` handles): first occurrence wins for duplicate names.
pub struct UnionSource {
    readers: Vec<SafetensorsReader>,
    entries: Vec<(String, TensorInfo)>,
    names: Vec<String>,
    name_to_shard: HashMap<String, usize>,
}

impl UnionSource {
    pub fn open(shard_paths: &[PathBuf]) -> Result<Self> {
        if shard_paths.is_empty() {
            return Err(StreamError::NoInputShards);
        }
        let mut readers = Vec::with_capacity(shard_paths.len());
        for sp in shard_paths {
            if !sp.exists() {
                return Err(StreamError::InputNotFound(sp.clone()));
            }
            readers.push(SafetensorsReader::open(sp)?);
        }
        let mut entries: Vec<(String, TensorInfo)> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        let mut name_to_shard: HashMap<String, usize> = HashMap::new();
        for (idx, reader) in readers.iter().enumerate() {
            for (name, info) in reader.header().iter() {
                if !name_to_shard.contains_key(name) {
                    name_to_shard.insert(name.clone(), idx);
                    names.push(name.clone());
                    entries.push((name.clone(), info.clone()));
                }
            }
        }
        Ok(Self {
            readers,
            entries,
            names,
            name_to_shard,
        })
    }
}

impl TensorSource for UnionSource {
    fn names(&self) -> &[String] {
        &self.names
    }
    fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.entries.iter().find(|(n, _)| n == name).map(|(_, i)| i)
    }
    fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let idx = self.name_to_shard[name];
        Ok(self.readers[idx].tensor_bytes(name)?)
    }
}

// --------------------------------------------------------------------------- //
// Public entry points.
// --------------------------------------------------------------------------- //

/// Run streaming quantization of one `input_path` → `output_path`, resumable.
pub fn stream_quantize(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    config: &QuantConfig,
) -> Result<StreamResult> {
    stream_quantize_with_progress(input_path, output_path, config, None)
}

/// Like [`stream_quantize`] with an optional `(cur, total)` progress callback
/// invoked after each tensor (mirrors reference `on_progress`).
pub fn stream_quantize_with_progress(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    config: &QuantConfig,
    on_progress: Option<&mut ProgressFn>,
) -> Result<StreamResult> {
    let source = SingleFileSource::open(input_path)?;
    stream_quantize_source(&source, output_path.as_ref(), config, on_progress, None)
}

/// Like [`stream_quantize_with_progress`] with a cooperative cancellation flag.
/// When the flag is set, the run stops at the next tensor boundary and returns
/// [`StreamError::Cancelled`], leaving a valid, resumable partial output.
pub fn stream_quantize_cancellable(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    config: &QuantConfig,
    on_progress: Option<&mut ProgressFn>,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<StreamResult> {
    let source = SingleFileSource::open(input_path)?;
    stream_quantize_source(
        &source,
        output_path.as_ref(),
        config,
        on_progress,
        Some(cancel),
    )
}

/// Stream several shards as ONE logical tensor sequence into a single output
/// file (reference `--output-mode single` for sharded inputs; replaces the
/// reference's temp-merge).
pub fn stream_quantize_shards(
    shard_paths: &[PathBuf],
    output_path: impl AsRef<Path>,
    config: &QuantConfig,
) -> Result<StreamResult> {
    stream_quantize_shards_with_progress(shard_paths, output_path, config, None)
}

/// Like [`stream_quantize_shards`] with an optional `(cur, total)` progress
/// callback invoked after each tensor of the union sequence.
pub fn stream_quantize_shards_with_progress(
    shard_paths: &[PathBuf],
    output_path: impl AsRef<Path>,
    config: &QuantConfig,
    on_progress: Option<&mut ProgressFn>,
) -> Result<StreamResult> {
    let source = UnionSource::open(shard_paths)?;
    stream_quantize_source(&source, output_path.as_ref(), config, on_progress, None)
}

/// Like [`stream_quantize_shards_with_progress`] with a cooperative
/// cancellation flag (see [`stream_quantize_cancellable`]).
pub fn stream_quantize_shards_cancellable(
    shard_paths: &[PathBuf],
    output_path: impl AsRef<Path>,
    config: &QuantConfig,
    on_progress: Option<&mut ProgressFn>,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<StreamResult> {
    let source = UnionSource::open(shard_paths)?;
    stream_quantize_source(
        &source,
        output_path.as_ref(),
        config,
        on_progress,
        Some(cancel),
    )
}

/// Stream-quantize a sharded model into a sharded OUTPUT directory
/// (port of `stream_quant.py::stream_quantize_sharded`).
///
/// Each input shard is quantized into its own output shard (never merged),
/// each individually resumable via its own `<shard>.quant-manifest.json`.
/// Afterwards `model.safetensors.index.json` and the non-weight sidecars are
/// copied verbatim, and a global `.quant-manifest.json` is written.
pub fn stream_quantize_sharded(
    model: &ShardedModel,
    output_dir: impl AsRef<Path>,
    config: &QuantConfig,
) -> Result<ShardedStreamResult> {
    stream_quantize_sharded_with_progress(model, output_dir, config, None)
}

/// Like [`stream_quantize_sharded`] with an optional `(shards_done,
/// total_shards)` progress callback invoked after each shard (mirrors the
/// reference, which passes `on_progress=None` to the per-shard runs and only
/// reports shard-level progress).
pub fn stream_quantize_sharded_with_progress(
    model: &ShardedModel,
    output_dir: impl AsRef<Path>,
    config: &QuantConfig,
    on_progress: Option<&mut ProgressFn>,
) -> Result<ShardedStreamResult> {
    stream_quantize_sharded_inner(model, output_dir, config, on_progress, None)
}

/// Like [`stream_quantize_sharded_with_progress`] with a cooperative
/// cancellation flag. The flag is checked both between shards and (via the
/// per-shard runs) between tensors, so cancellation stops at the nearest
/// tensor boundary, leaving every started shard valid and resumable.
pub fn stream_quantize_sharded_cancellable(
    model: &ShardedModel,
    output_dir: impl AsRef<Path>,
    config: &QuantConfig,
    on_progress: Option<&mut ProgressFn>,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<ShardedStreamResult> {
    stream_quantize_sharded_inner(model, output_dir, config, on_progress, Some(cancel))
}

fn stream_quantize_sharded_inner(
    model: &ShardedModel,
    output_dir: impl AsRef<Path>,
    config: &QuantConfig,
    mut on_progress: Option<&mut ProgressFn>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<ShardedStreamResult> {
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir)?;

    let total = model.shard_files.len();
    let mut per_shard: Vec<(String, Vec<String>)> = Vec::new();
    for (i, shard) in model.shard_files.iter().enumerate() {
        // Cooperative cancellation between shards.
        if let Some(flag) = cancel {
            if flag.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(StreamError::Cancelled { done: i, total });
            }
        }
        let in_path = model.model_dir.join(shard);
        let out_path = output_dir.join(shard);
        // Per-shard runs get NO tensor-level callback (reference passes
        // on_progress=None to the inner stream_quantize), but they DO honor
        // the cancellation flag so a Ctrl-C stops at a tensor boundary.
        let result = match cancel {
            Some(flag) => stream_quantize_cancellable(&in_path, &out_path, config, None, flag)?,
            None => stream_quantize(&in_path, &out_path, config)?,
        };
        per_shard.push((shard.clone(), result.done));
        if let Some(cb) = on_progress.as_deref_mut() {
            cb(i + 1, total);
        }
    }

    // Copy index json + sidecars unchanged (shard filenames / tensor names
    // stay valid).
    std::fs::copy(&model.index_path, output_dir.join(INDEX_NAME))?;
    for fname in &model.non_weight_files {
        let src = model.model_dir.join(fname);
        if src.is_file() {
            std::fs::copy(&src, output_dir.join(fname))?;
        }
    }

    // Global manifest (compact JSON, key order version/config_hash/output_dir/
    // shards; shards in shard order — mirrors json.dump of the reference dict).
    let config_hash = config.config_hash();
    let mut obj = serde_json::Map::new();
    obj.insert("version".into(), serde_json::Value::from(MANIFEST_VERSION));
    obj.insert(
        "config_hash".into(),
        serde_json::Value::from(config_hash.clone()),
    );
    obj.insert(
        "output_dir".into(),
        serde_json::Value::from(output_dir.to_string_lossy().into_owned()),
    );
    let mut shards_obj = serde_json::Map::new();
    for (shard, done) in &per_shard {
        shards_obj.insert(
            shard.clone(),
            serde_json::Value::Array(
                done.iter()
                    .map(|d| serde_json::Value::from(d.clone()))
                    .collect(),
            ),
        );
    }
    obj.insert("shards".into(), serde_json::Value::Object(shards_obj));
    let json = serde_json::to_vec(&serde_json::Value::Object(obj))
        .expect("global manifest serialization cannot fail");
    std::fs::write(output_dir.join(SHARDED_MANIFEST_NAME), json)?;

    Ok(ShardedStreamResult {
        config_hash,
        output_dir: output_dir.to_path_buf(),
        shards: per_shard,
    })
}

// --------------------------------------------------------------------------- //
// Core streaming body (shared by single-file and union sources).
// --------------------------------------------------------------------------- //

/// Cached output specs per processed weight (mirrors quantized_specs_cache in
/// reference stream_quant.py): computed once when the weight (or its earlier
/// bias) is reached, written at the weight's own file position.
///
/// Generalized across all four formats (plan Phase C.1, §3.2/§3.3). The
/// emission order is fixed and format-agnostic:
/// `weight`, `weight_scale`, [`weight_scale_2`], `comfy_quant`, [`input_scale`].
/// INT8 populates `q_dtype=I8`, `q_shape=[m,n]`, `scale_dtype=F32`,
/// `scale2=None`, `input_scale=(block mode)` — byte-identical to the pre-C.1
/// INT8 emission (proven by `phase5_stream_e2e.rs` whole-file parity).
struct WeightOutputs {
    /// Quantized weight payload (`<base>.weight`).
    q_bytes: Vec<u8>,
    /// Dtype of the quantized weight: I8 (INT8), F8_E4M3 (FP8/MXFP8), U8 (NVFP4).
    q_dtype: DType,
    /// Shape of the quantized weight. INT8/FP8 = `[m,n]`; MXFP8 = padded
    /// `[m_pad,n_pad]`; NVFP4 = packed `[m_pad, n_pad/2]`.
    q_shape: Vec<u64>,
    /// `<base>.weight_scale` payload.
    scale_bytes: Vec<u8>,
    /// Dtype of `weight_scale`: F32 (INT8/FP8), U8 (MXFP8 E8M0), F8_E4M3 (NVFP4).
    scale_dtype: DType,
    /// Shape of `weight_scale` (already squeezed for 1-element INT8/FP8 scales).
    scale_shape: Vec<u64>,
    /// Optional `<base>.weight_scale_2` (NVFP4 per-tensor scale only):
    /// `(bytes, dtype, shape)` = `(f32 le bytes, F32, [])`.
    scale2: Option<(Vec<u8>, DType, Vec<u64>)>,
    /// `<base>.comfy_quant` U8 JSON blob (family A or B).
    blob: Vec<u8>,
    /// Whether to emit `<base>.input_scale` (F32 `[]` = 1.0). INT8 block only;
    /// never for FP8/MXFP8/NVFP4 (plan §3.3).
    input_scale: bool,
}

fn stream_quantize_source<S: TensorSource + ?Sized>(
    input: &S,
    output_path: &Path,
    config: &QuantConfig,
    mut on_progress: Option<&mut ProgressFn>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<StreamResult> {
    // Phase 7.1 ConvRot validation, before ANY output file is created
    // (plan §3-G, decision Q3): silently emitting plain INT8 under a ConvRot
    // name would produce a model that looks pre-quantized-rotated but isn't.
    // Two things can be wrong with a ConvRot request:
    //   1. it is not INT8 row-wise — the reference gates the rotation on
    //      `self.convrot and self.scaling_mode == "row"`
    //      (learned_rounding.py:869), so every other combination is a
    //      misconfiguration, not a rotation;
    //   2. the group size is off the power-of-4 ladder, so `build_hadamard`
    //      has no regular Hadamard to return (the reference's scipy
    //      Sylvester fallback for other powers of two is not ported).
    if config.convrot {
        if config.format != Format::Int8 || config.scaling_mode != ScalingMode::Row {
            return Err(StreamError::ConvRotRequiresInt8Row {
                format: config.format.as_str(),
                scaling_mode: config.scaling_mode.as_str(),
            });
        }
        // The matrix itself is discarded here — validity is all that matters
        // up front; each rotated tensor rebuilds it (256×256, negligible next
        // to the rotation GEMM).
        crate::convrot::build_hadamard(config.convrot_group_size)
            .map_err(|_| StreamError::ConvRotBadGroupSize(config.convrot_group_size))?;
    }

    let names: Vec<String> = input.names().to_vec();
    // Reference: `total = len(names)` — the full tensor count, independent of
    // how many are already done from a resumed run.
    let total = names.len();

    // ---- manifest / resume bookkeeping ------------------------------------ //
    let mut state = StreamState::new(output_path, config.config_hash());
    let resumed = state.load_manifest(output_path);

    // Ensure manifest order includes any names from a prior run plus current ones.
    if resumed {
        for n in &names {
            if !state.order.contains(n) {
                state.order.push(n.clone());
            }
        }
    } else {
        state.order = names.clone();
    }

    // ---- file-level __metadata__ (plan Phase D.1, §3.3) ------------------- //
    // MXFP8/NVFP4 outputs carry `_quantization_metadata` describing exactly
    // the layers quantized in THIS file. Sharded mode runs this function per
    // shard, so per-shard metadata semantics fall out for free. INT8/FP8
    // never carry metadata (the INT8 reference rebuilds the header via
    // `_resolve_union_header`, which drops `__metadata__`; the ctq FP8
    // unified path only adds metadata behind the off-by-default
    // --save-quant-metadata flag — golden-verified absent).
    let file_metadata = build_file_metadata(input, config);

    // ---- open writer (resume appends; fresh truncates) --------------------- //
    // On resume the writer re-parses the existing header slot and preserves
    // its metadata, so the value written at fresh-open time survives
    // cancel/resume cycles unchanged.
    let mut writer = if resumed {
        IncrementalWriter::open_resume(output_path)?
    } else {
        IncrementalWriter::open_new_with(output_path, 1 << 16, file_metadata)?
    };

    // ---- classify tensors --------------------------------------------------- //
    // A 2D .weight that is NOT quantized (excluded or skip-heur or non-2D shape
    // rules) but has 2D shape gets cast to the output dtype like ctq's
    // cast_unquantized_weights. Everything else copies verbatim.
    let target_dtype = resolve_output_dtype(&config.orig_dtype);
    let mut skip_cast: HashSet<&str> = HashSet::new();

    for name in &names {
        if !name.ends_with(".weight") {
            continue;
        }
        let Some(info) = input.info(name) else {
            continue;
        };
        if info.shape.len() != 2 || info.shape.contains(&0) {
            continue;
        }
        if is_quantizable(config, name, &info.shape) {
            continue;
        }
        skip_cast.insert(name.as_str());
    }

    let remaining: Vec<String> = names
        .iter()
        .filter(|n| !state.done.contains(*n))
        .cloned()
        .collect();

    // ---- bias-correction calibration cache (Phase 7.3) --------------------- //
    // Unique in_features among ALL 2D `.weight` tensors (quantizable or not),
    // from one shared generator seeded once. This mirrors
    // `_build_torch_calibration_cache` exactly: it iterates every name ending
    // in `.weight` with a 2D shape and draws `randn(3072, in_features)` per
    // unique in_features — it does NOT filter by quantizability/exclusion.
    // (Skipping the draw for skipped weights would shift the RNG stream and
    // corrupt bias correction whenever a skipped weight sorts before a
    // quantized one — confirmed by probe.)
    //
    // The DRAW ORDER is format-dependent (plan §3.4): INT8/FP8 draw in file
    // order over all 2D weights; MXFP8/NVFP4 draw over sorted(weights-only).
    let calib = CalibCache::build(
        names.iter().map(|n| {
            let info = input.info(n);
            let nf = match info {
                Some(i) if n.ends_with(".weight") && i.shape.len() == 2 => {
                    Some(i.shape[1] as usize)
                }
                _ => None,
            };
            (n.clone(), nf)
        }),
        config.format.calib_order(),
        config.calib_seed as u64,
    );

    // Corrected-bias cache + cached weight outputs: quantizing a weight
    // computes its specs and its sibling bias's new bytes immediately (weight
    // may come before OR after the bias in file order). Values are written at
    // each tensor's own position.
    let mut weight_outputs: HashMap<String, WeightOutputs> = HashMap::new();
    let mut corrected_bias: HashMap<String, Vec<u8>> = HashMap::new();

    for name in &remaining {
        // Cooperative cancellation (Ctrl-C): stop at a tensor boundary. The
        // writer has already flushed the header after every completed tensor
        // and the manifest is saved per-tensor, so finalizing here leaves a
        // valid, resumable partial output.
        if let Some(flag) = cancel {
            if flag.load(std::sync::atomic::Ordering::Relaxed) {
                writer.finalize()?;
                state.save_manifest()?;
                return Err(StreamError::Cancelled {
                    done: state.done.len(),
                    total,
                });
            }
        }
        if corrected_bias.contains_key(name) {
            // Bias whose weight was already processed -> write corrected value.
            let info = input.info(name).expect("exists");
            writer.add_tensor(
                name,
                info.dtype,
                Some(&info.dtype_raw),
                &info.shape,
                &corrected_bias[name.as_str()],
            )?;
            finish_name(&mut state, name, total, &mut on_progress)?;
            continue;
        }

        let info_shape = input.info(name).map(|i| i.shape.as_slice()).unwrap_or(&[]);
        if is_quantizable(config, name, info_shape) && input.info(name).is_some() {
            // Weight whose bias came earlier: specs were cached then; write now.
            if !weight_outputs.contains_key(name) {
                let completed = compute_weight_outputs(
                    name,
                    input,
                    config,
                    &calib,
                    &mut weight_outputs,
                    &mut corrected_bias,
                    cancel,
                )?;
                if !completed {
                    // Ctrl-C during bias correction: stop at a tensor boundary.
                    writer.finalize()?;
                    state.save_manifest()?;
                    return Err(StreamError::Cancelled {
                        done: state.done.len(),
                        total,
                    });
                }
            }
            let o = &weight_outputs[name];
            writer.add_tensor(name, o.q_dtype, None, &o.q_shape, &o.q_bytes)?;
            let base = base_of(name);
            writer.add_tensor(
                &format!("{base}.weight_scale"),
                o.scale_dtype,
                None,
                &o.scale_shape,
                &o.scale_bytes,
            )?;
            if let Some((s2_bytes, s2_dtype, s2_shape)) = &o.scale2 {
                writer.add_tensor(
                    &format!("{base}.weight_scale_2"),
                    *s2_dtype,
                    None,
                    s2_shape,
                    s2_bytes,
                )?;
            }
            writer.add_tensor(
                &format!("{base}.comfy_quant"),
                DType::U8,
                None,
                &[o.blob.len() as u64],
                &o.blob,
            )?;
            if o.input_scale {
                writer.add_tensor(
                    &format!("{base}.input_scale"),
                    DType::F32,
                    None,
                    &[],
                    &1.0f32.to_le_bytes(),
                )?;
            }
        } else if name.ends_with(".bias") {
            // Bias reached BEFORE its weight: process (compute+cache) the
            // weight now but only write the bias here.
            let wname = format!("{}.weight", &name[..name.len() - ".bias".len()]);
            let wshape = input
                .info(&wname)
                .map(|i| i.shape.as_slice())
                .unwrap_or(&[]);
            if input.info(&wname).is_some() && is_quantizable(config, &wname, wshape) {
                let completed = compute_weight_outputs(
                    &wname,
                    input,
                    config,
                    &calib,
                    &mut weight_outputs,
                    &mut corrected_bias,
                    cancel,
                )?;
                if !completed {
                    // Ctrl-C during bias correction: stop at a tensor boundary.
                    writer.finalize()?;
                    state.save_manifest()?;
                    return Err(StreamError::Cancelled {
                        done: state.done.len(),
                        total,
                    });
                }
                let info = input.info(name).expect("bias exists");
                writer.add_tensor(
                    name,
                    info.dtype,
                    Some(&info.dtype_raw),
                    &info.shape,
                    &corrected_bias[name.as_str()],
                )?;
            } else {
                copy_tensor(name, input, &mut writer, target_dtype, &skip_cast)?;
            }
        } else {
            copy_tensor(name, input, &mut writer, target_dtype, &skip_cast)?;
        }
        finish_name(&mut state, name, total, &mut on_progress)?;
    }

    writer.finalize()?;
    state.save_manifest()?;

    Ok(StreamResult {
        config_hash: state.config_hash,
        order: state.order,
        done: {
            let mut v: Vec<String> = state.done.iter().cloned().collect();
            v.sort();
            v
        },
    })
}

/// Mark one tensor done and flush the manifest (mirrors per-tensor state.save).
/// When a progress callback is present it is invoked with
/// `(len(state.done), total)` after the manifest is saved — exactly like the
/// reference's `on_progress(len(state.done), total)`.
fn finish_name(
    state: &mut StreamState,
    name: &str,
    total: usize,
    on_progress: &mut Option<&mut ProgressFn>,
) -> Result<()> {
    state.done.insert(name.to_string());
    if !state.order.iter().any(|o| o == name) {
        state.order.push(name.to_string());
    }
    state.save_manifest()?;
    if let Some(cb) = on_progress {
        cb(state.done.len(), total);
    }
    Ok(())
}

/// ctq `constants.py::AVOID_KEY_NAMES` (verbatim). The MXFP8/NVFP4 dedicated
/// modules hardcode `exclude_patterns = list(AVOID_KEY_NAMES)` and skip any key
/// containing one of these substrings (`mxfp8_conversion.py:219`,
/// `nvfp4_conversion.py:221`). The FP8 unified path does NOT apply this list —
/// it only applies `--exclude-layers` (plan §3.6 / OQ-1).
const AVOID_KEY_NAMES: &[&str] = &[
    "norm",
    "bias",
    "embed_tokens",
    "lm_head",
    "shared",
    "patch_embedding",
    "audio_model.patch_embedding",
    "ref_conv",
    "control_adapter",
    "motion_encoder.enc.net_app",
    "face_encoder.conv",
    "pose_patch_embedding",
    "motion_encoder.enc.fc",
    "img_emb.proj",
    "k_norm",
    "q_norm",
    "motion_encoder.dec",
    "head.modulation",
    "casual_audio_encoder",
    "cond_encoder",
    "frame_packer",
    "norm_k",
    "norm_q",
    "tekken_model",
    "multi_modal_projector",
    "patch_conv",
    "ln_pre",
    "input_layernorm",
    "attention_norm",
    "post_attention_layernorm",
    "mm_in_projection_weight",
];

/// Effective block size for the skip-inefficient heuristic predicate, per
/// format AND (for FP8) per scaling mode (plan §3.6 + Phase C finding).
///
/// ctq `fp8_conversion.py:183-184` computes
/// `block_size = kwargs.get("block_size") or format_block_sizes["fp8"]=64` and
/// passes THAT to `should_skip_layer_for_performance`. The ctq CLI only
/// defaults `block_size=128` when block scaling is active, so tensor/row FP8
/// runs pass no block size → the heuristic uses ctq's fp8 default of **64**.
/// Golden proof: `linear_basic_bf16` `blocks.1.weight [128,64]` is BF16-skipped
/// in the `fp8` (block, bs=128) golden but F8_E4M3-quantized in
/// `fp8_tensor`/`fp8_row`. MXFP8=32 / NVFP4=16 from `constants.py:351/355`.
fn heur_block_size(config: &QuantConfig) -> usize {
    match config.format {
        Format::Int8 => config.block_size as usize,
        Format::Fp8E4m3 => match config.scaling_mode {
            ScalingMode::Block => config.block_size as usize,
            ScalingMode::Tensor | ScalingMode::Row => 64,
        },
        Format::Mxfp8 => 32,
        Format::Nvfp4 => 16,
    }
}

fn is_quantizable(config: &QuantConfig, name: &str, shape: &[u64]) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }
    if config.excluded(name) {
        return false;
    }
    // MXFP8/NVFP4 only: substring exclusion against ctq AVOID_KEY_NAMES.
    if config.format.carries_file_metadata() && AVOID_KEY_NAMES.iter().any(|pat| name.contains(pat))
    {
        return false;
    }
    if shape.len() != 2 || shape.contains(&0) {
        return false;
    }
    if config.skip_inefficient && should_skip_shape(shape, heur_block_size(config)) {
        return false;
    }
    true
}

/// Build the file-level `__metadata__` map for this run (plan Phase D.1, §3.3).
///
/// Returns `Some({"_quantization_metadata": <json string>})` for MXFP8/NVFP4
/// when at least one layer is quantized in THIS file, else `None` (INT8/FP8
/// never carry metadata; an all-skipped MXFP8/NVFP4 file matches ctq's
/// `if quant_metadata:` guard and carries no `__metadata__` at all).
///
/// The layer set is precomputed from the source header and is EXACTLY the set
/// `compute_weight_outputs` will quantize (same `is_quantizable` predicate),
/// so metadata is correct without waiting for the streaming loop. Sharded
/// mode runs `stream_quantize_source` per shard, giving per-shard metadata
/// semantics for free.
///
/// The JSON string is hand-built to byte-match ctq's
/// `json.dumps({"format_version": "1.0", "layers": quant_metadata})` with
/// default separators: outer keys sorted (`format_version` < `layers`), layer
/// keys in sorted order (ctq iterates `sorted(weight_keys)`), and inner key
/// order `format, group_size, orig_dtype, orig_shape` — identical to the
/// family-B blob, so the per-layer inner object reuses `encode_block_format`.
fn build_file_metadata<S: TensorSource + ?Sized>(
    input: &S,
    config: &QuantConfig,
) -> Option<Map<String, serde_json::Value>> {
    if !config.format.carries_file_metadata() {
        return None;
    }

    // Collect (base, pre-padding shape) for every quantizable layer.
    let mut layers: Vec<(String, Vec<u64>)> = Vec::new();
    for name in input.names() {
        let Some(info) = input.info(name) else {
            continue;
        };
        if !is_quantizable(config, name, &info.shape) {
            continue;
        }
        layers.push((base_of(name).to_string(), info.shape.clone()));
    }
    if layers.is_empty() {
        return None;
    }
    // ctq dict insertion order == iteration of sorted(weight_keys).
    layers.sort_by(|a, b| a.0.cmp(&b.0));

    let fmt_str = config.format.as_str();
    let group = config
        .format
        .fixed_group_size()
        .expect("family-B format has a fixed group size");
    let orig = resolve_orig_dtype_str(&config.orig_dtype);

    let inner: Vec<String> = layers
        .iter()
        .map(|(base, shape)| {
            let obj = encode_block_format(fmt_str, group, &orig, shape);
            // serde_json escapes the key string exactly like json.dumps.
            let key = serde_json::to_string(base).expect("string serialization cannot fail");
            format!(
                "{}: {}",
                key,
                std::str::from_utf8(&obj).expect("blob is ascii")
            )
        })
        .collect();

    let s = format!(
        r#"{{"format_version": "1.0", "layers": {{{}}}}}"#,
        inner.join(", ")
    );

    let mut meta = Map::new();
    meta.insert(
        "_quantization_metadata".to_string(),
        serde_json::Value::String(s),
    );
    Some(meta)
}

/// Castable float dtypes for skipped weights (mirrors `_CASTABLE_FLOAT_DTYPE_NAMES`
/// restricted to our supported set: bf16/f16/f32).
fn castable(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::Bf16)
}

/// Resolve "bfloat16"/"float16" to our DType (mirrors ctq resolve_output_dtype).
fn resolve_output_dtype(s: &str) -> Option<DType> {
    match s {
        "bfloat16" => Some(DType::Bf16),
        "float16" => Some(DType::F16),
        _ => None,
    }
}

/// Quantize one weight, compute its output specs (cached for later writing at
/// the weight's own file position), and if a sibling `.bias` exists compute +
/// cache its corrected bytes (mirrors process_weight in reference
/// stream_quant.py — which writes nothing itself).
///
/// Returns `Ok(true)` when the weight (and any sibling bias) was fully
/// computed, or `Ok(false)` when cooperative cancellation was requested during
/// the (long-running) bias-correction GEMM — the caller must then finalize the
/// writer, save the manifest, and report `StreamError::Cancelled`. Nothing is
/// written to disk mid-tensor, so a `false` return leaves the last completed
/// tensor as the valid resume point.
#[allow(clippy::type_complexity)]
fn compute_weight_outputs<S: TensorSource + ?Sized>(
    name: &str,
    input: &S,
    config: &QuantConfig,
    calib: &CalibCache,
    weight_outputs: &mut HashMap<String, WeightOutputs>,
    corrected_bias: &mut HashMap<String, Vec<u8>>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<bool> {
    let info = input.info(name).expect("checked caller");
    let (m, n) = (info.shape[0] as usize, info.shape[1] as usize);

    // Divisibility policy (plan §3.6): only INT8 errors on indivisible dims
    // when the skip heuristic is OFF. FP8 falls back to row-wise inside the
    // kernel; MXFP8/NVFP4 pad internally — neither ever errors here.
    if config.format == Format::Int8
        && !config.skip_inefficient
        && (m % config.block_size as usize != 0 || n % config.block_size as usize != 0)
    {
        return Err(StreamError::NotDivisible {
            m: m as u64,
            n: n as u64,
            block_size: config.block_size,
        });
    }

    let raw = input.tensor_bytes(name)?;
    let w_f32: Vec<f32> = decode_to_f32(info.dtype, raw);

    // ---- format routing (plan Phase C.2) ---------------------------------- //
    // Each branch produces the per-format output specs (`WeightOutputs`), the
    // dequantized weight `w_dq` (f32, original m×n, padding cropped) used for
    // sibling-bias correction (plan §3.5 / Phase C.4), and — ConvRot only —
    // the ROTATED weight `w_rot` that selects the ConvRot bias path.
    // Emission dtypes, shapes, and blob families follow the §3.3 byte
    // contract (golden headers).
    let orig = resolve_orig_dtype_str(&config.orig_dtype);
    let (outputs, w_dq, w_rot): (WeightOutputs, Vec<f32>, Option<Vec<f32>>) = match config.format {
        Format::Int8 => {
            let mode = match config.scaling_mode {
                ScalingMode::Tensor => crate::quant::ScalingMode::Tensor,
                ScalingMode::Row => crate::quant::ScalingMode::Row,
                ScalingMode::Block => crate::quant::ScalingMode::Block,
            };

            // ---- Phase 7.1: ConvRot (learned_rounding.py:864-884) --------- //
            // The rotation is decided PER TENSOR: `convrot` is set AND the
            // group size divides in_features (`N = W.shape[1]`). A tensor the
            // group size does not divide stays PLAIN row-wise INT8 — the
            // reference warns and leaves `convrot_applied=False`, so its blob
            // carries no convrot key either (byte-parity, not a style choice).
            // Validation already guaranteed `convrot ⇒ INT8 + row`, so the
            // plain fall-through below is exactly the pre-7.1 behaviour.
            let gs = config.convrot_group_size as usize;
            let rotated = config.convrot && n % gs == 0;
            if config.convrot && !rotated {
                eprintln!(
                    "warning: skipping ConvRot for {name}: in_features {n} not divisible by group size {gs}"
                );
            }

            let (qdata, scale, scale_shape, blob, w_dq, w_rot) = if rotated {
                // One call: rotate → row-wise INT8 quantize → rotated dequant.
                // It hands back `w_rot` too, so we never rotate twice.
                let r =
                    crate::convrot::convrot_int8_weight(&w_f32, m, n, config.convrot_group_size)?;
                // <base>.weight_scale F32, [m,1] for row mode (scalar squeeze
                // for a single row — normalize_tensorwise_scales parity).
                let scale_shape: Vec<u64> = if r.scale.len() == 1 {
                    vec![]
                } else {
                    r.scale_shape
                };
                // Family A with the ConvRot tail keys; `input_scale` is
                // emitted for block-wise INT8 only (fp8_conversion.py:597).
                let blob = encode_comfy_quant_int8_convrot(&orig, config.convrot_group_size);
                (
                    r.qdata,
                    r.scale,
                    scale_shape,
                    blob,
                    r.w_dq_rot,
                    Some(r.w_rot),
                )
            } else {
                let r = quantize_int8_weight(&w_f32, m, n, mode, config.block_size as usize);
                // <base>.weight_scale F32 with scalar squeeze for 1-element
                // scales (normalize_tensorwise_scales parity: [1]/[1,1] → []).
                let scale_shape: Vec<u64> = if r.scale.len() == 1 {
                    vec![]
                } else {
                    r.scale_shape.clone()
                };
                // <base>.comfy_quant blob via the comfy_schema encoder
                // (Phase 3.5): family-A key order format, orig_dtype,
                // group_size (block only).
                let comfy_fmt = match config.scaling_mode {
                    ScalingMode::Block => ComfyFormat::Int8Blockwise,
                    _ => ComfyFormat::Int8Tensorwise,
                };
                // NOTE (Phase 7.1, KNOWN DIVERGENCE — see
                // `tools/probe_convrot_blob_per_row.py`): the reference emits
                // `"per_row": true` for EVERY int8 row-wise layer, rotated or
                // not — `fp8_conversion.py:583-584` keys it off the SCALING
                // MODE, not off ConvRot. So a ConvRot run whose in_features is
                // not divisible by 256 really produces
                // `{"format": "int8_tensorwise", "orig_dtype": "...", "per_row": true}`
                // (no `convrot` key). The Phase 7.1 work order explicitly froze
                // the non-ConvRot row-wise blob, so we keep emitting it without
                // `per_row`; Phase 7.2 (golden generation + byte parity) owns
                // that fix.
                let blob = encode_comfy_quant(comfy_fmt, &orig, Some(config.block_size), None);
                let w_dq =
                    dequantize_int8(&r.qdata, &r.scale, m, n, mode, config.block_size as usize);
                (r.qdata, r.scale, scale_shape, blob, w_dq, None)
            };

            let input_scale = matches!(config.scaling_mode, ScalingMode::Block);
            let q_bytes: Vec<u8> = qdata.iter().flat_map(|v| v.to_le_bytes()).collect();
            let scale_bytes: Vec<u8> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();

            (
                WeightOutputs {
                    q_bytes,
                    q_dtype: DType::I8,
                    q_shape: vec![m as u64, n as u64],
                    scale_bytes,
                    scale_dtype: DType::F32,
                    scale_shape,
                    scale2: None,
                    blob,
                    input_scale,
                },
                w_dq,
                w_rot,
            )
        }

        Format::Fp8E4m3 => {
            let fp8_mode = match config.scaling_mode {
                ScalingMode::Tensor => Fp8ScalingMode::Tensor,
                ScalingMode::Row => Fp8ScalingMode::Row,
                ScalingMode::Block => Fp8ScalingMode::Block,
            };
            let r = quantize_fp8_weight(&w_f32, m, n, fp8_mode, config.block_size as usize);

            // <name> F8_E4M3 payload (kernel already emits one byte/element).
            let q_bytes = r.qdata.clone();

            // <base>.weight_scale F32, scalar-squeezed for 1-element scales
            // (normalize_tensorwise_scales parity, same as INT8).
            let scale_shape: Vec<u64> = if r.scale.len() == 1 {
                vec![]
            } else {
                r.scale_shape.clone()
            };
            let scale_bytes: Vec<u8> = r.scale.iter().flat_map(|v| v.to_le_bytes()).collect();

            // Family-A blob; the encoder drops group_size for tensor/row.
            let comfy_fmt = match config.scaling_mode {
                ScalingMode::Block => ComfyFormat::Fp8Blockwise,
                ScalingMode::Row => ComfyFormat::Fp8Rowwise,
                ScalingMode::Tensor => ComfyFormat::Fp8Tensor,
            };
            let blob = encode_comfy_quant(comfy_fmt, &orig, Some(config.block_size), None);

            // FP8 never emits input_scale in our parity contract (plan §3.3).
            let w_dq = dequantize_fp8(&w_f32, &r, m, n, fp8_mode, config.block_size as usize);
            (
                WeightOutputs {
                    q_bytes,
                    q_dtype: DType::F8E4M3,
                    q_shape: vec![m as u64, n as u64],
                    scale_bytes,
                    scale_dtype: DType::F32,
                    scale_shape,
                    scale2: None,
                    blob,
                    input_scale: false,
                },
                w_dq,
                None,
            )
        }

        Format::Mxfp8 => {
            let r = quantize_mxfp8_weight(&w_f32, m, n);

            // <name> F8_E4M3 payload at the PADDED shape; <base>.weight_scale
            // is the E8M0 tiled (to_blocked) U8 bytes as the kernel emits them.
            let q_bytes = r.qdata.clone();
            let q_shape = r.qdata_shape.clone();
            let scale_bytes = r.scale.clone();
            let scale_shape = r.scale_shape.clone();

            // Family-B blob carries the PRE-padding input shape.
            let blob =
                encode_comfy_quant(ComfyFormat::Mxfp8, &orig, None, Some(&[m as u64, n as u64]));

            let w_dq = dequantize_mxfp8(&r, m, n);
            (
                WeightOutputs {
                    q_bytes,
                    q_dtype: DType::F8E4M3,
                    q_shape,
                    scale_bytes,
                    scale_dtype: DType::U8,
                    scale_shape,
                    scale2: None,
                    blob,
                    input_scale: false,
                },
                w_dq,
                None,
            )
        }

        Format::Nvfp4 => {
            let r = quantize_nvfp4_weight(&w_f32, m, n);

            // <name> U8 packed payload at (m_pad, n_pad/2); <base>.weight_scale
            // is the E4M3 tiled (to_blocked) bytes; <base>.weight_scale_2 is the
            // per-tensor f32 scale.
            let q_bytes = r.qdata.clone();
            let q_shape = r.qdata_shape.clone();
            let scale_bytes = r.scale.clone();
            let scale_shape = r.scale_shape.clone();
            let scale2 = Some((
                r.per_tensor_scale.to_le_bytes().to_vec(),
                DType::F32,
                vec![],
            ));

            // Family-B blob carries the PRE-padding input shape.
            let blob =
                encode_comfy_quant(ComfyFormat::Nvfp4, &orig, None, Some(&[m as u64, n as u64]));

            let w_dq = dequantize_nvfp4(&r, m, n);
            (
                WeightOutputs {
                    q_bytes,
                    q_dtype: DType::U8,
                    q_shape,
                    scale_bytes,
                    scale_dtype: DType::F8E4M3,
                    scale_shape,
                    scale2,
                    blob,
                    input_scale: false,
                },
                w_dq,
                None,
            )
        }
    };

    weight_outputs.insert(name.to_string(), outputs);

    // ---- sibling-bias correction (Phase 7.3; per-format dequant C.4) ------ //
    // `w_dq` was produced by the format routing above via the format's own
    // dequant (plan §3.5): INT8 `dequantize_int8`, FP8 `dequantize_fp8`
    // (tensor/row = IEEE division, block = multiply), MXFP8 `dequantize_mxfp8`,
    // NVFP4 `dequantize_nvfp4`. The correction pipeline itself is format-agnostic.
    let bias_name = format!("{base}.bias", base = base_of(name));
    if input.info(&bias_name).is_some() && !corrected_bias.contains_key(&bias_name) {
        if let Some(x) = calib.get(n) {
            let bias_info = input.info(&bias_name).expect("checked");
            let bias_raw = input.tensor_bytes(&bias_name)?;
            // The reference correct biases of ANY dtype: it upcasts to f32 for
            // the math, then casts the result back to the bias's original dtype
            // (`(b_orig - corr).to(dtype=bias.dtype)`, stream_quant.py:145).
            // Real models (e.g. VibeVoice) ship BF16 biases, so decode f32/f16/
            // bf16 here and re-encode into the original dtype below. For F32 the
            // round trip is the identity, so byte-parity with the F32 goldens is
            // preserved exactly.
            let bias: Vec<f32> = match bias_info.dtype {
                DType::F32 => bias_raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                DType::F16 => bias_raw
                    .chunks_exact(2)
                    .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
                DType::Bf16 => bias_raw
                    .chunks_exact(2)
                    .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
                dt => {
                    return Err(StreamError::UnsupportedBiasDtype {
                        name: bias_name.clone(),
                        dtype: dt,
                    })
                }
            };
            // Phase 7.1: the ConvRot path takes PRECEDENCE over the generic
            // one — fp8_conversion.py:676-686 is an `if / elif`, so a
            // ConvRot-corrected bias is `b + adj` and never the generic
            // `b - mean(X @ err.T)`. `Some(w_rot)` means this tensor really
            // was rotated (in_features divisible by the group size).
            let corrected = match w_rot {
                Some(w_rot) => {
                    let s_count = x.len() / n; // CALIB_SAMPLES = 3072
                    let h = crate::convrot::build_hadamard(config.convrot_group_size)
                        .expect("convrot group size validated at config time");
                    // X_rot = rotate_activation(X, H, gs) —
                    // tensor_utils.py:181-183; both GEMMs then run on the
                    // ROTATED operands (learned_rounding.py:932-934).
                    let x_rot = crate::convrot::rotate_activation(
                        x,
                        &h,
                        s_count,
                        n,
                        config.convrot_group_size,
                    )
                    .expect("in_features divisibility checked when rotating the weight");
                    match crate::convrot::correct_bias_convrot(
                        &x_rot, &w_rot, &w_dq, &bias, m, n, s_count, cancel,
                    ) {
                        Some(c) => c,
                        // Ctrl-C during the bias-correction GEMM: abort.
                        None => return Ok(false),
                    }
                }
                None => match correct_bias(x, &w_f32, &w_dq, &bias, m, n, cancel) {
                    Some(c) => c,
                    // Ctrl-C during the bias-correction GEMM: abort promptly.
                    None => return Ok(false),
                },
            };
            // Cast back to the bias's original dtype (reference parity).
            let bytes: Vec<u8> = match bias_info.dtype {
                DType::F32 => corrected.iter().flat_map(|v| v.to_le_bytes()).collect(),
                DType::F16 => corrected
                    .iter()
                    .flat_map(|v| f32_to_f16_bits(*v).to_le_bytes())
                    .collect(),
                DType::Bf16 => corrected
                    .iter()
                    .flat_map(|v| f32_to_bf16_bits(*v).to_le_bytes())
                    .collect(),
                _ => unreachable!("non-float bias dtypes rejected above"),
            };
            corrected_bias.insert(bias_name, bytes);
        }
        // Missing calibration entry → torch reference warns and keeps the
        // original bias; our copy path handles that naturally.
    }
    Ok(true)
}

fn base_of(weight_name: &str) -> &str {
    // "<base>.weight" -> "<base>"; fall back to the full name when no suffix.
    weight_name.strip_suffix(".weight").unwrap_or(weight_name)
}

/// Mirror ctq's resolve_orig_dtype_str: bfloat16/float16 → torch-prefixed string.
fn resolve_orig_dtype_str(orig: &str) -> String {
    match orig {
        "bfloat16" => "torch.bfloat16".to_string(),
        "float16" => "torch.float16".to_string(),
        other => other.to_string(),
    }
}

/// Decode any supported float dtype to f32 values (for quantization input).
fn decode_to_f32(dtype: DType, raw: &[u8]) -> Vec<f32> {
    match dtype {
        DType::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::F16 | DType::U16 => raw
            .chunks_exact(2)
            .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        // U16 is tolerated as f16 storage per PHASE0 notes? No — U16 here means
        // numpy-written bf16 quirk from gen_golden inputs is BF16-tagged upstream;
        // treat U16 as bf16 raw bits (matches fixture reality where U16 == bf16 view).
        DType::Bf16 => raw
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        dt => panic!("unsupported weight dtype for quantization: {dt}"),
    }
}

fn copy_tensor<S: TensorSource + ?Sized>(
    name: &str,
    input: &S,
    writer: &mut IncrementalWriter,
    target_dtype: Option<DType>,
    skip_cast: &HashSet<&str>,
) -> Result<()> {
    let info = input.info(name).expect("exists");
    let data = input.tensor_bytes(name)?;

    // Skipped 2D .weight cast to output dtype (ctq cast_unquantized_weights):
    // only for castable float source dtypes differing from target.
    if skip_cast.contains(name) {
        if let (Some(target), true) = (target_dtype, castable(info.dtype)) {
            if info.dtype != target {
                let vals_f32 = match info.dtype {
                    DType::F32 => data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect::<Vec<_>>(),
                    DType::F16 | DType::U16 => data
                        .chunks_exact(2)
                        .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect::<Vec<_>>(),
                    DType::Bf16 => data
                        .chunks_exact(2)
                        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect::<Vec<_>>(),
                    _ => unreachable!("castable() gate"),
                };
                let out_bytes: Vec<u8> = match target {
                    DType::Bf16 => vals_f32
                        .iter()
                        .flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes())
                        .collect(),
                    DType::F16 => vals_f32
                        .iter()
                        .flat_map(|&v| f32_to_f16_bits(v).to_le_bytes())
                        .collect(),
                    DType::F32 => vals_f32.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    _ => unreachable!(),
                };
                writer.add_tensor(
                    name,
                    target,
                    Some(target.as_header_str()),
                    &info.shape,
                    &out_bytes,
                )?;
                return Ok(());
            }
        }
    }

    // Verbatim passthrough with original dtype string.
    writer.add_tensor(name, info.dtype, Some(&info.dtype_raw), &info.shape, data)?;
    Ok(())
}

// --------------------------------------------------------------------------- //
// Tests (plan Phase C.1 gate: INT8 emission layout provably unchanged).
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;
    use crate::st_io::header::TensorInfo;

    /// Minimal in-memory [`TensorSource`] for unit-testing the orchestrator
    /// without touching the filesystem for inputs.
    struct MemSource {
        names: Vec<String>,
        infos: HashMap<String, TensorInfo>,
        bytes: HashMap<String, Vec<u8>>,
    }

    impl MemSource {
        fn new() -> Self {
            Self {
                names: Vec::new(),
                infos: HashMap::new(),
                bytes: HashMap::new(),
            }
        }
        fn add(&mut self, name: &str, dtype: DType, shape: Vec<u64>, data: Vec<u8>) {
            let info = TensorInfo {
                dtype,
                dtype_raw: dtype.as_header_str().to_string(),
                shape,
                data_offsets: (0, data.len() as u64),
            };
            self.names.push(name.to_string());
            self.infos.insert(name.to_string(), info);
            self.bytes.insert(name.to_string(), data);
        }
    }

    impl TensorSource for MemSource {
        fn names(&self) -> &[String] {
            &self.names
        }
        fn info(&self, name: &str) -> Option<&TensorInfo> {
            self.infos.get(name)
        }
        fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
            Ok(self.bytes.get(name).map(|v| v.as_slice()).unwrap_or(&[]))
        }
    }

    /// Read back the output file's header as `(name, dtype_raw, shape)` in
    /// header (insertion) order.
    fn read_header(path: &Path) -> Vec<(String, String, Vec<u64>)> {
        let reader =
            crate::st_io::reader::SafetensorsReader::open(path).expect("output must be loadable");
        reader
            .header()
            .iter()
            .map(|(n, i)| (n.clone(), i.dtype_raw.clone(), i.shape.clone()))
            .collect()
    }

    /// C.1 gate: the generalized `WeightOutputs` + write block must emit the
    /// EXACT same INT8 tensor set, dtypes, shapes, and order as before C.1:
    /// `<name>` I8 `[m,n]`, `<base>.weight_scale` F32, `<base>.comfy_quant`
    /// U8, and (block mode) `<base>.input_scale` F32 `[]`. Whole-file INT8
    /// parity is additionally proven by `phase5_stream_e2e.rs`.
    #[test]
    fn weight_outputs_int8_layout_unchanged() {
        // One quantizable 2D weight (block mode, bs=128 → divisible) + bias.
        let (m, n) = (128usize, 128usize);
        let mut src = MemSource::new();
        // Deterministic non-zero bf16 weight.
        let w: Vec<u8> = (0..m * n)
            .flat_map(|i| {
                let v = ((i as f32) * 0.001).sin();
                crate::dtype::f32_to_bf16_bits(v).to_le_bytes()
            })
            .collect();
        src.add("blk.weight", DType::Bf16, vec![m as u64, n as u64], w);
        let b: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();
        src.add("blk.bias", DType::F32, vec![n as u64], b);

        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out.safetensors");
        let config = QuantConfig::default(); // INT8, block, bs=128, heur on
        stream_quantize_source(&src, &out, &config, None, None).unwrap();

        let hdr = read_header(&out);
        // Emission order at the weight's position: weight, weight_scale,
        // comfy_quant, input_scale (block). Bias is corrected in place.
        let by_name: HashMap<&str, &(String, String, Vec<u64>)> =
            hdr.iter().map(|e| (e.0.as_str(), e)).collect();

        let w_e = by_name["blk.weight"];
        assert_eq!(w_e.1, "I8", "INT8 weight dtype must stay I8");
        assert_eq!(w_e.2, vec![m as u64, n as u64], "INT8 weight shape [m,n]");

        let s_e = by_name["blk.weight_scale"];
        assert_eq!(s_e.1, "F32", "INT8 weight_scale dtype must stay F32");

        let c_e = by_name["blk.comfy_quant"];
        assert_eq!(c_e.1, "U8", "comfy_quant blob dtype must be U8");

        // Block mode emits input_scale (F32 scalar).
        let i_e = by_name["blk.input_scale"];
        assert_eq!(i_e.1, "F32");
        assert_eq!(i_e.2, Vec::<u64>::new(), "input_scale is a scalar []");

        // No weight_scale_2 for INT8.
        assert!(
            !by_name.contains_key("blk.weight_scale_2"),
            "INT8 must not emit weight_scale_2"
        );
    }

    /// INT8 with the skip heuristic OFF must ERROR on indivisible dims
    /// (plan §3.6: only INT8 keeps the `NotDivisible` error; FP8 falls back to
    /// row-wise and MXFP8/NVFP4 pad internally).
    #[test]
    fn int8_not_divisible_errors_when_heur_off() {
        let (m, n) = (130usize, 130usize); // not divisible by 128
        let mut src = MemSource::new();
        let w: Vec<u8> = (0..m * n)
            .flat_map(|i| crate::dtype::f32_to_bf16_bits((i as f32) * 0.001).to_le_bytes())
            .collect();
        src.add("blk.weight", DType::Bf16, vec![m as u64, n as u64], w);

        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out.safetensors");
        let mut config = QuantConfig::default(); // INT8, block, bs=128
        config.skip_inefficient = false; // heur OFF → must error, not skip

        let err = stream_quantize_source(&src, &out, &config, None, None)
            .expect_err("indivisible INT8 dims with heur off must error");
        assert!(
            matches!(err, StreamError::NotDivisible { .. }),
            "expected NotDivisible, got {err:?}"
        );
    }

    // ---- Phase D.1: _quantization_metadata builder ------------------------ //

    /// Locate `tests/golden/<name>` from a unit test (crate manifest dir is
    /// `crates/quant-core`).
    fn golden_dir(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p.join("tests/golden").join(name)
    }

    /// Extract the golden header's `__metadata__._quantization_metadata`
    /// string value for byte comparison.
    fn golden_quant_metadata(case: &str, fmt: &str) -> String {
        let path = golden_dir(case).join(format!("output_{fmt}.safetensors"));
        let reader = crate::st_io::reader::SafetensorsReader::open(&path)
            .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let meta = reader
            .header()
            .metadata()
            .unwrap_or_else(|| panic!("{case}/{fmt}: golden has no __metadata__"));
        match &meta["_quantization_metadata"] {
            serde_json::Value::String(s) => s.clone(),
            other => panic!("{case}/{fmt}: unexpected metadata value {other:?}"),
        }
    }

    fn family_b_config(format: Format) -> QuantConfig {
        let (target, block_size) = match format {
            Format::Mxfp8 => ("mxfp8", 32),
            Format::Nvfp4 => ("nvfp4", 16),
            _ => unreachable!(),
        };
        QuantConfig {
            format,
            target_format: target.into(),
            int8: false,
            scaling_mode: ScalingMode::Block,
            block_size,
            ..QuantConfig::default()
        }
    }

    /// D.1 gate: the hand-built `_quantization_metadata` string must be
    /// BYTE-IDENTICAL to the golden header's value (json.dumps default
    /// separators, sorted outer keys, sorted layer keys, family-B inner key
    /// order).
    #[test]
    fn quant_metadata_string_matches_golden_linear_basic_mxfp8() {
        let input = golden_dir("linear_basic_bf16").join("input.safetensors");
        let src = SingleFileSource::open(&input).unwrap();
        let config = family_b_config(Format::Mxfp8);

        let meta = build_file_metadata(&src, &config).expect("mxfp8 must build metadata");
        let ours = match &meta["_quantization_metadata"] {
            serde_json::Value::String(s) => s.clone(),
            other => panic!("unexpected metadata value {other:?}"),
        };
        assert_eq!(ours, golden_quant_metadata("linear_basic_bf16", "mxfp8"));
    }

    #[test]
    fn quant_metadata_string_matches_golden_linear_basic_nvfp4() {
        let input = golden_dir("linear_basic_bf16").join("input.safetensors");
        let src = SingleFileSource::open(&input).unwrap();
        let config = family_b_config(Format::Nvfp4);

        let meta = build_file_metadata(&src, &config).expect("nvfp4 must build metadata");
        let ours = match &meta["_quantization_metadata"] {
            serde_json::Value::String(s) => s.clone(),
            other => panic!("unexpected metadata value {other:?}"),
        };
        assert_eq!(ours, golden_quant_metadata("linear_basic_bf16", "nvfp4"));
    }

    /// D.1 gate: the layer set must exclude non-2D weights, heur-skipped
    /// weights, and AVOID_KEY_NAMES matches — exactly the layers that get
    /// quantized. conv_net: only `head` (conv.net.weight is 4D); odd_shapes:
    /// only `model.diffusion_model.double_block.0` (attn_norm.weight is 1D,
    /// odd.weight [130,130] is heur-skipped at block size 32/16).
    #[test]
    fn quant_metadata_excludes_skipped_and_non2d() {
        // conv_net: 4D conv weight + 2D head weight.
        let input = golden_dir("conv_net").join("input.safetensors");
        let src = SingleFileSource::open(&input).unwrap();
        for format in [Format::Mxfp8, Format::Nvfp4] {
            let config = family_b_config(format);
            let meta = build_file_metadata(&src, &config).expect("conv_net has one layer");
            let s = meta["_quantization_metadata"].as_str().unwrap();
            assert!(s.contains("\"head\""), "{format:?}: must include head: {s}");
            assert!(
                !s.contains("conv.net"),
                "{format:?}: must exclude 4D conv weight: {s}"
            );
            assert_eq!(s, golden_quant_metadata("conv_net", config.format.as_str()));
        }

        // odd_shapes: 1D norm weight + heur-skipped odd.weight [130,130].
        let input = golden_dir("odd_shapes").join("input.safetensors");
        let src = SingleFileSource::open(&input).unwrap();
        for format in [Format::Mxfp8, Format::Nvfp4] {
            let config = family_b_config(format);
            let meta = build_file_metadata(&src, &config).expect("odd_shapes has one layer");
            let s = meta["_quantization_metadata"].as_str().unwrap();
            assert!(
                s.contains("\"model.diffusion_model.double_block.0\""),
                "{format:?}: must include the double_block layer: {s}"
            );
            assert!(
                !s.contains("\"odd\""),
                "{format:?}: must exclude heur-skipped odd.weight: {s}"
            );
            assert!(
                !s.contains("attn_norm"),
                "{format:?}: must exclude 1D norm weight: {s}"
            );
            assert_eq!(
                s,
                golden_quant_metadata("odd_shapes", config.format.as_str())
            );
        }
    }

    /// INT8/FP8 never carry file metadata; an all-skipped MXFP8/NVFP4 input
    /// carries none either (ctq's `if quant_metadata:` guard).
    #[test]
    fn quant_metadata_none_for_int8_fp8_and_empty_layer_set() {
        let input = golden_dir("linear_basic_bf16").join("input.safetensors");
        let src = SingleFileSource::open(&input).unwrap();

        assert!(
            build_file_metadata(&src, &QuantConfig::default()).is_none(),
            "INT8 must not carry file metadata"
        );
        let mut fp8 = QuantConfig::default();
        fp8.format = Format::Fp8E4m3;
        fp8.target_format = "fp8".into();
        fp8.int8 = false;
        assert!(
            build_file_metadata(&src, &fp8).is_none(),
            "FP8 must not carry file metadata"
        );

        // All-skipped MXFP8 input (single tiny weight, heur on → skipped).
        let mut tiny = MemSource::new();
        tiny.add(
            "blk.weight",
            DType::Bf16,
            vec![16, 16],
            vec![0u8; 16 * 16 * 2],
        );
        assert!(
            build_file_metadata(&tiny, &family_b_config(Format::Mxfp8)).is_none(),
            "all-skipped MXFP8 input must carry no metadata"
        );
    }

    // ------------------------------------------------------------------ //
    // Phase 7.1: ConvRot wiring (decision Q3)
    // ------------------------------------------------------------------ //

    /// A ConvRot config is validated BEFORE the output file is created, so a
    /// rejected run never leaves an empty file that looks like a started one.
    fn convrot_config() -> QuantConfig {
        QuantConfig {
            convrot: true,
            scaling_mode: ScalingMode::Row,
            ..QuantConfig::default()
        }
    }

    /// Read one tensor's raw bytes out of the produced file.
    fn tensor_bytes_of(path: &Path, name: &str) -> Vec<u8> {
        let reader = crate::st_io::reader::SafetensorsReader::open(path)
            .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        reader
            .tensor_bytes(name)
            .unwrap_or_else(|_| panic!("missing tensor {name}"))
            .to_vec()
    }

    /// Build a bf16 weight matrix of shape `[m, n]` with deterministic,
    /// non-degenerate values.
    fn bf16_weight(m: usize, n: usize) -> Vec<u8> {
        (0..m * n)
            .flat_map(|i| {
                let v = ((i as f32) * 0.017).sin() + ((i % 7) as f32) * 0.003;
                crate::dtype::f32_to_bf16_bits(v).to_le_bytes()
            })
            .collect()
    }

    /// ConvRot is INT8 row-wise only (learned_rounding.py:869: the rotation is
    /// gated on `self.convrot and self.scaling_mode == "row"`). Any other
    /// combination is rejected before the output file exists.
    #[test]
    fn convrot_requires_int8_row_before_file_creation() {
        let src = MemSource::new(); // empty source: validation is first
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("never.safetensors");

        // (a) FP8 (fixed block scaling) + convrot.
        let mut fp8 = convrot_config();
        fp8.format = Format::Fp8E4m3;
        fp8.target_format = "fp8".into();
        fp8.int8 = false;
        let err = stream_quantize_source(&src, &out, &fp8, None, None).unwrap_err();
        assert!(
            matches!(err, StreamError::ConvRotRequiresInt8Row { .. }),
            "FP8 + convrot must be rejected, got: {err}"
        );
        let msg = err.to_string();
        assert!(msg.contains("fp8"), "error must name the format: {msg}");
        assert!(
            msg.contains("row"),
            "error must name the required scaling mode: {msg}"
        );

        // (b) INT8 block-wise + convrot.
        let block = QuantConfig {
            scaling_mode: ScalingMode::Block,
            ..convrot_config()
        };
        let err = stream_quantize_source(&src, &out, &block, None, None).unwrap_err();
        assert!(
            matches!(err, StreamError::ConvRotRequiresInt8Row { .. }),
            "INT8 block + convrot must be rejected, got: {err}"
        );

        // (c) INT8 tensor-wise + convrot.
        let tensor = QuantConfig {
            scaling_mode: ScalingMode::Tensor,
            ..convrot_config()
        };
        let err = stream_quantize_source(&src, &out, &tensor, None, None).unwrap_err();
        assert!(
            matches!(err, StreamError::ConvRotRequiresInt8Row { .. }),
            "INT8 tensor + convrot must be rejected, got: {err}"
        );

        // No output file may exist for a rejected config.
        assert!(
            !out.exists(),
            "validation must fire before the output file is created"
        );
    }

    /// A group size off the power-of-4 ladder (100, 512, 8) is rejected up
    /// front — `build_hadamard` has no regular Hadamard for it and the
    /// reference's scipy Sylvester fallback is not ported.
    #[test]
    fn convrot_rejects_bad_group_size_before_file_creation() {
        let src = MemSource::new();
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("never.safetensors");

        for gs in [100u32, 512, 8, 2, 0] {
            let config = QuantConfig {
                convrot_group_size: gs,
                ..convrot_config()
            };
            let err = stream_quantize_source(&src, &out, &config, None, None).unwrap_err();
            assert!(
                matches!(err, StreamError::ConvRotBadGroupSize(g) if g == gs),
                "group size {gs} must be rejected with ConvRotBadGroupSize, got: {err}"
            );
        }
        assert!(
            !out.exists(),
            "validation must fire before the output file is created"
        );

        // The whole ladder is accepted (validation-wise): 4/16/64/256/1024.
        for gs in [4u32, 16, 64, 256, 1024] {
            let config = QuantConfig {
                convrot_group_size: gs,
                ..convrot_config()
            };
            // Empty source → no ConvRotBadGroupSize; the run simply completes.
            stream_quantize_source(&src, &out, &config, None, None).unwrap();
        }
    }

    /// Per-tensor rotation decision: `in_features` divisible by the group size
    /// gets the ConvRot blob (`convrot`, `convrot_groupsize`, `per_row`); a
    /// tensor it does not divide stays PLAIN row-wise INT8 — the reference
    /// warns and leaves `convrot_applied=False`, so no convrot key at all.
    #[test]
    fn convrot_blob_per_tensor_divisibility() {
        let mut src = MemSource::new();
        // rot.weight [128, 256]: passes the heuristic (m,n >= 128, %128==0)
        // and 256 % 256 == 0 → rotated.
        src.add(
            "rot.weight",
            DType::Bf16,
            vec![128, 256],
            bf16_weight(128, 256),
        );
        // plain.weight [128, 128]: passes the heuristic, but 128 % 256
        // != 0 → stays plain row-wise INT8 (no convrot key).
        src.add(
            "plain.weight",
            DType::Bf16,
            vec![128, 128],
            bf16_weight(128, 128),
        );

        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out.safetensors");
        let config = convrot_config();
        stream_quantize_source(&src, &out, &config, None, None).unwrap();

        let rot = String::from_utf8(tensor_bytes_of(&out, "rot.comfy_quant")).unwrap();
        assert_eq!(
            rot,
            r#"{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "convrot": true, "convrot_groupsize": 256, "per_row": true}"#,
            "a rotated layer must carry the ConvRot keys"
        );
        for key in [
            r#""convrot": true"#,
            r#""convrot_groupsize": 256"#,
            r#""per_row": true"#,
        ] {
            assert!(rot.contains(key), "rot blob must contain {key}: {rot}");
        }

        let plain = String::from_utf8(tensor_bytes_of(&out, "plain.comfy_quant")).unwrap();
        assert_eq!(
            plain, r#"{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16"}"#,
            "a non-divisible layer must stay plain (no convrot key at all)"
        );

        // Row-wise INT8: [m,1] weight_scale, no input_scale (block only).
        let hdr: HashMap<String, Vec<u64>> = read_header(&out)
            .into_iter()
            .map(|(n, _d, s)| (n, s))
            .collect();
        assert_eq!(hdr["rot.weight_scale"], vec![128, 1]);
        assert!(!hdr.contains_key("rot.input_scale"));
        assert_eq!(hdr["rot.weight"], vec![128, 256]);
    }

    /// A ConvRot blob must still parse as `int8_tensorwise` (family A) so the
    /// `info` / `validate` passes keep working on rotated outputs — the
    /// reference tolerates the extra keys too.
    #[test]
    fn convrot_blob_parses_as_int8_tensorwise() {
        let blob = encode_comfy_quant_int8_convrot("torch.bfloat16", 256);
        let cfg = crate::comfy_schema::parse_blob(&blob).expect("convrot blob must parse");
        assert_eq!(cfg.format, ComfyFormat::Int8Tensorwise);
        assert_eq!(cfg.orig_dtype, "torch.bfloat16");
        assert_eq!(cfg.group_size, None, "convrot group size is its own key");
    }

    /// End-to-end ConvRot bias correction through the orchestrator: the
    /// corrected bias must be `b + mean_s(Y_ref - Y_qnt)` with
    /// `Y_ref = X_rot @ W_rot.T` and `Y_qnt = X_rot @ W_dq.T` — TWO GEMMs on
    /// the ROTATED operands that ADD (learned_rounding.py:930-939), never the
    /// generic `b - mean(X @ err.T)`. Verified against an independent f64
    /// recomputation of that formula, which pins the sign, the operand choice
    /// and the sample mean (kchunk128 vs naive f64 differs by ~1e-6 relative).
    #[test]
    fn convrot_bias_correction_uses_rotated_operands() {
        let (m, n, gs) = (8usize, 64usize, 64u32);
        let mut src = MemSource::new();
        src.add(
            "blk.weight",
            DType::Bf16,
            vec![m as u64, n as u64],
            bf16_weight(m, n),
        );
        let bias: Vec<f32> = (0..m).map(|i| (i as f32) * 0.05 - 0.2).collect();
        src.add(
            "blk.bias",
            DType::F32,
            vec![m as u64],
            bias.iter().flat_map(|v| v.to_le_bytes()).collect(),
        );

        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out.safetensors");
        // block_size is unused in row mode; it only feeds the heur-OFF
        // divisibility check, so set it to m to keep the fixture tiny.
        let config = QuantConfig {
            convrot: true,
            convrot_group_size: gs,
            scaling_mode: ScalingMode::Row,
            block_size: m as u32,
            skip_inefficient: false,
            ..QuantConfig::default()
        };
        stream_quantize_source(&src, &out, &config, None, None).unwrap();

        let got_raw = tensor_bytes_of(&out, "blk.bias");
        let got: Vec<f32> = got_raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got.len(), m);

        // ---- independent recomputation of b + mean(Y_ref - Y_qnt) ---- //
        let w = decode_to_f32(DType::Bf16, &bf16_weight(m, n));
        let h = crate::convrot::build_hadamard(gs).unwrap();
        let w_rot = crate::convrot::rotate_weight(&w, &h, m, n, gs).unwrap();
        let q = quantize_int8_weight(&w_rot, m, n, crate::quant::ScalingMode::Row, 128);
        let w_dq = dequantize_int8(
            &q.qdata,
            &q.scale,
            m,
            n,
            crate::quant::ScalingMode::Row,
            128,
        );
        let calib = CalibCache::build(
            src.names().iter().map(|nm| {
                let nf = src.info(nm).and_then(|i| {
                    // NOTE: `.then(|| ...)` (lazy) — `then_some(i.shape[1])`
                    // would index shape[1] on the 1-D bias too, panicking.
                    (nm.ends_with(".weight") && i.shape.len() == 2).then(|| i.shape[1] as usize)
                });
                (nm.clone(), nf)
            }),
            config.format.calib_order(),
            config.calib_seed as u64,
        );
        let x = calib.get(n).expect("calibration entry for in_features");
        let s_count = x.len() / n;
        assert_eq!(s_count, crate::bias_correction::CALIB_SAMPLES);
        let x_rot = crate::convrot::rotate_activation(x, &h, s_count, n, gs).unwrap();

        for i in 0..m {
            let mut acc = 0.0f64;
            for s in 0..s_count {
                let mut y_ref = 0.0f64;
                let mut y_qnt = 0.0f64;
                for k in 0..n {
                    let xv = x_rot[s * n + k] as f64;
                    y_ref += xv * (w_rot[i * n + k] as f64);
                    y_qnt += xv * (w_dq[i * n + k] as f64);
                }
                acc += y_ref - y_qnt;
            }
            let want = bias[i] as f64 + acc / s_count as f64;
            let tol = 1e-3 * want.abs().max(1.0);
            assert!(
                (got[i] as f64 - want).abs() <= tol,
                "bias[{i}]: got {} want {want} (the ConvRot path must ADD mean(Y_ref - Y_qnt))",
                got[i]
            );
        }
        // Sanity: the correction is a real adjustment, not a no-op, and it is
        // NOT the generic path's subtraction of the same magnitude.
        let generic_sign_ok = (0..m).any(|i| (got[i] - bias[i]).abs() > 0.0);
        assert!(
            generic_sign_ok,
            "the ConvRot correction must actually move the bias: {got:?} vs {bias:?}"
        );
    }

    /// Determinism: two runs with the same seed (and the default seed is
    /// pinned) produce byte-identical outputs, ConvRot included.
    #[test]
    fn convrot_determinism_same_seed_byte_identical() {
        let mut src = MemSource::new();
        src.add("rot.weight", DType::Bf16, vec![4, 256], bf16_weight(4, 256));

        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("a.safetensors");
        let b = tmp.path().join("b.safetensors");
        let config = convrot_config();
        stream_quantize_source(&src, &a, &config, None, None).unwrap();
        stream_quantize_source(&src, &b, &config, None, None).unwrap();
        assert_eq!(
            std::fs::read(&a).unwrap(),
            std::fs::read(&b).unwrap(),
            "two ConvRot runs with the same seed must be byte-identical"
        );
    }
}

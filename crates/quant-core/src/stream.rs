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
//! - the output NEVER carries `__metadata__`: the reference rebuilds the header
//!   via `_resolve_union_header`, which drops `__metadata__`, and then passes
//!   `header.get("__metadata__")` (always None) to the writer — confirmed by
//!   probe even for single-file inputs that have metadata.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::bias_correction::{correct_bias, CalibCache};
use crate::comfy_schema::{encode_comfy_quant, ComfyFormat};
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

    // ---- open writer (resume appends; fresh truncates) --------------------- //
    // The reference NEVER carries __metadata__ into the streaming output (the
    // union header drops it and `header.get("__metadata__")` is always None),
    // so a fresh file always starts without metadata — even for single-file
    // inputs that have metadata (probe-verified).
    let mut writer = if resumed {
        IncrementalWriter::open_resume(output_path)?
    } else {
        IncrementalWriter::open_new_with(output_path, 1 << 16, None)?
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
    // Each branch produces the per-format output specs (`WeightOutputs`) plus
    // the dequantized weight `w_dq` (f32, original m×n, padding cropped) used
    // for sibling-bias correction (plan §3.5 / Phase C.4). Emission dtypes,
    // shapes, and blob families follow the §3.3 byte contract (golden headers).
    let orig = resolve_orig_dtype_str(&config.orig_dtype);
    let (outputs, w_dq): (WeightOutputs, Vec<f32>) = match config.format {
        Format::Int8 => {
            let mode = match config.scaling_mode {
                ScalingMode::Tensor => crate::quant::ScalingMode::Tensor,
                ScalingMode::Row => crate::quant::ScalingMode::Row,
                ScalingMode::Block => crate::quant::ScalingMode::Block,
            };
            let r = quantize_int8_weight(&w_f32, m, n, mode, config.block_size as usize);

            // <name> int8 payload.
            let q_bytes: Vec<u8> = r.qdata.iter().flat_map(|v| v.to_le_bytes()).collect();

            // <base>.weight_scale F32 with scalar squeeze for 1-element scales
            // (normalize_tensorwise_scales parity: [1]/[1,1] → shape []).
            let scale_shape: Vec<u64> = if r.scale.len() == 1 {
                vec![]
            } else {
                r.scale_shape.clone()
            };
            let scale_bytes: Vec<u8> = r.scale.iter().flat_map(|v| v.to_le_bytes()).collect();

            // <base>.comfy_quant blob via the comfy_schema encoder (Phase 3.5):
            // family-A key order format, orig_dtype, group_size (block only).
            let comfy_fmt = match config.scaling_mode {
                ScalingMode::Block => ComfyFormat::Int8Blockwise,
                _ => ComfyFormat::Int8Tensorwise,
            };
            let blob = encode_comfy_quant(comfy_fmt, &orig, Some(config.block_size), None);
            let input_scale = matches!(config.scaling_mode, ScalingMode::Block);

            let w_dq = dequantize_int8(&r.qdata, &r.scale, m, n, mode, config.block_size as usize);
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
            let corrected = match correct_bias(x, &w_f32, &w_dq, &bias, m, n, cancel) {
                Some(c) => c,
                // Ctrl-C during the bias-correction GEMM: abort promptly.
                None => return Ok(false),
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
}

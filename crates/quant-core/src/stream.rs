//! Resumable streaming quantization orchestrator (plan Phase 5, step 4.3;
//! Phase 7.3 adds torch-parity bias correction).
//!
//! Port of reference `stream_quant.py::stream_quantize` (single-file variant):
//! - iterate tensors in file order
//! - 2D `.weight` tensors that are quantizable (not excluded, dims divisible by
//!   block size when heur is on) are INT8-quantized into
//!   `<name>` + `<base>.weight_scale` + `<base>.comfy_quant` + `<base>.input_scale`
//!   AND trigger correction of their sibling `.bias` via simulated-calibration
//!   mean output-error subtraction (`bias_correction` module)
//! - skipped 2D `.weight` tensors are cast to the output dtype (bf16/f16)
//!   when they're a castable float dtype and differ from it
//! - everything else passes through verbatim
//! - biases reached BEFORE their weight defer the weight processing so the
//!   corrected value is available at the bias's file position
//! - manifest saved after EVERY tensor; resume skips names in `done`
//! - config-hash mismatch → clean restart from zero

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::bias_correction::{correct_bias, CalibCache};
use crate::comfy_schema::{encode_comfy_quant, ComfyFormat};
use crate::dtype::{bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits, DType};
use crate::manifest::{QuantConfig, ScalingMode, StreamState};
use crate::quant::{dequantize_int8, quantize_int8_weight, should_skip_shape};
use crate::st_io::reader::SafetensorsReader;
use crate::st_io::writer::IncrementalWriter;

/// Output result of one full streaming run (mirrors reference return dict).
#[derive(Debug)]
pub struct StreamResult {
    pub config_hash: String,
    pub order: Vec<String>,
    pub done: Vec<String>,
}

/// Errors from the orchestrator.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("input not found: {0}")]
    InputNotFound(std::path::PathBuf),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    StIo(#[from] crate::st_io::Error),
    #[error(
        "INT8 block-wise requires dims divisible by block_size={block_size}, got ({m}, {n}); \
         enable skip_inefficient to copy such layers instead"
    )]
    NotDivisible { m: u64, n: u64, block_size: u32 },
}

pub type Result<T> = std::result::Result<T, StreamError>;

/// Cached output specs per processed weight (mirrors quantized_specs_cache in
/// reference stream_quant.py): computed once when the weight (or its earlier
/// bias) is reached, written at the weight's own file position.
struct WeightOutputs {
    q_bytes: Vec<u8>,
    scale_bytes: Vec<u8>,
    scale_shape: Vec<u64>,
    blob: Vec<u8>,
    input_scale: bool,
    m: u64,
    n: u64,
}

/// Run streaming quantization of `input_path` → `output_path`, resumable.
pub fn stream_quantize(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    config: &QuantConfig,
) -> Result<StreamResult> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();

    if !input_path.exists() {
        return Err(StreamError::InputNotFound(input_path.to_path_buf()));
    }

    let reader = SafetensorsReader::open(input_path)?;
    let names: Vec<String> = reader.header().names().cloned().collect();

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
    let mut writer = if resumed {
        IncrementalWriter::open_resume(output_path)?
    } else {
        let meta = reader.header().metadata().cloned();
        IncrementalWriter::open_new_with(output_path, 1 << 16, meta)?
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
        let Some(info) = reader.header().get(name) else {
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
    // Unique in_features among quantizable 2D weights, in file order, from one
    // shared generator seeded once (mirrors _build_torch_calibration_cache).
    let calib = CalibCache::build(
        names.iter().map(|n| {
            let info = reader.header().get(n);
            match info {
                Some(i) if is_quantizable(config, n, &i.shape) => Some(i.shape[1] as usize),
                _ => None,
            }
        }),
        config.calib_seed as u64,
    );

    // Corrected-bias cache + cached weight outputs: quantizing a weight
    // computes its specs and its sibling bias's new bytes immediately (weight
    // may come before OR after the bias in file order). Values are written at
    // each tensor's own position.
    let mut weight_outputs: HashMap<String, WeightOutputs> = HashMap::new();
    let mut corrected_bias: HashMap<String, Vec<u8>> = HashMap::new();

    for name in &remaining {
        if corrected_bias.contains_key(name.as_str()) || {
            // bias whose weight was already processed -> write corrected value
            name.ends_with(".bias") && corrected_bias.contains_key(name)
        } {
            let info = reader.header().get(name).expect("exists");
            writer.add_tensor(
                name,
                info.dtype,
                Some(&info.dtype_raw),
                &info.shape,
                &corrected_bias[name.as_str()],
            )?;
            finish_name(&mut state, name)?;
            continue;
        }

        let info_shape = reader
            .header()
            .get(name)
            .map(|i| i.shape.as_slice())
            .unwrap_or(&[]);
        if is_quantizable(config, name, info_shape) && reader.header().contains_key(name) {
            // Weight whose bias came earlier: specs were cached then; write now.
            if weight_outputs.contains_key(name) {
                let o = &weight_outputs[name];
                writer.add_tensor(name, DType::I8, None, &[o.m, o.n], &o.q_bytes)?;
                let base = base_of(name);
                writer.add_tensor(
                    &format!("{base}.weight_scale"),
                    DType::F32,
                    None,
                    &o.scale_shape,
                    &o.scale_bytes,
                )?;
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
            } else {
                compute_weight_outputs(
                    name,
                    &reader,
                    config,
                    &calib,
                    &mut weight_outputs,
                    &mut corrected_bias,
                )?;
                let o = &weight_outputs[name];
                writer.add_tensor(name, DType::I8, None, &[o.m, o.n], &o.q_bytes)?;
                let base = base_of(name);
                writer.add_tensor(
                    &format!("{base}.weight_scale"),
                    DType::F32,
                    None,
                    &o.scale_shape,
                    &o.scale_bytes,
                )?;
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
            }
        } else if name.ends_with(".bias") {
            // Bias reached BEFORE its weight: process (compute+cache) the
            // weight now but only write the bias here.
            let wname = format!("{}.weight", &name[..name.len() - ".bias".len()]);
            let wshape = reader
                .header()
                .get(&wname)
                .map(|i| i.shape.as_slice())
                .unwrap_or(&[]);
            if reader.header().contains_key(&wname) && is_quantizable(config, &wname, wshape) {
                compute_weight_outputs(
                    &wname,
                    &reader,
                    config,
                    &calib,
                    &mut weight_outputs,
                    &mut corrected_bias,
                )?;
                let info = reader.header().get(name).expect("bias exists");
                writer.add_tensor(
                    name,
                    info.dtype,
                    Some(&info.dtype_raw),
                    &info.shape,
                    &corrected_bias[name.as_str()],
                )?;
            } else {
                copy_tensor(name, &reader, &mut writer, target_dtype, &skip_cast)?;
            }
        } else {
            copy_tensor(name, &reader, &mut writer, target_dtype, &skip_cast)?;
        }
        finish_name(&mut state, name)?;
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
fn finish_name(state: &mut StreamState, name: &str) -> Result<()> {
    state.done.insert(name.to_string());
    if !state.order.iter().any(|o| o == name) {
        state.order.push(name.to_string());
    }
    state.save_manifest()?;
    Ok(())
}

fn is_quantizable(config: &QuantConfig, name: &str, shape: &[u64]) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }
    if config.excluded(name) {
        return false;
    }
    if shape.len() != 2 || shape.contains(&0) {
        return false;
    }
    if config.skip_inefficient && should_skip_shape(shape, config.block_size as usize) {
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
#[allow(clippy::type_complexity)]
fn compute_weight_outputs(
    name: &str,
    reader: &SafetensorsReader,
    config: &QuantConfig,
    calib: &CalibCache,
    weight_outputs: &mut HashMap<String, WeightOutputs>,
    corrected_bias: &mut HashMap<String, Vec<u8>>,
) -> Result<()> {
    let info = reader.header().get(name).expect("checked caller");
    let (m, n) = (info.shape[0] as usize, info.shape[1] as usize);
    if !config.skip_inefficient
        && (m % config.block_size as usize != 0 || n % config.block_size as usize != 0)
    {
        return Err(StreamError::NotDivisible {
            m: m as u64,
            n: n as u64,
            block_size: config.block_size,
        });
    }

    let raw = reader.tensor_bytes(name)?;
    let w_f32: Vec<f32> = decode_to_f32(info.dtype, raw);

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
    // family-A key order format, orig_dtype, group_size (block only). The
    // encoder emits group_size only for block-based formats, so passing the
    // block size unconditionally is safe for tensor/row modes.
    let comfy_fmt = match config.scaling_mode {
        ScalingMode::Block => ComfyFormat::Int8Blockwise,
        _ => ComfyFormat::Int8Tensorwise,
    };
    let orig = resolve_orig_dtype_str(&config.orig_dtype);
    let blob = encode_comfy_quant(comfy_fmt, &orig, Some(config.block_size), None);
    let input_scale = matches!(config.scaling_mode, ScalingMode::Block);

    weight_outputs.insert(
        name.to_string(),
        WeightOutputs {
            q_bytes,
            scale_bytes,
            scale_shape,
            blob,
            input_scale,
            m: m as u64,
            n: n as u64,
        },
    );

    // ---- sibling-bias correction (Phase 7.3) -------------------------------- //
    let bias_name = format!("{base}.bias", base = base_of(name));
    if reader.header().contains_key(&bias_name) && !corrected_bias.contains_key(&bias_name) {
        if let Some(x) = calib.get(n) {
            let w_dq = dequantize_int8(&r.qdata, &r.scale, m, n, mode, config.block_size as usize);
            let bias_info = reader.header().get(&bias_name).expect("checked");
            let bias_raw = reader.tensor_bytes(&bias_name)?;
            assert_eq!(
                bias_info.dtype,
                DType::F32,
                "fixtures use F32 biases; extend decode for others when needed"
            );
            let bias: Vec<f32> = bias_raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let bytes = correct_bias(x, &w_f32, &w_dq, &bias, m, n);
            corrected_bias.insert(bias_name, bytes);
        }
        // Missing calibration entry → torch reference warns and keeps the
        // original bias; our copy path handles that naturally.
    }
    Ok(())
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

fn copy_tensor(
    name: &str,
    reader: &SafetensorsReader,
    writer: &mut IncrementalWriter,
    target_dtype: Option<DType>,
    skip_cast: &HashSet<&str>,
) -> Result<()> {
    let info = reader.header().get(name).expect("exists");
    let data = reader.tensor_bytes(name)?;

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

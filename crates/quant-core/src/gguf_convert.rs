//! HF safetensors → GGUF conversion engine (Phase 10.2).
//!
//! Converts a single `.safetensors` file or a sharded HF model folder into a
//! GGUF v3 file using a chosen method from [`crate::gguf_registry`]. Built on
//! the pinned `rlx-gguf` crate (writer + GGML quant encoders).
//!
//! Scope (plan §1 / Phase 10.2):
//! - **Arch detection** from `config.json` (`architectures[0]`) → GGUF arch
//!   string + best-effort arch metadata (context length, embedding length,
//!   block count, ...).
//! - **Tensor name mapping** HF → GGUF for the dense llama/qwen/mistral/gemma
//!   family (the architectures the reference Unsloth GGUF path supports).
//!   Unmapped tensors pass through under their original name.
//! - **Per-tensor scheme** from the method's [`MethodPolicy`] (1-D → F32,
//!   embeddings → `embd_scheme`, rule overrides, else default).
//! - **Divisibility fallback**: if a tensor's element count doesn't divide the
//!   chosen scheme's block size, it falls back to F16 (mirrors
//!   `rlx-gguf-convert` and keeps the output valid GGUF).
//!
//! Parity boundary (plan 10.1 spike): legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 encoders
//! are byte-identical to llama.cpp; K-quants/IQ* use rlx-gguf's simpler
//! min/max search and are valid-but-not-byte-identical to `llama-quantize`.

use std::path::{Path, PathBuf};

use rlx_gguf::{quantize, GgmlType, GgufWriter, MetaValue};

use crate::discover::{classify_input, resolve_union, InputKind};
use crate::dtype::DType;
use crate::gguf_registry::{self, GgufScheme};
use crate::{gguf_quants, llama_policy, st_io::reader::SafetensorsReader};

/// Errors from GGUF conversion.
#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("input is not a usable safetensors file or sharded folder: {0}")]
    BadInput(String),
    #[error("unknown GGUF method '{0}'. Supported: {1}")]
    UnknownMethod(String, String),
    #[error("method '{0}' is an Unsloth Dynamic 2.0 per-layer variant and is not supported natively (proprietary heuristic). Use a plain method instead.")]
    DynamicMethod(String),
    #[error("method '{0}' cannot run: the backend has no encoder for it. {1}")]
    NoEncoder(String, String),
    #[error("tensor {name}: unsupported source dtype {dtype:?} for GGUF conversion")]
    BadDtype { name: String, dtype: DType },
    #[error("safetensors: {0}")]
    St(#[from] crate::st_io::error::Error),
    #[error("gguf: {0}")]
    Gguf(String),
    #[error("imatrix: {0}")]
    Imatrix(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("config.json: {0}")]
    Config(String),
}

/// Configuration for one HF → GGUF conversion.
#[derive(Debug, Clone)]
pub struct GgufConvertConfig {
    /// Registry method id (`"q4_k_m"`, `"f16"`, ...).
    pub method_id: String,
    /// Override the GGUF architecture string; else detected from config.json.
    pub arch: Option<String>,
    /// `general.name` metadata; else derived from the input base name.
    pub name: Option<String>,
    /// Importance matrix for the weighted K-quant path (Phase 4.4).
    /// When present and the method's scheme resolves to Q4_K/Q2_K/
    /// Q3_K/Q5_K/Q6_K for a tensor, that tensor is quantized with the
    /// ported llama.cpp weighted encoders (byte-parity tier); every other
    /// scheme keeps using rlx-gguf as before.
    pub imatrix: Option<crate::imatrix::Imatrix>,
}

/// Summary of a completed conversion.
#[derive(Debug, Clone)]
pub struct GgufConvertReport {
    pub output: PathBuf,
    pub method_id: String,
    pub arch: String,
    /// Total tensors written.
    pub tensors: usize,
    /// Tensors stored in the method's quantized scheme (or a rule override).
    pub quantized: usize,
    /// Tensors that fell back to F16 due to block-size divisibility.
    pub fallback_f16: usize,
    /// Tensors kept as F32 (1-D norms/biases/positions).
    pub kept_f32: usize,
    /// Output file size in bytes.
    pub output_bytes: u64,
    /// GGUF names of every tensor that fell back to F16 (Phase 2.3: silent
    /// degradation is not acceptable — the CLI prints one warning line per
    /// entry and tests assert the exact list).
    pub fallback_tensors: Vec<String>,
}

/// Convert a HF safetensors input (single file or sharded folder) to GGUF.
///
/// `on_progress`, when provided, is called as `(done, total)` after each
/// tensor is encoded — the same `(cur, total)` shape the streaming quantizer
/// uses, so the CLI can drive one progress bar.
pub fn convert_hf_to_gguf(
    input: &Path,
    output: &Path,
    cfg: &GgufConvertConfig,
    mut on_progress: Option<&mut dyn FnMut(usize, usize)>,
) -> Result<GgufConvertReport, GgufError> {
    // 1. Resolve the method (reject unknown, Dynamic 2.0, and no-encoder
    //    up front — Phase 2.1: every rejection names the specific cause).
    let entry = gguf_registry::get_method(&cfg.method_id).ok_or_else(|| {
        GgufError::UnknownMethod(
            cfg.method_id.clone(),
            gguf_registry::usable_ids().join(", "),
        )
    })?;
    if entry.method.dynamic_v2 {
        return Err(GgufError::DynamicMethod(cfg.method_id.clone()));
    }
    if let gguf_registry::BackendSupport::NoEncoder(reason) = entry.method.support {
        return Err(GgufError::NoEncoder(
            cfg.method_id.clone(),
            reason.to_string(),
        ));
    }

    // 2. Classify the input.
    let (kind, base_name) = classify_input(input);
    let kind = kind.ok_or_else(|| GgufError::BadInput(input.display().to_string()))?;
    let base_name = base_name.unwrap_or_else(|| "model".to_string());

    // 3. Collect the ordered (name, shard-index) list + open shard readers.
    let (names, readers) = collect_tensors(input, kind)?;
    let total = names.len();

    // 4. Arch detection + metadata from config.json (best-effort).
    let model_dir = match kind {
        InputKind::SingleFile => input.parent().map(|p| p.to_path_buf()),
        InputKind::ShardedFolder => Some(input.to_path_buf()),
    };
    let arch_info = model_dir
        .as_deref()
        .and_then(load_arch_info)
        .unwrap_or_default();
    let arch = cfg.arch.clone().unwrap_or_else(|| arch_info.arch.clone());

    // 5. Build the writer + metadata.
    let mut w = GgufWriter::new();
    w.set_arch(&arch);
    let general_name = cfg.name.clone().unwrap_or_else(|| base_name.clone());
    w.set_meta("general.name", MetaValue::String(general_name));
    w.set_meta(
        "general.quantized_by",
        MetaValue::String(format!("quantui-rs {}", env!("CARGO_PKG_VERSION"))),
    );
    // Arch-specific parameters (llama.cpp-compatible keys), best-effort.
    for (k, v) in arch_info.meta_pairs(&arch) {
        w.set_meta(k, v);
    }

    // 5.5 Phase 3.0: for composite methods, pre-compute the llama.cpp
    // policy state (attn-v / ffn-down counters) and the model facts the
    // policy tree consults (n_gqa from head counts, n_expert absent in
    // dense configs = 1).
    let engine = entry.policy.engine;
    let mut policy_state = if engine == llama_policy::LlamaPolicy::Flat {
        None
    } else {
        // Count over the GGUF-side names, in processing order.
        let gguf_names: Vec<String> = names
            .iter()
            .map(|(n, _)| hf_to_gguf_name(n).unwrap_or_else(|| n.clone()))
            .collect();
        let refs: Vec<&str> = gguf_names.iter().map(|s| s.as_str()).collect();
        Some(llama_policy::count_state(&refs, false))
    };
    let model_facts = llama_policy::ModelFacts {
        // n_gqa = head_count / head_count_kv (llama.cpp n_gqa()); defaults
        // to 1 (no GQA) when config.json lacks the keys.
        n_gqa: match (arch_info.head_count, arch_info.head_count_kv) {
            (Some(h), Some(kv)) if kv > 0 => (h / kv) as i32,
            _ => 1,
        },
        n_expert: 1, // dense configs; MoE support follows with Phase 3.5.
        is_70b_type: false,
    };

    // 6. Encode each tensor.
    let mut quantized = 0usize;
    let mut fallback_f16 = 0usize;
    let mut kept_f32 = 0usize;
    let mut fallback_tensors: Vec<String> = Vec::new();
    for (done, (name, shard_idx)) in names.iter().enumerate() {
        let reader = &readers[*shard_idx];
        let info = reader.header().get(name).expect("name came from header");
        let raw = reader.tensor_bytes(name)?;
        let ndim = info.shape.len();

        let gguf_name = hf_to_gguf_name(name).unwrap_or_else(|| name.clone());
        // Phase 3.0: composite methods resolve through the llama.cpp
        // policy engine (categories + counters + use_more_bits); simple
        // methods keep the flat registry engine.
        let scheme = match (&mut policy_state, engine) {
            (Some(state), _) => {
                let cat = llama_policy::tensor_get_category(&gguf_name);
                let ctx = llama_policy::PolicyCtx {
                    name: &gguf_name,
                    ndim,
                    category: cat,
                    facts: model_facts,
                    state,
                };
                let s = llama_policy::resolve(&engine, &ctx);
                llama_policy::advance(state, cat);
                s
            }
            (None, _) => gguf_registry::scheme_for(entry, &gguf_name, ndim),
        };
        let ggml = scheme_to_ggml(scheme);

        // Decode to f32 (GGUF encoders consume f32).
        let floats = decode_f32(info.dtype, raw).ok_or_else(|| GgufError::BadDtype {
            name: name.clone(),
            dtype: info.dtype,
        })?;

        // Phase 4.4: weighted path. When an imatrix is configured AND it
        // carries this tensor AND the scheme is one of the ported weighted
        // encoders (Q4_K / Q2_K / Q3_K / Q5_K / Q6_K — byte-parity tier vs
        // llama-quantize), quantize row by row with the shared per-tensor
        // weight vector, exactly as llama.cpp does (llama-quant.cpp:1260-1276
        // drives ggml_quantize_chunk per slab; ggml-quants.c:1626-1640
        // advances src per row while quant_weights stays at the entry base).
        let weights = cfg
            .imatrix
            .as_ref()
            .and_then(|im| im.weights_for(&gguf_name));
        let weighted = matches!(
            (scheme, weights),
            (GgufScheme::Q4K, Some(_))
                | (GgufScheme::Q2K, Some(_))
                | (GgufScheme::Q3K, Some(_))
                | (GgufScheme::Q5K, Some(_))
                | (GgufScheme::Q6K, Some(_))
        );
        if weighted {
            let n_per_row = info.shape.last().copied().unwrap_or(0) as usize;
            let nrows = floats.len() / n_per_row.max(1);
            // Size check mirrors llama-quant.cpp:1228: the entry must cover
            // ne[0] (ne[2]=1 for our 2-D dense case).
            let wv = weights.unwrap();
            if n_per_row == 0 || wv.len() != n_per_row || floats.len() % n_per_row != 0 {
                return Err(GgufError::Imatrix(format!(
                    "imatrix size {} != n_per_row {} for tensor '{gguf_name}'",
                    wv.len(),
                    n_per_row
                )));
            }
            let mut out = Vec::with_capacity(floats.len() / 2);
            for r in 0..nrows {
                let row = &floats[r * n_per_row..(r + 1) * n_per_row];
                let bytes = match scheme {
                    GgufScheme::Q4K => {
                        gguf_quants::quantize_row_q4_k_weighted(row, n_per_row, Some(wv))
                    }
                    GgufScheme::Q3K => {
                        gguf_quants::quantize_row_q3_k_weighted(row, n_per_row, Some(wv))
                    }
                    GgufScheme::Q5K => {
                        gguf_quants::quantize_row_q5_k_weighted(row, n_per_row, Some(wv))
                    }
                    GgufScheme::Q6K => {
                        gguf_quants::quantize_row_q6_k_weighted(row, n_per_row, Some(wv))
                    }
                    _ => gguf_quants::quantize_row_q2_k_weighted(row, n_per_row, Some(wv)),
                };
                out.extend(bytes);
            }
            quantized += 1;
            let shape: Vec<usize> = info.shape.iter().rev().map(|&d| d as usize).collect();
            w.add_tensor_bytes(&gguf_name, shape, ggml, out)
                .map_err(|e| GgufError::Gguf(e.to_string()))?;
            if let Some(cb) = on_progress.as_mut() {
                cb(done + 1, total);
            }
            continue;
        }

        // Encode; fall back to F16 if the element count doesn't divide the
        // scheme's block size (keeps the output valid GGUF). Phase 2.3:
        // the fallback must never be silent — one stderr warning per
        // degraded tensor, and the report carries the exact list.
        let (bytes, dtype) = match quantize(&floats, ggml) {
            Ok(b) => {
                if scheme != GgufScheme::F32 {
                    quantized += 1;
                } else {
                    kept_f32 += 1;
                }
                (b, ggml)
            }
            Err(e) => {
                eprintln!(
                    "warning: tensor '{gguf_name}' fell back to F16: \
                     method '{method}' scheme {scheme:?} cannot encode it ({e}); \
                     output is valid GGUF but this tensor is NOT {method}-quantized",
                    method = cfg.method_id,
                );
                let b =
                    quantize(&floats, GgmlType::F16).map_err(|e| GgufError::Gguf(e.to_string()))?;
                fallback_f16 += 1;
                fallback_tensors.push(gguf_name.clone());
                (b, GgmlType::F16)
            }
        };

        // GGUF stores dims reversed relative to HF row-major; data bytes are
        // unchanged (innermost HF dim stays contiguous == ggml ne0).
        let shape: Vec<usize> = info.shape.iter().rev().map(|&d| d as usize).collect();
        w.add_tensor_bytes(&gguf_name, shape, dtype, bytes)
            .map_err(|e| GgufError::Gguf(e.to_string()))?;

        if let Some(cb) = on_progress.as_mut() {
            cb(done + 1, total);
        }
    }

    // 7. Write the file.
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    w.write_to_path(output)
        .map_err(|e| GgufError::Gguf(e.to_string()))?;
    let output_bytes = std::fs::metadata(output)?.len();

    Ok(GgufConvertReport {
        output: output.to_path_buf(),
        method_id: cfg.method_id.clone(),
        arch,
        tensors: total,
        quantized,
        fallback_f16,
        kept_f32,
        output_bytes,
        fallback_tensors,
    })
}

// ─── tensor collection ──────────────────────────────────────────────

/// Open shard readers and return the ordered tensor list as
/// `(name, shard_index)` pairs (first-appearance order for sharded inputs).
type Collected = (Vec<(String, usize)>, Vec<SafetensorsReader>);

fn collect_tensors(input: &Path, kind: InputKind) -> Result<Collected, GgufError> {
    match kind {
        InputKind::SingleFile => {
            // A single file, or a folder holding exactly one safetensors.
            let file = if input.is_file() {
                input.to_path_buf()
            } else {
                let mut sts: Vec<PathBuf> = std::fs::read_dir(input)?
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
                    .collect();
                sts.sort();
                sts.pop()
                    .ok_or_else(|| GgufError::BadInput(input.display().to_string()))?
            };
            let reader = SafetensorsReader::open(&file)?;
            let names: Vec<(String, usize)> = reader
                .header()
                .names()
                .filter(|n| n.as_str() != "__metadata__")
                .map(|n| (n.clone(), 0usize))
                .collect();
            Ok((names, vec![reader]))
        }
        InputKind::ShardedFolder => {
            let model = crate::discover::discover_shards(input)
                .map_err(|e| GgufError::BadInput(e.to_string()))?;
            let shard_paths = model.shard_paths();
            let union = resolve_union(&shard_paths)?;
            let mut readers = Vec::with_capacity(shard_paths.len());
            for sp in &shard_paths {
                readers.push(SafetensorsReader::open(sp)?);
            }
            let names: Vec<(String, usize)> = union
                .entries
                .iter()
                .map(|(n, _)| {
                    let idx = union.name_to_shard[n];
                    (n.clone(), idx)
                })
                .collect();
            Ok((names, readers))
        }
    }
}

// ─── dtype decode ───────────────────────────────────────────────────

/// Decode a float tensor's raw bytes to f32. Returns `None` for dtypes that
/// have no meaningful float interpretation (ints, bool, fp8).
fn decode_f32(dtype: DType, raw: &[u8]) -> Option<Vec<f32>> {
    use crate::dtype::{bf16_bits_to_f32, f16_bits_to_f32};
    Some(match dtype {
        DType::F32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        DType::F16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_bits_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        // numpy-written bf16 files are tagged U16 (see dtype.rs tolerance).
        DType::Bf16 | DType::U16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        _ => return None,
    })
}

// ─── scheme → GgmlType ──────────────────────────────────────────────

fn scheme_to_ggml(s: GgufScheme) -> GgmlType {
    match s {
        GgufScheme::F32 => GgmlType::F32,
        GgufScheme::F16 => GgmlType::F16,
        GgufScheme::Bf16 => GgmlType::BF16,
        GgufScheme::Q8_0 => GgmlType::Q8_0,
        GgufScheme::Q4_0 => GgmlType::Q4_0,
        GgufScheme::Q4_1 => GgmlType::Q4_1,
        GgufScheme::Q5_0 => GgmlType::Q5_0,
        GgufScheme::Q5_1 => GgmlType::Q5_1,
        GgufScheme::Q2K => GgmlType::Q2K,
        GgufScheme::Q3K => GgmlType::Q3K,
        GgufScheme::Q4K => GgmlType::Q4K,
        GgufScheme::Q5K => GgmlType::Q5K,
        GgufScheme::Q6K => GgmlType::Q6K,
        GgufScheme::Q8K => GgmlType::Q8K,
        GgufScheme::Iq2Xxs => GgmlType::IQ2XXS,
        GgufScheme::Iq2Xs => GgmlType::IQ2XS,
        GgufScheme::Iq3Xxs => GgmlType::IQ3XXS,
        GgufScheme::Iq4Nl => GgmlType::IQ4NL,
        // Unsloth coverage plan Phase 3:
        GgufScheme::Iq1S => GgmlType::IQ1S,
        GgufScheme::Iq1M => GgmlType::IQ1M,
        GgufScheme::Iq2S => GgmlType::IQ2S,
        GgufScheme::Iq3S => GgmlType::IQ3S,
        GgufScheme::Iq4Xs => GgmlType::IQ4XS,
        GgufScheme::Tq1_0 => GgmlType::TQ1_0,
        GgufScheme::Tq2_0 => GgmlType::TQ2_0,
        GgufScheme::Q1_0 => GgmlType::Q1_0,
        GgufScheme::Q2_0 => GgmlType::Q2_0,
    }
}

// ─── HF → GGUF tensor name mapping ──────────────────────────────────

/// Map a HuggingFace tensor name to its GGUF (llama.cpp) equivalent for the
/// dense llama/qwen/mistral/gemma family. Returns `None` for names with no
/// known mapping (callers pass them through unchanged).
///
/// Port of the naming used by llama.cpp `convert_hf_to_gguf.py` for the
/// architectures the reference Unsloth GGUF path supports.
pub fn hf_to_gguf_name(name: &str) -> Option<String> {
    // Top-level (non-layer) tensors.
    match name {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        "model.norm.weight" => return Some("output_norm.weight".into()),
        _ => {}
    }

    // Layer tensors: model.layers.{i}.<rest>
    let rest = name.strip_prefix("model.layers.")?;
    let (idx_str, tail) = rest.split_once('.')?;
    let idx: u32 = idx_str.parse().ok()?;
    let blk = format!("blk.{idx}");

    // Split off the trailing .weight / .bias (if any) to map the core path.
    let (core, suffix) = if let Some(c) = tail.strip_suffix(".weight") {
        (c, ".weight")
    } else if let Some(c) = tail.strip_suffix(".bias") {
        (c, ".bias")
    } else {
        (tail, "")
    };

    let mapped = match core {
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        "self_attn.q_norm" => "attn_q_norm",
        "self_attn.k_norm" => "attn_k_norm",
        "self_attn.rotary_emb.inv_freq" => return None, // skip rotary cache
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        "mlp.gate" => "ffn_gate_inp",
        "input_layernorm" => "attn_norm",
        "post_attention_layernorm" => "ffn_norm",
        _ => return None,
    };
    Some(format!("{blk}.{mapped}{suffix}"))
}

// ─── config.json arch detection ─────────────────────────────────────

/// Architecture + numeric parameters parsed from `config.json` (best-effort).
#[derive(Debug, Clone, Default)]
pub struct ArchInfo {
    /// GGUF architecture string (e.g. `"llama"`, `"qwen2"`).
    pub arch: String,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub block_count: Option<u64>,
    pub feed_forward_length: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub rms_norm_eps: Option<f64>,
    pub vocab_size: Option<u64>,
}

impl ArchInfo {
    /// Emit llama.cpp-compatible `{arch}.*` metadata pairs for the values we
    /// managed to read.
    fn meta_pairs(&self, arch: &str) -> Vec<(String, MetaValue)> {
        let mut out = Vec::new();
        let a = arch.to_string();
        if let Some(v) = self.context_length {
            out.push((format!("{a}.context_length"), MetaValue::U64(v)));
        }
        if let Some(v) = self.embedding_length {
            out.push((format!("{a}.embedding_length"), MetaValue::U64(v)));
        }
        if let Some(v) = self.block_count {
            out.push((format!("{a}.block_count"), MetaValue::U64(v)));
        }
        if let Some(v) = self.feed_forward_length {
            out.push((format!("{a}.feed_forward_length"), MetaValue::U64(v)));
        }
        if let Some(v) = self.head_count {
            out.push((format!("{a}.attention.head_count"), MetaValue::U64(v)));
        }
        if let Some(v) = self.head_count_kv {
            out.push((format!("{a}.attention.head_count_kv"), MetaValue::U64(v)));
        }
        if let Some(v) = self.rms_norm_eps {
            out.push((
                format!("{a}.attention.layer_norm_rms_epsilon"),
                MetaValue::F32(v as f32),
            ));
        }
        out
    }
}

/// Map a HF `architectures[0]` class name to a GGUF arch string.
fn arch_from_classname(cls: &str) -> String {
    match cls {
        "LlamaForCausalLM" | "MistralForCausalLM" | "MixtralForCausalLM" => "llama".into(),
        "Qwen2ForCausalLM" | "Qwen3ForCausalLM" => "qwen2".into(),
        "GemmaForCausalLM" | "Gemma2ForCausalLM" => "gemma".into(),
        "PhiForCausalLM" | "Phi3ForCausalLM" => "phi".into(),
        "GPT2LMHeadModel" => "gpt2".into(),
        "BloomForCausalLM" => "bloom".into(),
        "FalconForCausalLM" => "falcon".into(),
        "StableLmForCausalLM" => "stablelm".into(),
        other => {
            // Fallback: lowercase the leading word before "For".
            let stem = other.split("For").next().unwrap_or(other);
            stem.to_lowercase()
        }
    }
}

/// Read `<dir>/config.json` and extract arch + numeric params. Returns `None`
/// if the file is absent or unparseable (conversion still proceeds with a
/// default arch).
fn load_arch_info(dir: &Path) -> Option<ArchInfo> {
    let path = dir.join("config.json");
    let raw = std::fs::read_to_string(&path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;

    let cls = v
        .get("architectures")
        .and_then(|a| a.get(0))
        .and_then(|s| s.as_str())
        .unwrap_or("LlamaForCausalLM");
    let arch = v
        .get("model_type")
        .and_then(|s| s.as_str())
        .map(|mt| {
            // Prefer model_type when it's a known GGUF arch; else class map.
            match mt {
                "llama" | "mistral" | "mixtral" => "llama".to_string(),
                "qwen2" | "qwen3" => "qwen2".to_string(),
                "gemma" | "gemma2" => "gemma".to_string(),
                "phi" | "phi3" => "phi".to_string(),
                "gpt2" => "gpt2".to_string(),
                _ => arch_from_classname(cls),
            }
        })
        .unwrap_or_else(|| arch_from_classname(cls));

    let u64 = |k: &str| v.get(k).and_then(|x| x.as_u64());
    let f64 = |k: &str| v.get(k).and_then(|x| x.as_f64());

    Some(ArchInfo {
        arch,
        context_length: u64("max_position_embeddings"),
        embedding_length: u64("hidden_size"),
        block_count: u64("num_hidden_layers"),
        feed_forward_length: u64("intermediate_size"),
        head_count: u64("num_attention_heads"),
        head_count_kv: u64("num_key_value_heads"),
        rms_norm_eps: f64("rms_norm_eps"),
        vocab_size: u64("vocab_size"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_mapping_dense_llama_family() {
        assert_eq!(
            hf_to_gguf_name("model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            hf_to_gguf_name("lm_head.weight").as_deref(),
            Some("output.weight")
        );
        assert_eq!(
            hf_to_gguf_name("model.norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.3.self_attn.q_proj.weight").as_deref(),
            Some("blk.3.attn_q.weight")
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.0.mlp.down_proj.weight").as_deref(),
            Some("blk.0.ffn_down.weight")
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.11.input_layernorm.weight").as_deref(),
            Some("blk.11.attn_norm.weight")
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.2.self_attn.k_proj.bias").as_deref(),
            Some("blk.2.attn_k.bias")
        );
        // Rotary cache is skipped.
        assert_eq!(
            hf_to_gguf_name("model.layers.0.self_attn.rotary_emb.inv_freq"),
            None
        );
        // Unknown passes through as None (caller keeps original name).
        assert_eq!(
            hf_to_gguf_name("model.layers.0.block_sparse_moe.experts.0.w1.weight"),
            None
        );
    }

    #[test]
    fn arch_classname_mapping() {
        assert_eq!(arch_from_classname("LlamaForCausalLM"), "llama");
        assert_eq!(arch_from_classname("Qwen2ForCausalLM"), "qwen2");
        assert_eq!(arch_from_classname("GemmaForCausalLM"), "gemma");
        assert_eq!(arch_from_classname("SomethingForCausalLM"), "something");
    }

    #[test]
    fn scheme_to_ggml_is_total() {
        // Every registry scheme must map to a GgmlType with an encoder.
        for s in [
            GgufScheme::F32,
            GgufScheme::F16,
            GgufScheme::Bf16,
            GgufScheme::Q8_0,
            GgufScheme::Q4_0,
            GgufScheme::Q4_1,
            GgufScheme::Q5_0,
            GgufScheme::Q5_1,
            GgufScheme::Q2K,
            GgufScheme::Q3K,
            GgufScheme::Q4K,
            GgufScheme::Q5K,
            GgufScheme::Q6K,
            GgufScheme::Q8K,
            GgufScheme::Iq2Xxs,
            GgufScheme::Iq2Xs,
            GgufScheme::Iq3Xxs,
            GgufScheme::Iq4Nl,
            // Phase 3 additions (Unsloth coverage plan):
            GgufScheme::Iq1S,
            GgufScheme::Iq1M,
            GgufScheme::Iq2S,
            GgufScheme::Iq3S,
            GgufScheme::Iq4Xs,
            GgufScheme::Tq1_0,
            GgufScheme::Tq2_0,
            GgufScheme::Q1_0,
            GgufScheme::Q2_0,
        ] {
            let _ = scheme_to_ggml(s);
        }
    }
}

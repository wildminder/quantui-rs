//! HF safetensors → GGUF conversion engine (Phase 10.2).
//!
//! Converts a single `.safetensors` file or a sharded HF model folder into a
//! GGUF v3 file using a chosen method from [`crate::gguf_registry`]. Built on
//! the pinned `rlx-gguf` crate (writer + GGML quant encoders).
//!
//! Scope (plan §1 / Phase 10.2):
//! - **Arch detection** from `config.json` and **tensor name mapping** HF →
//!   GGUF live in [`crate::gguf_names`] (extracted there, IMP-001); this
//!   module consumes them.
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
use crate::gguf_names::{hf_to_gguf_name, load_arch_info};
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
    /// Per-tensor recipe overrides (Phase 6.2, `--tensor-type-file`).
    /// When a rule matches the GGUF-side tensor name (search semantics,
    /// first-match-wins, llama-quant.cpp:713-727), its scheme wins over the
    /// method's own policy — including the composite `_M`/`_L` engines,
    /// which are skipped entirely for that tensor (upstream `manual` path).
    /// A matched recipe scheme still respects the F16 divisibility
    /// fallback and the 1-D F32 convention.
    pub recipe: Option<crate::gguf_recipe::TensorRecipe>,
    /// `--token-embedding-type`: overrides the scheme for the
    /// `token_embd.weight` / `per_layer_token_embd.weight` tensors
    /// (llama-quant.cpp:687-702). `per_layer_token_embd` still lets an
    /// explicit recipe rule name it (upstream `named` exception).
    pub token_embedding_type: Option<String>,
    /// `--output-tensor-type`: overrides the scheme for the
    /// `output.weight` tensor (llama-quant.cpp:704-706).
    pub output_tensor_type: Option<String>,
}

impl Default for GgufConvertConfig {
    /// Defaults mirror the CLI's: method `q4_k_m`, everything else off.
    /// Tests use `..Default::default()` so new optional fields (like
    /// `recipe`) do not churn every construction site.
    fn default() -> Self {
        Self {
            method_id: "q4_k_m".into(),
            arch: None,
            name: None,
            imatrix: None,
            recipe: None,
            token_embedding_type: None,
            output_tensor_type: None,
        }
    }
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
    /// Tensors demoted to a wider block-aligned scheme (or F16) because the
    /// GGUF row size `ne[0]` was not a multiple of the scheme's block size
    /// (llama-quant.cpp:372-425). Distinct from [`Self::fallback_f16`].
    pub row_fallback: usize,
    /// GGUF names of every tensor whose scheme was demoted because the row
    /// size `ne[0]` is not a multiple of the scheme's block size (port of
    /// llama-quantize `tensor_type_fallback`, llama-quant.cpp:372-425).
    /// Kept SEPARATE from [`Self::fallback_f16`]: that one means "the
    /// encoder rejected the tensor" (Phase 2.3, flat element count), this
    /// one means "the requested scheme cannot legally describe this row
    /// width" — a different defect with a different remedy.
    pub row_fallback_tensors: Vec<String>,
    /// Effective per-tensor scheme assignment in conversion order
    /// (Phase 6.3 `--emit-recipe`): `(gguf_name, scheme)` AFTER all
    /// overrides (recipe, category overrides) AND after the per-row
    /// block-size demotion — the demoted scheme is what actually lands in
    /// the file, so a recipe emitted from it round-trips byte-for-byte.
    pub effective_schemes: Vec<(String, GgufScheme)>,
}

/// Result of encoding ONE tensor (IMP-004 step 1, the extraction of the
/// conversion loop body): the payload plus everything the caller needs to
/// update the report, print warnings, and feed the writer. `warnings`
/// carries the stderr lines the inline code used to print directly — the
/// caller prints them in file order so sequential and (later) parallel
/// encoding produce identical stderr streams.
struct EncodedTensor {
    gguf_name: String,
    dtype: GgmlType,
    bytes: Vec<u8>,
    scheme: GgufScheme,
    /// True when the tensor was weighted-quantized (counted as quantized).
    weighted_done: bool,
    /// True when the tensor quantized under its own scheme.
    quantized: bool,
    kept_f32: bool,
    /// Phase 2.3/6.3: encoder rejected it or a demotion landed on F16.
    fell_back_f16: bool,
    /// Row-width demotion fired (the CAUSE counter, distinct from
    /// fell_back_f16 which counts the OUTCOME).
    row_demoted: bool,
    /// User-facing stderr lines, printed by the caller in file order.
    warnings: Vec<String>,
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
    let mut row_fallback = 0usize;
    let mut row_fallback_tensors: Vec<String> = Vec::new();
    let mut effective_schemes: Vec<(String, GgufScheme)> = Vec::new();
    for (done, (name, shard_idx)) in names.iter().enumerate() {
        let reader = &readers[*shard_idx];
        let encoded = encode_one_tensor(
            cfg,
            entry,
            engine,
            &mut policy_state,
            model_facts,
            reader,
            name,
            done,
            total,
        )?;

        // Report accounting + warning printing stay in file order here;
        // the parallel path (IMP-004 step 2) reuses the exact same logic.
        if encoded.row_demoted {
            row_fallback += 1;
            row_fallback_tensors.push(encoded.gguf_name.clone());
        }
        effective_schemes.push((encoded.gguf_name.clone(), encoded.scheme));
        for line in &encoded.warnings {
            eprintln!("{line}");
        }
        if encoded.weighted_done {
            quantized += 1;
        } else if encoded.fell_back_f16 {
            fallback_f16 += 1;
            fallback_tensors.push(encoded.gguf_name.clone());
        } else if encoded.quantized {
            quantized += 1;
        } else if encoded.kept_f32 {
            kept_f32 += 1;
        }

        let shape: Vec<usize> = reader
            .header()
            .get(name)
            .expect("name came from header")
            .shape
            .iter()
            .rev()
            .map(|&d| d as usize)
            .collect();
        w.add_tensor_bytes(&encoded.gguf_name, shape, encoded.dtype, encoded.bytes)
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
        row_fallback,
        row_fallback_tensors,
        effective_schemes,
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

/// Port of the early-return category overrides in llama_tensor_get_type
/// (llama-quant.cpp:688-706). Returns `Some(scheme)` when an override
/// applies — the caller must return it WITHOUT consulting the recipe or
/// the method's engine. The single exception (:688-699): a recipe rule
/// explicitly naming `per_layer_token_embd.weight` beats the embd
/// override (upstream comment: it is a "large separate table").
fn category_override(
    cfg: &GgufConvertConfig,
    gguf_name: &str,
    cat: llama_policy::TensorCategory,
) -> Result<Option<GgufScheme>, GgufError> {
    use llama_policy::TensorCategory;
    if let Some(embd_id) = cfg.token_embedding_type.as_deref() {
        if cat == TensorCategory::TokenEmbd {
            let named_by_recipe = gguf_name == "per_layer_token_embd.weight"
                && cfg
                    .recipe
                    .iter()
                    .flat_map(|r| r.rules.iter().map(|ru| &ru.regex))
                    .any(|re| re.is_match(gguf_name));
            if !named_by_recipe {
                return scheme_for_method_id(embd_id).map(Some);
            }
        }
    }
    if let Some(out_id) = cfg.output_tensor_type.as_deref() {
        if cat == TensorCategory::Output {
            return scheme_for_method_id(out_id).map(Some);
        }
    }
    Ok(None)
}

/// Port of `ggml_is_quantized(default_type)` (llama-quant.cpp:711): the
/// recipe and the method's engine only run when the method's default type
/// is quantized — an f16/f32/bf16 method ignores per-tensor recipes
/// entirely (upstream never enters the manual/impl block).
fn scheme_is_quantized(s: GgufScheme) -> bool {
    !matches!(s, GgufScheme::F32 | GgufScheme::F16 | GgufScheme::Bf16)
}

/// Phase 6.2: resolve a method-id override (`--token-embedding-type`,
/// `--output-tensor-type`) to its scheme. The id must be a usable registry
/// method; anything else is a hard usage error naming the id (the caller
/// maps it to exit 2 via GgufError::UnknownMethod semantics — here we
/// reuse the error type so the CLI's exit-code mapping picks it up).
fn scheme_for_method_id(id: &str) -> Result<GgufScheme, GgufError> {
    let entry = gguf_registry::get_method(id).ok_or_else(|| {
        GgufError::UnknownMethod(id.to_string(), gguf_registry::usable_ids().join(", "))
    })?;
    if entry.method.dynamic_v2
        || matches!(
            entry.method.support,
            gguf_registry::BackendSupport::NoEncoder(_)
        )
    {
        return Err(GgufError::UnknownMethod(
            id.to_string(),
            gguf_registry::usable_ids().join(", "),
        ));
    }
    Ok(entry.policy.default)
}

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

/// Number of elements in one quantized block (`ggml_blck_size`,
/// ggml.c type_traits table — the `blck_size` field).
///
/// Values mirror the PINNED encoder `rlx-gguf` 0.2.14 rather than
/// upstream llama.cpp where the two disagree, because rlx is what writes
/// the bytes: `Q2_0` is 128 here (rlx `q2_dequant.rs:28`) vs 64 in
/// llama.cpp (`ggml-common.h`), and `NVFP4` is 16 (rlx
/// `mx_dequant.rs:27`) vs 64 upstream.
///
/// The match is EXHAUSTIVE on purpose — no wildcard arm — so a new
/// `GgmlType` variant added by a rlx bump is a compile error here rather
/// than a silently wrong block size.
pub fn ggml_blck_size(t: GgmlType) -> usize {
    match t {
        // ── unquantized: every element is its own "block" ──
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => 1,
        GgmlType::I8 | GgmlType::I16 | GgmlType::I32 | GgmlType::I64 | GgmlType::F64 => 1,
        // ── 32-element blocks ──
        GgmlType::Q4_0
        | GgmlType::Q4_1
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::Q8_1
        | GgmlType::IQ4NL => 32,
        GgmlType::MXFP4 => 32,
        // ── 256-element blocks (K-quants, I-quants, ternary) ──
        GgmlType::Q2K
        | GgmlType::Q3K
        | GgmlType::Q4K
        | GgmlType::Q5K
        | GgmlType::Q6K
        | GgmlType::Q8K => 256,
        GgmlType::IQ2XXS
        | GgmlType::IQ2XS
        | GgmlType::IQ3XXS
        | GgmlType::IQ1S
        | GgmlType::IQ3S
        | GgmlType::IQ2S
        | GgmlType::IQ4XS
        | GgmlType::IQ1M => 256,
        GgmlType::TQ1_0 | GgmlType::TQ2_0 => 256,
        GgmlType::I8_S | GgmlType::FV5 | GgmlType::FV5B => 256,
        // ── 128-element blocks ──
        GgmlType::I2_S => 128,
        GgmlType::Q1_0 | GgmlType::Q2_0 => 128,
        // ── 16-element blocks ──
        GgmlType::NVFP4 => 16,
    }
}

/// Demote `target` when the GGUF row size `ne[0]` is not a multiple of the
/// scheme's block size.
///
/// Faithful port of `tensor_type_fallback`
/// (`docs/ref/llama.cpp/src/llama-quant.cpp:372-425`). Upstream calls it
/// from `llama_tensor_get_type` (:309), i.e. at type-selection time, and
/// `llama-quantize` therefore NEVER writes a tensor whose `ne[0]` is not
/// block-aligned.
///
/// Every demotion logs one warning line to stderr — silent degradation is
/// the bug we are fixing. Upstream `throw`s when no smaller type is
/// available; we return F16 and say so, because aborting a whole model
/// conversion over one odd-shaped conv kernel is worse than storing that
/// tensor unquantized (and matches the Phase 2.3 philosophy).
fn row_fallback_scheme(name: &str, ne0: usize, target: GgufScheme) -> GgufScheme {
    let qk_k = ggml_blck_size(scheme_to_ggml(target));
    // Fast path: the row is already block-aligned. Stays silent — a clean
    // conversion must produce NO output (existing tests assert this).
    if ne0 % qk_k == 0 {
        return target;
    }

    let new_type = match target {
        // Very-low-bit i-quants have nothing below them except IQ4_NL.
        GgufScheme::Iq1S
        | GgufScheme::Iq1M
        | GgufScheme::Iq2Xxs
        | GgufScheme::Iq2Xs
        | GgufScheme::Iq2S
        | GgufScheme::Iq3Xxs
        | GgufScheme::Iq3S
        | GgufScheme::Iq4Xs => GgufScheme::Iq4Nl,
        GgufScheme::Q2_0 | GgufScheme::Q2K | GgufScheme::Q3K => GgufScheme::Q4_0,
        // Ternary types: upstream has no smaller ternary, Q4_0 is the
        // narrowest legal 32-block fallback.
        GgufScheme::Tq1_0 | GgufScheme::Tq2_0 => GgufScheme::Q4_0,
        GgufScheme::Q4K => GgufScheme::Q5_0,
        GgufScheme::Q5K => GgufScheme::Q5_1,
        GgufScheme::Q6K => GgufScheme::Q8_0,
        _ => {
            if qk_k <= 32 {
                // Already the narrowest block available — nothing to demote
                // to (upstream: `return qtype; // TODO: what to do here?`).
                target
            } else {
                // Upstream throws `format("unsupported quantization type ...")`.
                GgufScheme::F16
            }
        }
    };

    // Second check (llama-quant.cpp:411-421): the demoted type must also
    // divide the row, otherwise there is no legal quantized type at all
    // ("most likely, this tensor's first dimension is not divisible by 32").
    let mut return_type = new_type;
    let unusual = ne0 % ggml_blck_size(scheme_to_ggml(return_type)) != 0;
    if unusual {
        // Upstream: `return_type = GGML_TYPE_F16;`
        return_type = GgufScheme::F16;
    }

    // Upstream logs ONE line in three parts (:379, :419, :422) — kept as a
    // single line here so per-tensor greps stay 1:1 with tensor count.
    eprintln!(
        "warning: {name:<36} - ncols {ne0:>6} not divisible by {qk_k:>3} \
         (required for type {tgt:>7}) {unusual}-> falling back to {ret:>7}",
        tgt = scheme_type_name(target),
        unusual = if unusual {
            "(WARNING: must use F16 due to unusual shape) "
        } else {
            ""
        },
        ret = scheme_type_name(return_type),
    );

    return_type
}

/// Upstream `ggml_type_name()` spelling, so the warning line above is
/// byte-comparable with `llama-quantize` output.
fn scheme_type_name(s: GgufScheme) -> &'static str {
    match s {
        GgufScheme::F32 => "F32",
        GgufScheme::F16 => "F16",
        GgufScheme::Bf16 => "BF16",
        GgufScheme::Q8_0 => "Q8_0",
        GgufScheme::Q4_0 => "Q4_0",
        GgufScheme::Q4_1 => "Q4_1",
        GgufScheme::Q5_0 => "Q5_0",
        GgufScheme::Q5_1 => "Q5_1",
        GgufScheme::Q2K => "Q2_K",
        GgufScheme::Q3K => "Q3_K",
        GgufScheme::Q4K => "Q4_K",
        GgufScheme::Q5K => "Q5_K",
        GgufScheme::Q6K => "Q6_K",
        GgufScheme::Q8K => "Q8_K",
        GgufScheme::Iq2Xxs => "IQ2_XXS",
        GgufScheme::Iq2Xs => "IQ2_XS",
        GgufScheme::Iq3Xxs => "IQ3_XXS",
        GgufScheme::Iq4Nl => "IQ4_NL",
        GgufScheme::Iq1S => "IQ1_S",
        GgufScheme::Iq1M => "IQ1_M",
        GgufScheme::Iq2S => "IQ2_S",
        GgufScheme::Iq3S => "IQ3_S",
        GgufScheme::Iq4Xs => "IQ4_XS",
        GgufScheme::Tq1_0 => "TQ1_0",
        GgufScheme::Tq2_0 => "TQ2_0",
        GgufScheme::Q1_0 => "Q1_0",
        GgufScheme::Q2_0 => "Q2_0",
    }
}

// ─── per-tensor encoding (IMP-004 step 1) ──────────────────────────

// ─── per-tensor encoding (IMP-004 step 1) ──────────────────────────

/// Encode ONE tensor: resolve its GGUF-side name and scheme (advancing
/// the policy counters — this part is order-sensitive and stays in the
/// sequential driver), apply the row-width demotion, and quantize the
/// payload. Returns everything the driver needs; warnings come back as
/// strings so the driver prints them in file order regardless of where
/// encoding ran (IMP-004).
#[allow(clippy::too_many_arguments)]
fn encode_one_tensor(
    cfg: &GgufConvertConfig,
    entry: &gguf_registry::RegistryEntry,
    engine: llama_policy::LlamaPolicy,
    policy_state: &mut Option<llama_policy::PolicyState>,
    model_facts: llama_policy::ModelFacts,
    reader: &SafetensorsReader,
    name: &str,
    _done: usize,
    _total: usize,
) -> Result<EncodedTensor, GgufError> {
    let info = reader.header().get(name).expect("name came from header");
    let raw = reader.tensor_bytes(name)?;
    let ndim = info.shape.len();

    let gguf_name = hf_to_gguf_name(name).unwrap_or_else(|| name.to_string());
    let mut warnings: Vec<String> = Vec::new();
    let mut row_demoted = false;

    // Phase 3.0 + 6.2: per-tensor type resolution — a faithful port of
    // llama_tensor_get_type (llama-quant.cpp:683-739):
    //   1. 1-D → F32 (never quantized — the shared convention);
    //   2. --token-embedding-type / --output-tensor-type return EARLY
    //      (:688-706): neither the recipe nor the method's engine runs
    //      for those tensors. Exception: a recipe rule explicitly
    //      naming per_layer_token_embd beats the embd override
    //      (:688-699 `named` — it is a "large separate table");
    //   3. the recipe's first matching rule = manual mode (:713-727):
    //      it skips the engine INCLUDING its counter advancement —
    //      upstream `manual = true` never calls
    //      llama_tensor_get_type_impl, which owns the ++qs.i_* counter
    //      increments (:570, :634, :730-731) — and the bare default
    //      applies when no rule matches;
    //   4. the method's own engine: the llama_policy engine for
    //      composite methods (categories + counters + use_more_bits),
    //      the flat registry rules otherwise.
    // The manual/engine block only runs when the method's default type
    // is quantized (:711) — f16/f32/bf16 methods ignore the recipe
    // entirely, exactly like upstream. The category overrides at step
    // 2 sit BEFORE that gate, so they apply even to f16 methods.
    let cat = llama_policy::tensor_get_category(&gguf_name);
    let scheme = if ndim < 2 {
        GgufScheme::F32
    } else if let Some(s) = category_override(cfg, &gguf_name, cat)? {
        s
    } else if !scheme_is_quantized(entry.policy.default) {
        // Upstream skips the manual+engine block when the method's
        // default type is not quantized (:711) — the recipe is inert.
        entry.policy.default
    } else if let Some(s) = cfg.recipe.as_ref().and_then(|r| r.scheme_for(&gguf_name)) {
        // Manual mode — the engine and its counters are skipped.
        s
    } else {
        match policy_state.as_mut() {
            Some(state) => {
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
            None => gguf_registry::scheme_for(entry, &gguf_name, ndim),
        }
    };

    // Per-row block-size demotion — port of `tensor_type_fallback`
    // (llama-quant.cpp:372-425), which upstream calls from
    // `llama_tensor_get_type` (:309), i.e. at TYPE-SELECTION time. It
    // must therefore run BEFORE the imatrix/weighted dispatch below
    // (a demoted tensor may no longer be an imatrix scheme at all) and
    // before `quantize()`.
    //
    // Why this exists: the GGUF contract requires ne[0] % blck_size == 0
    // per ROW (gguf.cpp:724 rejects it, :1409 asserts it), while the
    // pinned rlx-gguf 0.2.14 `quantize()` only checks the FLAT element
    // count (rlx quantize.rs:199 `check_div(name, n, blk)`). Conv1d /
    // ConvTranspose1d weights (VibeVoice-1.5B: 102 tensors with kernel
    // sizes 4/7/8/10/16 → ne[0] = 4/7/8/10/16) have a divisible flat
    // count but NOT a divisible row, so they were silently quantized
    // into blocks straddling row boundaries. llama-quantize never
    // writes such a tensor.
    let ne0 = info.shape.last().copied().unwrap_or(0) as usize;
    let pre = scheme;
    let scheme = if ne0 == 0 {
        scheme
    } else {
        let demoted = row_fallback_scheme(&gguf_name, ne0, scheme);
        if demoted != scheme {
            row_demoted = true;
        }
        demoted
    };
    // A demotion that lands on F16 has the SAME user-visible outcome as
    // the Phase 2.3 fallback — the tensor is not in the method's scheme —
    // so it is reported the same way (`fallback_f16`, the CLI's "were NOT
    // quantized with '<method>'" summary). `row_fallback` above is what
    // records the CAUSE: a row width no quantized type can describe,
    // discovered at type-selection time rather than by the encoder.
    let demoted_to_f16 = pre != GgufScheme::F16 && scheme == GgufScheme::F16;

    // Phase 6.3: effective scheme is recorded by the DRIVER after this
    // returns (it owns the report).
    let ggml = scheme_to_ggml(scheme);

    // Decode to f32 (GGUF encoders consume f32).
    let floats = decode_f32(info.dtype, raw).ok_or_else(|| GgufError::BadDtype {
        name: name.to_string(),
        dtype: info.dtype,
    })?;

    // Phase 4.4 entry semantics (llama-quant.cpp:1222-1251, :803-822):
    // - no imatrix configured → legacy behaviour: everything goes
    //   through the unweighted encoders (rlx for IQ*; the Option-None
    //   `av_x + |x|` path for K-quants) — byte-identical to the
    //   pre-Phase-4 output, and what the method-matrix sweep locks.
    // - imatrix configured but no entry for this tensor:
    //     * IQ-scheme tensor → HARD ERROR (very-low-bit garbage;
    //       llama-quant.cpp:1245-1251 bails out the same way);
    //     * Q2_K under q2_k_s → HARD ERROR (:818);
    //     * any other K-quant → warn + proceed unweighted (:1226
    //       logs "did not find weights" at INFO);
    //     * token_embd / output are exempt from the hard errors (:804)
    //       and a SIZE-mismatched token_embd entry is dropped with a
    //       note (:1238 — "tok_embd should be ignored in this case").
    let exempt = matches!(
        gguf_name.as_str(),
        "token_embd.weight" | "per_layer_token_embd.weight" | "output.weight"
    );
    let is_iq_scheme = matches!(
        scheme,
        GgufScheme::Iq1S
            | GgufScheme::Iq1M
            | GgufScheme::Iq2Xxs
            | GgufScheme::Iq2Xs
            | GgufScheme::Iq2S
            | GgufScheme::Iq3Xxs
            | GgufScheme::Iq3S
            | GgufScheme::Iq4Nl
            | GgufScheme::Iq4Xs
    );
    let mut weights = cfg
        .imatrix
        .as_ref()
        .and_then(|im| im.weights_for(&gguf_name));
    if cfg.imatrix.is_some() {
        let n_per_row = info.shape.last().copied().unwrap_or(0) as usize;
        if let Some(wv) = weights {
            if n_per_row == 0 || wv.len() != n_per_row {
                if matches!(
                    gguf_name.as_str(),
                    "token_embd.weight" | "per_layer_token_embd.weight"
                ) {
                    warnings.push(format!(
                        "note: imatrix size {} != n_per_row {} for tensor '{gguf_name}' — \
                         quantizing it without weights (llama-quantize ignores tok_embd mismatches)",
                        wv.len(),
                        n_per_row
                    ));
                    weights = None;
                } else {
                    return Err(GgufError::Imatrix(format!(
                        "imatrix size {} != n_per_row {} for tensor '{gguf_name}'",
                        wv.len(),
                        n_per_row
                    )));
                }
            }
        }
        if weights.is_none() {
            // llama-quant.cpp:803-822 hard-requires an entry for IQ
            // schemes (and Q2_K under the Q2_K_S ftype — but our
            // registry has no q2_k_s method, Unsloth ships q2_k /
            // q2_k_l, so that branch cannot occur here). token_embd /
            // output are always exempt (:804-806).
            let requires = is_iq_scheme && !exempt;
            if requires {
                return Err(GgufError::Imatrix(format!(
                    "Missing importance matrix for tensor '{gguf_name}' in a very low-bit \
                     quantization (method '{}'); the result would be garbage, so bailing out",
                    cfg.method_id
                )));
            }
            // Upstream logs "did not find weights" (llama-quant.cpp:1226)
            // for every tensor lacking an entry. We note 2-D tensors
            // only — 1-D norms/biases never carry imatrix entries, and a
            // note per norm would be pure noise.
            if ndim >= 2 {
                warnings.push(format!(
                    "note: did not find weights for '{gguf_name}' — quantizing it without \
                     an importance matrix (llama-quantize logs the same)"
                ));
            }
        }
    }
    let weighted = matches!(
        (scheme, weights),
        (GgufScheme::Q4K, Some(_))
            | (GgufScheme::Q2K, Some(_))
            | (GgufScheme::Q3K, Some(_))
            | (GgufScheme::Q5K, Some(_))
            | (GgufScheme::Q6K, Some(_))
            | (GgufScheme::Iq2Xxs, Some(_))
            | (GgufScheme::Iq2Xs, Some(_))
            | (GgufScheme::Iq2S, Some(_))
            | (GgufScheme::Iq3Xxs, Some(_))
            | (GgufScheme::Iq3S, Some(_))
            | (GgufScheme::Iq1S, Some(_))
            | (GgufScheme::Iq1M, Some(_))
            | (GgufScheme::Iq4Nl, Some(_))
            | (GgufScheme::Iq4Xs, Some(_))
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
                // IQ family: weights are REQUIRED upstream (NULL is
                // GGML_ASSERTed, ggml-quants.c:3302/:3480) — the driver
                // only routes here when the entry exists.
                GgufScheme::Iq2Xxs => {
                    crate::gguf_iq_quants::quantize_row_iq2_xxs_weighted(row, n_per_row, wv)
                }
                GgufScheme::Iq2Xs => {
                    crate::gguf_iq_quants::quantize_row_iq2_xs_weighted(row, n_per_row, wv)
                }
                // iq2_s/iq3_xxs/iq3_s accept Option (their *_ref
                // paths pass NULL upstream); the driver always has
                // weights here.
                GgufScheme::Iq2S => {
                    crate::gguf_iq_quants::quantize_row_iq2_s_weighted(row, n_per_row, Some(wv))
                }
                GgufScheme::Iq3Xxs => {
                    crate::gguf_iq_quants::quantize_row_iq3_xxs_weighted(row, n_per_row, Some(wv))
                }
                GgufScheme::Iq3S => {
                    crate::gguf_iq_quants::quantize_row_iq3_s_weighted(row, n_per_row, Some(wv))
                }
                GgufScheme::Iq1S => {
                    crate::gguf_iq_quants::quantize_row_iq1_s_weighted(row, n_per_row, wv)
                }
                GgufScheme::Iq1M => {
                    crate::gguf_iq_quants::quantize_row_iq1_m_weighted(row, n_per_row, Some(wv))
                }
                GgufScheme::Iq4Nl => {
                    crate::gguf_iq_quants::quantize_row_iq4_nl_weighted(row, n_per_row, Some(wv))
                }
                GgufScheme::Iq4Xs => {
                    crate::gguf_iq_quants::quantize_row_iq4_xs_weighted(row, n_per_row, Some(wv))
                }
                _ => gguf_quants::quantize_row_q2_k_weighted(row, n_per_row, Some(wv)),
            };
            out.extend(bytes);
        }
        return Ok(EncodedTensor {
            gguf_name,
            dtype: ggml,
            bytes: out,
            scheme,
            weighted_done: true,
            quantized: false,
            kept_f32: false,
            fell_back_f16: false,
            row_demoted,
            warnings,
        });
    }

    // Encode; fall back to F16 if the element count doesn't divide the
    // scheme's block size (keeps the output valid GGUF). Phase 2.3:
    // the fallback must never be silent — one stderr warning per
    // degraded tensor, and the report carries the exact list.
    let (bytes, dtype, result) = match quantize(&floats, ggml) {
        Ok(b) => {
            let result = if demoted_to_f16 {
                warnings.push(format!(
                    "warning: tensor '{gguf_name}' fell back to F16: \
                     method '{method}' scheme {pre:?} cannot describe a row of {ne0} elements; \
                     output is valid GGUF but this tensor is NOT {method}-quantized",
                    method = cfg.method_id,
                ));
                "f16"
            } else if scheme != GgufScheme::F32 {
                "quantized"
            } else {
                "f32"
            };
            (b, ggml, result)
        }
        Err(e) => {
            warnings.push(format!(
                "warning: tensor '{gguf_name}' fell back to F16: \
                 method '{method}' scheme {scheme:?} cannot encode it ({e}); \
                 output is valid GGUF but this tensor is NOT {method}-quantized",
                method = cfg.method_id,
            ));
            let b = quantize(&floats, GgmlType::F16).map_err(|e| GgufError::Gguf(e.to_string()))?;
            return Ok(EncodedTensor {
                gguf_name,
                dtype: GgmlType::F16,
                bytes: b,
                scheme,
                weighted_done: false,
                quantized: false,
                kept_f32: false,
                fell_back_f16: true,
                row_demoted,
                warnings,
            });
        }
    };

    Ok(EncodedTensor {
        gguf_name,
        dtype,
        bytes,
        scheme,
        weighted_done: false,
        quantized: result == "quantized",
        kept_f32: result == "f32",
        fell_back_f16: result == "f16",
        row_demoted,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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

//! ComfyUI's **actual** loader contract, transcribed from its source.
//!
//! # Why this module exists
//!
//! `validator::validate_comfy_quant` checks OUR contract — the reference
//! encoder's (`ctq`/`convert_to_quant`) notion of a well-formed
//! `.comfy_quant` blob. It is deliberately strict: it requires `orig_dtype`,
//! rejects unknown shapes of metadata, and so on.
//!
//! ComfyUI's loader is a *different and looser* contract, and being stricter
//! than the consumer is a bug in both directions: it rejects files ComfyUI
//! loads fine, and it happily passes files ComfyUI cannot load at all. This
//! module encodes the consumer's rules so `validate --comfy` can answer the
//! only question that matters — *will ComfyUI load this?*
//!
//! # Provenance
//!
//! Every rule below cites the ComfyUI revision on this machine
//! (`C:\AI\ComfyUI\ComfyUI`). The load path is
//! `comfy/ops.py::load_comfy_weight` (the shared state-dict body used by both
//! `manual_load` and the fused-expert path).
//!
//! Two facts established by exhaustive search of that tree, recorded here
//! because they are load-bearing and NOT obvious from the blob format:
//!
//! 1. **`__metadata__` is optional.** `comfy/utils.py:156` returns
//!    `header.get("__metadata__", {})`, and `comfy/utils.py:130` skips the
//!    key in the tensor loop. A file with no `__metadata__` loads fine.
//! 2. **`yue2_format` is never read.** `grep -rn yue2_format` over the entire
//!    ComfyUI tree, all file types, returns zero hits — as do `convrot_int8`,
//!    `vae_source`, and `vae_dtype`. The 7 metadata keys on a ComfyUI-produced
//!    file are producer-side provenance, not loader input. Only
//!    `metadata["config"]` is consumed, and only for VAE/DiT configs.

use std::collections::BTreeSet;

use crate::dtype::DType;
use crate::st_io::SafetensorsReader;

/// Format strings present in ComfyUI's `QUANT_ALGOS`
/// (`comfy/quant_ops.py:212-256`).
///
/// The load path does `QUANT_ALGOS[module.quant_format]` with **no `.get()` and
/// no normalization** (`comfy/ops.py:1206`), so a format string outside this
/// set raises a bare `KeyError` at load time — the file does not degrade, it
/// simply fails to open.
///
/// Note `mxfp8` is registered only under `_CK_MXFP8_AVAILABLE`
/// (`comfy/quant_ops.py:230-236`); we accept it and note the conditional
/// rather than pretending it is unconditional.
pub const COMFY_QUANT_ALGOS: [&str; 7] = [
    "float8_e4m3fn",
    "float8_e5m2",
    "nvfp4",
    "mxfp8",
    "int8_tensorwise",
    "convrot_w4a4",
    "asym_w4a8_int8",
];

/// The on-disk storage dtype ComfyUI's `QUANT_ALGOS[fmt]["storage_t"]` expects.
///
/// FP8 formats accept EITHER native `F8E4M3`/U8-on-disk: ComfyUI reads them
/// through a dtype *view* (`comfy/ops.py:1210`: "fp8 dtype views handle both
/// legacy uint8-on-disk and native fp8"), so both wire types are loadable.
/// `storage_t` is `torch.float8_e4m3fn` for both `mxfp8` and `float8_e4m3fn`.
fn expected_storage_dtypes(format: &str) -> &'static [DType] {
    match format {
        "int8_tensorwise" | "convrot_w4a4" | "asym_w4a8_int8" => &[DType::I8],
        "nvfp4" => &[DType::U8],
        // FP8 family: native fp8 or legacy uint8-on-disk, per ops.py:1210.
        "float8_e4m3fn" | "float8_e5m2" | "mxfp8" => &[DType::F8E4M3, DType::U8],
        _ => &[],
    }
}

/// Sibling scale tensors ComfyUI pops for a given format.
///
/// Transcribed from the `if/elif` chain in `comfy/ops.py:1211-1271`. Every one
/// of these raises `ValueError(f"Missing ... for layer {layer_name}")` when
/// absent, so they are hard requirements, not advisories.
fn required_scales(format: &str) -> &'static [&'static str] {
    match format {
        "float8_e4m3fn" | "float8_e5m2" => &["weight_scale"],
        "mxfp8" => &["weight_scale"],
        "nvfp4" => &["weight_scale", "weight_scale_2"],
        "int8_tensorwise" => &["weight_scale"],
        "convrot_w4a4" => &["weight_scale"],
        "asym_w4a8_int8" => &["weight_s_rel", "weight_s_channel"],
        _ => &[],
    }
}

/// The five tensors ComfyUI's YuE2 model probe requires
/// (`comfy/model_detection.py:1182-1186`). Absence means the file cannot be
/// detected as a YuE2 model, so it loads as "unsupported".
///
/// Detection is by TENSOR NAME ONLY — no metadata key participates. The
/// `key_prefix` used there is empty for a diffusion model.
pub const YUE2_PROBE_TENSORS: [&str; 5] = [
    "vae2llm.weight",
    "llm2vae.weight",
    "latent_pos_embed.pe",
    "model.layers.0.self_attn.qkv_proj.weight",
    "time_embedder.mlp.0.weight",
];

/// Result of one contract check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractIssue {
    /// ComfyUI raises on this. The file will not load.
    Fatal(String),
    /// ComfyUI tolerates it, but it is worth knowing.
    Advisory(String),
}

/// Validate one file against ComfyUI's loader contract.
///
/// This deliberately does NOT reuse `validate_comfy_quant`'s stricter
/// reference-encoder rules — the whole point is to model the consumer.
pub fn check_comfy_loadable(path: &std::path::Path) -> Vec<ContractIssue> {
    let mut issues = Vec::new();

    let reader = match SafetensorsReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            issues.push(ContractIssue::Fatal(format!(
                "invalid safetensors header (ComfyUI cannot open the file): {e}"
            )));
            return issues;
        }
    };

    let header = reader.header();
    let names: Vec<String> = header.names().cloned().collect();
    let name_set: BTreeSet<&str> = names.iter().map(|s| s.as_str()).collect();
    let algos: BTreeSet<&str> = COMFY_QUANT_ALGOS.iter().copied().collect();

    // ---- 1. every .comfy_quant blob must parse and be an object ---------- //
    // ComfyUI: `json.loads(layer_conf.numpy().tobytes())` (ops.py:1192).
    let mut blob_formats: BTreeSet<String> = BTreeSet::new();
    for name in names.iter().filter(|n| n.ends_with(".comfy_quant")) {
        let prefix = name.trim_end_matches(".comfy_quant");
        let raw = match reader.tensor_bytes(name) {
            Ok(r) => r,
            Err(e) => {
                issues.push(ContractIssue::Fatal(format!(
                    "{name}: cannot read blob payload: {e}"
                )));
                continue;
            }
        };
        // ComfyUI does exactly this: `json.loads(layer_conf.numpy().tobytes())`
        // (ops.py:1192), then `.get("format")` (ops.py:1197). It does NOT use a
        // typed schema, and in particular it does NOT require `orig_dtype` —
        // `comfy_schema::parse_blob` does, because it models the reference
        // ENCODER's contract, not the consumer's. Using it here would reject
        // ComfyUI's own output, so read the JSON directly.
        let json: serde_json::Value = match serde_json::from_slice(raw) {
            Ok(v) => v,
            Err(e) => {
                issues.push(ContractIssue::Fatal(format!(
                    "{name}: blob is not decodable JSON ({e}); ComfyUI raises \
                     'Unknown quantization format' at ops.py:1203"
                )));
                continue;
            }
        };
        if !json.is_object() {
            issues.push(ContractIssue::Fatal(format!(
                "{name}: blob is a JSON {} not an object; ComfyUI's \
                 layer_conf.get(\"format\") would yield None -> ValueError \
                 (ops.py:1203).",
                match json {
                    serde_json::Value::Array(_) => "array",
                    serde_json::Value::Null => "null",
                    _ => "scalar",
                }
            )));
            continue;
        }
        {
            // ---- 2. format must be present and a QUANT_ALGOS key -------- //
            let f = match json.get("format").and_then(|v| v.as_str()) {
                Some(f) => f.to_string(),
                None => {
                    issues.push(ContractIssue::Fatal(format!(
                        "{name}: no string \"format\" key; ComfyUI raises \
                         'Unknown quantization format for layer' (ops.py:1203)."
                    )));
                    continue;
                }
            };
            if !algos.contains(f.as_str()) {
                issues.push(ContractIssue::Fatal(format!(
                    "{name}: format {f:?} is not a key in ComfyUI's \
                     QUANT_ALGOS (comfy/quant_ops.py:212-256). The load \
                     path does QUANT_ALGOS[format] with no .get() \
                     (ops.py:1206), so this raises KeyError and the file \
                     will NOT load. Known keys: {:?}.",
                    COMFY_QUANT_ALGOS
                )));
            } else {
                blob_formats.insert(f.clone());

                // ---- 3. required sibling scales ----------------------- //
                for suffix in required_scales(&f) {
                    let key = format!("{prefix}.{suffix}");
                    if !name_set.contains(key.as_str()) {
                        issues.push(ContractIssue::Fatal(format!(
                            "{name}: format {f:?} requires sibling \
                             {key}, which ComfyUI pops unconditionally \
                             (ops.py:1211-1271) and raises ValueError \
                             when it is missing."
                        )));
                    }
                }

                // ---- 4. convrot_groupsize must be an int -------------- //
                // ComfyUI: `int(layer_conf.get("convrot_groupsize",
                // params_conf.get("convrot_groupsize", 256)))`
                // (ops.py:1234-1236) — a non-integer here raises.
                if let Some(gs) = json.get("convrot_groupsize") {
                    if gs.as_i64().is_none() {
                        issues.push(ContractIssue::Fatal(format!(
                            "{name}: convrot_groupsize {gs} is not an \
                             integer; ComfyUI calls int() on it \
                             (ops.py:1234) and raises."
                        )));
                    }
                }

                // ---- 5. payload dtype vs storage_t -------------------- //
                let want = format!("{prefix}.weight");
                if let Some(info) = header.get(&want) {
                    let ok = expected_storage_dtypes(&f).contains(&info.dtype);
                    if !ok {
                        issues.push(ContractIssue::Fatal(format!(
                            "{want}: dtype {:?} cannot back a {f:?} \
                             quantized weight; ComfyUI wraps it in a \
                             QuantizedTensor typed by \
                             QUANT_ALGOS[format]['storage_t'] (ops.py:1277).",
                            info.dtype
                        )));
                    }
                } else {
                    issues.push(ContractIssue::Fatal(format!(
                        "{prefix}: has a .comfy_quant but no .weight \
                         tensor; ComfyUI returns early with weight=None \
                         and logs 'Missing weight for layer' (ops.py:1173-1177)."
                    )));
                }
            }
        }
    }

    // ---- 6. a YuE2 diffusion model needs all five probe tensors ---------- //
    // Only meaningful when the file actually looks like a YuE2 model; if none
    // of the probes are present it is simply some other architecture, which is
    // not a contract violation. Report the near-miss case.
    let probe_hits = YUE2_PROBE_TENSORS
        .iter()
        .copied()
        .filter(|t| name_set.contains(t) || names.iter().any(|n| n.ends_with(t)))
        .count();
    if probe_hits > 0 && probe_hits < YUE2_PROBE_TENSORS.len() {
        let missing: Vec<&str> = YUE2_PROBE_TENSORS
            .iter()
            .copied()
            .filter(|t| !name_set.contains(t) && !names.iter().any(|n| n.ends_with(t)))
            .collect();
        issues.push(ContractIssue::Advisory(format!(
            "looks like a YuE2 model but {}/5 detection probes are present \
             (comfy/model_detection.py:1182-1186). Missing: {:?}. ComfyUI \
             keys detection on tensor names only — no metadata key \
             participates — so an incomplete set will not be detected as YuE2.",
            probe_hits, missing
        )));
    }

    // ---- 7. __metadata__ is NOT required ------------------------------- //
    // Deliberately no check. `header.get("__metadata__", {})` (utils.py:156)
    // makes it optional, and `yue2_format` is never read by any ComfyUI
    // version we can find. Asserting its absence would be inventing a
    // requirement the consumer does not have.
    if !blob_formats.is_empty() {
        issues.push(ContractIssue::Advisory(format!(
            "quantized formats present: {:?} — checked against ComfyUI's \
             loader contract (ops.py:1190-1279). This is static contract \
             conformance, NOT proof of load: only a real ComfyUI load settles \
             whether the runtime applies the convrot inverse rotation.",
            blob_formats
        )));
    }

    issues
}

/// True when no [`ContractIssue::Fatal`] was found.
pub fn is_comfy_loadable(issues: &[ContractIssue]) -> bool {
    !issues.iter().any(|i| matches!(i, ContractIssue::Fatal(_)))
}

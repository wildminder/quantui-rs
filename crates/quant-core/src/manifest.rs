//! QuantConfig + resumable-run manifest (plan Phase 5, steps 4.1/4.2).
//!
//! Port of reference `tensor_quant.py::QuantConfig` (config-relevant fields +
//! `config_hash`) and `stream_quant.py::_StreamState` manifest persistence:
//! - config_hash: sha256 of `json.dumps(payload, sort_keys=True)` [:16]
//! - manifest JSON: `{version, config_hash, order, done}` with compact separators
//! - save = write `.tmp` then atomic rename

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Scaling mode for the streaming quantizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingMode {
    Tensor,
    Row,
    Block,
}

impl ScalingMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ScalingMode::Tensor => "tensor",
            ScalingMode::Row => "row",
            ScalingMode::Block => "block",
        }
    }
}

/// Calibration RNG draw order for the bias-correction cache (plan §3.4).
///
/// INT8/FP8 draw over ALL 2D `.weight` keys in FILE order (mirrors ctq
/// `convert_to_fp8_scaled`); MXFP8/NVFP4 draw over `sorted(2D .weight keys)`
/// only (mirrors the dedicated ctq format modules).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibOrder {
    FileOrderAll2D,
    SortedWeightsOnly,
}

/// Target quantization format family (plan Phase A.1). INT8 is the shipped
/// streaming path; FP8/MXFP8/NVFP4 kernels exist and are being wired in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Format {
    #[default]
    Int8,
    Fp8E4m3,
    Mxfp8,
    Nvfp4,
}

impl Format {
    /// The `target_format` string carried in the config-hash payload.
    pub fn as_str(&self) -> &'static str {
        match self {
            Format::Int8 => "int8",
            Format::Fp8E4m3 => "fp8",
            Format::Mxfp8 => "mxfp8",
            Format::Nvfp4 => "nvfp4",
        }
    }

    /// Block size used by the skip-inefficient heuristic predicate. INT8/FP8
    /// use the config's (user-selectable) block size; MXFP8/NVFP4 have fixed
    /// format block sizes (32 / 16).
    pub fn heur_block_size(&self, config_block_size: u32) -> u32 {
        match self {
            Format::Int8 | Format::Fp8E4m3 => config_block_size,
            Format::Mxfp8 => 32,
            Format::Nvfp4 => 16,
        }
    }

    /// Calibration RNG draw order for this format family (plan §3.4).
    pub fn calib_order(&self) -> CalibOrder {
        match self {
            Format::Int8 | Format::Fp8E4m3 => CalibOrder::FileOrderAll2D,
            Format::Mxfp8 | Format::Nvfp4 => CalibOrder::SortedWeightsOnly,
        }
    }

    /// Whether this format's output carries file-level `__metadata__`
    /// (`_quantization_metadata`). Only MXFP8/NVFP4 (plan §3.3).
    pub fn carries_file_metadata(&self) -> bool {
        matches!(self, Format::Mxfp8 | Format::Nvfp4)
    }

    /// Fixed group size for the comfy_quant blob, if the format has one.
    /// INT8/FP8 group size is mode-dependent (handled at emission); MXFP8=32,
    /// NVFP4=16.
    pub fn fixed_group_size(&self) -> Option<u32> {
        match self {
            Format::Int8 | Format::Fp8E4m3 => None,
            Format::Mxfp8 => Some(32),
            Format::Nvfp4 => Some(16),
        }
    }
}

/// Streaming-quantization configuration — mirrors `QuantConfig`'s hash-relevant
/// fields exactly. Field names in [`QuantConfig::config_hash`] must match the
/// Python dict keys character-for-character.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Format family (plan Phase A.1). Drives kernel routing, calibration
    /// draw order, and the config-hash payload. Defaults to INT8 — the
    /// shipped streaming path.
    pub format: Format,
    pub target_format: String, // "int8"
    pub int8: bool,            // true
    pub scaling_mode: ScalingMode,
    pub block_size: u32,           // 128
    pub no_learned_rounding: bool, // --simple, true
    pub convrot: bool,             // false
    pub convrot_group_size: u32,   // 256
    /// Output dtype for skipped-weight casting: "bfloat16" | "float16".
    pub orig_dtype: String,
    pub skip_inefficient: bool, // --heur
    /// Pinned calibration seed (parity contract with the whole-file baseline).
    pub calib_seed: i64,
    /// Optional exclude-layers regex; None disables matching.
    pub exclude_layers: Option<String>,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            format: Format::Int8,
            target_format: "int8".into(),
            int8: true,
            scaling_mode: ScalingMode::Block,
            block_size: 128,
            no_learned_rounding: true,
            convrot: false,
            convrot_group_size: 256,
            orig_dtype: "bfloat16".into(),
            skip_inefficient: true,
            calib_seed: 233983427,
            exclude_layers: None,
        }
    }
}

impl QuantConfig {
    /// sha256(json.dumps(payload, sort_keys=True))[:16] — byte-parity port.
    ///
    /// Python `json.dumps` default separators are `", "` / `": "`; booleans are
    /// lowercase; ints plain decimal. We hand-build that exact string to avoid
    /// serde_json formatting drift.
    pub fn config_hash(&self) -> String {
        let payload = format!(
            concat!(
                r#"{{"block_size": {}, "calib_seed": {}, "convrot": {}, "#,
                r#""convrot_group_size": {}, "int8": {}, "no_learned_rounding": {}, "#,
                r#""scaling_mode": "{}", "skip_inefficient": {}, "target_format": "{}"}}"#
            ),
            self.block_size,
            self.calib_seed,
            self.convrot,
            self.convrot_group_size,
            self.int8,
            self.no_learned_rounding,
            self.scaling_mode.as_str(),
            self.skip_inefficient,
            self.target_format,
        );
        let digest = Sha256::digest(payload.as_bytes());
        hex(&digest)[..16].to_string()
    }

    /// Regex layer-exclusion mirroring `QuantConfig.excluded`: search semantics,
    /// invalid pattern → never exclude.
    pub fn excluded(&self, name: &str) -> bool {
        let Some(pattern) = &self.exclude_layers else {
            return false;
        };
        regex::Regex::new(pattern)
            .map(|re| re.is_match(name))
            .unwrap_or(false)
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// --------------------------------------------------------------------------- //
// Manifest persistence (_StreamState port)
// --------------------------------------------------------------------------- //

pub const MANIFEST_VERSION: u64 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub struct ManifestPayload {
    pub version: u64,
    pub config_hash: String,
    #[serde(default)]
    pub order: Vec<String>,
    #[serde(default)]
    pub done: Vec<String>,
}

/// Incremental progress bookkeeping for one streaming run.
pub struct StreamState {
    pub manifest_path: PathBuf,
    pub config_hash: String,
    pub order: Vec<String>,
    pub done: std::collections::HashSet<String>,
}

impl StreamState {
    pub fn new(output_path: impl AsRef<Path>, config_hash: String) -> Self {
        let manifest_path = output_path.as_ref().to_path_buf();
        let mut s = manifest_path.into_os_string();
        s.push(".quant-manifest.json");
        Self {
            manifest_path: PathBuf::from(s),
            config_hash,
            order: Vec::new(),
            done: std::collections::HashSet::new(),
        }
    }

    /// Load an existing manifest if it matches this run's config hash AND both
    /// the manifest and partial output exist. Returns `true` when resumed.
    /// Any parse failure or hash mismatch → clean restart (`false`).
    pub fn load_manifest(&mut self, output_path: &Path) -> bool {
        if !output_path.exists() || !self.manifest_path.exists() {
            return false;
        }
        let Ok(text) = std::fs::read_to_string(&self.manifest_path) else {
            return false;
        };
        let Ok(data) = serde_json::from_str::<ManifestPayload>(&text) else {
            return false;
        };
        if data.config_hash != self.config_hash {
            // Different config → do not trust the partial file; restart clean.
            return false;
        }
        self.order = data.order;
        self.done = data.done.into_iter().collect();
        true
    }

    /// Atomic save: compact JSON to `.tmp`, then rename over the manifest.
    pub fn save_manifest(&self) -> std::io::Result<()> {
        let payload = ManifestPayload {
            version: MANIFEST_VERSION,
            config_hash: self.config_hash.clone(),
            order: self.order.clone(),
            done: {
                let mut v: Vec<String> = self.done.iter().cloned().collect();
                v.sort();
                v
            },
        };
        // Compact separators (serde_json default).
        let json = serde_json::to_vec(&payload).expect("manifest serialization cannot fail");
        let tmp = {
            let mut s = self.manifest_path.clone().into_os_string();
            s.push(".tmp");
            PathBuf::from(s)
        };
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, &self.manifest_path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_enum_properties() {
        // as_str = the target_format strings (plan §3.2).
        assert_eq!(Format::Int8.as_str(), "int8");
        assert_eq!(Format::Fp8E4m3.as_str(), "fp8");
        assert_eq!(Format::Mxfp8.as_str(), "mxfp8");
        assert_eq!(Format::Nvfp4.as_str(), "nvfp4");

        // heur block sizes: INT8/FP8 follow the config block size;
        // MXFP8=32, NVFP4=16 (plan §3.2 / ground truth #5).
        assert_eq!(Format::Int8.heur_block_size(128), 128);
        assert_eq!(Format::Int8.heur_block_size(64), 64);
        assert_eq!(Format::Fp8E4m3.heur_block_size(128), 128);
        assert_eq!(Format::Fp8E4m3.heur_block_size(256), 256);
        assert_eq!(Format::Mxfp8.heur_block_size(128), 32);
        assert_eq!(Format::Nvfp4.heur_block_size(128), 16);

        // Calibration draw order per format family (plan §3.4).
        assert_eq!(Format::Int8.calib_order(), CalibOrder::FileOrderAll2D);
        assert_eq!(Format::Fp8E4m3.calib_order(), CalibOrder::FileOrderAll2D);
        assert_eq!(Format::Mxfp8.calib_order(), CalibOrder::SortedWeightsOnly);
        assert_eq!(Format::Nvfp4.calib_order(), CalibOrder::SortedWeightsOnly);

        // File-level metadata only for MXFP8/NVFP4 (plan §3.3).
        assert!(!Format::Int8.carries_file_metadata());
        assert!(!Format::Fp8E4m3.carries_file_metadata());
        assert!(Format::Mxfp8.carries_file_metadata());
        assert!(Format::Nvfp4.carries_file_metadata());

        // Fixed group sizes (plan §3.2).
        assert_eq!(Format::Int8.fixed_group_size(), None);
        assert_eq!(Format::Fp8E4m3.fixed_group_size(), None);
        assert_eq!(Format::Mxfp8.fixed_group_size(), Some(32));
        assert_eq!(Format::Nvfp4.fixed_group_size(), Some(16));

        // Default format is INT8 (old configs / manifests stay INT8).
        assert_eq!(Format::default(), Format::Int8);
        assert_eq!(QuantConfig::default().format, Format::Int8);
    }

    #[test]
    fn config_hash_matches_python_reference() {
        // Vector captured from docs/ref QuantConfig defaults via venv python:
        //   QuantConfig().config_hash() == "56920c6553cfa241"
        let c = QuantConfig::default();
        assert_eq!(c.config_hash(), "56920c6553cfa241");
    }

    #[test]
    fn config_hash_changes_with_relevant_fields() {
        let mut c = QuantConfig::default();
        let base = c.config_hash();
        c.block_size = 64;
        assert_ne!(c.config_hash(), base);
        c.block_size = 128;
        c.scaling_mode = ScalingMode::Row;
        assert_ne!(c.config_hash(), base);
    }

    #[test]
    fn manifest_roundtrip_and_hash_guard() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("model.safetensors");

        let mut st = StreamState::new(&out, "deadbeefdeadbeef".into());
        assert!(!st.load_manifest(&out), "no files yet → no resume");

        // Simulate a partial run.
        std::fs::write(&out, b"partial").unwrap();
        st.order = ["a.weight", "a.bias"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        st.done.insert("a.weight".to_string());
        st.save_manifest().unwrap();

        // Resume path: same hash → resumed with same state.
        let mut st2 = StreamState::new(&out, "deadbeefdeadbeef".into());
        assert!(st2.load_manifest(&out));
        assert_eq!(st2.order.len(), 2);
        assert!(st2.done.contains("a.weight"));
        assert!(!st2.done.contains("a.bias"));

        // Hash mismatch → clean restart signal.
        let mut st3 = StreamState::new(&out, "ffffffffffffffff".into());
        assert!(!st3.load_manifest(&out));

        // Corrupt manifest → clean restart.
        std::fs::write(st2.manifest_path.clone(), b"{broken").unwrap();
        let mut st4 = StreamState::new(&out, "deadbeefdeadbeef".into());
        assert!(!st4.load_manifest(&out));
    }

    #[test]
    fn truncated_tmp_is_never_loaded() {
        // A leftover .tmp file (crash between write and rename) is ignored.
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("m.safetensors");
        std::fs::write(&out, b"x").unwrap();
        let state = StreamState::new(&out, "aa".repeat(8));
        let tmp_path = {
            let mut s = state.manifest_path.clone().into_os_string();
            s.push(".tmp");
            PathBuf::from(s)
        };
        std::fs::write(
            &tmp_path,
            b"{\"version\":1,\"config_hash\":\"aabbccdd00112233\"}",
        )
        .unwrap();
        let mut st = StreamState::new(&out, "aabbccdd00112233".into());
        assert!(!st.load_manifest(&out));
    }

    #[test]
    fn excluded_regex_semantics() {
        let mut c = QuantConfig::default();
        c.exclude_layers = Some("attn_norm|text_embed".into());
        assert!(c.excluded("blocks.0.attn_norm.weight"));
        assert!(!c.excluded("blocks.1.ff.weight"));
        c.exclude_layers = Some("[invalid".into());
        assert!(
            !c.excluded("attn_norm.weight"),
            "invalid regex never excludes"
        );
        c.exclude_layers = None;
        assert!(!c.excluded("anything"));
    }

    // tempfile is only needed by tests here but lives in dev-deps of the crate.
    use tempfile;
}

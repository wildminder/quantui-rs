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

use crate::quality::Quality;

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

    /// Calibration RNG draw order for this format family (plan §3.4, revised
    /// by the E.5 discriminating fixture).
    ///
    /// The split is NOT "INT8+FP8 vs MXFP8+NVFP4" — it is **which reference
    /// implementation owns the format's bias correction**:
    ///
    /// * `Int8` — bias correction comes from the quantui reference streaming
    ///   path (the reference stream_quant.py's `_build_torch_calibration_cache`),
    ///   which reads the header RAW (`read_safetensors_header`) and walks
    ///   `names` in **on-disk file order**.
    /// * `Fp8E4m3` / `Mxfp8` / `Nvfp4` — bias correction comes from ctq
    ///   (`formats/{fp8,mxfp8,nvfp4}_conversion.py`). Every ctq loader builds
    ///   its key list from `safetensors.safe_open(...).keys()`, which returns
    ///   keys **sorted alphabetically** (verified deterministic across files
    ///   and repeated opens — `tools/probe_safe_open_keys.py`), NOT in file
    ///   order. FP8 then iterates that sorted `all_keys`; MXFP8/NVFP4
    ///   pre-filter and `sorted()` it. Same result: **sorted order**.
    ///
    /// The two orders only differ when a file's 2D `.weight` tensors are NOT
    /// in alphabetical on-disk order — which `safetensors.torch.save_file`
    /// usually hides (it groups by dtype and sorts within each group) but is
    /// perfectly legal and occurs in real checkpoints. `tests/golden/
    /// sharded_unsorted` is the fixture that pins this down: it fails under
    /// the wrong assignment for FP8.
    ///
    /// NOTE: ctq's FP8 in `low_memory=True` mode builds `_all_keys` from the
    /// raw header (file order) instead. The goldens — and this port — use the
    /// default `low_memory=False` (sorted) path.
    pub fn calib_order(&self) -> CalibOrder {
        match self {
            Format::Int8 => CalibOrder::FileOrderAll2D,
            Format::Fp8E4m3 | Format::Mxfp8 | Format::Nvfp4 => CalibOrder::SortedWeightsOnly,
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
    /// Opt-in quality refinement (Tier 2). `Exact` — the default — keeps this
    /// format byte-exact with the references; every other variant changes
    /// output bytes on purpose and is only reachable through an explicit
    /// opt-in format id. See [`crate::quality`].
    pub quality: Quality,
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
            quality: Quality::Exact,
        }
    }
}

impl QuantConfig {
    /// sha256(json.dumps(payload, sort_keys=True))[:16] — byte-parity port.
    ///
    /// Python `json.dumps` default separators are `", "` / `": "`; booleans are
    /// lowercase; ints plain decimal. We hand-build that exact string to avoid
    /// serde_json formatting drift.
    ///
    /// Per-format payload values (plan §3.2): the same 9 keys for every
    /// format; `target_format`/`int8`/`scaling_mode`/`block_size` vary. The
    /// INT8 branch reads the raw fields exactly as before — the captured
    /// Python vector `56920c6553cfa241` must never move. Non-INT8 payloads
    /// have no external reference vector (the reference only ever ran INT8);
    /// they only need to be deterministic, distinct per effective config, and
    /// collision-free vs INT8 (guaranteed by distinct `target_format`).
    ///
    /// # Tier 2: the `quality_tuning` key is CONDITIONAL
    ///
    /// A 10th key, `quality_tuning`, is appended **only when
    /// [`Quality::id()`] is `Some`** — i.e. never for
    /// [`Quality::Exact`]. All four committed vectors were produced by the
    /// 9-key payload, so an unconditional key would silently break every one
    /// of them, including the Python-captured INT8 one. Do not "simplify" this
    /// into an unconditional field.
    ///
    /// `sort_keys=True` ordering puts `quality_tuning` after
    /// `no_learned_rounding` and before `scaling_mode`. That placement is
    /// load-bearing for the digest, and is asserted by
    /// `quality_key_sorts_between_no_learned_rounding_and_scaling_mode`.
    ///
    /// This key is also the **only** resume-safety guard: `load_manifest`
    /// trusts a partial output if and only if the hash matches, so two
    /// quality variants sharing a hash would let a re-run resume into an
    /// artifact built by a different algorithm.
    pub fn config_hash(&self) -> String {
        let (target_format, int8, scaling_mode, block_size): (&str, bool, &str, u32) =
            match self.format {
                Format::Int8 => (
                    &self.target_format,
                    self.int8,
                    self.scaling_mode.as_str(),
                    self.block_size,
                ),
                Format::Fp8E4m3 => ("fp8", false, self.scaling_mode.as_str(), self.block_size),
                Format::Mxfp8 => ("mxfp8", false, "block", 32),
                Format::Nvfp4 => ("nvfp4", false, "block", 16),
            };
        // `None` for `Quality::Exact` keeps the payload byte-identical to the
        // pre-Tier-2 9-key form. See the doc comment above.
        let quality_suffix = match self.quality.id() {
            None => String::new(),
            Some(id) => format!(r#""quality_tuning": "{id}", "#),
        };
        let payload = format!(
            concat!(
                r#"{{"block_size": {}, "calib_seed": {}, "convrot": {}, "#,
                r#""convrot_group_size": {}, "int8": {}, "no_learned_rounding": {}, "#,
                "{}",
                r#""scaling_mode": "{}", "skip_inefficient": {}, "target_format": "{}"}}"#
            ),
            block_size,
            self.calib_seed,
            self.convrot,
            self.convrot_group_size,
            int8,
            self.no_learned_rounding,
            quality_suffix,
            scaling_mode,
            self.skip_inefficient,
            target_format,
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

    /// Compile `--exclude-layers` ONCE and report whether it is usable.
    ///
    /// `excluded` swallows a compile error with `unwrap_or(false)`, so an
    /// invalid pattern silently excludes NOTHING and the run emits a full
    /// quantized file with no diagnostic. That behaviour is deliberate and
    /// pinned by `excluded_regex_semantics` (the Python original is not on
    /// this box, so the oracle cannot be re-checked — do not quietly change
    /// the semantics). This accessor exists so the CLI can WARN instead of
    /// leaving the user to discover the mistake from the file size.
    ///
    /// Returns the compile error, if any. `Ok(())` when no pattern is set.
    pub fn exclude_layers_status(&self) -> Result<(), regex::Error> {
        match &self.exclude_layers {
            None => Ok(()),
            Some(p) => regex::Regex::new(p).map(|_| ()),
        }
    }

    /// How many tensors the exclusion actually removes, for the CLI summary.
    /// Counting is done against the real names so a no-op pattern is visible
    /// as `0` rather than silently producing an oversized artifact.
    pub fn count_excluded<'a>(&self, names: impl Iterator<Item = &'a str>) -> usize {
        names.filter(|n| self.excluded(n)).count()
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
    use sha2::{Digest, Sha256};

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

        // Calibration draw order: INT8 follows the quantui reference streaming
        // path (raw header -> file order); every ctq-owned format (FP8/MXFP8/
        // NVFP4) goes through `safe_open.keys()`, which is SORTED (E.5).
        assert_eq!(Format::Int8.calib_order(), CalibOrder::FileOrderAll2D);
        assert_eq!(Format::Fp8E4m3.calib_order(), CalibOrder::SortedWeightsOnly);
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
        // Vector captured from the reference QuantConfig defaults:
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

    /// Captured deterministic vectors for the non-INT8 formats (plan A.2).
    /// The payload strings are hand-built `json.dumps(sort_keys=True)`
    /// mirrors; these vectors pin them against accidental drift. The INT8
    /// vector above is the only externally-referenced one (Python ref);
    /// these are internal stability pins.
    #[test]
    fn config_hash_fp8_vector() {
        let mut c = QuantConfig::default();
        c.format = Format::Fp8E4m3;
        c.target_format = "fp8".into();
        c.int8 = false;
        assert_eq!(c.config_hash(), "5f14780b1bcf30f2");
        // Scaling mode is hash-relevant for FP8 (tensor/row/block).
        c.scaling_mode = ScalingMode::Row;
        assert_ne!(c.config_hash(), "5f14780b1bcf30f2");
    }

    #[test]
    fn config_hash_mxfp8_vector() {
        let mut c = QuantConfig::default();
        c.format = Format::Mxfp8;
        c.target_format = "mxfp8".into();
        c.int8 = false;
        // Fixed block_size=32 / scaling_mode=block regardless of config fields.
        assert_eq!(c.config_hash(), "cff3b89365c9544d");
        c.block_size = 64; // ignored for the hash payload
        c.scaling_mode = ScalingMode::Row; // ignored for the hash payload
        assert_eq!(c.config_hash(), "cff3b89365c9544d");
    }

    #[test]
    fn config_hash_nvfp4_vector() {
        let mut c = QuantConfig::default();
        c.format = Format::Nvfp4;
        c.target_format = "nvfp4".into();
        c.int8 = false;
        assert_eq!(c.config_hash(), "95ede677cf402b53");
        c.block_size = 64; // ignored for the hash payload
        assert_eq!(c.config_hash(), "95ede677cf402b53");
    }

    #[test]
    fn config_hash_pairwise_distinct_across_formats() {
        let mut c = QuantConfig::default();
        let int8 = c.config_hash();
        c.format = Format::Fp8E4m3;
        let fp8 = c.config_hash();
        c.format = Format::Mxfp8;
        let mxfp8 = c.config_hash();
        c.format = Format::Nvfp4;
        let nvfp4 = c.config_hash();
        let all = [int8, fp8, mxfp8, nvfp4];
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(all[i], all[j], "formats {i} and {j} collide");
            }
        }
    }

    // ----------------------------------------------------------------- //
    // Tier 2: the conditional `quality_tuning` key
    // ----------------------------------------------------------------- //

    /// Parity preservation — the most important test in this step.
    ///
    /// `Quality::default()` is `Exact`, and `Exact.id()` is `None`, so the
    /// default payload is the original 9-key string byte-for-byte. All four
    /// committed vectors must therefore reproduce exactly. **Do not edit these
    /// expected values to make a change pass** — the INT8 one is captured from
    /// the Python reference and is not ours to move.
    #[test]
    fn default_quality_is_exact_and_preserves_every_pinned_hash() {
        assert_eq!(Quality::default(), Quality::Exact);

        let mut c = QuantConfig::default();
        assert_eq!(
            c.config_hash(),
            "56920c6553cfa241",
            "INT8 (Python-captured)"
        );

        c.format = Format::Fp8E4m3;
        c.target_format = "fp8".into();
        c.int8 = false;
        assert_eq!(c.config_hash(), "5f14780b1bcf30f2", "FP8 block");

        c.format = Format::Mxfp8;
        c.target_format = "mxfp8".into();
        assert_eq!(c.config_hash(), "cff3b89365c9544d", "MXFP8");

        c.format = Format::Nvfp4;
        c.target_format = "nvfp4".into();
        assert_eq!(c.config_hash(), "95ede677cf402b53", "NVFP4");
    }

    /// Resume safety. `load_manifest` trusts a partial output if and only if
    /// the hash matches, and nothing else. Without distinct hashes per quality
    /// variant, a plain NVFP4 run that is interrupted and re-run with
    /// `nvfp4_l2` would resume into the byte-exact partial file and mix two
    /// algorithms in one artifact — silently, with no error.
    ///
    /// # The expected digests here are SORTED-order, and that is deliberate
    ///
    /// The Tier 2 plan predicted `ab2267a626535821` / `4bb443c5e08167c6` /
    /// `f0cb29848982c322` for these three variants. Those values are **wrong**:
    /// they were computed with `quality_tuning` appended *last*, whereas the
    /// payload is a `json.dumps(sort_keys=True)` mirror — the same rule under
    /// which all four committed vectors above reproduce exactly. Sorted order
    /// places the key between `no_learned_rounding` and `scaling_mode`.
    ///
    /// The plan's own text got this right and its own arithmetic did not: it
    /// warned "verify the emitted string, do not assume", then verified against
    /// the wrong string. Pairwise distinctness — the property this test exists
    /// to protect — holds under either ordering, so the bug was invisible to
    /// the property it was checking.
    #[test]
    fn quality_variants_get_pairwise_distinct_hashes() {
        let mut c = QuantConfig::default();
        c.format = Format::Nvfp4;
        c.target_format = "nvfp4".into();
        c.int8 = false;

        let base = c.config_hash();
        assert_eq!(base, "95ede677cf402b53");

        c.quality = Quality::Nvfp4L2ScaleSearch;
        let l2 = c.config_hash();
        assert_eq!(l2, "85cf59c986e5d1a4");
        assert_ne!(
            base, l2,
            "nvfp4_l2 must not resume into a plain nvfp4 partial"
        );

        c.quality = Quality::Nvfp4HessianScaleSearch;
        let hess = c.config_hash();
        assert_eq!(hess, "47437e5e8343725b");
        assert_ne!(l2, hess);

        let mut m = QuantConfig::default();
        m.format = Format::Mxfp8;
        m.target_format = "mxfp8".into();
        m.int8 = false;
        m.quality = Quality::Mxfp8E8m0Compensated;
        assert_eq!(m.config_hash(), "ee7269a90020e9b5");
    }

    /// `sort_keys=True` places `quality_tuning` after `no_learned_rounding`
    /// and before `scaling_mode`. The digest depends on that placement, so
    /// assert the emitted string rather than trusting the ordering.
    #[test]
    fn quality_key_sorts_between_no_learned_rounding_and_scaling_mode() {
        let mut c = QuantConfig::default();
        c.quality = Quality::Nvfp4L2ScaleSearch;
        // Mirror of the format! payload with the suffix spliced in, so a
        // reordering of the format string is caught rather than silently
        // changing every quality digest.
        let expected = concat!(
            r#"{"block_size": 128, "calib_seed": 233983427, "convrot": false, "#,
            r#""convrot_group_size": 256, "int8": true, "no_learned_rounding": true, "#,
            r#""quality_tuning": "nvfp4_l2", "#,
            r#""scaling_mode": "block", "skip_inefficient": true, "target_format": "int8"}"#,
        );
        assert_eq!(
            c.config_hash(),
            super::hex(&Sha256::digest(expected.as_bytes()))[..16]
        );
    }

    /// The converse of the fix, pinned: an *unconditional* key would move the
    /// Python-captured INT8 vector. Shown here so a future "cleanup" cannot
    /// reintroduce it believing the test suite was merely noisy.
    #[test]
    fn unconditional_quality_key_would_move_the_python_vector() {
        // What the payload would hash to if the key were always emitted.
        let with_key = concat!(
            r#"{"block_size": 128, "calib_seed": 233983427, "convrot": false, "#,
            r#""convrot_group_size": 256, "int8": true, "no_learned_rounding": true, "#,
            r#""quality_tuning": "exact", "#,
            r#""scaling_mode": "block", "skip_inefficient": true, "target_format": "int8"}"#,
        );
        let wrong = hex(&Sha256::digest(with_key.as_bytes()))[..16].to_string();
        assert_ne!(
            wrong, "56920c6553cfa241",
            "this control is only meaningful while the unconditional form differs"
        );
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

    /// The `unwrap_or(false)` in `excluded` is a SILENT no-op on a typo, which
    /// is the whole reason `exclude_layers_status` exists. Pin both halves:
    /// the status must reject what `excluded` quietly ignores, and a valid
    /// pattern must report Ok.
    #[test]
    fn exclude_layers_status_catches_what_excluded_swallows() {
        let mut c = QuantConfig::default();
        c.exclude_layers = Some("[invalid".into());
        assert!(
            c.exclude_layers_status().is_err(),
            "an invalid pattern must be REPORTED, not silently ignored"
        );
        // And the swallow itself is unchanged — this test documents the
        // behaviour the CLI warning exists to surface, not to endorse.
        assert!(!c.excluded("attn_norm.weight"));

        c.exclude_layers = Some("attn_norm".into());
        assert!(c.exclude_layers_status().is_ok());
        c.exclude_layers = None;
        assert!(c.exclude_layers_status().is_ok(), "no pattern is always ok");
    }

    /// The pattern that reproduces ComfyUI's own `transformer_blocks`-only
    /// selection on Qwen-Image 2.1, verified against the real tensor names:
    /// it must hit every non-block 2-D weight, and must NOT touch anything
    /// inside `transformer_blocks`.
    #[test]
    fn transformer_blocks_only_exclusion_is_exact() {
        let mut c = QuantConfig::default();
        c.exclude_layers =
            Some(r"^(img_in|modulation|norm_out|proj_out|time_text_embed|txt_in)\.".into());
        // Every non-block 2-D weight in that model.
        for n in [
            "img_in.weight",
            "modulation.1.weight",
            "norm_out.linear.weight",
            "proj_out.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_2.weight",
            "txt_in.in_layer.weight",
            "txt_in.out_layer.weight",
        ] {
            assert!(c.excluded(n), "{n} should be excluded");
        }
        // Nothing inside the blocks may be caught — this is the direction
        // that would silently change 192 tensors if the regex were loose.
        for n in [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.47.img_mlp.gate_up.weight",
            "transformer_blocks.9.attn.to_out.0.weight",
        ] {
            assert!(!c.excluded(n), "{n} must NOT be excluded");
        }
    }

    #[test]
    fn count_excluded_reports_zero_for_a_noop_pattern() {
        let names = [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
        ];
        let mut c = QuantConfig::default();
        c.exclude_layers = Some("[invalid".into());
        assert_eq!(
            c.count_excluded(names.iter().copied()),
            0,
            "a swallowed compile error must be visible as 0, not as a mystery file size"
        );
        c.exclude_layers = Some("attn\\.to_q".into());
        assert_eq!(c.count_excluded(names.iter().copied()), 1);
    }

    // tempfile is only needed by tests here but lives in dev-deps of the crate.
    use tempfile;
}

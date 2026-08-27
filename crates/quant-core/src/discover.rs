//! Input discovery + output auto-naming (plan Phase 6, steps 5.1 + 5.3).
//!
//! Ports the reference classification / sharded-folder logic:
//! - `quant_methods.py::classify_input` + `is_sharded_folder` (single file vs
//!   HuggingFace sharded folder, plus the auto-naming base name)
//! - `worker_ctq.py::discover_shards` (parse `model.safetensors.index.json`,
//!   enumerate shards in first-appearance order + non-weight sidecars)
//! - `stream_quant.py::_resolve_union_header` (merge shard headers into one
//!   union, first-occurrence-wins shard mapping)
//! - `run_config.py::ctq_quant_tags` / `ctq_output_stem` / `output_state` /
//!   `suggest_comfy_output` (the 6-combination auto-naming)
//!
//! All functions are pure path/string logic except `discover_shards` and
//! `resolve_union`, which read the filesystem.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use crate::st_io::header::TensorInfo;
use crate::st_io::reader::SafetensorsReader;
use crate::st_io::Error as StError;

/// HuggingFace sharded-model index filename (`quant_methods.py::INDEX_NAME`).
pub const INDEX_NAME: &str = "model.safetensors.index.json";

/// A bare HuggingFace shard marker (e.g. `model-00001-of-00003`) carries no
/// semantic name of its own; when one is used as a single-file input we fall
/// back to the parent folder name for auto-naming (`_SHARD_NAME_RE`).
const SHARD_NAME_RE: &str = r"^model-\d+-of-\d+$";

/// Input classification result (`classify_input`'s `kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    SingleFile,
    ShardedFolder,
}

// --------------------------------------------------------------------------- //
// Path helpers (lexical, no symlink resolution — mirrors os.path.abspath).
// --------------------------------------------------------------------------- //

/// Lexical normalization: collapse `.` / `..` / duplicate separators.
fn norm_path(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                // pop() returns false at a prefix/root/empty path; nothing to do.
                let _ = out.pop();
            }
            c => out.push(c.as_os_str()),
        }
    }
    out
}

/// Absolute path without symlink resolution (mirrors `os.path.abspath`).
fn abspath(p: &Path) -> PathBuf {
    let base = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().expect("current_dir").join(p)
    };
    norm_path(&base)
}

/// File stem (basename without final extension).
fn stem_of(p: &Path) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Basename of a path (final component; tolerates trailing separators).
fn basename_of(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

// --------------------------------------------------------------------------- //
// Input classification (quant_methods.py).
// --------------------------------------------------------------------------- //

/// True iff `path` is a directory containing `model.safetensors.index.json`.
pub fn is_sharded_folder(path: impl AsRef<Path>) -> bool {
    let p = path.as_ref();
    p.is_dir() && p.join(INDEX_NAME).is_file()
}

/// Classify an input path → `(kind, base_name)`.
///
/// Mirrors `quant_methods.py::classify_input`:
/// - a `.safetensors` file → `SingleFile`, base = its stem, **unless** the stem
///   is a bare HF shard marker (`model-00001-of-00003`) → base = parent folder;
/// - a folder with `model.safetensors.index.json` → `ShardedFolder`, base = folder;
/// - a folder holding exactly one `.safetensors` → `SingleFile`, base = folder;
/// - anything else → `(None, None)`.
pub fn classify_input(path: impl AsRef<Path>) -> (Option<InputKind>, Option<String>) {
    let p = path.as_ref();
    let s = p.to_string_lossy();
    if s.is_empty() {
        return (None, None);
    }
    if p.is_file() {
        if !s.ends_with(".safetensors") {
            return (None, None);
        }
        let stem = stem_of(p);
        let re = regex::Regex::new(SHARD_NAME_RE).expect("static regex");
        let base = if re.is_match(&stem) {
            // Shard marker → use the parent folder name (more meaningful).
            basename_of(abspath(p).parent().unwrap_or_else(|| Path::new("")))
        } else {
            stem
        };
        return (Some(InputKind::SingleFile), Some(base));
    }
    if p.is_dir() {
        if p.join(INDEX_NAME).is_file() {
            return (Some(InputKind::ShardedFolder), Some(basename_of(p)));
        }
        let sts: Vec<String> = match std::fs::read_dir(p) {
            Ok(rd) => rd
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_file())
                .filter_map(|e| {
                    let n = e.file_name().to_string_lossy().into_owned();
                    if n.ends_with(".safetensors") {
                        Some(n)
                    } else {
                        None
                    }
                })
                .collect(),
            Err(_) => return (None, None),
        };
        if sts.len() == 1 {
            return (Some(InputKind::SingleFile), Some(basename_of(p)));
        }
        return (None, None);
    }
    (None, None)
}

// --------------------------------------------------------------------------- //
// Sharded-model discovery (worker_ctq.py::discover_shards).
// --------------------------------------------------------------------------- //

/// Resolved HuggingFace sharded model, ready for per-shard quantization.
#[derive(Debug, Clone)]
pub struct ShardedModel {
    /// Absolute path to the HF model folder.
    pub model_dir: PathBuf,
    /// Absolute path to `model.safetensors.index.json`.
    pub index_path: PathBuf,
    /// tensor name → shard filename (relative to `model_dir`).
    pub weight_map: HashMap<String, String>,
    /// Unique, order-preserving shard filenames (relative), in first-appearance
    /// order of `weight_map.values()`.
    pub shard_files: Vec<String>,
    /// Files to copy verbatim (config.json, tokenizer*, ...).
    pub non_weight_files: Vec<String>,
}

impl ShardedModel {
    /// Absolute paths of the shards, in discovery order.
    pub fn shard_paths(&self) -> Vec<PathBuf> {
        self.shard_files
            .iter()
            .map(|s| self.model_dir.join(s))
            .collect()
    }
}

/// Errors from shard discovery.
#[derive(Debug, thiserror::Error)]
pub enum DiscoverError {
    #[error("index json not found: {0}")]
    IndexNotFound(PathBuf),
    #[error("index json {path:?} is not valid JSON: {message}")]
    IndexJson { path: PathBuf, message: String },
    #[error("index json {path:?} has no usable \"weight_map\" object")]
    NoWeightMap { path: PathBuf },
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    St(#[from] StError),
}

/// Parse `model.safetensors.index.json` and enumerate shards + sidecar files.
///
/// Faithful port of `worker_ctq.py::discover_shards`: shard order follows first
/// appearance in `weight_map.values()`; any file in `model_dir` that is not the
/// index json and not a referenced shard is a non-weight file to copy verbatim;
/// orphan `.safetensors` not in the index are ignored.
pub fn discover_shards(model_dir: impl AsRef<Path>) -> Result<ShardedModel, DiscoverError> {
    let model_dir = model_dir.as_ref().to_path_buf();
    let index_path = model_dir.join(INDEX_NAME);
    if !index_path.is_file() {
        return Err(DiscoverError::IndexNotFound(index_path));
    }
    let raw = std::fs::read_to_string(&index_path).map_err(|source| DiscoverError::Io {
        path: index_path.clone(),
        source,
    })?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| DiscoverError::IndexJson {
            path: index_path.clone(),
            message: e.to_string(),
        })?;
    let weight_map_obj = value
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| DiscoverError::NoWeightMap {
            path: index_path.clone(),
        })?;

    let mut weight_map = HashMap::new();
    let mut shard_files: Vec<String> = Vec::new();
    for (tensor, shard) in weight_map_obj {
        let shard_name = shard
            .as_str()
            .ok_or_else(|| DiscoverError::IndexJson {
                path: index_path.clone(),
                message: format!("weight_map[{tensor:?}] is not a string"),
            })?
            .to_string();
        if !shard_files.contains(&shard_name) {
            shard_files.push(shard_name.clone());
        }
        weight_map.insert(tensor.clone(), shard_name);
    }

    // Non-weight sidecars: sorted dir entries that are files, not the index,
    // not a referenced shard, and not an orphan .safetensors.
    let mut entries: Vec<String> = std::fs::read_dir(&model_dir)
        .map_err(|source| DiscoverError::Io {
            path: model_dir.clone(),
            source,
        })?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    entries.sort();
    let mut non_weight_files = Vec::new();
    for entry in entries {
        let full = model_dir.join(&entry);
        if !full.is_file() {
            continue;
        }
        if entry == INDEX_NAME || shard_files.contains(&entry) {
            continue;
        }
        if entry.ends_with(".safetensors") {
            continue; // orphan shard not referenced by the index
        }
        non_weight_files.push(entry);
    }

    Ok(ShardedModel {
        model_dir,
        index_path,
        weight_map,
        shard_files,
        non_weight_files,
    })
}

// --------------------------------------------------------------------------- //
// Union header across shards (stream_quant.py::_resolve_union_header).
// --------------------------------------------------------------------------- //

/// Merged header across several shards for single-output streaming.
///
/// `entries` is the ordered union of tensor descriptors (first-appearance order
/// across shards); `name_to_shard[i]` is the shard index each name lives in
/// (first occurrence wins, mirroring HF weight_map precedence).
pub struct UnionHeader {
    pub entries: Vec<(String, TensorInfo)>,
    pub name_to_shard: HashMap<String, usize>,
}

impl UnionHeader {
    /// Ordered tensor names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(n, _)| n.as_str())
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.entries.iter().find(|(n, _)| n == name).map(|(_, i)| i)
    }
}

/// Merge several shard headers into one union header for streaming.
///
/// NOTE: the reference `_resolve_union_header` captures the first `__metadata__`
/// but `stream_quantize` then overwrites it with `header.get("__metadata__")`
/// (always `None`, since the union never carries `__metadata__`). The net effect
/// — confirmed by probe — is that the streaming output NEVER carries metadata.
/// We therefore do not surface metadata here at all.
pub fn resolve_union(shard_paths: &[PathBuf]) -> Result<UnionHeader, StError> {
    let mut entries: Vec<(String, TensorInfo)> = Vec::new();
    let mut name_to_shard: HashMap<String, usize> = HashMap::new();
    for (idx, sp) in shard_paths.iter().enumerate() {
        let reader = SafetensorsReader::open(sp)?;
        for (name, info) in reader.header().iter() {
            if !name_to_shard.contains_key(name) {
                name_to_shard.insert(name.clone(), idx);
                entries.push((name.clone(), info.clone()));
            }
        }
    }
    Ok(UnionHeader {
        entries,
        name_to_shard,
    })
}

// --------------------------------------------------------------------------- //
// Auto-naming (run_config.py).
// --------------------------------------------------------------------------- //

/// Output-field classification (`run_config.py::output_state`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputState {
    Empty,
    DirPath,
    FilePath,
}

/// Classify the output field: `empty` | `file_path` | `dir_path`.
pub fn output_state(out: &str) -> OutputState {
    if out.is_empty() {
        return OutputState::Empty;
    }
    let p = Path::new(out);
    if p.is_dir() {
        return OutputState::DirPath;
    }
    if out.ends_with('/') || out.ends_with('\\') {
        return OutputState::DirPath;
    }
    match p.extension() {
        Some(e) if !e.is_empty() => OutputState::FilePath,
        _ => OutputState::DirPath,
    }
}

/// Short, filesystem-safe tags describing a ctq configuration
/// (`run_config.py::ctq_quant_tags`). The format id is the primary descriptor;
/// extra tags only capture options that change the emitted artifact.
pub fn ctq_quant_tags(
    fmt: &str,
    convrot_group_size: Option<&str>,
    simple: bool,
    low_memory: bool,
    calib_samples: &str,
    heur: bool,
) -> Vec<String> {
    let mut tags = vec![fmt.to_string()];
    if fmt == "int8_convrot" {
        if let Some(gs) = convrot_group_size {
            if !gs.is_empty() {
                tags.push(format!("gs{gs}"));
            }
        }
    }
    if simple {
        tags.push("simple".to_string());
    }
    if low_memory {
        tags.push("lowmem".to_string());
    }
    if !calib_samples.is_empty() {
        tags.push(format!("calib{calib_samples}"));
    }
    if heur {
        tags.push("heur".to_string());
    }
    tags
}

/// Build a meaningful output filename stem: `<base>-<quant_tags>`.
pub fn ctq_output_stem(base: &str, quant_tags: &[String]) -> String {
    format!("{base}-{}", quant_tags.join("-"))
}

/// ComfyUI/ctq 6-combination auto-naming (`run_config.py::suggest_comfy_output`).
///
/// Returns the suggested output path, or `None` when the field should be left
/// untouched (explicit `.safetensors` file, or a directory destination).
pub fn suggest_comfy_output(
    inp: &str,
    output: &str,
    quant_tags: &[String],
    output_mode: &str,
) -> Option<PathBuf> {
    let inp = inp.trim();
    let output = output.trim();
    if inp.is_empty() {
        return None;
    }
    let (kind, base) = classify_input(inp);
    let base = base?; // unusable input
    let state = output_state(output);
    if state == OutputState::FilePath {
        return None; // Cases B / E: an explicit .safetensors file wins.
    }
    let stem = ctq_output_stem(&base, quant_tags);
    if kind == Some(InputKind::SingleFile) {
        let target_dir = if state == OutputState::Empty {
            abspath(Path::new(inp))
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_default()
        } else {
            PathBuf::from(output)
        };
        return Some(target_dir.join(format!("{stem}.safetensors")));
    }
    // sharded_folder
    if output_mode == "single" {
        let parent = abspath(Path::new(inp))
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        return Some(parent.join(format!("{stem}.safetensors")));
    }
    // sharded (default): output is a directory.
    if state == OutputState::Empty {
        let parent = abspath(Path::new(inp))
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        return Some(parent.join(&stem));
    }
    // dir_path (Case F): the directory IS the destination; leave unchanged.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn write(p: &Path, content: &[u8]) {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(p, content).unwrap();
    }

    // ---------------- classify_input ---------------- //

    #[test]
    fn classify_single_file() {
        let d = tmpdir();
        let f = d.path().join("mymodel.safetensors");
        write(&f, b"x");
        assert_eq!(
            classify_input(&f),
            (Some(InputKind::SingleFile), Some("mymodel".into()))
        );
    }

    #[test]
    fn classify_shard_marker_uses_parent_folder() {
        let d = tmpdir();
        let model_dir = d.path().join("some_model");
        let f = model_dir.join("model-00001-of-00003.safetensors");
        write(&f, b"x");
        assert_eq!(
            classify_input(&f),
            (Some(InputKind::SingleFile), Some("some_model".into()))
        );
    }

    #[test]
    fn classify_sharded_folder() {
        let d = tmpdir();
        let dir = d.path().join("sharded_model");
        write(&dir.join(INDEX_NAME), b"{}");
        assert_eq!(
            classify_input(&dir),
            (Some(InputKind::ShardedFolder), Some("sharded_model".into()))
        );
        assert!(is_sharded_folder(&dir));
    }

    #[test]
    fn classify_folder_with_one_safetensors() {
        let d = tmpdir();
        let dir = d.path().join("onefolder");
        write(&dir.join("only.safetensors"), b"x");
        assert_eq!(
            classify_input(&dir),
            (Some(InputKind::SingleFile), Some("onefolder".into()))
        );
        assert!(!is_sharded_folder(&dir));
    }

    #[test]
    fn classify_unusable() {
        let d = tmpdir();
        // Non-existent file.
        assert_eq!(
            classify_input(d.path().join("nope.safetensors")),
            (None, None)
        );
        // Folder with two safetensors and no index.
        let dir = d.path().join("twofolder");
        write(&dir.join("a.safetensors"), b"x");
        write(&dir.join("b.safetensors"), b"x");
        assert_eq!(classify_input(&dir), (None, None));
        // Non-safetensors file.
        let txt = d.path().join("readme.txt");
        write(&txt, b"x");
        assert_eq!(classify_input(&txt), (None, None));
        // Empty string.
        assert_eq!(classify_input(""), (None, None));
    }

    // ---------------- discover_shards ---------------- //

    fn write_shard(dir: &Path, name: &str, tensors: &[&str]) {
        // Minimal valid safetensors: header with the given tensor names (F32 [1]).
        let mut obj = serde_json::Map::new();
        let mut offset = 0u64;
        let mut data = Vec::new();
        for t in tensors {
            let mut entry = serde_json::Map::new();
            entry.insert("dtype".into(), "F32".into());
            entry.insert("shape".into(), serde_json::json!([1]));
            entry.insert(
                "data_offsets".into(),
                serde_json::json!([offset, offset + 4]),
            );
            obj.insert(t.to_string(), serde_json::Value::Object(entry));
            offset += 4;
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(obj)).unwrap();
        let header = crate::st_io::align_header_to_8(&header);
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&data);
        write(&dir.join(name), &bytes);
    }

    #[test]
    fn discover_shards_order_and_sidecars() {
        let d = tmpdir();
        let dir = d.path().join("model");
        // index references shard-2 first, then shard-1 → first-appearance order.
        let index = serde_json::json!({
            "metadata": {"total_size": 8},
            "weight_map": {
                "b.weight": "model-00002-of-00002.safetensors",
                "a.weight": "model-00001-of-00002.safetensors",
                "c.weight": "model-00002-of-00002.safetensors"
            }
        });
        write(&dir.join(INDEX_NAME), index.to_string().as_bytes());
        write_shard(&dir, "model-00001-of-00002.safetensors", &["a.weight"]);
        write_shard(
            &dir,
            "model-00002-of-00002.safetensors",
            &["b.weight", "c.weight"],
        );
        write(&dir.join("config.json"), b"{}");
        write(&dir.join("tokenizer.json"), b"{}");
        // Orphan shard not in the index → ignored.
        write_shard(&dir, "orphan.safetensors", &["z.weight"]);

        let m = discover_shards(&dir).unwrap();
        assert_eq!(
            m.shard_files,
            vec![
                "model-00002-of-00002.safetensors",
                "model-00001-of-00002.safetensors"
            ]
        );
        assert_eq!(m.weight_map["a.weight"], "model-00001-of-00002.safetensors");
        assert_eq!(m.weight_map["b.weight"], "model-00002-of-00002.safetensors");
        assert_eq!(m.non_weight_files, vec!["config.json", "tokenizer.json"]);
        assert_eq!(m.shard_paths().len(), 2);
    }

    #[test]
    fn discover_shards_missing_index() {
        let d = tmpdir();
        let dir = d.path().join("empty_model");
        std::fs::create_dir_all(&dir).unwrap();
        match discover_shards(&dir) {
            Err(DiscoverError::IndexNotFound(p)) => assert!(p.ends_with(INDEX_NAME)),
            other => panic!("expected IndexNotFound, got {other:?}"),
        }
    }

    #[test]
    fn discover_shards_bad_json() {
        let d = tmpdir();
        let dir = d.path().join("bad");
        write(&dir.join(INDEX_NAME), b"{not json");
        assert!(matches!(
            discover_shards(&dir),
            Err(DiscoverError::IndexJson { .. })
        ));
        // No weight_map.
        write(&dir.join(INDEX_NAME), br#"{"metadata":{}}"#);
        assert!(matches!(
            discover_shards(&dir),
            Err(DiscoverError::NoWeightMap { .. })
        ));
    }

    // ---------------- resolve_union ---------------- //

    #[test]
    fn resolve_union_first_wins_and_order() {
        let d = tmpdir();
        // shard1: dup + a ; shard2: dup (different) + z.
        let p1 = d.path().join("shard1.safetensors");
        let p2 = d.path().join("shard2.safetensors");
        write_shard_at(&p1, &["dup.weight", "a.bias"]);
        write_shard_at(&p2, &["dup.weight", "z.bias"]);

        let union = resolve_union(&[p1.clone(), p2.clone()]).unwrap();
        let names: Vec<&str> = union.names().collect();
        // First-appearance order: shard1's tensors, then shard2's new ones.
        assert_eq!(names, vec!["dup.weight", "a.bias", "z.bias"]);
        assert_eq!(union.name_to_shard["dup.weight"], 0); // first wins
        assert_eq!(union.name_to_shard["a.bias"], 0);
        assert_eq!(union.name_to_shard["z.bias"], 1);
    }

    fn write_shard_at(path: &Path, tensors: &[&str]) {
        let mut obj = serde_json::Map::new();
        let mut offset = 0u64;
        let mut data = Vec::new();
        for t in tensors {
            let mut entry = serde_json::Map::new();
            entry.insert("dtype".into(), "F32".into());
            entry.insert("shape".into(), serde_json::json!([1]));
            entry.insert(
                "data_offsets".into(),
                serde_json::json!([offset, offset + 4]),
            );
            obj.insert(t.to_string(), serde_json::Value::Object(entry));
            offset += 4;
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(obj)).unwrap();
        let header = crate::st_io::align_header_to_8(&header);
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&data);
        write(path, &bytes);
    }

    // ---------------- naming ---------------- //

    #[test]
    fn quant_tags_matrix() {
        // Captured verbatim from reference run_config.ctq_quant_tags.
        assert_eq!(
            ctq_quant_tags("fp8_e4m3", None, false, false, "", false),
            vec!["fp8_e4m3"]
        );
        assert_eq!(
            ctq_quant_tags("int8_block", None, false, false, "", false),
            vec!["int8_block"]
        );
        assert_eq!(
            ctq_quant_tags("int8_block", None, false, false, "", true),
            vec!["int8_block", "heur"]
        );
        assert_eq!(
            ctq_quant_tags("int8_block", None, true, false, "", false),
            vec!["int8_block", "simple"]
        );
        assert_eq!(
            ctq_quant_tags("int8_block", None, true, true, "512", true),
            vec!["int8_block", "simple", "lowmem", "calib512", "heur"]
        );
        assert_eq!(
            ctq_quant_tags("int8_convrot", Some("256"), false, false, "", false),
            vec!["int8_convrot", "gs256"]
        );
        assert_eq!(
            ctq_quant_tags("int8_convrot", Some("128"), true, false, "", false),
            vec!["int8_convrot", "gs128", "simple"]
        );
        assert_eq!(
            ctq_quant_tags("int8_convrot", None, false, false, "", false),
            vec!["int8_convrot"]
        );
    }

    #[test]
    fn output_stem() {
        assert_eq!(
            ctq_output_stem(
                "mymodel",
                &["fp8_e4m3".into(), "simple".into(), "heur".into()]
            ),
            "mymodel-fp8_e4m3-simple-heur"
        );
        assert_eq!(
            ctq_output_stem("base", &["int8_block".into()]),
            "base-int8_block"
        );
    }

    #[test]
    fn output_state_matrix() {
        assert_eq!(output_state(""), OutputState::Empty);
        assert_eq!(output_state("/some/dir/"), OutputState::DirPath);
        assert_eq!(output_state("file.safetensors"), OutputState::FilePath);
        assert_eq!(output_state("bare_stem"), OutputState::DirPath);
        let d = tmpdir();
        assert_eq!(
            output_state(&d.path().to_string_lossy()),
            OutputState::DirPath
        );
    }

    #[test]
    fn suggest_comfy_output_matrix() {
        let d = tmpdir();
        let single = d.path().join("mymodel.safetensors");
        write(&single, b"x");
        let shard_dir = d.path().join("sharded_model");
        write(&shard_dir.join(INDEX_NAME), b"{}");
        let tags: Vec<String> = vec!["fp8_e4m3".into(), "simple".into()];

        // single_file, empty output → sibling .safetensors.
        assert_eq!(
            suggest_comfy_output(&single.to_string_lossy(), "", &tags, "sharded"),
            Some(d.path().join("mymodel-fp8_e4m3-simple.safetensors"))
        );
        // sharded_folder, single mode → sibling .safetensors in parent.
        assert_eq!(
            suggest_comfy_output(&shard_dir.to_string_lossy(), "", &tags, "single"),
            Some(d.path().join("sharded_model-fp8_e4m3-simple.safetensors"))
        );
        // sharded_folder, sharded mode, empty → sibling directory.
        assert_eq!(
            suggest_comfy_output(&shard_dir.to_string_lossy(), "", &tags, "sharded"),
            Some(d.path().join("sharded_model-fp8_e4m3-simple"))
        );
        // explicit .safetensors output wins → None.
        assert_eq!(
            suggest_comfy_output(
                &shard_dir.to_string_lossy(),
                &single.to_string_lossy(),
                &tags,
                "single"
            ),
            None
        );
        // unusable input → None.
        assert_eq!(
            suggest_comfy_output(
                &d.path().join("nope.safetensors").to_string_lossy(),
                "",
                &tags,
                "single"
            ),
            None
        );
    }
}

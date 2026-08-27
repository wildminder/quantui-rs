//! Profiles + recent-jobs persistence (plan Phase 9.4 / step 8.4).
//!
//! Port of the reference `profiles_store.py` storage model, adapted to the
//! standalone CLI. One JSON document at `~/.quantui-rs/store.json` (overridable
//! via `QUANTUI_RS_CONFIG_DIR`), shape::
//!
//! ```json
//! {
//!   "profiles": {"<name>": { ...fields... }, ...},
//!   "recents": [ {"ts": ..., "family": ..., "method": ..., "output": ...,
//!                 "status": ..., "exit_code": 0, "duration_s": 12.3}, ... ]
//! }
//! ```
//!
//! `recents` is capped at [`MAX_RECENTS`] (20, newest first). Reads are
//! best-effort: a missing or corrupt file falls back to an empty store.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Env var that overrides the config directory (reference uses
/// `UNSLOTH_QUANT_CONFIG_DIR`; we use a CLI-specific name).
pub const CONFIG_ENV_VAR: &str = "QUANTUI_RS_CONFIG_DIR";
pub const STORE_FILENAME: &str = "store.json";
pub const MAX_RECENTS: usize = 20;

/// One finished quantization run (reference `RunRecord`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RunRecord {
    #[serde(default)]
    pub ts: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub method: String,
    #[serde(default)]
    pub output: String,
    /// `"success" | "failed" | "stopped"`.
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub exit_code: i32,
    #[serde(default)]
    pub duration_s: f64,
}

/// The whole store document.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Store {
    #[serde(default)]
    pub profiles: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub recents: Vec<RunRecord>,
}

impl Store {
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Resolve the config directory: env override wins, else `~/.quantui-rs`.
pub fn config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var(CONFIG_ENV_VAR) {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    home_dir().join(".quantui-rs")
}

/// Best-effort home directory (no external crate).
fn home_dir() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Resolve the JSON store path.
pub fn store_path(dir_override: Option<&Path>) -> PathBuf {
    let dir = match dir_override {
        Some(p) => p.to_path_buf(),
        None => config_dir(),
    };
    dir.join(STORE_FILENAME)
}

/// Load the full store document; corrupt/missing file → empty store.
pub fn load_store(config_dir: Option<&Path>) -> Store {
    let path = store_path(config_dir);
    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(_) => return Store::empty(),
    };
    match serde_json::from_str::<Store>(&raw) {
        Ok(s) => s,
        Err(_) => Store::empty(),
    }
}

/// Persist the whole store document (creates the directory if needed).
/// Returns an error on I/O failure — callers that must not crash wrap this.
pub fn save_store(store: &Store, config_dir: Option<&Path>) -> std::io::Result<()> {
    let path = store_path(config_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(store)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&path, json)
}

/// Prepend a recent run, capping the list at [`MAX_RECENTS`] (newest first).
pub fn add_recent(store: &mut Store, record: RunRecord) {
    store.recents.insert(0, record);
    store.recents.truncate(MAX_RECENTS);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn record(ts: &str) -> RunRecord {
        RunRecord {
            ts: ts.into(),
            family: "ctq".into(),
            method: "int8_block".into(),
            output: "/tmp/out.safetensors".into(),
            status: "success".into(),
            exit_code: 0,
            duration_s: 1.5,
        }
    }

    #[test]
    fn save_load_round_trip() {
        let dir = tmp();
        let mut store = Store::empty();
        store
            .profiles
            .insert("fast".into(), serde_json::json!({"scaling_mode": "block"}));
        add_recent(&mut store, record("2026-08-27T10:00:00"));

        save_store(&store, Some(dir.path())).unwrap();
        let loaded = load_store(Some(dir.path()));
        assert_eq!(loaded, store);
    }

    #[test]
    fn missing_file_loads_empty() {
        let dir = tmp();
        let loaded = load_store(Some(dir.path()));
        assert_eq!(loaded, Store::empty());
    }

    #[test]
    fn corrupt_file_recovers_to_empty() {
        let dir = tmp();
        std::fs::write(dir.path().join(STORE_FILENAME), "{not valid json!!").unwrap();
        let loaded = load_store(Some(dir.path()));
        assert_eq!(loaded, Store::empty());
    }

    #[test]
    fn partial_json_recovers_missing_fields() {
        let dir = tmp();
        // Valid JSON but missing `recents` → serde defaults fill it in.
        std::fs::write(
            dir.path().join(STORE_FILENAME),
            r#"{"profiles": {"a": {}}}"#,
        )
        .unwrap();
        let loaded = load_store(Some(dir.path()));
        assert!(loaded.profiles.contains_key("a"));
        assert!(loaded.recents.is_empty());
    }

    #[test]
    fn recents_capped_at_max_newest_first() {
        let mut store = Store::empty();
        for i in 0..(MAX_RECENTS + 5) {
            add_recent(&mut store, record(&format!("ts{i}")));
        }
        assert_eq!(store.recents.len(), MAX_RECENTS);
        // Newest first: the last-added record is at index 0.
        assert_eq!(store.recents[0].ts, format!("ts{}", MAX_RECENTS + 4));
    }
}

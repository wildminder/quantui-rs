//! Typed errors for safetensors IO. Never panics on malformed input.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("cannot resume {path:?}: file too small ({size} bytes, need >= 8)")]
    TooSmallToResume { path: PathBuf, size: u64 },

    #[error("cannot resume {path:?}: invalid header slot {slot} (must be {align}-aligned and within 8..={max})")]
    InvalidSlot {
        path: PathBuf,
        slot: u64,
        align: usize,
        max: u64,
    },

    #[error("{path:?}: header JSON parse failed: {message}")]
    HeaderJson { path: PathBuf, message: String },

    #[error("{path:?}: header entry {name:?} is not a valid tensor descriptor: {message}")]
    BadTensorEntry {
        path: PathBuf,
        name: String,
        message: String,
    },

    #[error("{path:?}: unknown dtype string {dtype:?}")]
    UnknownDtype { path: PathBuf, dtype: String },

    #[error("{path:?}: data_offsets [{start}, {end}] exceed data region (len {data_len})")]
    OffsetsOutOfRange {
        path: PathBuf,
        start: u64,
        end: u64,
        data_len: u64,
    },

    #[error("{path:?}: tensor byte length {len} does not match shape/dtype product {expected}")]
    SizeMismatch {
        path: PathBuf,
        len: u64,
        expected: u64,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

//! Memory-mapped safetensors reader with lazy per-tensor zero-copy views.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use super::error::{Error, Result};
use super::header::{Header, TensorInfo};
use super::HEADER_ALIGN;

/// Opened, memory-mapped safetensors file. Tensor data is accessed as
/// zero-copy byte slices — Phase 2 kernels take `&[u8]` directly.
pub struct SafetensorsReader {
    path: std::path::PathBuf,
    header: Header,
    slot: u64,
    mmap: Mmap,
}

impl SafetensorsReader {
    /// Open and parse the file. Validates offsets against the actual file size.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        // SAFETY: mmap of a read-only file handle; no concurrent mutation is
        // expected in our CLI usage (single-writer model).
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let size = mmap.len() as u64;

        if size < 8 {
            return Err(Error::TooSmallToResume {
                path: path.to_path_buf(),
                size,
            });
        }
        let slot = u64::from_le_bytes(mmap[0..8].try_into().expect("8 bytes"));
        if slot < HEADER_ALIGN as u64 || slot > size - 8 || (slot % HEADER_ALIGN as u64) != 0 {
            return Err(Error::InvalidSlot {
                path: path.to_path_buf(),
                slot,
                align: super::HEADER_ALIGN,
                max: size - 8,
            });
        }
        let raw = &mmap[8..8 + slot as usize];
        // Trailing spaces are valid JSON whitespace; rstrip then parse
        // (mirrors reference `json.loads(raw.rstrip(b" "))`).
        let trimmed = trim_trailing_spaces(raw);
        let header = Header::parse_json_bytes(trimmed, path)?;
        header.validate(path, size - 8 - slot)?;
        Ok(Self {
            path: path.to_path_buf(),
            header,
            slot,
            mmap,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Byte offset where the tensor data region begins.
    pub fn data_start(&self) -> u64 {
        8 + self.slot
    }

    /// Raw bytes of one tensor. Zero-copy view into the mmap.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.header.get(name).ok_or_else(|| Error::BadTensorEntry {
            path: self.path.clone(),
            name: name.to_string(),
            message: "tensor not found".into(),
        })?;
        let (start, end) = info.data_offsets;
        let base = self.data_start() as usize;
        Ok(&self.mmap[base + start as usize..base + end as usize])
    }

    /// Iterate all tensors as (name, info, bytes).
    pub fn iter_tensors(&self) -> impl Iterator<Item = (&String, &TensorInfo, &[u8])> {
        let base = self.data_start() as usize;
        self.header.iter().map(move |(name, info)| {
            let (s, e) = info.data_offsets;
            (name, info, &self.mmap[base + s as usize..base + e as usize])
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn trim_trailing_spaces(bytes: &[u8]) -> &[u8] {
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1] == b' ' {
        end -= 1;
    }
    &bytes[..end]
}

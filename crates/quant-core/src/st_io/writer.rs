//! Incremental, resumable safetensors writer.
//!
//! Faithful port of reference `IncrementalSafetensorsWriter`:
//! - `open_new`: truncate + write empty header slot (default 64 KiB, 8-aligned)
//! - `open_resume`: parse existing slot, position data cursor at end
//! - `add_tensor`: append data at cursor; rewrite header slot after each add
//!   (Python default `flush_every=1`)
//! - Slot growth: when the serialized header outgrows the slot, the whole file
//!   is rewritten with a doubled slot (data preserved)
//! - `finalize`: last header rewrite + flush + fsync

use std::fs::{File, OpenOptions};
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};

use serde_json::Map;

use super::error::{Error, Result};
use super::header::{Header, TensorInfo};
use super::HEADER_ALIGN;
use crate::dtype::DType;

pub struct IncrementalWriter {
    path: PathBuf,
    fh: Option<File>,
    slot: u64,
    data_len: u64,
    header: Header,
    done: Vec<String>,
}

impl IncrementalWriter {
    /// Start a new file (truncates any existing content). Mirrors Python
    /// `open(path, "w")` defaults: initial_slot = 64 KiB, flush_every = 1.
    pub fn open_new(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_new_with(path, 1 << 16, None)
    }

    pub fn open_new_with(
        path: impl AsRef<Path>,
        initial_slot: usize,
        metadata: Option<Map<String, serde_json::Value>>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let slot: u64 = ((initial_slot.max(HEADER_ALIGN) / HEADER_ALIGN) * HEADER_ALIGN) as u64;
        #[allow(unused_mut)]
        let mut header = Header::default();
        if let Some(meta) = &metadata {
            header.set_metadata(meta.clone());
        }
        let mut w = Self {
            path,
            fh: None,
            slot,
            data_len: 0,
            header,
            done: Vec::new(),
        };
        // Truncate and write the initial empty-slot header.
        let file = File::create(&w.path).map_err(|source| Error::Io {
            path: w.path.clone(),
            source,
        })?;
        w.fh = Some(file);
        w.write_header_slot()?;
        Ok(w)
    }

    /// Resume an existing incremental file. Mirrors Python `open(path, "a")`:
    /// parses the header slot, records existing tensors as done, positions the
    /// data cursor at the current end of the data region.
    pub fn open_resume(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let size = std::fs::metadata(&path)
            .map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?
            .len();
        if size < 8 {
            return Err(Error::TooSmallToResume {
                path: path.clone(),
                size,
            });
        }

        let mut raw_file = File::open(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        use std::io::Read;
        let mut len_buf = [0u8; 8];
        raw_file
            .read_exact(&mut len_buf)
            .map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
        let slot = u64::from_le_bytes(len_buf);
        if slot < HEADER_ALIGN as u64 || slot > size - 8 || (slot % HEADER_ALIGN as u64) != 0 {
            return Err(Error::InvalidSlot {
                path: path.clone(),
                slot,
                align: HEADER_ALIGN,
                max: size - 8,
            });
        }
        let mut raw = vec![0u8; slot as usize];
        raw_file.read_exact(&mut raw).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        drop(raw_file);

        // rstrip spaces then parse (reference does json.loads(raw.rstrip(b" "))).
        let trimmed: &[u8] = {
            let mut end = raw.len();
            while end > 0 && raw[end - 1] == b' ' {
                end -= 1;
            }
            &raw[..end]
        };
        let header = Header::parse_json_bytes(trimmed, &path)?;

        let on_disk = size.saturating_sub(8 + slot);

        // Reconcile the header against the bytes actually on disk. A crashed
        // writer may have flushed a header that lists tensors whose payloads
        // were only partially written (or not written at all) — e.g. a golden
        // file truncated mid-data-section. Keep only the contiguous prefix of
        // tensors whose payloads fully fit within `on_disk`; drop the rest so
        // they are re-appended on resume. This mirrors the real crash contract
        // (the last flushed header lists only completed tensors) and makes
        // resume well-defined even from a truncated file. Tensors are appended
        // contiguously, so the kept prefix ends exactly at the re-append cursor
        // and reproduces the original offsets byte-for-byte.
        let mut kept: Vec<(String, TensorInfo)> = Vec::new();
        let mut cursor = 0u64;
        for (name, info) in header.iter() {
            let (start, end) = info.data_offsets;
            if start == cursor && end <= on_disk {
                cursor = end;
                kept.push((name.clone(), info.clone()));
            } else {
                break;
            }
        }
        let mut reconciled = Header::default();
        if let Some(meta) = header.metadata() {
            reconciled.set_metadata(meta.clone());
        }
        for (name, info) in &kept {
            reconciled.insert(name.clone(), info.clone());
        }
        let data_len = cursor;
        let done: Vec<String> = kept.iter().map(|(n, _)| n.clone()).collect();

        let fh = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
        // Discard any partially-written trailing bytes so re-appends start from
        // a clean cursor (file length = header slot + kept data prefix).
        fh.set_len(8 + slot + data_len).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let mut w = Self {
            path,
            fh: Some(fh),
            slot,
            data_len,
            header: reconciled,
            done,
        };
        // Mirror reference `_resume`: seek to end so appends land after existing data.
        if let Some(f) = w.fh.as_mut() {
            f.seek(std::io::SeekFrom::End(0))
                .map_err(|source| Error::Io {
                    path: w.path.clone(),
                    source,
                })?;
        }
        Ok(w)
    }

    /// Byte offset where the tensor data region begins.
    pub fn data_start(&self) -> u64 {
        8 + self.slot
    }

    /// Tensors already present (from this session or a resumed one).
    pub fn done(&self) -> &[String] {
        &self.done
    }

    fn serialize_header(&self) -> Vec<u8> {
        self.header.serialize_json()
    }

    fn write_header_slot(&mut self) -> Result<()> {
        // Reference flushes buffered tensor data BEFORE growing the slot,
        // because growth reopens with "wb" (truncate). Our File handle writes
        // are unbuffered (no BufWriter), matching that ordering requirement.
        let hdr = self.serialize_header();
        if hdr.len() as u64 > self.slot {
            self.grow_slot(hdr.len())?;
        }
        let padded_len = self.slot as usize;
        let mut buf = Vec::with_capacity(8 + padded_len);
        buf.extend_from_slice(&(self.slot).to_le_bytes());
        buf.extend_from_slice(&hdr);
        buf.resize(8 + padded_len, b' ');

        let f = self.fh.as_mut().expect("file open");
        f.seek(std::io::SeekFrom::Start(0))
            .and_then(|_| f.write_all(&buf))
            .and_then(|_| f.seek(std::io::SeekFrom::End(0)))
            .map_err(|source| Error::Io {
                path: self.path.clone(),
                source,
            })?;
        Ok(())
    }

    /// Rewrite the whole file with a larger header slot (rare path).
    fn grow_slot(&mut self, needed: usize) -> Result<()> {
        let mut new_slot = self.slot;
        while new_slot < (needed + HEADER_ALIGN) as u64 {
            new_slot *= 2;
        }
        new_slot = (new_slot / HEADER_ALIGN as u64) * HEADER_ALIGN as u64;

        // Read existing data region.
        let old_data_start = (8 + self.slot) as usize;
        let data: Vec<u8> = {
            let mut f = File::open(&self.path).map_err(|source| Error::Io {
                path: self.path.clone(),
                source,
            })?;
            use std::io::Read;
            f.seek(std::io::SeekFrom::Start(old_data_start as u64))
                .map_err(|source| Error::Io {
                    path: self.path.clone(),
                    source,
                })?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).map_err(|source| Error::Io {
                path: self.path.clone(),
                source,
            })?;
            buf
        };

        self.slot = new_slot;
        // Reopen truncated ("wb" semantics), rewrite slot, re-append data.
        let f = File::create(&self.path).map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        self.fh = Some(f);
        self.write_header_slot()?;
        let ds = self.data_start();
        let f = self.fh.as_mut().expect("file open");
        f.seek(std::io::SeekFrom::Start(ds))
            .and_then(|_| f.write_all(&data))
            .and_then(|_| f.seek(std::io::SeekFrom::End(0)))
            .map_err(|source| Error::Io {
                path: self.path.clone(),
                source,
            })?;
        Ok(())
    }

    /// Append a tensor. Returns true if written, false if skipped (name already
    /// present — resume idempotency).
    pub fn add_tensor(
        &mut self,
        name: &str,
        dtype: DType,
        dtype_raw_override: Option<&str>,
        shape: &[u64],
        data: &[u8],
    ) -> Result<bool> {
        if self.done.iter().any(|n| n == name) {
            return Ok(false);
        }
        let start = self.data_len;
        let end = start + data.len() as u64;
        let raw = dtype_raw_override
            .unwrap_or_else(|| dtype.as_header_str())
            .to_string();

        // Write payload first (mirrors reference order: data then header update).
        let ds = self.data_start() + start;
        let f = self.fh.as_mut().expect("file open");
        f.seek(std::io::SeekFrom::Start(ds))
            .and_then(|_| f.write_all(data))
            .map_err(|source| Error::Io {
                path: self.path.clone(),
                source,
            })?;

        self.header.insert(
            name.to_string(),
            TensorInfo {
                dtype,
                dtype_raw: raw,
                shape: shape.to_vec(),
                data_offsets: (start, end),
            },
        );
        self.data_len = end;
        self.done.push(name.to_string());

        // Python default flush_every=1 → rewrite slot after every add.
        self.write_header_slot()?;
        Ok(true)
    }

    /// Final header rewrite + flush + fsync + close.
    pub fn finalize(mut self) -> Result<()> {
        if self.fh.is_none() {
            return Ok(());
        }
        self.write_header_slot()?;
        let f = self.fh.take().expect("file open");
        f.sync_all().map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        Ok(())
    }
}

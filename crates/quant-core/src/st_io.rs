//! Safetensors IO: header parsing, memmap reads, incremental fixed-slot writer.
//!
//! Faithful port of the reference `incremental_safetensors.py` +
//! `_align_header_to_8` from `comfy_quant_schema.py`.
//!
//! File format:
//! ```text
//! u64 LE header_len | JSON header (name -> {dtype, shape, data_offsets}) | tensor buffers
//! ```
//!
//! The writer materializes the header into a FIXED-SIZE slot (default 64 KiB,
//! 8-aligned, doubled when outgrown) so it can be rewritten in place while
//! tensor data appends at the end — a crash mid-run always leaves a loadable
//! prefix, and resume re-reads the slot to reconstruct state.

pub mod error;
pub mod header;
pub mod reader;
pub mod writer;

pub use error::{Error, Result};
pub use header::Header;
pub use reader::SafetensorsReader;
pub use writer::IncrementalWriter;

/// Slot alignment used by both the reference writer and this port.
pub const HEADER_ALIGN: usize = 8;

/// Pad a serialized safetensors JSON header with spaces so the following
/// data section begins at an 8-byte-aligned offset.
///
/// Verbatim port of `_align_header_to_8` from the reference
/// `quantui/comfy_quant_schema.py`:
/// `pad = (8 - len(header) % 8) % 8; header + b" " * pad`.
///
/// Used by the reference one-shot writer (`write_safetensors`) — the ctq
/// golden outputs are written this way (minimal space padding, verified in
/// `tests/phase4_header_align.rs`). The incremental writer does NOT use it:
/// it materializes the header into a fixed pre-aligned slot instead.
pub fn align_header_to_8(header_bytes: &[u8]) -> Vec<u8> {
    let pad = (HEADER_ALIGN - (header_bytes.len() % HEADER_ALIGN)) % HEADER_ALIGN;
    let mut out = Vec::with_capacity(header_bytes.len() + pad);
    out.extend_from_slice(header_bytes);
    out.resize(header_bytes.len() + pad, b' ');
    out
}

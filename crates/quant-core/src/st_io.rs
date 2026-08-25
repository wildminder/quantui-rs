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

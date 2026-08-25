# PHASE 1 NOTES — safetensors IO layer

**Date:** 2026-08-25
**Status:** COMPLETE — all gates green (build / clippy -D warnings / fmt --check / test: 10 passed)
**Note:** Implemented directly by team-lead after repeated provider 503 failures killed the engineer's runs mid-task.

## What was built

| File | Contents |
|---|---|
| `crates/quant-core/src/st_io/mod.rs` (was st_io.rs) | Module wiring, `HEADER_ALIGN=8` |
| `crates/quant-core/src/dtype.rs` | DType enum mirroring `_TORCH_DTYPE_TO_STR` + U16 tolerance; elem_size table |
| `crates/quant-core/src/st_io/error.rs` | Typed thiserror errors: TooSmallToResume, InvalidSlot, HeaderJson, BadTensorEntry, UnknownDtype, OffsetsOutOfRange, SizeMismatch |
| `crates/quant-core/src/st_io/header.rs` | Insertion-order-preserving header (serde_json preserve_order), compact serialize (`separators=(",",":")` semantics), `_align_header_to_8` port, validate() |
| `crates/quant-core/src/st_io/reader.rs` | memmap2 zero-copy reader with per-tensor `&[u8]` views |
| `crates/quant-core/src/st_io/writer.rs` | Faithful `IncrementalSafetensorsWriter` port: 64KiB default slot, in-place slot rewrite per add_tensor (flush_every=1), doubling grow_slot preserving data, resume-by-reopen |
| `crates/quant-core/tests/phase1_st_io.rs` | 10 tests |

## Test evidence

```
cargo test --workspace → 10 passed, 0 failed
```

Critical parity results:
- **writer_rebuilds_golden_outputs_byte_identically**: all 3 golden outputs rebuilt from manifest tensor order → **byte-for-byte identical** to Python-produced files (including the emergent 65536-byte slot on output files vs 384-byte minimal header on inputs)
- **kill_sim_resume_from_own_half_written_file**: crash-sim after tensor #1 → resume → equals golden exactly
- **kill_sim_resume_from_python_truncated_partial**: golden truncated mid-data-section with partial header → resume → equals original golden byte-for-byte. This proves cross-language crash-recovery compatibility.
- **header_roundtrip_exact_bytes_for_all_goldens**: parse→serialize == original header bytes (modulo space padding) for all 6 files
- **safetensors_crate_reads_our_writer_output**: official HF crate deserializes everything we write

## Format discoveries during the port

1. **Slot-size asymmetry confirmed live**: input goldens (written by plain safetensors lib) have compact headers (e.g. 384B); output goldens (incremental writer) use a preallocated 64KiB slot. The Rust port reproduces this naturally from the default initial_slot — no hardcoding.
2. **JSON float formatting caveat**: reference headers contain only strings/integers (no floats), so serde_json formatting matches Python's for our corpus. If metadata ever carries floats, revisit.
3. **Resume contract detail**: Python clamps negative data_len to 0 when file < 8+slot; Rust mirrors via saturating_sub.
4. **U16 elem_size = 2 bytes** (uint16), not 4 — caught by the UINT16 tolerance test.
5. Reference `add_tensor` writes payload BEFORE updating the header — order preserved in the port so a crash between the two leaves the old header valid (payload orphaned but harmless).

## Deviations from plan
None material. `dtype_raw_override` parameter added to `add_tensor` so callers can pass through exotic dtype strings verbatim (needed for byte-parity of passthrough tensors).

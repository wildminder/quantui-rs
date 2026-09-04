//! quant-core: core library for the quantui-rs CLI.
//!
//! Module layout mirrors plan §3 (docs/plans/2026-08-25-quantui-rust-rewrite-plan.md).
//! All phases have landed (master plan Phases 0-11 + the all-formats plan
//! Phases A-E), so every module below is fully implemented — nothing is
//! stubbed or gated behind a phase marker any more.

pub mod comfy_schema;
pub mod convrot;
pub mod dtype;
pub mod manifest;
pub mod quant;
pub mod quant_fp8;
pub mod quant_mxfp8;
pub mod quant_nvfp4;
pub mod stream;
// Enabled early: Phase 1 (safetensors IO) scaffolding created in Phase 0 per spec.
pub mod bias_correction;
pub mod discover;
// IQ-family lattice infrastructure (Unsloth plan Phase 4.3 IQ slice).
// gguf_iq_grid is the runtime port (init/kmap/neighbors);
// gguf_iq_tables_data is generated — see tools/extract_iq_tables.py.
pub mod gguf_convert;
mod gguf_iq_grid;
pub mod gguf_iq_quants;
mod gguf_iq_tables_data;
pub mod gguf_names;
pub mod gguf_quants;
pub mod gguf_recipe;
pub mod gguf_registry;
pub mod gguf_verify;
pub mod imatrix;
pub mod llama_policy;
pub mod st_io;
pub mod torch_rng;
pub mod validator;

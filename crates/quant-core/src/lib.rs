//! quant-core: core library for the quantui-rs CLI.
//!
//! Module layout mirrors plan §3 (docs/plans/2026-08-25-quantui-rust-rewrite-plan.md).
//! Only modules needed for the current phase are enabled; the rest are stubbed out
//! with TODO(phase-N) markers and will be uncommented as phases land.

pub mod comfy_schema;
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
pub mod gguf_convert;
pub mod gguf_registry;
pub mod st_io;
pub mod torch_rng;
pub mod validator;

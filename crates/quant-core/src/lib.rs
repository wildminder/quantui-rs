//! quant-core: core library for the quantui-rs CLI.
//!
//! Module layout mirrors plan §3 (docs/plans/2026-08-25-quantui-rust-rewrite-plan.md).
//! All phases have landed (master plan Phases 0-11 + the all-formats plan
//! Phases A-E), so every module below is fully implemented — nothing is
//! stubbed or gated behind a phase marker any more.

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
pub mod llama_policy;
pub mod st_io;
pub mod torch_rng;
pub mod validator;
